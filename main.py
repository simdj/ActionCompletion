from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import logging
import argparse
import copy

# from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset


from utils import progress_bar, set_logging_defaults
from vidaug import augmentors as va

from datasets.video_to_frames import UCF101Dataset
from datasets.rgbd_ac import RGBD_AC_Dataset

from models.encoder import FrameSequenceEncoder
from models.alignment import batch_get_alignment
from models.memory_module import MemoryModule
from baseline import BaseLineClassifier

from config import CONFIG

import random


def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification')
    parser.add_argument('--data', type=str, default='rgbd_ac', help='ucf101/hmdb51/ugbd_ac/self')

    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--backbone', type=str, default='resnet50', help='vgg/resnet/c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    # parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--sgpu', type=int, default=0, help='GPU id')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPU')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')

    parser.add_argument('--log', type=str, default='./log/', help='log directory')
    parser.add_argument('--log_file_name', type=str, default='log.csv', help='log file name')    
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    # parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--bs', type=int, default=4, help='mini-batch size')
    
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    # parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    # parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--cnn_freeze', type=bool, default=True, help='freeze cnn encoder')

    parser.add_argument('--memory_start_epoch', type=int, default=10, help='epoch when memory moddule is used')
    parser.add_argument('--ref_usage_cnt', type=int, default=5, help='number of reference memory to use')

    args = parser.parse_args()
    return args


def make_batch(samples):
	inputs = [sample[0] for sample in samples]
	actions = [sample[1] for sample in samples]
	moments = [sample[2] for sample in samples]

	padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
	return {
		'frame_seq': padded_inputs.contiguous(),
		'action': torch.stack(actions).contiguous(),
		'moment': torch.stack(moments).contiguous()
	}



def train(model, device, loader, optimizer, loss_fn, epoch):
	cnn_encoder, rnn_decoder = model
	cnn_encoder.train()
	rnn_decoder.train()

	loss_list = []
	acc_list = []
	score_list = []

	accum_loss = 0.0
	accum_acc = 0.0
	accum_mae = 0.0
	time_epoch_start = time.time()
	for batch_idx, data in enumerate(loader):
		time_batch_start = time.time()
		batch_frame_seq = data['frame_seq'].to(device)
		batch_action = data['action']
		batch_moment = data['moment']

		complete_mask = batch_moment>0
		label = (batch_moment>0).float().to(device)
		# label = torch.tensor(batch_moment>0, dtype=torch.float).to(device).detach().item()
		# data ready 

		optimizer.zero_grad()
		# feed forward
		cnn_feat_seq = cnn_encoder(batch_frame_seq)
		pred = rnn_decoder(cnn_feat_seq)

		loss = loss_fn(pred, label)

		loss.backward()
		optimizer.step()

		loss_value = loss.item()
		err = torch.abs(pred-label)
		acc = torch.mean((err<0.5).float()).detach().item()		
		mae_score = torch.mean(err).detach().item()


		accum_loss += loss_value
		accum_acc += acc
		accum_mae += mae_score
		
		loss_list.append(loss_value)
		acc_list.append(acc)
		score_list.append(mae_score)

		progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f | ACC: %.3f' 
			% ( accum_loss/(batch_idx+1), accum_mae/(batch_idx+1), accum_acc/(batch_idx+1))
		)
		
	
	logger = logging.getLogger('train')
	logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} ACC: {:.2f}'.format(
		epoch, time.time() - time_epoch_start, np.mean(loss_list), np.mean(score_list), np.mean(acc_list)
	))
def train_with_memory(memory, model, device, loader, optimizer, loss_fn, epoch):
	cnn_encoder, rnn_decoder = model
	cnn_encoder.train()
	rnn_decoder.train()

	
	loss_list = []
	memory_loss_list = []
	acc_list = []
	score_list = []

	accum_loss = 0.0
	accum_memory_loss = 0.0
	accum_acc = 0.0
	accum_mae = 0.0


	time_epoch_start = time.time()
	for batch_idx, data in enumerate(loader):
		time_batch_start = time.time()
		batch_frame_seq = data['frame_seq'].to(device)
		batch_action = data['action']
		batch_moment = data['moment']

		complete_mask = batch_moment>0
		label = (batch_moment>0).float().to(device)
		# label = torch.tensor(batch_moment>0, dtype=torch.float).to(device).detach().item()
		# data ready 

		optimizer.zero_grad()
		# feed forward
		cnn_feat_seq = cnn_encoder(batch_frame_seq)
		pred = rnn_decoder(cnn_feat_seq)
		loss = loss_fn(pred, label)


		seq_len = cnn_feat_seq.size(1)
		feat_dim = cnn_feat_seq.size(2)
		

		# get memory (action_class_idx)
		# extract corresponing memory cnn_feat_seq given query (cnn_feat_seq)
		# loss_with_memory = loss_fn(pred(corresponing_cnn_feat_seq), label)
		with torch.no_grad():
			nb_memory = args.ref_usage_cnt
			reference_memory_cnn_feat_seq = memory.get_batch_memory(batch_action, nb_memory=nb_memory).to(device)
			# # reference_memory_cnn_feat_seq shape: (batch_size x nb_memory x seq_len x dim)
			# reference_memory_cnn_feat_seq = reference_memory_cnn_feat_seq[:,0,:,:] # only one memory
			# # reference_memory_cnn_feat_seq shape: (batch_size x seq_len x dim)
			reference_memory_cnn_feat_seq = reference_memory_cnn_feat_seq.reshape(-1,seq_len,feat_dim)
			
		# TODO - broadcasting cnn_feat_seq 		
		batch_cnn_feat_seq = cnn_feat_seq.unsqueeze(1).repeat(1,nb_memory,1,1) # current impl : copy		
		batch_cnn_feat_seq = batch_cnn_feat_seq.reshape(-1,seq_len,feat_dim)

		aligned_label = label.unsqueeze(1).repeat(1,nb_memory).reshape(-1)



		aligned_cnn_feat_seq_by_memory= batch_get_alignment(batch_cnn_feat_seq, reference_memory_cnn_feat_seq.detach())
		aligned_pred = rnn_decoder(aligned_cnn_feat_seq_by_memory)
		loss_with_memory = loss_fn(aligned_pred, aligned_label)
				
		# total_loss
		loss += loss_with_memory
		memory_loss_value = loss_with_memory.item()
		accum_memory_loss += memory_loss_value
		memory_loss_list.append(memory_loss_value)




		loss.backward()
		optimizer.step()

		loss_value = loss.item()
		err = torch.abs(pred-label)
		acc = torch.mean((err<0.5).float()).detach().item()		
		mae_score = torch.mean(err).detach().item()


		accum_loss += loss_value
		accum_acc += acc
		accum_mae += mae_score
		
		loss_list.append(loss_value)
		acc_list.append(acc)
		score_list.append(mae_score)

		progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f | ACC: %.3f | accum_memory_loss: %.3f ' 
			% ( accum_loss/(batch_idx+1), accum_mae/(batch_idx+1), accum_acc/(batch_idx+1), accum_memory_loss/(batch_idx+1) )
		)
		
	
	logger = logging.getLogger('train')
	logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.3f} MAE: {:.2f} ACC: {:.2f} memory loss: {:.3f}'.format(
		epoch, time.time() - time_epoch_start, np.mean(loss_list), np.mean(score_list), np.mean(acc_list), np.mean(memory_loss_list)
	))

def val_with_memory(memory, model, device, loader, optimizer, loss_fn, epoch):
	cnn_encoder, rnn_decoder = model
	cnn_encoder.eval()
	rnn_decoder.eval()

	
	loss_list = []
	acc_list = []
	score_list = []

	aligned_mae_list = []
	aligned_acc_list = []

	
	val_aligned_acc = 0.0
	val_aligned_mae = 0.0

	val_loss = 0.0
	val_acc = 0.0
	val_mae = 0.0
	time_epoch_start = time.time()
	with torch.no_grad():
		for batch_idx, data in enumerate(loader):
			time_batch_start = time.time()
			batch_frame_seq = data['frame_seq'].to(device)
			batch_action = data['action']
			batch_moment = data['moment']

			complete_mask = batch_moment>0
			label = (batch_moment>0).float().to(device)
			# data ready 

			# # # optimizer.zero_grad()
			# feed forward
			cnn_feat_seq = cnn_encoder(batch_frame_seq)
			pred = rnn_decoder(cnn_feat_seq)

			loss = loss_fn(pred, label)

			# # # loss.backward()
			# # # optimizer.step()
			
			seq_len = cnn_feat_seq.size(1)
			feat_dim = cnn_feat_seq.size(2)
			# get memory (action_class_idx)
			# extract corresponing memory cnn_feat_seq given query (cnn_feat_seq)
			# loss_with_memory = loss_fn(pred(corresponing_cnn_feat_seq), label)
			nb_memory = args.ref_usage_cnt
			reference_memory_cnn_feat_seq = memory.get_batch_memory(batch_action, nb_memory=nb_memory).to(device)
			# # reference_memory_cnn_feat_seq shape: (batch_size x nb_memory x seq_len x dim)
			# reference_memory_cnn_feat_seq = reference_memory_cnn_feat_seq[:,0,:,:] # only one memory
			# # reference_memory_cnn_feat_seq shape: (batch_size x seq_len x dim)
			reference_memory_cnn_feat_seq = reference_memory_cnn_feat_seq.reshape(-1,seq_len,feat_dim)
				
			# TODO - broadcasting cnn_feat_seq 		
			batch_cnn_feat_seq = cnn_feat_seq.unsqueeze(1).repeat(1,nb_memory,1,1) # current impl : copy		
			batch_cnn_feat_seq = batch_cnn_feat_seq.reshape(-1,seq_len,feat_dim)

			aligned_label = label.unsqueeze(1).repeat(1,nb_memory).reshape(-1)



			aligned_cnn_feat_seq_by_memory= batch_get_alignment(batch_cnn_feat_seq, reference_memory_cnn_feat_seq.detach())
			aligned_pred = rnn_decoder(aligned_cnn_feat_seq_by_memory)

			err_aligned = torch.abs(aligned_label-aligned_pred)
			mae_aligned = torch.mean(err_aligned).detach().item()
			acc_aligned = torch.mean((err_aligned<0.5).float()).detach().item()

			val_aligned_acc += acc_aligned
			val_aligned_mae += mae_aligned
			aligned_mae_list.append(mae_aligned)
			aligned_acc_list.append(acc_aligned)



			loss_value = loss.item()
			err = torch.abs(pred-label)
			acc = torch.mean((err<0.5).float()).detach().item()
			
			mae_score = torch.mean(err).detach().item()

			val_loss += loss_value
			val_acc += acc
			val_mae += mae_score


			
			loss_list.append(loss_value)
			acc_list.append(acc)
			score_list.append(mae_score)

			progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f | ACC: %.3f | MAE(align): %.3f | ACC(align): %.3f ' 
				% ( val_loss/(batch_idx+1), val_mae/(batch_idx+1), val_acc/(batch_idx+1) , val_aligned_mae/(batch_idx+1), val_aligned_acc/(batch_idx+1) )
			)
	
	# final log
	
	logger = logging.getLogger('val')
	logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} ACC: {:.2f} MAE(align): {:.2f} ACC(align): {:.2f}'.format(
		epoch, time.time() - time_epoch_start, np.mean(loss_list), np.mean(score_list), np.mean(acc_list), np.mean(aligned_mae_list), np.mean(aligned_acc_list)
	))

	final_score = np.mean(score_list) # MAE

	return final_score


def val(model, device, loader, optimizer, loss_fn, epoch):
	cnn_encoder, rnn_decoder = model
	cnn_encoder.eval()
	rnn_decoder.eval()

	loss_list = []
	acc_list = []
	score_list = []

	val_loss = 0.0
	val_acc = 0.0
	val_mae = 0.0
	time_epoch_start = time.time()
	with torch.no_grad():
		for batch_idx, data in enumerate(loader):
			time_batch_start = time.time()
			batch_frame_seq = data['frame_seq'].to(device)
			batch_action = data['action']
			batch_moment = data['moment']

			complete_mask = batch_moment>0
			label = (batch_moment>0).float().to(device)
			# data ready 

			# # # optimizer.zero_grad()
			# feed forward
			cnn_feat_seq = cnn_encoder(batch_frame_seq)
			pred = rnn_decoder(cnn_feat_seq)

			loss = loss_fn(pred, label)

			# # # loss.backward()
			# # # optimizer.step()


			loss_value = loss.item()
			err = torch.abs(pred-label)
			acc = torch.mean((err<0.5).float()).detach().item()
			
			mae_score = torch.mean(err).detach().item()

			val_loss += loss_value
			val_acc += acc
			val_mae += mae_score
			
			loss_list.append(loss_value)
			acc_list.append(acc)
			score_list.append(mae_score)

			progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f | ACC: %.3f' 
				% ( val_loss/(batch_idx+1), val_mae/(batch_idx+1), val_acc/(batch_idx+1) )
			)
	
	# final log
	
	logger = logging.getLogger('val')
	logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} ACC: {:.2f}'.format(
		epoch, time.time() - time_epoch_start, np.mean(loss_list), np.mean(score_list), np.mean(acc_list)
	))

	final_score = np.mean(score_list) # MAE

	return final_score

def save_checkpoint(ckpt_path, model, optimizer, epoch, performance):
	print("Saving..")
	cnn_encoder, rnn_decoder = model
	state = {
		# 'config': config.samples,
		'encoder': cnn_encoder.state_dict(),
		'decoder': rnn_decoder.state_dict(),
		'optimizer': optimizer.state_dict(),
		'performance': performance,
		'epoch' : epoch,
		'rng_state' : torch.get_rng_state()
	}
	torch.save(state, ckpt_path)

def load_checkpoint(ckpt_path, model, optimizer):
	print("==> Resuming from ckpt ", ckpt_path)
	ckpt = torch.load(ckpt_path)
	cnn_encoder, rnn_decoder = model
	cnn_encoder.load_state_dict(ckpt['encoder'])
	rnn_decoder.load_state_dict(ckpt['decoder'])
	optimizer.load_state_dict(ckpt['optimizer'])
	start_epoch = ckpt['epoch']
	ckpt_performance = ckpt['performance']
	ckpt_rng_state = ckpt['rng_state']
	torch.set_rng_state(ckpt_rng_state)
	return model, optimizer, start_epoch, ckpt_performance, ckpt_rng_state



def get_video_augments():

	sometimes50 = lambda aug: va.Sometimes(0.5, aug)
	sometimes10 = lambda aug: va.Sometimes(0.1, aug)

	video_augments = va.Sequential([
		va.RandomCrop(size=(224,224)),
		va.RandomRoate(degrees=10),
		sometimes50(va.HorizontalFlip()),
		sometimes50(va.InvertColor()),
		# sometimes10(va.RandomShear(x=10,y=10)),
		# sometimes10(va.RandomTranslate(x=10,y=10)),
		sometimes10(va.GaussianBlur(sigma=1)),
		sometimes10(va.Pepper(ratio=99)),
		sometimes10(va.Salt(ratio=99))
		
	])
	# List of augmenters: https://github.com/okankop/vidaug
	# Affine
	# 	* RandomRotate
	#     * RandomResize
	#     * RandomTranslate
	#     * RandomShear
	# Crop
		# CenterCrop
		# CornerCrop
		# RandomCrop
	# Flip
	# Geometric
	# * GaussianBlur
	#    * ElasticTransformation
	#    * PiecewiseAffineTransform
	#    * Superpixel
	# Intensity
	# 	InvertColor
	# 	* Add
	# 	* Multiply
	# 	* Pepper
	# 	* Salt
	# temporal
	return video_augments


if __name__ == "__main__":
	args = parse_args()
	print(vars(args))

	logdir = os.path.join(args.log)
	set_logging_defaults(logdir, args)
	logger = logging.getLogger('main')
	logname = os.path.join(logdir, args.log_file_name)


	use_cuda = torch.cuda.is_available()
	device = "cpu"
	if use_cuda:
		print('gpu cnt', torch.cuda.device_count())
		gpu_idx = args.sgpu
		device = torch.device('cuda:'+str(gpu_idx))
	if torch.cuda.device_count()>1:
		print('device', device)
	# device="cpu"
	model_name = args.backbone
	batch_size = args.bs
	num_epoch = args.num_epoch
	cnn_freeze = args.cnn_freeze
	num_workers = args.num_workers
	
	input_size = CONFIG.IMAGE_SIZE
	cnn_embed_dim = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE


	frame_sequence_encoder = FrameSequenceEncoder(model_name=model_name, use_pretrained=True, feature_extract=cnn_freeze)
	completion_classifier = BaseLineClassifier(embed_dim=cnn_embed_dim)

	# if use_cuda and torch.cuda.device_count()>1:
	# 	frame_sequence_encoder = nn.DataParallel(frame_sequence_encoder)
	# 	completion_classifier = nn.DataParallel(completion_classifier)

	frame_sequence_encoder.to(device)
	completion_classifier.to(device)

	# data_augments = get_video_augments()
	data_augments = None
	data_transforms = {
		"train": transforms.Compose([
			# transforms.RandomResizedCrop(input_size),
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

		]),
		"val": transforms.Compose([
			transforms.Resize(input_size),
			transforms.CenterCrop(input_size),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}
	


	if args.data=='ucf101':
		class_idx_filename = 'completion_all_classInd.txt'
		# class_idx_filename = 'completion_blowing_classInd.txt'		
		train_dataset = UCF101Dataset("../data/ucf101", class_idx_filename=class_idx_filename
			, train=True, transforms_=data_transforms["train"], augments_=data_augments)
		test_dataset = UCF101Dataset("../data/ucf101", class_idx_filename=class_idx_filename
			, train=False, transforms_=data_transforms["val"])
	elif args.data=='rgbd_ac':
		class_idx_filename = 'completion_all_classInd.txt'
		# class_idx_filename = 'completion_open_classInd.txt'
		train_dataset = RGBD_AC_Dataset("../data/RGBD-AC", class_idx_filename=class_idx_filename
			, train=True, transforms_=data_transforms["train"], augments_=data_augments)
		test_dataset = RGBD_AC_Dataset("../data/RGBD-AC", class_idx_filename=class_idx_filename
			, train=False, transforms_=data_transforms["val"])
	elif args.data =='self':
		train_dataset = 
	else:
		raise ValueError('Specify data(ucf101/hmdb51/rgbd-ac')


	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers
		, shuffle=True, collate_fn=make_batch)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers
		,  shuffle=True, collate_fn=make_batch)
	
	
	model = [frame_sequence_encoder, completion_classifier]
	


	params_to_update = list(frame_sequence_encoder.parameters())+list(completion_classifier.parameters())
	
	# TODO reduction
	loss_fn = nn.BCELoss()

	optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)



	mm = MemoryModule(class_set=train_dataset.class_set, capacity_per_class=10)

	best_MAE = np.inf # MAE
	memory_using_start = args.memory_start_epoch
	for epoch in range(num_epoch):
		if epoch < memory_using_start:
			train(model, device, train_dataloader, optimizer, loss_fn, epoch)
			current_MAE = val(model, device, test_dataloader, optimizer, loss_fn, epoch)
		else:		
			mm.construct_positive_memory_fast( device, train_dataloader, model)
			train_with_memory(mm, model, device, train_dataloader, optimizer, loss_fn, epoch)
			current_MAE = val_with_memory(mm,model, device, test_dataloader, optimizer, loss_fn, epoch)
	
		
		
		if current_MAE < best_MAE:
			save_checkpoint(os.path.join(logdir, 'ckpt.t7'), model, optimizer, epoch, best_MAE)
			best_MAE = current_MAE





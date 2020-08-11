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

from datasets.video_to_frames import UCF101Dataset

from models.FeatureExtracter import CNNEncoder
# from models.alignment import Alignment
from models.memory_module import MemoryModule
from baseline import BaseLineClassifier

import random


def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--backbone', type=str, default='resnet', help='vgg/resnet/c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, default='./log/', help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--num_epoch', type=int, default=150, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=4, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--cnn_freeze', type=bool, default=True, help='freeze cnn encoder')
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



def train_given_memory(memory, model, device, loader, optimizer, loss_fn, epoch):
	cnn_encoder, rnn_decoder = model
	cnn_encoder.train()
	rnn_decoder.train()

	losses = []
	scores = []

	train_loss = 0.0
	train_mae = 0.0
	time_epoch_start = time.time()
	for batch_idx, data in enumerate(loader):
		time_batch_start = time.time()
		batch_frame_seq = data['frame_seq'].to(device)
		batch_action = data['action']
		batch_moment = data['moment']

		# complete_mask = batch_moment>0
		label = (batch_moment>0).float().to(device)
		






		optimizer.zero_grad()
		# feed forward
		cnn_feat_seq = cnn_feature_extracter(batch_frame_seq)
		pred = completion_classifier(cnn_feat_seq)

		loss = loss_fn(pred, label)

		loss.backward()
		optimizer.step()

		loss_val = loss.item()
		
		mae_score = torch.mean(torch.abs(pred-label)).item()

		train_loss += loss_val
		train_mae += mae_score
		
		losses.append(loss_val)
		scores.append(mae_score)

		progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f' 
			% ( train_loss/(batch_idx+1), train_mae/(batch_idx+1)))
		
	print('Epoch time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} '.format(
		time.time() - time_epoch_start, np.mean(losses), np.mean(scores)
	))
	logger = logging.getLogger('train')
	logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} '.format(
		epoch, time.time() - time_epoch_start, np.mean(losses), np.mean(scores)
	))



def train(model, device, loader, optimizer, loss_fn, epoch):
	cnn_encoder, rnn_decoder = model
	cnn_encoder.train()
	rnn_decoder.train()

	losses = []
	scores = []

	train_loss = 0.0
	train_mae = 0.0
	time_epoch_start = time.time()
	for batch_idx, data in enumerate(loader):
		time_batch_start = time.time()
		batch_frame_seq = data['frame_seq'].to(device)
		batch_action = data['action']
		batch_moment = data['moment']

		complete_mask = batch_moment>0
		label = (batch_moment>0).float().to(device)
		# label = torch.tensor(batch_moment>0, dtype=torch.float).to(device).detach()
		# data ready 

		optimizer.zero_grad()
		# feed forward
		cnn_feat_seq = cnn_feature_extracter(batch_frame_seq)
		pred = completion_classifier(cnn_feat_seq)

		loss = loss_fn(pred, label)

		loss.backward()
		optimizer.step()

		loss_val = loss.item()
		
		mae_score = torch.mean(torch.abs(pred-label)).item()

		train_loss += loss_val
		train_mae += mae_score
		
		losses.append(loss_val)
		scores.append(mae_score)

		progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f' 
			% ( train_loss/(batch_idx+1), train_mae/(batch_idx+1)))
		
	print('Epoch time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} '.format(
		time.time() - time_epoch_start, np.mean(losses), np.mean(scores)
	))
	logger = logging.getLogger('train')
	logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} '.format(
		epoch, time.time() - time_epoch_start, np.mean(losses), np.mean(scores)
	))

def val(model, device, loader, optimizer, loss_fn, epoch):
	cnn_encoder, rnn_decoder = model
	cnn_encoder.eval()
	rnn_decoder.eval()

	losses = []
	scores = []

	val_loss = 0.0
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
			cnn_feat_seq = cnn_feature_extracter(batch_frame_seq)
			pred = completion_classifier(cnn_feat_seq)

			loss = loss_fn(pred, label)

			# # # loss.backward()
			# # # optimizer.step()

			loss_val = loss.item()
			
			mae_score = torch.mean(torch.abs(pred-label)).item()

			val_loss += loss_val
			val_mae += mae_score
			
			losses.append(loss_val)
			scores.append(mae_score)

			progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f' 
				% ( val_loss/(batch_idx+1), val_mae/(batch_idx+1)))
	
	# final log
	print('Epoch time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} '.format(
		time.time() - time_epoch_start, np.mean(losses), np.mean(scores)
	))
	logger = logging.getLogger('val')
	logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} '.format(
		epoch, time.time() - time_epoch_start, np.mean(losses), np.mean(scores)
	))

	final_score = np.mean(scores) # MAE

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





if __name__ == "__main__":
	args = parse_args()
	print(vars(args))

	logdir = os.path.join(args.log)
	set_logging_defaults(logdir, args)
	logger = logging.getLogger('main')
	logname = os.path.join(logdir, 'log.csv')

	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda:0' if use_cuda else "cpu")
	# device="cpu"
	model_name = args.backbone
	batch_size = args.bs
	num_epoch = args.num_epoch
	cnn_freeze = args.cnn_freeze
	
	cnn_feature_extracter = CNNEncoder(model_name=model_name, use_pretrained=True, feature_extract=cnn_freeze).to(device)

	input_size = cnn_feature_extracter.input_size
	cnn_embed_dim = cnn_feature_extracter.cnn_embed_dim

	completion_classifier = BaseLineClassifier(embed_dim=cnn_embed_dim).to(device)


	data_transforms = {
		"train": transforms.Compose([
			transforms.RandomResizedCrop(input_size),
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
	


	

	train_dataset = UCF101Dataset("../data/ucf101", train=True, transforms_=data_transforms["train"])
	test_dataset = UCF101Dataset("../data/ucf101", train=False, transforms_=data_transforms["val"])

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_batch)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_batch)
	
	model = [cnn_feature_extracter, completion_classifier]
	params_to_update = list(cnn_feature_extracter.parameters())+list(completion_classifier.parameters())
	
	# TODO reduction
	loss_fn = nn.BCELoss()

	optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)



	mm = MemoryModule(nb_class=train_dataset.nb_class, capacity_per_class=2)

	best_MAE = np.inf # MAE
	for epoch in range(num_epoch):
		train(model, device, train_dataloader, optimizer, loss_fn, epoch)
		mm.construct_positive_memory_fast( device, train_dataloader, model)
		current_MAE = val(model, device, test_dataloader, optimizer, loss_fn, epoch)
		if current_MAE < best_MAE:
			save_checkpoint(os.path.join(logdir, 'ckpt.t7'), model, optimizer, epoch, best_MAE)





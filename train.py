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
from datasets.self_supervised_data import Self_Supervised_Dataset

from models.encoder import FrameSequenceEncoder
from models.alignment import batch_get_alignment
from models.memory_module import MemoryModule
# from baseline import BaseLineClassifier
from models.decoder import BaseLineClassifier

from config import CONFIG, parse_args, set_config_with_args

from easydict import EasyDict as edict
import random

def save_checkpoint(ckpt_path, model, optimizer, epoch, performance):
    print("Saving..", ckpt_path)
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


def get_dataloader(data_dir, task_spec=None):
    input_size = CONFIG.IMAGE_SIZE
    
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

    batch_size = CONFIG.TRAIN.BATCH_SIZE
    if CONFIG.DATA.DATASET=='ucf101':
        class_idx_filename = 'completion_all_classInd.txt'
        # class_idx_filename = 'completion_blowing_classInd.txt'        
        train_dataset = UCF101Dataset(data_dir, class_idx_filename=class_idx_filename, train=True, transforms_=data_transforms["train"])
        test_dataset  = UCF101Dataset(data_dir, class_idx_filename=class_idx_filename, train=False, transforms_=data_transforms["val"])
    elif CONFIG.DATA.DATASET=='rgbd_ac':
        class_idx_filename = 'completion_all_classInd.txt'
        # class_idx_filename = 'completion_open_classInd.txt'
        # data_dir : ../data/RGBD-AC
        train_dataset = RGBD_AC_Dataset(data_dir, class_idx_filename=class_idx_filename, train=True, transforms_=data_transforms["train"])
        test_dataset  = RGBD_AC_Dataset(data_dir, class_idx_filename=class_idx_filename, train=False, transforms_=data_transforms["val"])
    elif CONFIG.DATA.DATASET =='self':
        train_dataset = Self_Supervised_Dataset(data_dir, video_len=CONFIG.SELF_LEARN.VIDEO_LEN, train=True, transforms_=data_transforms['train'])
        test_dataset  = Self_Supervised_Dataset(data_dir, video_len=CONFIG.SELF_LEARN.VIDEO_LEN, train=False, transforms_=data_transforms['val'])
    else:
        raise ValueError('Specify data(ucf101/hmdb51/rgbd-ac/self')
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=True, collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS)

    return train_dataloader, test_dataloader


# def get_model_for_self_learning():

#     model_name  = CONFIG.MODEL.BASE_MODEL.NETWORK
#     cnn_freeze = CONFIG.MODEL.BASE_MODEL.FREEZE
#     cnn_embed_dim =CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE

#     frame_sequence_encoder = FrameSequenceEncoder(model_name=model_name, use_pretrained=True, base_feature_extract=cnn_freeze, embedder_feature_extract=False)
#     completion_classifier = BaseLineClassifier(embed_dim=cnn_embed_dim)

#     model_to_return = [frame_sequence_encoder, completion_classifier]
#     return model_to_return

# def get_model_for_transfer_learning():
#     model_name  = CONFIG.MODEL.BASE_MODEL.NETWORK
#     cnn_embed_dim =CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE

#     frame_sequence_encoder = FrameSequenceEncoder(model_name=model_name, use_pretrained=True, base_feature_extract=True, embedder_feature_extract=True )
#     completion_classifier = BaseLineClassifier(embed_dim=cnn_embed_dim)

#     model_to_return = [frame_sequence_encoder, completion_classifier]
#     return model_to_return

def get_model(base_freeze, embedder_freeze):
    model_name  = CONFIG.MODEL.BASE_MODEL.NETWORK
    cnn_embed_dim =CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE

    frame_sequence_encoder = FrameSequenceEncoder(model_name=model_name, use_pretrained=True, base_feature_extract=base_freeze, embedder_feature_extract=embedder_freeze)
    completion_classifier = BaseLineClassifier(embed_dim=cnn_embed_dim)

    model_to_return = [frame_sequence_encoder, completion_classifier]
    return model_to_return

def get_model_loss_optim(task_type):
    device = CONFIG.DEVICE
    
    # model
    model = None
    if task_type=='self':
        model = get_model(base_freeze=CONFIG.MODEL.BASE_MODEL.FREEZE, embedder_freeze=False)   
    elif task_type=='basic':
        model = get_model(base_freeze=CONFIG.MODEL.BASE_MODEL.FREEZE, embedder_freeze=False) 
    elif task_type=='finetune':
        model = get_model(base_freeze=True, embedder_freeze=False) 
    else:
        raise ValueError("[get_model_loss_optim] specify task_type, ")
    frame_sequence_encoder, completion_classifier = model
    
    frame_sequence_encoder.cuda()
    completion_classifier.cuda() 


    # define loss_fn
    loss_fn = nn.BCELoss()
    # loss_fn = loss_fn.cuda()




    # define optim
    params_to_update = list(frame_sequence_encoder.parameters())+list(completion_classifier.parameters())
    # optimizer = optim.Adam(params_to_update, lr=CONFIG.OPTIMIZER.LR, weight_decay=CONFIG.OPTIMIZER.WD)
    optimizer = optim.SGD(params_to_update, lr=CONFIG.OPTIMIZER.LR, momentum=CONFIG.OPTIMIZER.MOMENTUM, weight_decay=CONFIG.OPTIMIZER.WD)


    return model,loss_fn, optimizer






def train_one_epoch(model, device, loader, optimizer, loss_fn, epoch):
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
    
    # if CONFIG.test_data == None:
    #     for batch_idx, data in enumerate(loader):
    #         CONFIG.test_data = data
    #         break


    
    for batch_idx, data in enumerate(loader):
    # for batch_idx in range(10):
    #     data = CONFIG.test_data
        time_batch_start = time.time()
        



        batch_frame_seq = data['frame_seq'].to(device)
        # batch_frame_seq = data['frame_seq'].cuda()
        batch_action = data['action']
        batch_moment = data['moment']
        # print('batch_frame_seq.shape',batch_frame_seq.shape)
        complete_mask = batch_moment>0
        label = (batch_moment>0).float().to(device)
        # label = (batch_moment>0).float().cuda()


        # data ready 
        optimizer.zero_grad()
        # feed forward
        cnn_feat_seq = cnn_encoder(batch_frame_seq)
        pred = rnn_decoder(cnn_feat_seq)
        
        # optimize
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()

        # log
        loss_value = loss.item()
        err = torch.abs(pred-label)
        acc = torch.mean((err<0.5).float()).detach().item()     
        mae_score = torch.mean(err).detach().item()


        a=[batch_idx, label.detach().cpu().numpy(), pred.detach().cpu().numpy(), err.detach().cpu().numpy()]
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
            # batch_frame_seq = data['frame_seq'].to(device)
            batch_frame_seq = data['frame_seq'].cuda()
            batch_action = data['action']
            batch_moment = data['moment']

            complete_mask = batch_moment>0
            # label = (batch_moment>0).float().to(device)
            label = (batch_moment>0).float().cuda()
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




def do_learning(params):
    '''
    input: params: edict
    '''
    model, loss_fn, optimizer  = get_model_loss_optim(params.task_type)

    if params.pre_trained_ckpt_path:
        model, optimizer, start_epoch, _, _ = load_checkpoint(params.pre_trained_ckpt_path, model, optimizer)


    train_dataloader, test_dataloader = get_dataloader(data_dir='../data/ucf101', task_spec=None)

    # fill up params
    params.model = model
    
    params.loss_fn = loss_fn
    params.optimizer = optimizer

    params.train_dataloader = train_dataloader
    params.test_dataloader = test_dataloader

    best_MAE = np.inf 
    device = CONFIG.DEVICE
    for epoch in range(CONFIG.TRAIN.NUM_EPOCH):
        train_one_epoch(params.model, device, params.train_dataloader, params.optimizer, params.loss_fn, epoch)

        # current_MAE = val(params.model, device, params.test_dataloader, params.optimizer, params.loss_fn, epoch)
        
        # if current_MAE < best_MAE:
        #     save_checkpoint(params.save_ckpt_path, params.model, params.optimizer, epoch, best_MAE)
        #     best_MAE = current_MAE



def do_basic_learning(save_ckpt_path):
    params = edict()
    params.task_type='basic'
    params.pre_trained_ckpt_path = None
    params.save_ckpt_path = save_ckpt_path
    do_learning(params)




def do_self_learning(save_ckpt_path='./self_model.ckpt'):
    # dataloader
    # if CONFIG.DATA.DATASET != 'self':
    #     raise ValueError("[self training] invalid config and function call. check CONFIG.DATA.DATASET")
    params = edict()
    params.task_type='self'
    params.pre_trained_ckpt_path = None
    params.save_ckpt_path = save_ckpt_path
    do_learning(params)    
    


def do_finetune_learning(pre_trained_ckpt_path, save_ckpt_path):
    params = edict()
    params.task_type='finetune'
    params.pre_trained_ckpt_path = pre_trained_ckpt_path
    params.save_ckpt_path = save_ckpt_path
    do_learning(params)    
    



def experiment_self_learning():
    args = parse_args()
    set_config_with_args(args)    

    self_trained_model_ckpt_path = './self_model.ckpt'
    
    CONFIG.DATA.DATASET = 'self'
    print("experiment_self_learning")
    print(CONFIG)
    print("\n\n")
    do_self_learning(save_ckpt_path=self_trained_model_ckpt_path)
    


def experiment_basic():
    args = parse_args()
    set_config_with_args(args)    

    print("=======basic=======")
    print(CONFIG)
    original_dataset = CONFIG.DATA.DATASET
    single_model_ckpt_path = './final_model_single.ckpt'
    do_basic_learning(save_ckpt_path=single_model_ckpt_path)


def experiment_finetune():
    args = parse_args()
    set_config_with_args(args)    

    original_dataset = CONFIG.DATA.DATASET

    print("=======transfer======")

    self_trained_model_ckpt_path = './self_model.ckpt'
    transfered_model_ckpt_path = './final_model_transfer.ckpt'
    
    # CONFIG.DATA.DATASET = 'self'
    # print(CONFIG)
    # do_self_learning(save_ckpt_path=self_trained_model_ckpt_path)
    
    CONFIG.DATA.DATASET = original_dataset
    print(CONFIG)
    do_finetune_learning(pre_trained_ckpt_path=self_trained_model_ckpt_path, save_ckpt_path=transfered_model_ckpt_path)


def whole_process():
    args = parse_args()
    set_config_with_args(args)    

    original_dataset = CONFIG.DATA.DATASET

    print("=======transfer======")

    self_trained_model_ckpt_path = './self_model.ckpt'
    transfered_model_ckpt_path = './final_model_transfer.ckpt'
    
    CONFIG.DATA.DATASET = 'self'
    print(CONFIG)
    do_self_learning(save_ckpt_path=self_trained_model_ckpt_path)
    
    CONFIG.DATA.DATASET = original_dataset
    print(CONFIG)
    do_finetune_learning(pre_trained_ckpt_path=self_trained_model_ckpt_path, save_ckpt_path=transfered_model_ckpt_path)




if __name__ == "__main__":

    args = parse_args()
    # if args.experiment_type =='basic':
    # experiment_basic()
    # elif args.experiment_type =='finetune':
    #     experiment_finetune()
    # elif args.experiment_type =='self':
    #     experiment_self_learning()
    # elif args.experiment_type =='whole':
    #     whole_process()
    # else:
    #     raise ValueError("speicfy experiment type")
    experiment_self_learning()
    
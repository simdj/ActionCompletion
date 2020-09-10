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


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP





# CONFIG.N_GPU = 4

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


def get_dataset(data_dir, task_spec=None):
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
    

    return train_dataset, test_dataset
    



def get_model(base_freeze, embedder_freeze):
    model_name  = CONFIG.MODEL.BASE_MODEL.NETWORK
    cnn_embed_dim =CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE

    frame_sequence_encoder = FrameSequenceEncoder(model_name=model_name, use_pretrained=True, base_feature_extract=base_freeze, embedder_feature_extract=embedder_freeze)
    completion_classifier = BaseLineClassifier(embed_dim=cnn_embed_dim)

    model_to_return = [frame_sequence_encoder, completion_classifier]
    return model_to_return

def get_model_loss_optim(task_type):
    
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
    
    # frame_sequence_encoder.cuda()
    # completion_classifier.cuda() 

    # define loss_fn
    loss_fn = nn.BCELoss()
    # loss_fn = loss_fn.cuda()

    # define optim
    params_to_update = list(frame_sequence_encoder.parameters())+list(completion_classifier.parameters())
    # optimizer = optim.Adam(params_to_update, lr=CONFIG.OPTIMIZER.LR, weight_decay=CONFIG.OPTIMIZER.WD)
    optimizer = optim.SGD(params_to_update, lr=CONFIG.OPTIMIZER.LR, momentum=CONFIG.OPTIMIZER.MOMENTUM, weight_decay=CONFIG.OPTIMIZER.WD)

    return model,loss_fn, optimizer

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


def save_checkpoint_distributed(rank, world_size, ckpt_path, model, optimizer, epoch, performance):
    
    print(f"Running save_checkpoint_distributed() on rank {rank}.")
    if rank ==0:
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
    # use a barrier() to prevent other processes loading the model !before! process 0 saves it
    dist.barrier()


def load_checkpoint_distributed(rank, world_size, ckpt_path, ddp_model, optimizer):
    print(f"Running load_checkpoint_distributed() on rank {rank}.")
    print("==> Resuming from ckpt ", ckpt_path)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # ddp

    ckpt = torch.load(ckpt_path, map_location=map_location) # ddp
    cnn_encoder, rnn_decoder = ddp_model
    cnn_encoder.load_state_dict(ckpt['encoder'])
    rnn_decoder.load_state_dict(ckpt['decoder'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']
    ckpt_performance = ckpt['performance']
    ckpt_rng_state = ckpt['rng_state']
    torch.set_rng_state(ckpt_rng_state)
    return ddp_model, optimizer, start_epoch, ckpt_performance, ckpt_rng_state




def train_one_epoch(rank, model, loader, optimizer, loss_fn, epoch):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    res = edict()
    res.loss_list, res.acc_list, res.score_list = [], [], []
    res.accum_loss, res.accum_acc, res.accum_mae = 0.0, 0.0, 0.0
    
    time_epoch_start = time.time()
    for batch_idx, data in enumerate(loader):
        time_batch_start = time.time()
        
        batch_frame_seq = data['frame_seq'].cuda()
        batch_action = data['action']
        batch_moment = data['moment']
        complete_mask = batch_moment>0
        label = (batch_moment>0).float().cuda()
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


        res.accum_loss += loss_value
        res.accum_acc += acc
        res.accum_mae += mae_score
        
        res.loss_list.append(loss_value)
        res.acc_list.append(acc)
        res.score_list.append(mae_score)

        if rank==0:
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f | ACC: %.3f' 
                % ( res.accum_loss/(batch_idx+1), res.accum_mae/(batch_idx+1), res.accum_acc/(batch_idx+1))
            )
    if rank==0:
        logger = logging.getLogger('train')
        logger.info('{} Epoch {}time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} ACC: {:.2f}'.format(
            rank, epoch, time.time() - time_epoch_start, np.mean(res.loss_list), np.mean(res.score_list), np.mean(res.acc_list)
        ))
    

def val(rank, model, loader, optimizer, loss_fn, epoch):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    res = edict()
    res.loss_list = []
    res.acc_list = []
    res.score_list = []

    res.val_loss = 0.0
    res.val_acc = 0.0
    res.val_mae = 0.0
    time_epoch_start = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            time_batch_start = time.time()
            # batch_frame_seq = data['frame_seq'].to(device)
            batch_frame_seq = data['frame_seq'].cuda()
            batch_action = data['action']
            batch_moment = data['moment']

            complete_mask = batch_moment>0
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

            res.val_loss += loss_value
            res.val_acc += acc
            res.val_mae += mae_score
            
            res.loss_list.append(loss_value)
            res.acc_list.append(acc)
            res.score_list.append(mae_score)
            if rank==0:
                progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f | ACC: %.3f' 
                    % ( res.val_loss/(batch_idx+1), res.val_mae/(batch_idx+1), res.val_acc/(batch_idx+1) )
                )
    
    # final log
    if rank==0:
        logger = logging.getLogger('val')
        logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.2f} MAE: {:.2f} ACC: {:.2f}'.format(
            epoch, time.time() - time_epoch_start, np.mean(res.loss_list), np.mean(res.score_list), np.mean(res.acc_list)
        ))

    final_score = np.mean(res.score_list) # MAE

    return final_score


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def do_learning(rank):
    '''
    input: params: edict
    '''
    args = parse_args()
    set_config_with_args(args)    
    # ------------------------------------------------------------------------------------------------------
    # device configuration
    # ------------------------------------------------------------------------------------------------------
    print(f"running DDP on rank {rank}")    
    setup(rank, world_size=CONFIG.N_GPU)
    # TODO - cpu compability code needed
    # CONFIG.DEVICE = torch.cuda.device(rank)
    torch.cuda.set_device(rank)    
    

    # ------------------------------------------------------------------------------------------------------
    # learning configuration
    # ------------------------------------------------------------------------------------------------------
    original_dataset = CONFIG.DATA.DATASET
    single_model_ckpt_path = './final_model_single.ckpt'

    params = edict()
    params.task_type='self'
    params.pre_trained_ckpt_path = None
    params.save_ckpt_path = single_model_ckpt_path



    # ------------------------------------------------------------------------------------------------------
    # model, loss, optimizier
    # ------------------------------------------------------------------------------------------------------  
    model, loss_fn, optimizer  = get_model_loss_optim(params.task_type)
    if params.pre_trained_ckpt_path:
        model, optimizer, start_epoch, _, _ = load_checkpoint(params.pre_trained_ckpt_path, model, optimizer)
    
    frame_sequence_encoder, completion_classifier = model
    
    # ------------------------------------------------------------------------------------------------------
    # DDP model
    # ------------------------------------------------------------------------------------------------------
    frame_sequence_encoder.to(rank)
    completion_classifier.to(rank)

    frame_sequence_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(frame_sequence_encoder)
    completion_classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(completion_classifier)

    DDP_frame_sequence_encoder = DDP(frame_sequence_encoder, device_ids=[rank], find_unused_parameters=True)
    DDP_completion_classifier = DDP(completion_classifier, device_ids=[rank], find_unused_parameters=True)

    
    model = [DDP_frame_sequence_encoder, DDP_completion_classifier]


    # ------------------------------------------------------------------------------------------------------
    # DDP data loader
    # ------------------------------------------------------------------------------------------------------
    train_dataset, test_dataset = get_dataset(data_dir=CONFIG.DATA.DATA_DIR, task_spec=CONFIG.SELF_LEARN.TASK_SPEC)
    print("train_dataset", train_dataset)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
        num_replicas=CONFIG.N_GPU, rank=rank, shuffle=True)
    test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset, 
        num_replicas=CONFIG.N_GPU, rank=rank, shuffle=True)


    batch_size = CONFIG.TRAIN.BATCH_SIZE
    train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size/CONFIG.N_GPU), 
        collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS, sampler=train_sampler, pin_memory=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=int(batch_size/CONFIG.N_GPU), 
        collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS, sampler=test_sampler, pin_memory=True)

    
    # ------------------------------------------------------------------------------------------------------
    # learning loop
    # ------------------------------------------------------------------------------------------------------


    # fill up params
    params.model = model
    
    params.loss_fn = loss_fn
    params.optimizer = optimizer

    params.train_dataloader = train_dataloader
    params.test_dataloader = test_dataloader

    best_MAE = np.inf 
    # device = CONFIG.DEVICE
    for epoch in range(CONFIG.TRAIN.NUM_EPOCH):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        train_one_epoch(rank, params.model, params.train_dataloader, params.optimizer, params.loss_fn, epoch)


        current_MAE = val(rank, params.model, params.test_dataloader, params.optimizer, params.loss_fn, epoch)
        
        # if current_MAE < best_MAE:
        #     save_checkpoint(params.save_ckpt_path, params.model, params.optimizer, epoch, best_MAE)
        #     best_MAE = current_MAE





if __name__ == "__main__":
    # if args.experiment_type =='basic':
    #     experiment_basic()
    # elif args.experiment_type =='finetune':
    #     experiment_finetune()
    # elif args.experiment_type =='self':
    #     experiment_self_learning()
    # elif args.experiment_type =='whole':
    #     whole_process()
    # else:
    #     raise ValueError("speicfy experiment type")
    # mp.spawn(do_learning, args=(), nprocs=CONFIG.N_GPU, join=True)
    args = parse_args()
    set_config_with_args(args)
    mp.spawn(do_learning, args=(), nprocs=4, join=True)

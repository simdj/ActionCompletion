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

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR



from utils import progress_bar
# from vidaug import augmentors as va

# from datasets.video_to_frames import UCF101Dataset
# from datasets.rgbd_ac import RGBD_AC_Dataset
# from datasets.self_supervised_data import Self_Supervised_Dataset


# from models.alignment import batch_get_alignment


from models.algo import get_model, get_model_loss_optim, save_checkpoint_distributed, load_checkpoint_distributed



from config import CONFIG, parse_args, set_config_with_args

from easydict import EasyDict as edict
import random




from utils import *
from datasets.mngt import get_dataset




def train_one_epoch(rank, model, loader, optimizer, loss_fn, epoch):
    # cnn_encoder, rnn_decoder = model
    # cnn_encoder.train()
    # rnn_decoder.train()
    # model.pre_processing_at_training_epoch()
    # print(model.module)
    model.module.pre_processing_at_training_epoch()


    model.train()

    res = edict()
    res.loss_list, res.acc_list, res.score_list = [], [], []
    res.sum_loss, res.sum_acc, res.sum_mae = 0.0, 0.0, 0.0
    
    time_epoch_start = time.time()
    for batch_idx, data in enumerate(loader):
        time_batch_start = time.time()
        
        batch_frame_seq = data['frame_seq'].cuda()
        batch_action = data['action'].cuda()
        batch_moment = data['moment']
        complete_mask = batch_moment>0
        label = (batch_moment>0).float().cuda()
        # data ready 


        optimizer.zero_grad()
        # feed forward
        pred = model(batch_frame_seq, batch_action)


        # optimize
        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()


        # log
        loss_value = loss.item()
        err = torch.abs(pred-label)
        acc = torch.mean((err<0.5).float()).detach().item()     
        mae_score = torch.mean(err).detach().item()


        res.sum_loss += loss_value
        res.sum_acc += acc
        res.sum_mae += mae_score
        
        res.loss_list.append(loss_value)
        res.acc_list.append(acc)
        res.score_list.append(mae_score)

        if rank==0:
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f | ACC: %.3f' 
                % ( res.sum_loss/(batch_idx+1), res.sum_mae/(batch_idx+1), res.sum_acc/(batch_idx+1))
            )
        
    if rank==0:
        logger = logging.getLogger('train')
        logger.info('{} Epoch {}time: {:.2f} s.   LOSS: {:.4f} MAE: {:.4f} ACC: {:.4f}'.format(
            rank, epoch, time.time() - time_epoch_start, np.mean(res.loss_list), np.mean(res.score_list), np.mean(res.acc_list)
        ))

    
    

def val(rank, model, loader, loss_fn, epoch):
    # cnn_encoder, rnn_decoder = model
    # cnn_encoder.eval()
    # rnn_decoder.eval()
    model.eval()

    res = edict()
    res.loss_list, res.acc_list, res.score_list = [], [], []
    res.sum_loss, res.sum_acc, res.sum_mae = 0.0, 0.0, 0.0

    time_epoch_start = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            time_batch_start = time.time()
            # batch_frame_seq = data['frame_seq'].to(device)
            batch_frame_seq = data['frame_seq'].cuda()
            batch_action = data['action'].cuda()
            batch_moment = data['moment']

            complete_mask = batch_moment>0
            label = (batch_moment>0).float().cuda()

            # data ready 

            # # # optimizer.zero_grad()
            # feed forward
            pred = model(batch_frame_seq, batch_action)
            # cnn_feat_seq = cnn_encoder(batch_frame_seq)
            # pred = rnn_decoder(cnn_feat_seq, batch_action)

            loss = loss_fn(pred, label)

            # # # loss.backward()
            # # # optimizer.step()


            loss_value = loss.item()
            err = torch.abs(pred-label)
            acc = torch.mean((err<0.5).float()).detach().item()
            
            mae_score = torch.mean(err).detach().item()

            res.sum_loss += loss_value
            res.sum_acc += acc
            res.sum_mae += mae_score

            res.loss_list.append(loss_value)
            res.acc_list.append(acc)
            res.score_list.append(mae_score)
            if rank==0:
                progress_bar(batch_idx, len(loader), 'Loss: %.3f | MAE: %.3f | ACC: %.3f' 
                    % ( res.sum_loss/(batch_idx+1), res.sum_mae/(batch_idx+1), res.sum_acc/(batch_idx+1) )
                )
    
    # final log
    if rank==0:
        logger = logging.getLogger('val')
        logger.info('Epoch {} time: {:.2f} s.   LOSS: {:.4f} MAE: {:.4f} ACC: {:.4f}'.format(
            epoch, time.time() - time_epoch_start, np.mean(res.loss_list), np.mean(res.score_list), np.mean(res.acc_list)
        ))

    # final_score = np.mean(res.score_list) # MAE
    final_score = np.mean(res.loss_list) # BCE

    return final_score


import pandas as pd
def evaulate_performance( model, loader,  result_csv_path):   
    print("evaluation started --> ", result_csv_path)
    model.eval()
    time_epoch_start = time.time()
    df_res = pd.DataFrame()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            time_batch_start = time.time()
            # batch_frame_seq = data['frame_seq'].to(device)
            batch_frame_seq = data['frame_seq'].cuda()
            batch_action = data['action'].cuda()
            batch_moment = data['moment']
            batch_video_name = data['video_name']

            complete_mask = batch_moment>0
            label = (batch_moment>0).float().cuda()



            # data ready 

            # feed forward
            pred = model(batch_frame_seq, batch_action).detach().cpu().tolist()

            df = pd.DataFrame({
                "video": batch_video_name,
                "class": torch.argmax(batch_action, dim=-1).reshape(-1).detach().cpu().numpy(),
                "label": label.reshape(-1).detach().cpu().numpy(),
                "prediction": pred
                })
            
            df_res=df_res.append(df, ignore_index=True)
        
    
    df_res.to_csv(result_csv_path)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12466'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def do_learning(rank):
    args = parse_args()
    set_config_with_args(args)    
    
    # ------------------------------------------------------------------------------------------------------
    # device configuration
    # ------------------------------------------------------------------------------------------------------
    print(f"running DDP on rank {rank}")    
    setup(rank, world_size=CONFIG.N_GPU)
    torch.cuda.set_device(rank)    

    # ------------------------------------------------------------------------------------------------------
    # DDP data loader
    # ------------------------------------------------------------------------------------------------------
    train_dataset, test_dataset = get_dataset(data_dir=CONFIG.DATA.DATA_DIR, task_spec=CONFIG.SELF_LEARN.TASK_SPEC)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
        num_replicas=CONFIG.N_GPU, rank=rank, shuffle=True)
    # test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset, 
    #     num_replicas=CONFIG.N_GPU, rank=rank, shuffle=True)


    batch_size = CONFIG.TRAIN.BATCH_SIZE
    train_dataloader = DataLoader(train_dataset, batch_size=int(batch_size/CONFIG.N_GPU), 
        collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS, sampler=train_sampler, pin_memory=True)
    # test_dataloader  = DataLoader(test_dataset,  batch_size=int(batch_size/CONFIG.N_GPU), 
    #     collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS, sampler=test_sampler, pin_memory=True)

    # same validation data
    test_dataloader  = DataLoader(test_dataset,  batch_size=int(batch_size/CONFIG.N_GPU), 
        collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)

    # ------------------------------------------------------------------------------------------------------
    # learning configuration
    # ------------------------------------------------------------------------------------------------------
    original_dataset = CONFIG.DATA.DATASET
    
    params = edict()
    params.model_type=CONFIG.MODEL_TYPE

    params.model_ckpt_save_path = CONFIG.CHECKPOINT.MODEL_SAVE_PATH
    params.model_ckpt_load_path = CONFIG.CHECKPOINT.MODEL_LOAD_PATH
    params.memory_ckpt_save_path = CONFIG.CHECKPOINT.MEMORY_SAVE_PATH
    params.memory_ckpt_load_path = CONFIG.CHECKPOINT.MEMORY_LOAD_PATH

    params.enc_conv_embedder_freeze = False


    params.loader = train_dataloader
    params.memory_capacity_per_class = CONFIG.MEMORY.CAPACITY_PER_CLASS


    # ------------------------------------------------------------------------------------------------------
    # model, loss, optimizier
    # ------------------------------------------------------------------------------------------------------  
    model, loss_fn, optimizer  = get_model_loss_optim(params)
    
    # ------------------------------------------------------------------------------------------------------
    # DDP model
    # ------------------------------------------------------------------------------------------------------
    model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    DDP_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model = DDP_model

    if params.model_ckpt_load_path:
        model, optimizer, start_epoch, _, _ = load_checkpoint_distributed(rank, params, model, optimizer)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ------------------------------------------------------------------------------------------------------
    # learning loop
    # ------------------------------------------------------------------------------------------------------

    best_MAE = np.inf 
    for epoch in range(CONFIG.TRAIN.NUM_EPOCH):
        train_sampler.set_epoch(epoch)
        # test_sampler.set_epoch(epoch)
        train_one_epoch(rank, model, train_dataloader, optimizer, loss_fn, epoch)
        current_MAE = val(rank, model, test_dataloader, loss_fn, epoch)
        
        if current_MAE > 0.8:
            break # no hope.... end loop
 
        lr_scheduler.step()
        
        if current_MAE < best_MAE:
            save_checkpoint_distributed(rank, params, model, optimizer, epoch, best_MAE)
            best_MAE = current_MAE


def do_inference():
    args = parse_args()
    set_config_with_args(args)    
    
    rank=0
    setup(rank, world_size=1)
    torch.cuda.set_device(rank)

    # ------------------------------------------------------------------------------------------------------
    # data loader
    # ------------------------------------------------------------------------------------------------------
    train_dataset, test_dataset = get_dataset(data_dir=CONFIG.DATA.DATA_DIR, task_spec=CONFIG.SELF_LEARN.TASK_SPEC)
        
    batch_size = CONFIG.TRAIN.BATCH_SIZE
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
        collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, 
        collate_fn=make_batch, num_workers=CONFIG.NUM_WORKERS, pin_memory=True)

    

    # ------------------------------------------------------------------------------------------------------
    # model configuration
    # ------------------------------------------------------------------------------------------------------
    params = edict()
    params.model_type=CONFIG.MODEL_TYPE
    params.enc_conv_embedder_freeze = True
    # params.model_ckpt_save_path = CONFIG.CHECKPOINT.MODEL_SAVE_PATH
    params.model_ckpt_load_path = CONFIG.CHECKPOINT.MODEL_LOAD_PATH
    # params.memory_ckpt_save_path = CONFIG.CHECKPOINT.MEMORY_SAVE_PATH
    params.memory_ckpt_load_path = CONFIG.CHECKPOINT.MEMORY_LOAD_PATH

    params.loader = train_dataloader
    params.memory_capacity_per_class = CONFIG.MEMORY.CAPACITY_PER_CLASS


    # ------------------------------------------------------------------------------------------------------
    # model, loss, optimizier
    # ------------------------------------------------------------------------------------------------------  
    model, loss_fn, optimizer  = get_model_loss_optim(params)
    model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    DDP_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model = DDP_model
    
    if params.model_ckpt_load_path:
        model, optimizer, start_epoch, _, _ = load_checkpoint_distributed(rank, params, model, optimizer)
    
    evaulate_performance(model, test_dataloader, CONFIG.EVALUATION.RESULT_PATH)
        
        
if __name__ == "__main__":
    
    args = parse_args()
       
    if args.mode=='train':
        mp.spawn(do_learning, args=(), nprocs=4, join=True)
    else:
        do_inference()

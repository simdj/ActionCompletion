# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration of an experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict

import logging
import argparse
import os
from utils import progress_bar, set_logging_defaults

import torch

from datetime import datetime

import numpy as np
# reproducible
torch.manual_seed(0)
# np.random.seed(0) <-- force all rank proc to generate same batch....
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Video Classification')
    parser.add_argument('--experiment_type', type=str, default='transfer', help='basic/transfer')

    parser.add_argument('--data', type=str, default='self', help='ucf101/hmdb51/ugbd_ac/self')
    parser.add_argument('--video_len', type=int, default=64, help='video length')


    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--backbone', type=str, default='resnet50', help='vgg/resnet/c3d/r3d/r21d')
    parser.add_argument('--dataset', type=str, default='ucf101', help='ucf101/hmdb51')
    parser.add_argument('--split', type=str, default='1', help='dataset split')
    # parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--sgpu', type=int, default=0, help='GPU id')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPU')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')

    parser.add_argument('--log', type=str, default='./log/', help='log directory')
    parser.add_argument('--log_file_name', type=str, default='log.csv', help='log file name')    
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    # parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--bs', type=int, default=64, help='mini-batch size')
    
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loading workers')
    # parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    # parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--cnn_freeze', type=bool, default=True, help='freeze cnn encoder')
    parser.add_argument('--decoder_rnn_layer', type=int, default=1, help='nb of decoder_rnn_layer')

    parser.add_argument('--memory_start_epoch', type=int, default=10, help='epoch when memory moddule is used')
    parser.add_argument('--ref_usage_cnt', type=int, default=5, help='number of reference memory to use')

    args = parser.parse_args()
    return args

def set_config_with_args(args):
    CONFIG.NUM_WORKERS = args.num_workers

    
    # log <- datetime info
    log_datetime =  datetime.now().strftime('%m_%d_%H_%M')
    
    CONFIG.LOG.DIR = os.path.join(args.log, log_datetime, args.experiment_type)

    # logdir = os.path.join(CONFIG.LOG.DIR)
    set_logging_defaults(CONFIG.LOG.DIR, args)
    logger = logging.getLogger('main')
    # logname = os.path.join(logdir, args.log_file_name)

    
    CONFIG.MODEL.BASE_MODEL.FREEZE = args.cnn_freeze

    CONFIG.MODEL.DECODER.RNN_LAYER = args.decoder_rnn_layer

    CONFIG.OPTIMIZER.LR = args.lr
    CONFIG.OPTIMIZER.WD = args.wd

    
    CONFIG.TRAIN.NUM_EPOCH = args.num_epoch
    CONFIG.TRAIN.BATCH_SIZE = args.bs


    CONFIG.SELF_LEARN.VIDEO_LEN = args.video_len
    

    use_cuda = torch.cuda.is_available()
    device = "cpu"
    if use_cuda:
        # print('gpu cnt', torch.cuda.device_count())
        CONFIG.USE_CUDA = True
        CONFIG.N_GPU = torch.cuda.device_count()
        
    #     gpu_idx = args.sgpu
    #     device = torch.device('cuda:'+str(gpu_idx))
        
    # CONFIG.DEVICE = device

    CONFIG.DATA.DATASET = args.data


# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ******************************************************************************
# params
# ******************************************************************************
# CONFIG.DEVICE = 'cpu'
CONFIG.NUM_WORKERS = 4
CONFIG.IMAGE_SIZE = 224
CONFIG.N_GPU = 1
CONFIG.USE_CUDA = False

# CONFIG.NUM_EPOCH = 20

# ******************************************************************************
# DATA
# ******************************************************************************
CONFIG.DATA = edict()
CONFIG.DATA.DATASET = 'ucf101'
CONFIG.DATA.DATA_DIR = '../data/ucf101'
CONFIG.DATA.ACTION_CLASS_NUM = 10
CONFIG.test_data = None


# # ******************************************************************************
# # Training params
# # ******************************************************************************

# # Number of training steps.
CONFIG.TRAIN = edict()
CONFIG.TRAIN.NUM_EPOCH = 2
# # Number of samples in each batch.
CONFIG.TRAIN.BATCH_SIZE = 8
# # Number of frames to use while training.
# CONFIG.TRAIN.NUM_FRAMES = 20
# CONFIG.TRAIN.VISUALIZE_INTERVAL = 200



# ******************************************************************************
# self-learning params
# ******************************************************************************
CONFIG.SELF_LEARN = edict()
CONFIG.SELF_LEARN.VIDEO_LEN = 16
CONFIG.SELF_LEARN.TASK_SPEC = None




# ******************************************************************************
# model params
# ******************************************************************************
CONFIG.MODEL = edict()




CONFIG.MODEL.BASE_MODEL = edict()
CONFIG.MODEL.BASE_MODEL.NETWORK = 'resnet50'
CONFIG.MODEL.BASE_MODEL.FREEZE = True



CONFIG.MODEL.CONV_EMBEDDER_MODEL = edict()
# List of conv layers defined as (channels, kernel_size, activate).
CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS = [
    (256, 3, True),
    (256, 3, True),
]
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD = 'max_pool'
# List of fc layers defined as (channels, activate).
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS = [
    (256, True),
    (256, True),
]
CONFIG.MODEL.CONV_EMBEDDER_MODEL.CAPACITY_SCALAR = 2
CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE = 128
CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE = False
CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_RATE = 0.0
CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_SPATIAL = False
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_DROPOUT_RATE = 0.1
CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN = True

# Conv followed by GRU Embedder
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL = edict()
# List of conv layers defined as (channels, kernel_size, activate).
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.CONV_LAYERS = [(512, 3, True),
                                                   (512, 3, True)]
# List of fc layers defined as (channels, activate).
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.GRU_LAYERS = [
    128,
]
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.DROPOUT_RATE = 0.0
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.USE_BN = True
# sim add
CONFIG.MODEL.CONV_EMBEDDER_MODEL.INPUT_CHANNEL_BASE=1024 #resnet50-> 1024, 

CONFIG.MODEL.L2_REG_WEIGHT = 0.00001


CONFIG.MODEL.DECODER = edict()
CONFIG.MODEL.DECODER.RNN_LAYER = 1

# ******************************************************************************
# Optimizer params
# ******************************************************************************
CONFIG.OPTIMIZER = edict()
CONFIG.OPTIMIZER.TYPE = 'Adam'
CONFIG.OPTIMIZER.LR = 1e-3
CONFIG.OPTIMIZER.WD = 1e-3
CONFIG.OPTIMIZER.MOMENTUM  = 9e-1
# # Supported optimizers are: AdamOptimizer, MomentumOptimizer
# CONFIG.OPTIMIZER.TYPE = 'AdamOptimizer'

# CONFIG.OPTIMIZER.LR = edict()
# # Initial learning rate for optimizer.
# CONFIG.OPTIMIZER.LR.INITIAL_LR = 0.0001
# # Learning rate decay strategy.
# # Currently Supported strategies: fixed, exp_decay, manual
# CONFIG.OPTIMIZER.LR.DECAY_TYPE = 'fixed'
# CONFIG.OPTIMIZER.LR.EXP_DECAY_RATE = 0.97
# CONFIG.OPTIMIZER.LR.EXP_DECAY_STEPS = 1000
# CONFIG.OPTIMIZER.LR.MANUAL_LR_STEP_BOUNDARIES = [5000, 10000]
# CONFIG.OPTIMIZER.LR.MANUAL_LR_DECAY_RATE = 0.1
# CONFIG.OPTIMIZER.LR.NUM_WARMUP_STEPS = 0

# ******************************************************************************
# Experiment params
# ******************************************************************************
CONFIG.LOG = edict()
CONFIG.LOG.DIR = './log/'
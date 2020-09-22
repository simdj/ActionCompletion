

#!/bin/bash

set -e
# set -x



# virtualenv -p python3 env 
source pytorch-env/bin/activate

# pip install -r requirements.txt
# pip install -e .


# video data preprocess (video -> rbg frame file list)

# python datasets/video_to_frames.py
# python __practice.py
# python -m main --lr 1e-2 --wd 1e-2 --num_epoch 70 --memory_start_epoch 10 --log './log/result1/'
# python -m main --lr 1e-2 --wd 1e-3 --num_epoch 70 --memory_start_epoch 10 --log './log/result2/'
# python -m main --lr 1e-3 --wd 1e-2 --num_epoch 100 --memory_start_epoch 10 --log './log/result3/'
# python -m main --lr 1e-3 --wd 1e-3 --num_epoch 100 --memory_start_epoch 10 --log './log/result4/'
# python -m main --data 'rgbd_ac' --lr 1e-2 --wd 1e-4 --num_epoch 50 --memory_start_epoch 10 --log './log/rgbd_ac/w_memory/'
# python -m main --data 'rgbd_ac' --lr 1e-2 --wd 1e-4 --num_epoch 50 --memory_start_epoch 60 --log './log/rgbd_ac/womemory/'
# python -m main --data 'rgbd_ac' --lr 1e-3 --wd 1e-4 --num_epoch 50 --memory_start_epoch 10 --log './log/rgbd_ac/w_memory/'
# python -m main --data 'rgbd_ac' --lr 1e-3 --wd 1e-4 --num_epoch 50 --memory_start_epoch 60 --log './log/rgbd_ac/womemory/'


# python -m main --data 'ucf101' --lr 1e-2 --wd 1e-4 --num_epoch 50 --memory_start_epoch 10 --log './log/ucf101/w_memory/'
# python -m main --data 'ucf101' --lr 1e-2 --wd 1e-4 --num_epoch 50 --memory_start_epoch 60 --log './log/ucf101/womemory/'
# python -m main --data 'ucf101' --lr 1e-3 --wd 1e-4 --num_epoch 50 --memory_start_epoch 10 --log './log/ucf101/w_memory/'
# python -m main --data 'ucf101' --lr 1e-3 --wd 1e-4 --num_epoch 50 --memory_start_epoch 60 --log './log/ucf101/womemory/'

# python -m experiment --data 'ucf101' --lr 1e-2 --wd 1e-4 --num_epoch 50 --memory_start_epoch 10 --log './log/ss/w_memory/lr2'
# python -m experiment --data 'ucf101' --lr 1e-2 --wd 1e-4 --num_epoch 50 --memory_start_epoch 60 --log './log/ss/womemory/lr2'
# python -m experiment --data 'ucf101' --lr 1e-3 --wd 1e-4 --num_epoch 50 --memory_start_epoch 60 --log './log/ss/womemory/lr3'
# python -m experiment --data 'ucf101' --lr 1e-3 --wd 1e-4 --num_epoch 50 --memory_start_epoch 10 --log './log/ss/w_memory/lr3'



# python -m train --data 'self'  --num_epoch 2
# python -m train --experiment_type 'basic' --data 'ucf101'  --num_epoch 30 --bs 16 --lr 1e-2 --wd 1e-3 --log './log/'
# python -m train --experiment_type 'self' --data 'ucf101'  --num_epoch 30 --bs 16 --lr 1e-1 --wd 1e-5 --log './log/'


# python -m train --experiment_type 'basic' --data 'ucf101'  --num_epoch 30 --lr 1e-3 --wd 1e-3 --log './log/'
# python -m train --experiment_type 'transfer' --data 'ucf101'  --num_epoch 30 --lr 1e-3 --wd 1e-3 --log './log/'
# python -m train --experiment_type 'basic' --data 'ucf101'  --num_epoch 2 --bs 16 --lr 1e-2 --wd 1e-3 --log './log/'



# python -m train_ddp --num_epoch 50 --bs 16 --lr 1e-1 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 32
# python -m train_ddp --num_epoch 50 --bs 16 --lr 1e-1 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 16

# python -m train_ddp --num_epoch 50 --bs 16 --lr 1e-1 --wd 1e-5 --log './log/' --decoder_rnn_layer 2 --video_len 32
# python -m train_ddp --num_epoch 50 --bs 16 --lr 1e-2 --wd 1e-5 --log './log/' --decoder_rnn_layer 2 --video_len 32

# python -m train_ddp --num_epoch 100 --bs 8 --lr 1e-3 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64
# python -m train_ddp --num_epoch 100 --bs 8 --lr 1e-2 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64
# python -m train_ddp --num_epoch 100 --bs 4 --lr 1e-2 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64
# python -m train_ddp --num_epoch 100 --bs 4 --lr 1e-3 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64


# python -m train_ddp --data ucf101 --num_epoch 100 --bs 4 --lr 1e-2 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64 
# python -m train_ddp --data ucf101 --num_epoch 100 --bs 4 --lr 1e-3 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64 
# python -m train_ddp --data ucf101 --num_epoch 100 --bs 8 --lr 1e-2 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64 
# python -m train_ddp --data ucf101 --num_epoch 100 --bs 8 --lr 1e-3 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64 

python -m train_ddp --data ucf101 --num_epoch 100 --bs 16 --lr 1e-2 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64 
python -m train_ddp --data ucf101 --num_epoch 100 --bs 16 --lr 1e-3 --wd 1e-5 --log './log/' --decoder_rnn_layer 1 --video_len 64 

python -m train_ddp --data ucf101 --num_epoch 100 --bs 16 --lr 1e-2 --wd 1e-3 --log './log/' --decoder_rnn_layer 1 --video_len 64 
python -m train_ddp --data ucf101 --num_epoch 100 --bs 16 --lr 1e-3 --wd 1e-3 --log './log/' --decoder_rnn_layer 1 --video_len 64 




deactivate
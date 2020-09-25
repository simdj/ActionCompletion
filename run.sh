#!/bin/bash

set -e
# set -x
# 


# virtualenv -p python3 env 
source pytorch-env/bin/activate

# pip install -r requirements.txt
# pip install -e .



for data in 'ucf101' 'rgbd_ac'
do
	for bs in '4' '8' '16'
	do 
		for lr in '1e-3' '1e-2'
		do
			for wd in '1e-5' '1e-4'
			do
				echo ${data} ${lr} ${wd}
				mkdir -p ./ckpt/lr_${lr}_wd_${wd}
				ckpt=./ckpt/lr_${lr}_wd_${wd}


				python -m train_ddp --mode 'train' --model_type 'base' --model_save_ckpt  ${ckpt}/${data}_base_net.ckpt  --data ${data} --num_epoch 30 --bs ${bs} --lr ${lr} --wd ${wd} --log './log/' --video_len 64 
				python -m train_ddp --mode 'train' --model_type 'base_memory' --model_save_ckpt  ${ckpt}/${data}_sim_net.ckpt --memory_save_ckpt  ${ckpt}/${data}_sim_memory.ckpt --data ${data} --num_epoch 30 --bs ${bs} --lr ${lr} --wd ${wd} --log './log/' --video_len 64 


				python -m train_ddp --mode 'test' --model_type 'base' --model_load_ckpt  ${ckpt}/${data}_base_net.ckpt  --result_csv  ${ckpt}/${data}_base_result.csv --data ${data} --num_epoch 30 --bs 4 --lr ${lr} --wd ${wd} --log './log/' --video_len 64 				
				python -m train_ddp --mode 'test' --model_type 'base_memory' --model_load_ckpt  ${ckpt}/${data}_sim_net.ckpt --memory_load_ckpt  ${ckpt}/${data}_sim_memory.ckpt --result_csv  ${ckpt}/${data}_sim_result.csv --data ${data} --num_epoch 30 --bs 4 --lr ${lr} --wd ${wd} --log './log/' --video_len 64 


			done
		done
	done
done

deactivate
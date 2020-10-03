#!/bin/bash

set -e
# set -x
# 


# virtualenv -p python3 env 
source pytorch-env/bin/activate

# pip install -r requirements.txt
# pip install -e .

num_epoch=25

for data in 'rgbd_ac' 
do
	for bs in '8' '4'
	do
		for lr in '5e-3' 
		do 
			for wd in '1e-5' '1e-3' 
			do
				for video_len in '64'
				do
					for self_learn in 'True' 'False'
					do
						for split in  '2' '3' '1'
						do 
							echo ${data} ${lr} ${wd} ${split}
							mkdir -p ./ckpt/lr_${lr}_wd_${wd}_self_${self_learn}_len_${video_len}_split${split}
							ckpt=./ckpt/lr_${lr}_wd_${wd}_self_${self_learn}_len_${video_len}_split${split}


							python -m train_ddp --mode 'train' --model_type 'base_memory' --self_learn ${self_learn} --model_save_ckpt  ${ckpt}/${data}_sim_net.ckpt --memory_save_ckpt  ${ckpt}/${data}_sim_memory.ckpt --data ${data} --split ${split} --num_epoch ${num_epoch} --bs ${bs} --lr ${lr} --wd ${wd} --log './log/' --video_len ${video_len}
							python -m train_ddp --mode 'train' --model_type 'base'		  --self_learn ${self_learn} --model_save_ckpt  ${ckpt}/${data}_base_net.ckpt  --data ${data} --split ${split} --num_epoch ${num_epoch} --bs ${bs} --lr ${lr} --wd ${wd} --log './log/' --video_len ${video_len}
							

							python -m train_ddp --mode 'test' --model_type 'base_memory' --self_learn ${self_learn} --model_load_ckpt  ${ckpt}/${data}_sim_net.ckpt --memory_load_ckpt  ${ckpt}/${data}_sim_memory.ckpt --result_csv  ${ckpt}/${data}_sim_result.csv --data ${data} --split ${split} --num_epoch ${num_epoch} --bs 4 --lr ${lr} --wd ${wd} --log './log/' --video_len ${video_len}
							python -m train_ddp --mode 'test' --model_type 'base' 		 --self_learn ${self_learn} --model_load_ckpt  ${ckpt}/${data}_base_net.ckpt  --result_csv  ${ckpt}/${data}_base_result.csv --data ${data} --split ${split} --num_epoch ${num_epoch} --bs 4 --lr ${lr} --wd ${wd} --log './log/' --video_len ${video_len}				
						done
						
					done

				done
			done
		done
	done
done

deactivate
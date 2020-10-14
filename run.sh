#!/bin/bash

set -e
# set -x
# 


# virtualenv -p python3 env 
source pytorch-env/bin/activate

# pip install -r requirements.txt
# pip install -e .

num_epoch=15
data='rgbd_ac'

bs='8'
for lr in '5e-3' 
do 
	for wd in '1e-4' '1e-3'
	do
		for cnn_embedding_dim in '32' '64' 
		do 
			for cnn_num_context in '2' '4'
			do			
				for video_len in '64'
				do
					for self_learn in 'True' 'False'
					do
						for split in  '1'
						do 
							echo ${data} ${lr} ${wd} ${cnn_embedding_dim} ${cnn_num_context} ${video_len} ${split} 
							# mkdir -p ./ckpt/lr_${lr}_wd_${wd}_self_${self_learn}_len_${video_len}_split${split}
							experiment_name=bs_${bs}_lr_${lr}_wd_${wd}_self_${self_learn}_len_${video_len}_emb${cnn_embedding_dim}_context${cnn_num_context}_split${split}
							echo ${experiment_name}
							ckpt=./ckpt/${experiment_name}
							log_dir=./log/${experiment_name}
							mkdir -p ${log_dir}
							mkdir -p ${ckpt}


							python -m train_ddp --mode 'train' --model_type 'base'		  --self_learn ${self_learn} --model_save_ckpt  ${ckpt}/${data}_base_net.ckpt  --data ${data} --split ${split} --num_epoch ${num_epoch} --bs ${bs} --lr ${lr} --wd ${wd} --log ${log_dir} --video_len ${video_len} --cnn_embedding_dim ${cnn_embedding_dim} --cnn_num_context ${cnn_num_context}
							python -m train_ddp --mode 'train' --model_type 'base_memory' --self_learn ${self_learn} --model_save_ckpt  ${ckpt}/${data}_sim_net.ckpt --memory_save_ckpt  ${ckpt}/${data}_sim_memory.ckpt --data ${data} --split ${split} --num_epoch ${num_epoch} --bs ${bs} --lr ${lr} --wd ${wd} --log ${log_dir} --video_len ${video_len} --cnn_embedding_dim ${cnn_embedding_dim} --cnn_num_context ${cnn_num_context}
							

							python -m train_ddp --mode 'test' --model_type 'base' 		 --self_learn ${self_learn} --model_load_ckpt  ${ckpt}/${data}_base_net.ckpt  --result_csv  ${ckpt}/${data}_base_result.csv --data ${data} --split ${split} --num_epoch ${num_epoch} --bs 4 --lr ${lr} --wd ${wd} --log ${log_dir} --video_len ${video_len}				--cnn_embedding_dim ${cnn_embedding_dim} --cnn_num_context ${cnn_num_context}
							python -m train_ddp --mode 'test' --model_type 'base_memory' --self_learn ${self_learn} --model_load_ckpt  ${ckpt}/${data}_sim_net.ckpt --memory_load_ckpt  ${ckpt}/${data}_sim_memory.ckpt --result_csv  ${ckpt}/${data}_sim_result.csv --data ${data} --split ${split} --num_epoch ${num_epoch} --bs 4 --lr ${lr} --wd ${wd} --log ${log_dir} --video_len ${video_len} --cnn_embedding_dim ${cnn_embedding_dim} --cnn_num_context ${cnn_num_context}
						done
						
					done

				done
			done
		done
	done
done

deactivate
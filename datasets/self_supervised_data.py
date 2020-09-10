# video processing
"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
# import uuid
# import tempfile

import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
from skimage import io as image_io
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms
from PIL import Image

import cv2

from video_generation.video_generator import generate_rotating_video
from video_generation.SS_TASK import generate_rotation_task, get_rotation_task

class Self_Supervised_Dataset(Dataset):
	def __init__(self, root_dir, split='1', train=True, video_len=100, task_idx=0, transforms_=None):
		self.root_dir = root_dir
		self.split = split
		self.train = train

		self.video_len = video_len
		self.task_idx = task_idx 

		self.transforms_ = transforms_

		self.init_with_ucf101()

		self.init_task_info(self.task_idx)

		self.class_set = [0]  # TODO action class		
	

	def init_task_info(self, task_idx=0):
		if task_idx==-1:
			rotation_task = generate_rotation_task()
		else:
			rotation_task = get_rotation_task(task_idx)

		self.positive_ratio = rotation_task.positive_ratio
		self.list_positive_schedule_skeleton = rotation_task.pos
		self.list_negative_schedule_skeleton = rotation_task.neg
		# self.positive_ratio = 0.5
		# self.list_positive_schedule_skeleton = [[0,90]]
		# self.list_negative_schedule_skeleton = [[0,50], [0,-90], [0,-50], [0,30,-30]]
		print('train:', self.train, ' -self supervised task', self.positive_ratio, self.list_positive_schedule_skeleton, self.list_negative_schedule_skeleton)

	
	def init_with_ucf101(self):
		annotation_split_path = os.path.join(self.root_dir, 'completion_annotation.txt')
		# 	[ex] Basketball/v_Basketball_g02_c06 101 2/1/1		
		completion_annotation = pd.read_csv(annotation_split_path, header=None, sep=' ')
		# [split] 0: drop / 1:train / 2:test
		split_idx = int(self.split)-1 # assert split=[1-3]
		train_test_split = completion_annotation[2].str.split('/').str[split_idx].astype('int32')
		self.train_split = completion_annotation[train_test_split==1].reset_index()
		self.test_split = completion_annotation[train_test_split==2].reset_index()


	def __len__(self):
		if self.train:
			return len(self.train_split)
		else:
			return len(self.test_split)


	def __getitem__(self, idx):		
		return self.get_item_rotation_task(idx)


	def get_item_rotation_task(self,idx, is_random=False):
		if self.train:
			item = self.train_split.iloc[idx]
		else:
			item = self.test_split.iloc[idx]
		
		# parsing for select a target frame dir 
		video_name = item[0]
		frames_dir = video_name.split("/").pop()
		frames_dir = os.path.join(self.root_dir, 'jpegs_256', frames_dir)
		# randoly choose one of frame in the frames_dir
		randomly_choosen_frame_file = random.choice(os.listdir(frames_dir)) # fully random
		# randomly_choosen_frame_file = random.choice(os.listdir(frames_dir)[:1]) # deterministic

		query_image_path = os.path.join(frames_dir, randomly_choosen_frame_file)

		
		#  positive:negative
		flag_sample_positive = np.random.rand() < self.positive_ratio
		


		if flag_sample_positive:
			target_list_schedule_skelton = self.list_positive_schedule_skeleton
		else:
			target_list_schedule_skelton = self.list_negative_schedule_skeleton
		
		
		# choose one  schedule_skeleton randomly
		schedule_skeleton = random.choice(target_list_schedule_skelton)
		# augment
		schedule = np.array(schedule_skeleton)
		# schedule = schedule + np.random.randn(*schedule.shape) # add some noise
		
		

		pil_seq, detail_scheule = generate_rotating_video(query_image_path, schedule, self.video_len)	
		

		# transforms
		if self.transforms_:
			video_tensor = torch.stack([self.transforms_(img) for img in pil_seq])
		else:
			video_tensor = torch.stack([transforms.ToTensor()(img) for img in pil_seq])
		

		label_completeness = True
		if not flag_sample_positive:
			label_completeness = False
		if idx<100:
			if label_completeness:

				save_pil_list(pil_seq, 'pos.avi')
			else:
				save_pil_list(pil_seq, 'neg.avi')

		action_class_tensor = torch.tensor(0)
		label_tensor = torch.tensor(label_completeness)
		return video_tensor, action_class_tensor, label_tensor

def save_pil_list(pil_list, video_save_path='sample_vid.avi'):
	resized_pil_list = [pil.resize((224,224))for pil in pil_list]
	# convert pil->cv2 : https://pythonpath.wordpress.com/2012/09/17/pil-image-to-cv2-image/
	cv2_list = [cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR) for pil in resized_pil_list ]	
	# cv2 image shape : H x W x C	
	video_tensor = torch.stack([torch.tensor(cv2_img) for cv2_img in cv2_list])
	# write_video expects video_tensor : N x H x W x C
	torchvision.io.write_video(video_save_path, video_tensor, fps=25)
	
	# # image save example
	# resized_pil_list[0].save('pil.png')
	# cv2.imwrite('cv2.png', cv2_list[0])


def make_batch(samples):
	batch_video = [sample[0] for sample in samples]
	batch_label_completeness = [sample[1] for sample in samples]

	# padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
	return {
		'batch_video': torch.stack(batch_video).contiguous(),
		'batch_label_completeness': torch.stack(batch_label_completeness).contiguous()
	}



if __name__ == "__main__":
	data_transforms = {
		"train": transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

		]),
		"val": transforms.Compose([
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}


	# train_dataset = RGBD_AC_Dataset("../../data/RGBD-AC", class_idx_filename=class_idx_filename, train=True, split='1')
	train_dataset = Self_Supervised_Dataset('../data/ucf101', video_len=50, train=True, transforms_=data_transforms['train'])
	test_dataset = Self_Supervised_Dataset('../data/ucf101', video_len=50, train=False, transforms_=data_transforms['val'])

	# root_dir, split='1', data_len, video_len, task, train=True):
	print(len(train_dataset))
	train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn = make_batch)
	test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn = make_batch)
	
	for data in train_dataloader:
		batch_video = data['batch_video']
		batch_label_completeness = data['batch_label_completeness']

		print(batch_video.shape)
		print(batch_label_completeness, batch_label_completeness.shape)
		
		break

	# for data in test_dataloader:
	# 	batch_video = data['batch_video']
	# 	batch_label_completeness = data['batch_label_completeness']

	# 	print(batch_video.shape)
	# 	print(batch_label_completeness, batch_label_completeness.shape)
		
	# 	break
	


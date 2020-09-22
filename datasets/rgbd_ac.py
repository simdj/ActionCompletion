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
from torchvision import transforms
from PIL import Image

from utils import *

class RGBD_AC_Dataset(Dataset):
	"""UCF101 dataset for action completion detection. 
	# the action class index starts from 0 .
	"""

	def __init__(self, root_dir, split='1', class_idx_filename='completion_all_classInd.txt',
	 train=True, transforms_=None, augments_=None, max_video_len=64, read_from_images=True, class_num=10):
		# self.label_category = get_ucf101_action_completeness_category()
		self.root_dir  = root_dir
		self.split = split
		self.train = train

		self.transforms_ = transforms_
		self.augments_ = augments_
		
		self.max_video_len = max_video_len
		self.read_from_images = read_from_images
		self.class_num = class_num

		self.toPIL = transforms.ToPILImage()
		
		class_idx_path = os.path.join(root_dir, class_idx_filename)
		class_idx_data = pd.read_csv(class_idx_path, header=None, sep=' ')
		self.class_idx2label = class_idx_data.set_index(0)[1]
		self.class_label2idx = class_idx_data.set_index(1)[0]
		# class_idx_file content 
		# 	a01 RGB-switch
		# 	a02 RGB-plug
		# self.class_idx2label['a01'] -> RBG-switch
		# self.class_label2idx['RGB-switch'] -> 1
		# self.class_set = self.class_idx2label.values
		

		# self.nb_class = len(self.class_idx2label)

		annotation_split_path = os.path.join(self.root_dir, 'RGBD-AC_completion_moment_annotations.txt')
		# completion_annotation example
		# <videoName> <completionAnnotation> <train-test flag per split>
		# 	[ex] c_a01s02e02 79 1/2/1/1/1/1/1/1		
		completion_annotation = pd.read_csv(annotation_split_path, header=None, sep=' ')
		completion_annotation['class_idx_str']=completion_annotation[0].str[2:5] 
		completion_annotation['class_idx']=completion_annotation[0].str[4].astype('int32')
		completion_annotation['class_label']=self.class_idx2label[completion_annotation['class_idx_str']].values
		
		
		# video_name = completion_annotation[0] # string
		# moment = completion_annotation[1] # int32

		# [filter] annotation with only interested class
		interested_action_list = self.class_idx2label.values
		annotation_action_pd = completion_annotation['class_label']
		completion_annotation = completion_annotation[annotation_action_pd.isin(interested_action_list)]

		self.class_set = completion_annotation['class_idx'].unique()
		print('class set', self.class_set)
		
	
		# [split] 0: drop / 1:train / 2:test
		split_idx = int(self.split)-1 # assert split=[1-3]
		train_test_split = completion_annotation[2].str.split('/').str[split_idx].astype('int32')
		self.train_split = completion_annotation[train_test_split==1].reset_index()
		self.test_split = completion_annotation[train_test_split==2].reset_index()

	def convert_videoname_to_class(self, videoname):

		pass


	def __len__(self):
		if self.train:
			return len(self.train_split)
		else:
			return len(self.test_split)


	def __getitem__(self,idx):
		"""
		returns
			frame sequence (tensor) : [channel x teime x height x width]
			class_idx (tensor) : class index [0-C]
			completeness (tensor) : is_complete [0,1]
		"""
		return self.get_item_from_images(idx)
		

	

	def get_item_from_images(self, idx):
		if self.train:
			item = self.train_split.iloc[idx]
		else:
			item = self.test_split.iloc[idx]

		# video_name = item[0] # ex 1) c_a01s03e06 2) n_a01s06e04
		video_name = item[0]
		moment = item[1]
		class_name = item['class_label']
		class_idx = item['class_idx']
		
		
		frames_dir = os.path.join(self.root_dir, class_name, video_name, 'RGB_images')		
		# example : RGBD-AC/RGB-open/c_a03s02e01/RGB_images
		template_image_data = image_io.imread(os.path.join(frames_dir, 'rgb00001.jpg'))

		length = len([ x for x in os.listdir(frames_dir)])
		height, width, channel = template_image_data.shape
		# load
		num_sampled = self.max_video_len
		sample_idx_list = get_sample_idx_list(max(length, num_sampled),num_sampled)

		video_data = np.zeros([self.max_video_len, height, width, channel], dtype=np.uint8)
		for video_idx, frame_idx in enumerate(sample_idx_list):
			if frame_idx+1 > length:
				break
			image_path = os.path.join(frames_dir, 'rgb{0:05d}.jpg'.format(frame_idx)) # starts with 0
			image_data = image_io.imread(image_path)
			video_data[video_idx,:,:,:] = image_data
		
		if self.augments_:
			video_data = self.augments_(video_data)
		
		if self.transforms_:
			trans_video = []
			for frame in video_data:
				frame = self.toPIL(frame)
				frame = self.transforms_(frame)
				trans_video.append(frame)
			video_tensor = torch.stack(trans_video)
		else:
			video_tensor = torch.tensor(video_data)
		class_one_hot_tensor = torch.nn.functional.one_hot(torch.LongTensor([class_idx]), self.class_num)
		return video_tensor, class_one_hot_tensor, torch.tensor(int(moment))






if __name__ == "__main__":

	# video_path = "../data/ucf101/video/Basketball/v_Basketball_g01_c01.avi"
	# frame_save_path = "./test/"

	# extract_frames_from_video(video_path, frame_save_path)
	# import time
	# start=time.time()
	# for _ in range(1):
	# 	unittest_dataset_dataloader_images()
	# print('images', time.time()-start)

	# start=time.time()
	# for _ in range(1):
	# 	unittest_dataset_dataloader_videos()
	# print('videos', time.time()-start)

	# unittest_dataset_dataloader_images()	
	# class_idx_filename = 'completion_blowing_classInd.txt'
	class_idx_filename = 'completion_all_classInd.txt'
	class_idx_filename = 'completion_open_classInd.txt'
	train_dataset = RGBD_AC_Dataset("../../data/RGBD-AC", class_idx_filename=class_idx_filename, train=True, split='1')
	print(len(train_dataset))
	train_dataloader = DataLoader(train_dataset, batch_size=1)
	for data in train_dataloader:
		# print(data)
		break


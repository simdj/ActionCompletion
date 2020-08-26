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



def extract_and_save_frames_from_video(video_path, frame_save_path):
	video_data = skvideo.io.vread(video_path)
	for i,frame in enumerate(video_data):
		frame_i_save_path = os.path.join(frame_save_path, str(i)+'.png')
		skvideo.io.vwrite(frame_i_save_path, frame)


def extract_frame_seq_from_video(video_path, frame_save_path, transforms_=None):
	print(video_path)
	fn_toPIL = transforms.ToPILImage()
	video_data = skvideo.io.vread(video_path)
	length, height, width, channel = video_data.shape
	print(video_data.shape)


	for i,frame in enumerate(video_data):
		# frame = fn_toPIL(frame)
		# if transforms_:
		# 	frame = transforms_(frame)
		frame_i_save_path = os.path.join(frame_save_path, str(i)+'.png')
		print(frame_i_save_path)
		skvideo.io.vwrite(frame_i_save_path, frame)






class UCF101Dataset(Dataset):
	"""UCF101 dataset for action completion detection. 
	# the action class index starts from 0 .
	"""

	def __init__(self, root_dir, split='1', class_idx_filename='completion_all_classInd.txt',
	 train=True, transforms_=None, augments_=None, max_video_len=64, read_from_images=True):
		# self.label_category = get_ucf101_action_completeness_category()
		self.root_dir  = root_dir
		self.split = split
		self.train = train
		self.transforms_ = transforms_
		self.augments_ = augments_
		
		self.max_video_len = max_video_len
		self.read_from_images = read_from_images

		self.toPIL = transforms.ToPILImage()
		
		class_idx_path = os.path.join(root_dir, class_idx_filename)
		class_idx_data = pd.read_csv(class_idx_path, header=None, sep=' ')
		self.class_idx2label = class_idx_data.set_index(0)[1]
		self.class_label2idx = class_idx_data.set_index(1)[0]

		# print(self.class_idx2label)
		# print(self.class_label2idx)
		# self.nb_class = len(self.class_idx2label)

		annotation_split_path = os.path.join(self.root_dir, 'completion_annotation.txt')
		# completion_annotation example
		# <videoName> <completionAnnotation> <train-test flag per split>
		# 	[ex] Basketball/v_Basketball_g02_c06 101 2/1/1		
		completion_annotation = pd.read_csv(annotation_split_path, header=None, sep=' ')
		# filter annotation with only interested class
		interested_action_list = self.class_idx2label.values
		annotation_action_pd = completion_annotation[0].str.split('/').str[0]
		completion_annotation = completion_annotation[annotation_action_pd.isin(interested_action_list)]

		self.class_set = self.class_label2idx.values
		print('class set', self.class_set)
		

		
		
		# [split] 0: drop / 1:train / 2:test
		split_idx = int(self.split)-1 # assert split=[1-3]
		train_test_split = completion_annotation[2].str.split('/').str[split_idx].astype('int32')
		self.train_split = completion_annotation[train_test_split==1].reset_index()
		self.test_split = completion_annotation[train_test_split==2].reset_index()


	# def padding_tensor(sequences):
	# 	"""
	# 	:param sequences: list of tensor (each tensor has variable-length)
	# 	"""
	# 	num_seq = len(sequences)
	# 	max_seq_len = max([s.size(0) for s in sequences])
	# 	out_dims = (num_seq, max_seq_len)
	# 	out_tensor = sequences[0].data.new(*out_dims).fill_(0)
	# 	mask = sequences[0].data.new(*out_dims).fill_(0)
	# 	for i, tensor_of_seq in enumerate(sequences):
	# 		each_tensor_len = tensor_of_seq.size(0)
	# 		out_tensor[i,:length] = tensor_of_seq
	# 		mask[i,:length]=1
	# 	return out_tensor, mask

	# 	# pad sequences, make al the same length, pack_padded_sequence
	# 	# run through LSTM, use pad_packed_sequence,
	# 	#  flatten all outputs and label, mask out padded outputs, 
	# 	#  calculate cross-entropy
	# 	# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e


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
		# NOTE 1. from video?
		# NOTE 2. from image list?
		if self.read_from_images:
			return self.get_item_from_images(idx)
		else:
			return self.get_item_from_videos(idx)
		

	def get_item_from_videos(self, idx):
		"""
		returns
			frame sequence (tensor) : [channel x teime x height x width]
			class_idx (tensor) : class index [0-C]
			completeness (tensor) : is_complete [0,1]
		"""
		# NOTE 1. from video?
		if self.train:
			item = self.train_split.iloc[idx]
		else:
			item = self.test_split.iloc[idx]

		video_name = item[0]
		moment = item[1]

		class_name = video_name[:video_name.find('/')]
		class_idx = self.class_label2idx[class_name]
		# load video
		video_path = os.path.join(self.root_dir, 'video', video_name+'.avi')
		video_data = skvideo.io.vread(video_path)
		length, height, width, channel = video_data.shape
		if length<self.max_video_len:
			# pad zero image
			pad_shape = [self.max_video_len-length, height, width, channel]
			pad_nd_arr = np.zeros(pad_shape, dtype=np.uint8) # dtype is important (transform func assume)
			video_data = np.concatenate((video_data,pad_nd_arr), axis=0)
			length, height, width, channel = video_data.shape
		

		# sampling frame
		# note that input video has 25 FPS
		# num_sampled = int(max(1,min(50,length*self.sample_rate))) # num_sampled in [1-50]
		# num_sampled = min(self.max_video_len, length) # video length > 50 -> only 50 frames are sampled
		num_sampled = self.max_video_len
		sample_idx_list = get_sample_idx_list(length, num_sampled)
		video_data = [video_data[idx] for idx in sample_idx_list]


		# transform 		
		if self.transforms_:
			trans_video = []
			for frame in video_data:
				# random.seed
				frame = self.toPIL(frame)
				frame = self.transforms_(frame)
				trans_video.append(frame)
			# # (T x C X H x W) to (C X T x H x W)
			# video_tensor = torch.stack(trans_video).permute([1, 0, 2, 3])
			# (T x C X H x W) 
			video_tensor = torch.stack(trans_video)
		else:
			# TODO
			video_tensor = torch.tensor(video_data)

		return video_tensor, torch.tensor(class_idx), torch.tensor(int(moment))


	def get_item_from_images(self, idx):
		if self.train:
			item = self.train_split.iloc[idx]
		else:
			item = self.test_split.iloc[idx]

		video_name = item[0] # ex) SoccerPenalty/v_SoccerPenalty_g01_c02
		moment = item[1]

		class_name = video_name[:video_name.find('/')]
		class_idx = self.class_label2idx[class_name]
		
		frames_dir = video_name.split("/").pop()
		frames_dir = os.path.join(self.root_dir, 'jpegs_256', frames_dir)		
		template_image_data = image_io.imread(os.path.join(frames_dir, 'frame000001.jpg'))

		length = len([ x for x in os.listdir(frames_dir)])
		height, width, channel = template_image_data.shape
		# load
		num_sampled = self.max_video_len
		sample_idx_list = get_sample_idx_list(max(length, num_sampled),num_sampled)

		video_data = np.zeros([self.max_video_len, height, width, channel], dtype=np.uint8)
		for video_idx, frame_idx in enumerate(sample_idx_list):
			if frame_idx+1 > length:
				break
			image_path = os.path.join(frames_dir, 'frame{0:06d}.jpg'.format(frame_idx+1)) # starts with 1
			image_data = image_io.imread(image_path)
			video_data[video_idx,:,:,:] = image_data
		
		# first_image = video_data[0]
		# img = Image.fromarray(first_image, 'RGB')
		# img.save('before.png')
		# img.show()
		
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
		return video_tensor, torch.tensor(class_idx), torch.tensor(int(moment))







	def get_frame_path(self, data_dir_root, annotation_entry, frame_idx):
		# data_dir_root = "../data/ucf101/jpegs_256/"
		# annotation_entry = "SoccerPenalty/v_SoccerPenalty_g01_c02 111 2/1/1"
		# get_frame_path(data_dir_root, annotation_entry, 142)
		# output: ../data/ucf101/jpegs_256/v_SoccerPenalty_g01_c02/frame000142.jpg

		frames_dir = annotation_entry.split(" ")[0].split("/").pop()		 
		frame_file_name = 'frame{0:06d}.jpg'.format(frame_idx) # 0000xx

		frame_path = os.path.join(data_dir_root, frames_dir, frame_file_name)
		return frame_path
		



	
		

def get_sample_idx_list(length, num_sampled):
	interval = length//num_sampled
	offset = np.random.randint(interval)
	sampled_idx_list = [interval*idx+offset for idx in range(num_sampled)]
	return sampled_idx_list




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


def unittest_dataset_dataloader_videos():
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

	class_idx_filename = 'completion_blowing_classInd.txt'
	train_dataset = UCF101Dataset("../../data/ucf101", class_idx_filename=class_idx_filename, train=True, transforms_=data_transforms["train"])
	
	# class_idx_filename = 'completion_all_classInd.txt'
	# train_dataset = UCF101Dataset("../../data/ucf101", class_idx_filename=class_idx_filename, train=True)
	print(len(train_dataset))
	train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, collate_fn=make_batch)
	for data in train_dataloader:
		pass
		batch_frame_seq = data['frame_seq']
		batch_action = data['action']
		batch_moment = data['moment']
		print(batch_frame_seq.shape, batch_action, batch_moment)

		break
def get_video_augments():
	from vidaug import augmentors as va
	sometimes50 = lambda aug: va.Sometimes(0.5, aug)
	sometimes10 = lambda aug: va.Sometimes(0.1, aug)

	video_augments = va.Sequential([
		va.RandomCrop(size=(224,224)),
		# va.RandomRotate(degrees=10),
		sometimes50(va.HorizontalFlip()),
		sometimes50(va.InvertColor()),
		# sometimes10(va.RandomShear(x=10,y=10)),
		# sometimes10(va.RandomTranslate(x=10,y=10)),
		sometimes10(va.GaussianBlur(sigma=1)),
		sometimes10(va.Pepper(ratio=99)),
		sometimes10(va.Salt(ratio=99))
		
	])
	# List of augmenters: https://github.com/okankop/vidaug
	# Affine
	# 	* RandomRotate
	#     * RandomResize
	#     * RandomTranslate
	#     * RandomShear
	# Crop
		# CenterCrop
		# CornerCrop
		# RandomCrop
	# Flip
	# Geometric
	# * GaussianBlur
	#    * ElasticTransformation
	#    * PiecewiseAffineTransform
	#    * Superpixel
	# Intensity
	# 	InvertColor
	# 	* Add
	# 	* Multiply
	# 	* Pepper
	# 	* Salt
	# temporal
	return video_augments

def unittest_dataset_dataloader_images():
	data_transforms = {
		"train": transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.ToTensor(),
			# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

		]),
		"val": transforms.Compose([
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}
	# va = get_video_augments()
	va = None

	class_idx_filename = 'completion_blowing_classInd.txt'
	class_idx_filename = 'completion_all_classInd.txt'
	train_dataset = UCF101Dataset("../../data/ucf101"
		, class_idx_filename=class_idx_filename,
		 train=True, transforms_=data_transforms["train"], augments_=va
		, read_from_images=True)
	
	# class_idx_filename = 'completion_all_classInd.txt'
	# train_dataset = UCF101Dataset("../../data/ucf101", class_idx_filename=class_idx_filename, train=True)
	print(len(train_dataset))
	train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, collate_fn=make_batch)
	for data in train_dataloader:
		pass
		
		# batch_frame_seq = data['frame_seq']
		# batch_action = data['action']
		# batch_moment = data['moment']
		# print(batch_frame_seq[0][0].numpy())
		# first_image = batch_frame_seq[0][0].numpy()
		# img = Image.fromarray(first_image, 'RGB')
		# img.save('after.png')
		# img.show()
		# print(batch_frame_seq.shape, batch_action, batch_moment)

		break








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

	unittest_dataset_dataloader_images()	

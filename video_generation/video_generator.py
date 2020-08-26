# task video generation
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
from torchvision import transforms, datasets
from PIL import Image as PILImage

import cv2

def save_video_from_pil_list(pil_list, video_path):
	(w,h)  = pil_list[0].size
	size = (w,h)
	out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

	for img in pil_list:
		out.write(np.array(img)[:,:,::-1].copy())
	out.release()

def save_video_from_cv_image_list(cv_image_list, video_path):
	(h,w) = cv_image_list[0].shape[:2]
	size = (w,h)
	out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

	for img in cv_image_list:
		out.write(img)
	out.release()


def center_crop_pil_image(pil_img, crop_ratio=0.6):
	w,h = pil_img.size
	new_w = int(w*crop_ratio)
	new_h = int(h*crop_ratio)
	left, right = (w - new_w)//2, (w+new_w)//2
	top, bottom = (h - new_h)//2, (h+new_h)//2
	return pil_img.crop((left, top, right, bottom))


def center_crop_cv2_image(cv2_img, crop_ratio=0.75):
	(h,w) = cv2_img.shape[:2]
	new_w = int(w*crop_ratio)
	new_h = int(h*crop_ratio)
	left, right = (w - new_w)//2, (w+new_w)//2
	top, bottom = (h - new_h)//2, (h+new_h)//2

	return cv2_img[top:bottom, left:right]


def get_detail_milestone_simple(milestone, video_len):
	# near-uniformly 
	# simple - add remaining time to the randomly chosen entry
	list_duration = []
	sub_video_len = video_len // (len(milestone)-1)
	len_remains = video_len % (len(milestone)-1)
	
	list_duration = [sub_video_len]*(len(milestone)-1)
	randomly_sampled_idx = random.randint(0,len(list_duration)-1)
	list_duration[randomly_sampled_idx] = list_duration[randomly_sampled_idx]+len_remains
	return list_duration


def get_detail_milestone_random(milestone, video_len):
	num_cut = len(milestone)-2
	cut_range = range(1,video_len-1)
	# list_cut : list of start/ending location of clip
	list_cut = random.sample(cut_range, num_cut)
	list_cut = [0]+sorted(list_cut)+[video_len-1]
	list_duration = [end-start for (start,end) in zip(list_cut[:-1],list_cut[1:])]
	return list_duration


def get_detail_milestone_random_gaussian(milestone, video_len):
	num_event = len(milestone)-1
	prob_event = np.random.normal(1,0.3,num_event)
	prob_event = np.exp(prob_event)
	prob_event /= np.sum(prob_event)

	list_duration = [int(video_len*prob) for prob in prob_event]
	len_remains = video_len - sum(list_duration)
	randomly_sampled_idx = random.randint(0,len(list_duration)-1)
	list_duration[randomly_sampled_idx] = list_duration[randomly_sampled_idx]+len_remains

	print(list_duration)


	return list_duration




def get_detail_milestone(rotation_milestone, video_len, is_random=False):
	if len(rotation_milestone)<2:
		raiseValueerror("milestone should be a list (len>=2)")
	if is_random:
		list_of_sub_video_len = get_detail_milestone_random_gaussian(rotation_milestone, video_len)
	else:
		list_of_sub_video_len = get_detail_milestone_simple(rotation_milestone, video_len)
	
	detail_schedule = []
	for i, (start, end, duration) in enumerate(zip(rotation_milestone[:-1], rotation_milestone[1:],list_of_sub_video_len)):
		if i<len(list_of_sub_video_len)-1:
			detail_schedule.append(np.linspace(start,end, num=duration, endpoint=False))
		else:
			detail_schedule.append(np.linspace(start,end, num=duration, endpoint=True))
	detail_schedule = np.concatenate(detail_schedule, axis=0)
	return detail_schedule


def generate_rotating_video(query_image_path, rotation_milestone, video_len, is_random=False):
	"""
	arg: query_image_path
	arg: rotation_milestone: - [0,90,0], [0,10,20,30,20,10]
	arg: video len: length of video to handle schedule - 64, 180,  #TODO - fix expression
	return : image_list: [PIL Image, ...]
	# return: videos : numpy ndarray [Lx?]
	return: rotation_milestone_detail: [rotation_degree, ...]
	"""
	query_image = PILImage.open(query_image_path)
	rotation_milestone_detail = get_detail_milestone(rotation_milestone, video_len, is_random=is_random)
	image_list = [query_image.rotate(degree) for degree in rotation_milestone_detail]
	image_list = [center_crop_pil_image(img) for img in image_list]
	# np_video = np.stack([np.array(img, dtype=np.uint8) for img in image_list])
	# np_video = np.stack([np.array(img) for img in image_list])
	# print(np_video.shape)
	return image_list, rotation_milestone_detail



def generate_resizing_video(query_image_path, resize_milestone, video_len, is_random=False):
	"""
	arg: query_image_path: PIL.Image
	arg: resize_milestone: - [1,2,1], [1, 2, 3] [1,0.1, 2]
	arg: video len: length of video to handle schedule - 64, 180,  #TODO - fix expression
	return : image_list: [PIL Image, ...]
	return: resize_milestone_detail: [resize_degree, ...]
	"""
	resize_milestone_detail = get_detail_milestone(resize_milestone, video_len, is_random=is_random)
	# TODO
	# image_list = [query_image_path.rotate(degree) for degree in resize_milestone_detail]
	# return image_list, resize_milestone_detail

	image_list = [center_crop_cv2_image(img) for img in image_list]


def generate_translating_video(query_image_path, translation_milestone, video_len, is_random=False):
	"""
	arg: query_image_path:
	arg: translation_milestone: - [(0,0), (x,y)],  [(0,0), (x,y), (0,0)], [(0,0),(x1,y1), (x2,y2), (x3,y3)]
	arg: video len: length of video to handle schedule - 64, 180,  #TODO - fix expression
	return : image_list: [PIL Image, ...]
	return: resize_milestone_detail: [resize_degree, ...]
	"""
	image_data = cv2.imread(query_image_path)
	num_rows, num_cols = image_data.shape[:2]

	
	translation_x_milestone= [int(x*num_rows) for (x,y) in translation_milestone]
	translation_y_milestone= [int(y*num_cols) for (x,y) in translation_milestone]

	translation_x_milestone_detail = get_detail_milestone(translation_x_milestone, video_len, is_random=is_random)
	translation_y_milestone_detail = get_detail_milestone(translation_y_milestone, video_len, is_random=is_random)

	image_list = []
	for tx,ty in zip(translation_x_milestone_detail, translation_y_milestone_detail):
		translation_matrix = np.float32([ [1,0,tx], [0,1,ty] ])
		output_shape = (num_cols , num_rows) # same shape with original -> crop!
		translated_image = cv2.warpAffine(image_data,translation_matrix, output_shape  )
		image_list.append(translated_image)

	
	translation_milestone_detail = [(x,y) for (x,y) in zip(translation_x_milestone_detail,translation_y_milestone_detail)]
	image_list = [center_crop_cv2_image(img) for img in image_list]
	return image_list, translation_milestone_detail



def test(query_image_path):
	# query_image_pil = Image.open(query_image_path)
	
	
	video_len = 100
	list_positive_schedule= [[0,180,0], [0,175,0], [0,185,0]]
	list_negative_schedule = [[0,90,0], [0,180,250], [0,180]]

	positive_schedule = [0,180,0]	
	negative_schedule = [0,70,0]

	# positive
	num_positive = 5
	num_negative = 10
	
	video, detail_schedule = generate_rotating_video(query_image_path, positive_schedule, video_len=video_len, is_random=True)
	save_video_from_pil_list(video, 'test_rotation.avi')
	# for i in range(num_positive):
	# 	video, detail_schedule = generate_rotating_video(query_image_path, positive_schedule, video_len=video_len, is_random=True)
	# 	save_video_from_pil_list(video, 'positive'+str(i)+'.avi')

	# for i in range(num_negative):
	# 	video, detail_schedule = generate_rotating_video(query_image_path, negative_schedule, video_len=video_len, is_random=True)
	# 	save_video_from_pil_list(video, 'netative'+str(i)+'.avi')
	

	positive_schedule = [(0,0), (0.1,0.1)]
	video, detail_schedule = generate_translating_video(query_image_path, positive_schedule, video_len)
	save_video_from_cv_image_list(video, 'test_translation.avi')


	
	
	

	
	

if __name__ =="__main__":
	query_image_path = "../datasets/test.jpg"
	test(query_image_path)


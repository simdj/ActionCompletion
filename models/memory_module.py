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
import pickle

# from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset


from utils import progress_bar

import torch

import random

class MemoryModule():
	def __init__(self, class_set, capacity_per_class=5):
		self.class_set = class_set
		
		self.capacity_per_class = capacity_per_class
		self.memory = dict({x:[] for x in self.class_set })
		

	def add(self, cnn_feat_seq, class_idx, add_prob=0.1):
		target_memory = self.memory[class_idx]

		if len(target_memory) < self.capacity_per_class:
			target_memory.append(cnn_feat_seq)
		else:
			# memory is full -> coin toss 
			if random.random()  < add_prob: # coin toss pass -> this seq would added
				random.shuffle(target_memory)
				target_memory.pop() # randomly pop existing memory
				target_memory.append(cnn_feat_seq)



	def get_memory(self, class_idx, nb_memory=-1):
		if nb_memory==-1:
			nb_memory=self.capacity_per_class
		
		memory_list = random.choices(self.memory[class_idx], k=min(nb_memory, self.capacity_per_class))
		return torch.stack(memory_list)
	
	
	def get_batch_memory(self, batch_class_one_hot, nb_memory=-1):
		'''
		batch_class_one_hot : batch_size x 1 x |class|
		'''
		# return: batch x memory_number x seq_len x dim
		# return: list of list of torch.tensor ==> call torch.stack and to(device)


		if nb_memory==-1:
			nb_memory=self.capacity_per_class
		
		batch_class_index = torch.argmax(batch_class_one_hot, dim=-1)
		# -> shape: batch_size x 1
		
		batch_memory_list = [ self.get_memory(int(idx.item()), nb_memory) for idx in batch_class_index ]
		return torch.stack(batch_memory_list)

	def clear(self):
		self.memory.clear()
		self.memory = dict({x:[] for x in self.class_set })

	def construct_positive_memory_all(self, device, loader, model):
		"""
			As iterating over all action seqs, fill memory with embedded of positive action seq 
			
		# """
		# self.clear()

		# cnn_encoder, _ = model
		# cnn_encoder.eval()

		# time_memory_build_start = time.time()
		# with torch.no_grad():
		# 	for batch_idx, data in enumerate(loader):
		# 		batch_moment = data['moment']
		# 		positive_mask = batch_moment>0

		# 		if len(positive_mask)==0:
		# 			continue
				
		# 		positive_batch_action = data['action'][positive_mask].tolist()
		# 		positive_batch_frame_seq = data['frame_seq'][positive_mask].to(device)
				
		# 		positive_batch_cnn_feat_seq = cnn_encoder(positive_batch_frame_seq)
		# 		for cnn_feat_seq, action_class in zip(positive_batch_cnn_feat_seq, positive_batch_action ):
		# 			self.add(cnn_feat_seq, action_class, add_prob=0.1)
				
		# 		# progress_bar(batch_idx, len(loader), '')
			
		# print('Memory Construction Done. Elapsed time: {:.2f} s. '.format(time.time() - time_memory_build_start))
		pass
			
	def construct_positive_memory_fast(self, loader, cnn_encoder):
		

		memory_full_state = dict( {x:0 for x in self.class_set } )

		self.clear()

		time_memory_build_start = time.time()
		with torch.no_grad():
			for batch_idx, data in enumerate(loader):
				batch_moment = data['moment']
				positive_mask = batch_moment>0

				if sum(positive_mask)==0:
					continue
				
				positive_batch_action = data['action'][positive_mask]
				positive_batch_class_index = torch.argmax(positive_batch_action, dim=-1).reshape(-1).tolist()

				positive_batch_frame_seq = data['frame_seq'][positive_mask].cuda()
				positive_batch_cnn_feat_seq = cnn_encoder(positive_batch_frame_seq)
				
				for cnn_feat_seq, action_class_index in zip(positive_batch_cnn_feat_seq, positive_batch_class_index ):
					if memory_full_state[action_class_index]<self.capacity_per_class:
						self.add(cnn_feat_seq, action_class_index, add_prob=1.0)
						memory_full_state[action_class_index] +=1 

				current_memory_size = sum(memory_full_state.values())
				full_size = len(self.class_set)*self.capacity_per_class
				if current_memory_size == full_size:
					break
				# progress_bar(batch_idx, len(loader), '')
				# print(self.memory)
				# print(memory_full_state, current_memory_size, full_size)
			
		# print('Memory Construction Done. Elapsed time: {:.2f} s. '.format(time.time() - time_memory_build_start))

	




					

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

# from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset


# from utils import progress_bar, set_logging_defaults



import random

class MemoryModule():
	def __init__(self, nb_class, capacity_per_class=5):
		self.nb_class = nb_class
		self.capacity_per_class = capacity_per_class
		self.memory = dict({x:[] for x in range(self.nb_class) })

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


	def get_memory(self, class_idx, nb_memory=1):
		return random.choices(self.memory[class_idx], k=min(nb_memory, self.capacity_per_class))
	
	def clear(self):
		self.memory.clear()
		self.memory = dict({x:[] for x in range(self.nb_class) })

	def construct_positive_memory_all(self, device, loader, model):
		"""
			As iterating over all action seqs, fill memory with embedded of positive action seq 
			
		"""
		self.clear()

		cnn_encoder, _ = model
		cnn_encoder.eval()

		time_memory_build_start = time.time()
		with torch.no_grad():
			for batch_idx, data in enumerate(loader):
				batch_moment = data['moment']
				positive_mask = batch_moment>0

				if len(positive_mask)==0:
					continue
				
				positive_batch_action = data['action'][positive_mask].tolist()
				positive_batch_frame_seq = data['frame_seq'][positive_mask].to(device)
				
				positive_batch_cnn_feat_seq = cnn_encoder(positive_batch_frame_seq)
				for cnn_feat_seq, action_class in zip(positive_batch_cnn_feat_seq, positive_batch_action ):
					self.add(cnn_feat_seq, action_class, add_prob=0.1)
				
				# progress_bar(batch_idx, len(loader), '')
			
		print('Epoch time: {:.2f} s. '.format(time.time() - time_memory_build_start))
			
	def construct_positive_memory_fast(self, device, loader, model):

		memory_full_state = dict( {x:0 for x in range(self.nb_class) } )

		self.clear()

		cnn_encoder, _ = model
		cnn_encoder.eval()

		time_memory_build_start = time.time()
		with torch.no_grad():
			for batch_idx, data in enumerate(loader):
				batch_moment = data['moment']
				positive_mask = batch_moment>0

				if len(positive_mask)==0:
					continue
				
				positive_batch_action = data['action'][positive_mask].tolist()
				positive_batch_frame_seq = data['frame_seq'][positive_mask].to(device)
				
				positive_batch_cnn_feat_seq = cnn_encoder(positive_batch_frame_seq)
				for cnn_feat_seq, action_class in zip(positive_batch_cnn_feat_seq, positive_batch_action ):
					if memory_full_state[action_class]<self.capacity_per_class:
						self.add(cnn_feat_seq, action_class, add_prob=1.0)
						memory_full_state[action_class] +=1 

				memory_meet_capacity_count=sum([x[1]==self.capacity_per_class for x in memory_full_state.items()])
				is_full_memory = memory_meet_capacity_count==self.nb_class
				if is_full_memory:
					break
				# progress_bar(batch_idx, len(loader), '')
			
		print('Epoch time: {:.2f} s. '.format(time.time() - time_memory_build_start))


					

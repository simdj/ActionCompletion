# some codes are borrowed from the below link
# https://github.com/google-research/google-research/tree/master/tcc
# author = {Dwibedi, Debidatta and Aytar, Yusuf and Tompson, Jonathan and Sermanet, Pierre and Zisserman, Andrew},
# title = {Temporal Cycle-Consistency Learning},
# booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
# month = {June},
# year = {2019},
# }
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F


def pairwise_l2_distance(embs1, embs2):
	"""
	compute pairwisel2 distance b/w all rows of embs1 and embs2
	
	param: embs1: shape: (seq_len1 x dim)
	param: embs2: shape: (seq_len2 x dim)
	return: dist: shape: (seq_len1 x seq_len2)

	"""

	
	# norm1: shape: (seq_len x 1) # 1 for sum of sqaure	
	# norm2: shape: (1 x seq_len) # 1 for sum of sqaure	
	norm1 = torch.sum(embs1*embs1, dim=1).reshape(-1,1)
	norm2 = torch.sum(embs2*embs2, dim=1).reshape(1,-1)
	# sum_of_norm1_norm2 = norm1 + norm2 # broadcast
	# (a-b)^2 = a^2+b^2-2ab
	dist = norm1 + norm2 - 2.0*torch.matmul(embs1, embs2.T)
	dist = torch.max(dist,torch.zeros_like(dist))
	return dist

def get_scaled_similarity(embs1, embs2, similarity_type, temperature):
	"""
		param: embs1: shape: (seq_len1 x dim)
		param: embs2: shape: (seq_len2 x dim)
	"""
	
	if similarity_type =='cosine':
		sim_matrix = torch.matmul(embs1, embs2.T)
	elif similarity_type =='l2':
		sim_matrix = -1.0*pairwise_l2_distance(embs1,embs2)
	else:
		raise ValueError('similarity_type can either be l2 or cosine.')
	
	dim = embs1.size(1)
	# TODO - google code ref
	
	# Scale the distance  by number of dimension. 
	# This normalization helps with optimization. (refer to google's tcc)
	sim_matrix = torch.div(sim_matrix, dim)
	# Scale the distance by a temperature that helps with how soft/hard the
	# alignment should be.
	sim_matrix = torch.div(sim_matrix, temperature)
	return sim_matrix

def get_alignment(embs1, embs2, similarity_type='l2', temperature=0.1):
	seq_len = embs1.size(1)
	sim_l2 = get_scaled_similarity(embs1,embs2, similarity_type, temperature)
	softmaxed_sim_l2 = F.softmax(sim_l2, dim=1)

	nn_embs = torch.matmul(softmaxed_sim_l2, embs2)
	return nn_embs


if __name__ == "__main__":
	# test
	seq_len1 = 10
	seq_len2 = 5
	dim=8


	a=torch.randn(seq_len1, dim)
	b=torch.randn(seq_len2, dim)
	print(a)
	print(b)
	
	
	a_map_to_b= get_alignment(a,b)
	print(a_map_to_b)
	print(a_map_to_b.shape)



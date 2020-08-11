from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy






# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
        
    def forward(self, x):
        return x


class CNNEncoder(nn.Module):
	def __init__(self, model_name='resnet', use_pretrained=True, feature_extract=True):
		super(CNNEncoder, self).__init__()
		
		self.input_size = 0
		self.cnn_embed_dim = 0
		self.cnn_backbone = None

		if model_name =='resnet':
			cnn_backbone = models.resnet18(pretrained=use_pretrained)
			self.input_size = 224
			self.cnn_embed_dim = cnn_backbone.fc.in_features
			
			self.cnn_backbone = cnn_backbone
			self.cnn_backbone.fc = IdentityLayer() # delete last layer
		elif model_name == 'vgg':
			cnn_backbone = models.vgg11_bn(pretrained=use_pretrained)
			
			self.input_size = 224
			self.cnn_embed_dim = cnn_backbone.classifier[6].in_features
			
			self.cnn_backbone = cnn_backbone
			self.cnn_backbone.classifier[6]=IdentityLayer() # delete last layer
		
		# feature layer freeze or not
		if feature_extract:
			for param in self.cnn_backbone.parameters():
				param.requires_grad=False

	def forward(self, frame_seq):
		"""
		param: frame_seq : shape(batch x seq_len x C x H x W)
		return: cnn_feqt_seq: shape(batch x seq_len x embed_size)
		"""
		cnn_embed_seq = []
		for t in range(frame_seq.size(1)):
			x = self.cnn_backbone(frame_seq[:,t,:,:,:])
			x = x.view(x.size(0),-1)
			cnn_embed_seq.append(x)
		
		# list2tensor  by stacking
		cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
		# swap (time) and (batch dim) 
		cnn_embed_seq = cnn_embed_seq.transpose_(0,1)
		# cnn_embed_seq: shape (batch, seq_len, embed_size)
		return cnn_embed_seq

if __name__ =="__main__":
	fe = CNNEncoder()
	print(fe.cnn_backbone)
	

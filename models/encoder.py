from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy



from easydict import EasyDict as edict


from config import CONFIG



class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
        
    def forward(self, x):
        return x
class FrameSequenceEncoder(nn.Module):
	def __init__(self, model_name='reesnet', use_pretrained=True, feature_extract=True):
		super(FrameSequenceEncoder, self).__init__()
		self.base_encoder = BaseEncoder(model_name, use_pretrained, feature_extract)
		self.conv_embedder = ConvEmbedder()


	def forward(self, frame_seq, num_frames_of_context=2):
		"""
			frame_seq: shape: (Batch x seq_len x C x H x W)
			emb_output : shape : (Batch x seq_len//num_frames_of_context x emb_dim)
		"""
		cnn_feat_seq = self.base_encoder(frame_seq)
		emb_output = self.conv_embedder(cnn_feat_seq,num_frames_of_context)
		return emb_output



class BaseEncoder(nn.Module):
	def __init__(self, model_name='resnet', use_pretrained=True, feature_extract=True):
		super(BaseEncoder, self).__init__()
		
		self.input_size = 0
		self.cnn_embed_dim = 0
		self.cnn_backbone = None

		
		if model_name == 'resnet' or model_name == 'resnet50':
			cnn_backbone = models.resnet50(pretrained=use_pretrained)
			self.input_size = 224
			self.cnn_embed_dim = cnn_backbone.fc.in_features
			
			self.cnn_backbone = cnn_backbone
			# # last layer
			# self.cnn_backbone.layer4 = IdentityLayer() 
			# self.cnn_backbone.avgpool = IdentityLayer()
			# self.cnn_backbone.fc = IdentityLayer()

			# # conv4 out
			self.cnn_backbone.layer4 = IdentityLayer() 
			self.cnn_backbone.avgpool = IdentityLayer()
			self.cnn_backbone.fc = IdentityLayer()			
		elif model_name=='resnet18':
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
		return: cnn_feat_seq: shape(batch x seq_len x embed_size)
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


# CONFIG = edict()
# CONFIG.MODEL = edict()
# CONFIG.MODEL.CONV_EMBEDDER_MODEL = edict()
# # List of conv layers defined as (channels, kernel_size, activate).
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS = [
#     (256, 3, True),
#     (256, 3, True),
# ]
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD = 'max_pool'
# # List of fc layers defined as (channels, activate).
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS = [
#     (256, True),
#     (256, True),
# ]
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.CAPACITY_SCALAR = 2
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE = 128
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE = False
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_RATE = 0.0
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_SPATIAL = False
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_DROPOUT_RATE = 0.1
# CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN = True

# CONFIG.MODEL.CONV_EMBEDDER_MODEL.INPUT_CHANNEL_BASE=1024 #resnet50-> 1024, 

# CONFIG.MODEL.L2_REG_WEIGHT = 0.00001

def get_conv_bn_layers(in_channel, conv_params, use_bn, conv_dims):
	if conv_dims ==1:
		conv_layer = nn.Conv1d
		bn = nn.BatchNorm1d
	elif conv_dims ==2:
		conv_layer = nn.Conv2d
		bn = nn.BatchNorm2d
	elif conv_dims==3:
		conv_layer = nn.Conv3d
		bn = nn.BatchNorm3d
	else:
		raise ValueError('Invalid number of conv_dims')

	l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT # TODO check l2 regularization
	total_layers = []
	conv_layers = []
	bn_layers = []
	activations = []
	channels = [in_channel] + [ channel for channel, kernel_size,activate in conv_params]
	# for channels, kernel_size, activate  in conv_params:
		# conv
		# in_channels, out_channels = channels
	for i in range(len(channels)-1):
		in_channel = channels[i]
		out_channel = channels[i+1]
		kernel_size = conv_params[i][1]

		conv_layers.append(conv_layer(in_channel, out_channel, kernel_size, padding=1))
	for out_channel, _, _  in conv_params:
		# bn
		if use_bn:
			bn_layers.append(bn(num_features=out_channel))
	
	for _, _, activate  in conv_params:
		# activation
		if activate:
			activation = nn.ReLU()
		else:
			activation = nn.Identify()
		activations.append(activation)

	for i in range(len(conv_layers)):
		total_layers.append(conv_layers[i])
		if use_bn:
			total_layers.append(bn_layers[i])
		total_layers.append(activations[i])
	# return conv_layers, bn_layers, activations
	return nn.Sequential(*total_layers)


def get_fc_layers(in_channel, fc_params):
	channels = [in_channel] + [ channel for channel, activate in fc_params]
	activations = [activate for channel, activate in fc_params]
	
	layers = []
	for i in range(len(channels)-1):
		in_channel = channels[i]
		out_channel = channels[i+1]
		activate = activations[i]
		
		layers.append(nn.Linear(in_channel, out_channel))
		if activate:
			layers.append(nn.ReLU())
	return nn.Sequential(*layers)
	

class ConvEmbedder(nn.Module):
	def __init__(self):
		super(ConvEmbedder, self).__init__()
		conv_params = CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS
		fc_params = CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS
		self.use_bn = CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN
		l2_reg_weight = CONFIG.MODEL.L2_REG_WEIGHT
		embedding_size = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
		cap_scalar = CONFIG.MODEL.CONV_EMBEDDER_MODEL.CAPACITY_SCALAR

		# 3dconv
		conv_params = [(cap_scalar*x[0], x[1], x[2]) for x in conv_params]
		self.conv_layer_input_channel = CONFIG.MODEL.CONV_EMBEDDER_MODEL.INPUT_CHANNEL_BASE
		self.conv3d_layers = get_conv_bn_layers(
			in_channel=self.conv_layer_input_channel, conv_params=conv_params, use_bn=self.use_bn, conv_dims=3)

		# global max pooling (in forward function)

		# fc
		fc_params = [(cap_scalar*x[0], x[1]) for x in fc_params]
		fc_layer_input_dim = conv_params[-1][0]
		self.fc_layers = get_fc_layers(fc_layer_input_dim, fc_params)
		
		# embed
		num_features_before_embedding = fc_params[-1][0]
		self.embedding_dim = CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
		self.embedding_layer = nn.Linear(num_features_before_embedding, self.embedding_dim)

	def forward(self, cnn_feat_seq, num_frames_of_context=2, shape_check=False):
		"""
		 cnn_feat_seq shape: {BS x seq_len} x  C x H x W 
			# seq_len = num_context x num_frames_of_context
			num_context = seq_len//num_frames_of_context 
		--> (context split) reshape needed  (BS x num_context) x num_frames_of_context x C x H x W
		--> (3d conv ready) transpose N x C x Depth x H x W
				--> (BS x num_context) x C x num_frames_of_context x  H x W

				global max pool 
		"""
		if shape_check:
			print('conv embedder\'s input shape', cnn_feat_seq.shape)
		batch_size, seq_len, cnn_feat_dim = cnn_feat_seq.shape
		cnn_feat_seq = cnn_feat_seq.reshape(*[batch_size,seq_len, self.conv_layer_input_channel,14,14])
		batch_size, total_frame_cnt, c, h, w = cnn_feat_seq.shape
		if shape_check:
			print('conv embedder\'s input shape un-flatten', cnn_feat_seq.shape)
		
		num_context = total_frame_cnt//num_frames_of_context
		new_shape = [batch_size*num_context, num_frames_of_context, c, h, w]
		
		cnn_feat_seq  = cnn_feat_seq.reshape(*new_shape)
		if shape_check:
			print('conv embedder\'s input shape un-flatten + split context(k)', cnn_feat_seq.shape)
		cnn_feat_seq = torch.transpose(cnn_feat_seq,1,2)
		
		res = cnn_feat_seq
		
		if shape_check:
			print('conv embedder\'s input shape un-flatten + split context(k) + transpose for 3d convolution', cnn_feat_seq.shape)
		
		res = self.conv3d_layers(res)
		if shape_check:
			print('after 3d conv', res.shape)
		# global max pool
		res = F.max_pool3d(res, kernel_size=res.size()[2:])
		res = torch.squeeze(res)
		if shape_check:
			print('after global max pool', res.shape)

		res = self.fc_layers(res)
		if shape_check:
			print('after fc', res.shape)

		res = self.embedding_layer(res)
		if shape_check:
			print('after embed', res.shape)

		# batch
		res = res.reshape((*[-1, num_context, self.embedding_dim]))
		if shape_check:
			print('after unfold to batch', res.shape)
		return res
		
		# example k=2
		# new_shape [56, 2, 1024, 14, 14]
		# cnn_feat_seq shape torch.Size([56, 1024, 2, 14, 14])
		# after 3d conv torch.Size([56, 512, 2, 14, 14])
		# after global max pool torch.Size([56, 512])
		# after fc torch.Size([56, 512])
		# after embed torch.Size([56, 128])
		# after unfold to batch torch.Size([8, 7, 128])

		



if __name__ =="__main__":

	bs = 8
	seq_len=14
	k=2

	x = torch.randn(bs, seq_len, 3, 224, 224)

	base_enc = BaseEncoder('resnet50')
	ce = ConvEmbedder()
	cnn_feat_seq = base_enc(x)
	emb_output = ce(cnn_feat_seq,k, shape_check=True)

# <example result>
# bs = 8
# seq_len=14
# k=2
# x = torch.randn(bs, seq_len, 3, 224, 224)
# base_enc = BaseEncoder('resnet50')
# ce = ConvEmbedder()
# cnn_feat_seq = base_enc(x)
# emb_output = ce(cnn_feat_seq,k)
# conv embedder's input shape torch.Size([8, 14, 200704])
# conv embedder's input shape un-flatten torch.Size([8, 14, 1024, 14, 14])
# conv embedder's input shape un-flatten + split context(k) torch.Size([56, 2, 1024, 14, 14])
# conv embedder's input shape un-flatten + split context(k) + transpose for 3d convolution torch.Size([56, 1024, 2, 14, 14])
# after 3d conv torch.Size([56, 512, 2, 14, 14])
# after global max pool torch.Size([56, 512])
# after fc torch.Size([56, 512])
# after embed torch.Size([56, 128])
# after unfold to batch torch.Size([8, 7, 128])
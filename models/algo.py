from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from config import CONFIG


from models.encoder import FrameSequenceEncoder
from models.alignment import batch_get_alignment
from models.memory_module import MemoryModule
# from baseline import BaseLineClassifier
from models.decoder import BaseLineClassifier



class BASE(nn.Module):
	def __init__(self, enc_conv_embedder_freeze):
		super(BASE, self).__init__()

		model_name  = CONFIG.MODEL.BASE_MODEL.NETWORK
		cnn_embed_dim =CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
		action_class_num = CONFIG.DATA.ACTION_CLASS_NUM
		self.enc = FrameSequenceEncoder(model_name=model_name, embedder_freeze=enc_conv_embedder_freeze)
		self.dec = BaseLineClassifier(action_class_num=action_class_num, embed_dim=cnn_embed_dim)

	def forward(self, seq_emb_frame, class_index):
		feat_seq = self.enc(seq_emb_frame)
		return self.dec(feat_seq, class_index)


class BASE_MEMORY(nn.Module):
	def __init__(self, enc_conv_embedder_freeze):
		super(BASE_MEMORY, self).__init__()

		model_name  = CONFIG.MODEL.BASE_MODEL.NETWORK
		cnn_embed_dim =CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE

		self.enc = FrameSequenceEncoder(model_name=model_name, embedder_freeze=enc_conv_embedder_freeze)
		self.dec = BaseLineClassifier(embed_dim=cnn_embed_dim)
		# TODO
		self.memory = None

	def forward(self, seq_emb_frame, class_index):
		feat_seq = self.enc(seq_emb_frame)
		# TODO memory
		return self.dec(feat_seq, class_index)


def get_model(model_type, enc_conv_embedder_freeze):
	if model_type=='base':
		return BASE(enc_conv_embedder_freeze=enc_conv_embedder_freeze)
	elif model_type=='base_memory':
		return BASE_MEMORY(enc_conv_embedder_freeze=enc_conv_embedder_freeze)
	else:
		raise ValueError("[get_model] specify model_type")


from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn


from models.encoder import FrameSequenceEncoder
from models.alignment import batch_get_alignment
from models.memory_module import MemoryModule
# from baseline import BaseLineClassifier
from models.decoder import BaseLineClassifier



class BASE(nn.Module):
	def __init__(self):
		super(BASE, self).__init()
		self.enc = FrameSequenceEncoder()
		self.dec = BaseLineClassifier()

	def forward(self, seq_emb_frame):
		return self.dec(self.enc(seq_emb_frame))
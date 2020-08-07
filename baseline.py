# seq(frame) -> seq(emb(frame)) -> completeness

import torch
import torch.nn as nn
import torch.nn.functional as F 

class BaseLineClassifier(nn.Module):
	"""
	classifier seq(emb(frame)) - > completeness	
	integrate attention lstm / completion lstm
	implementation of weakly-supervised ... (ICCVW 2019)
	"""
	def __init__(self, emb_dim=128, hid_dim=64):
		super(BaseLineClassifier, self).__init__()
		self.hid_dim = hid_dim
		self.attentionLSTM = nn.LSTM(emb_dim, hid_dim)
		self.hidden2attion_score = nn.Linear(hid_dim, 1)
		self.completionLSTM = nn.LSTM(emb_dim, hid_dim)
		self.hidden2completion_score = nn.Linear(hid_dim,1)

	def forward(self, seq_emb_frame):
		# seq_emb_frame - [batch_size x seq_len x hidden_dim]
		attention_list = []
		completion_list = []
		
		# for emb_frame in seq_emb_frame:
		batch_size = seq_emb_frame.shape[0]
		seq_len = seq_emb_frame.shape[1]
		
		lstm_out, (ht, ct) = self.attentionLSTM(seq_emb_frame)
		attention_score = self.hidden2attion_score(lstm_out).squeeze()
		
		attention_score_softmax = F.softmax(attention_score, dim=1)
		

		lstm_out, (ht, ct) = self.completionLSTM(seq_emb_frame)
		completion_score = self.hidden2completion_score(lstm_out).squeeze()
		
		# # <supervised setting>
		# frame_level_complete_pred = torch.sigmoid(attention_score_softmax * completion_score)
		# print(frame_level_complete_pred)
		# print(frame_level_complete_pred.shape)

		
		# # <weakly-supervised setting>
		video_level_complete_pred = torch.sigmoid(torch.sum(attention_score_softmax * completion_score, dim=1))
		final_score = video_level_complete_pred
	
		return final_score



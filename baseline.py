# seq(frame) -> seq(emb(frame)) -> completeness

import torch
import torch.nn as nn
import torch.nn.functional as F 



class BaseLineClassifier(nn.Module):
	"""
	classifier seq(emb(frame)) - > completeness	
	# # <weakly-supervised setting>
	input: seq of embedd_feature
	output : scalar 0~1: non-complete / complete
	
	"""
	def __init__(self, embed_dim=128, hidden_dim=64, use_mean=False):
		super(BaseLineClassifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.use_mean = use_mean

		self.completionLSTM = nn.LSTM(embed_dim, hidden_dim)
		self.hidden2completion_score = nn.Linear(hidden_dim,1)

	def forward(self, seq_emb_frame):
		# seq_emb_frame - [batch_size x seq_len x embed_dim]
		# print(seq_emb_frame.shape)
		completion_list = []
		
		# for emb_frame in seq_emb_frame:
		batch_size = seq_emb_frame.shape[0]
		seq_len = seq_emb_frame.shape[1]
		
		
		
		lstm_out, (ht, ct) = self.completionLSTM(seq_emb_frame)
		# lstm_out: shape: (batch_size x seq_len x hidden_dim)
		completion_score = self.hidden2completion_score(lstm_out)
		# lstm_out: shape: (batch_size x seq_len x1)
		completion_score = completion_score.squeeze(2)
		# completion_score: shape: (batch_size x seq_len)
		
		
		if self.use_mean:
			video_level_complete_pred = torch.sigmoid(torch.mean(completion_score, dim=1)) # == torch.sum(score*uniform_dist, dim=1)
		else:
			# consider last step only
			video_level_complete_pred = torch.sigmoid(completion_score[:,-1]) # == torch.sum(score*uniform_dist, dim=1)
		final_score = video_level_complete_pred
	
		return final_score


class AttentiveClassifier(nn.Module):
	"""
	classifier seq(emb(frame)) - > completeness	
	integrate attention lstm / completion lstm
	implementation of weakly-supervised ... (ICCVW 2019)
	"""
	def __init__(self, embed_dim=128, hidden_dim=64):
		super(AttentiveClassifier, self).__init__()
		self.hidden_dim = hidden_dim

		self.attentionLSTM = nn.LSTM(embed_dim, hidden_dim)
		self.hidden2attion_score = nn.Linear(hidden_dim, 1)
		
		self.completionLSTM = nn.LSTM(embed_dim, hidden_dim)
		self.hidden2completion_score = nn.Linear(hidden_dim,1)

	def forward(self, seq_emb_frame):
		# seq_emb_frame - [batch_size x seq_len x hidden_dim]
		# print(seq_emb_frame.shape)
		attention_list = []
		completion_list = []
		
		# for emb_frame in seq_emb_frame:
		batch_size = seq_emb_frame.shape[0]
		seq_len = seq_emb_frame.shape[1]
		
		attention_lstm_out, (ht, ct) = self.attentionLSTM(seq_emb_frame) 
		# attention_lstm_out: shape: (batch_size x seq_len x hidden_dim)		
		attention_score = self.hidden2attion_score(attention_lstm_out) 
		# attention_score shape: (batch_size x seq_len x 1)
		attention_score = attention_score.squeeze(2)		
		attention_score_softmax = F.softmax(attention_score, dim=1)
		# attention_score shape: (batch_size x seq_len)
		

		completion_lstm_out, (ht, ct) = self.completionLSTM(seq_emb_frame) # shape: (batch_size x seq_len x hidden_dim)
		completion_score = self.hidden2completion_score(completion_lstm_out)
		completion_score = completion_score.squeeze(2)
		
		# # <supervised setting>
		# frame_level_complete_pred = torch.sigmoid(attention_score_softmax * completion_score)
		# print(frame_level_complete_pred)
		# print(frame_level_complete_pred.shape)

		
		# # <weakly-supervised setting>
		video_level_complete_pred = torch.sigmoid(torch.sum(attention_score_softmax * completion_score, dim=1))
		final_score = video_level_complete_pred
	
		return final_score


if __name__ =="__main__":
	# seq_emb_frame - [batch_size x seq_len x hidden_dim]
	# print(seq_emb_frame.shape)
	bs=4
	seq_len=7
	embed_dim = 8
	
	clf = BaseLineClassifier(embed_dim=embed_dim, hidden_dim=64)
	seq_emb_frame = torch.randn(bs,seq_len,embed_dim)

	pred =clf(seq_emb_frame)
	print(pred)

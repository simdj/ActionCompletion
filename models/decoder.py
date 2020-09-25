# seq(frame) -> seq(emb(frame)) -> completeness

import torch
import torch.nn as nn
import torch.nn.functional as F 

from config import CONFIG


class BaseLineClassifier(nn.Module):
	"""
	classifier seq(emb(frame)) - > completeness	
	# # <weakly-supervised setting>
	input: seq of embedd_feature
	output : scalar 0~1: non-complete / complete
	
	"""
	def __init__(self, action_class_num=1, embed_dim=128, hidden_dim=32, use_mean=False, one_layer=False):
		super(BaseLineClassifier, self).__init__()
		self.action_class_dim = action_class_num
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.use_mean = use_mean
		self.one_layer = one_layer

		
		lstm_input_dim = self.embed_dim
		self.completionLSTM = nn.LSTM(lstm_input_dim, hidden_dim, CONFIG.MODEL.DECODER.RNN_LAYER)

		mlp_input_dim = hidden_dim + self.action_class_dim
		self.hidden2completion_score = MLP(mlp_input_dim, [hidden_dim, 1])
		
		# print(self.completionLSTM)
		# print(self.hidden2completion_score)

	def forward(self, seq_emb_frame, action_class_one_hot):
		# seq_emb_frame - [batch_size x seq_len x embed_dim]
		# action_class_one_hot - [batch_size x 1 x action_class_dim]
		
		batch_size = seq_emb_frame.shape[0]
		seq_len = seq_emb_frame.shape[1]
		
		lstm_out, (ht,ct) = self.completionLSTM(seq_emb_frame)
		# lstm_out : (batch_size x seq_len x hidden_dim)	

		# action_class_one_hot = F.one_hot(action_class, self.action_class_dim)
		# action_class_one_hot : (batch_size x self.action_class_dim)
		action_class_one_hot_seq = action_class_one_hot.repeat(1,seq_len,1).float()
		# action_class_one_hot_seq : (batch_size x seq_len x self.action_class_dim)

		
		
		# append 'action class one hot vector'
		completion_classifier_input = torch.cat([lstm_out, action_class_one_hot_seq], dim=2)
		# completion_classifier_input : (batch_size x seq_len x (hidden_dim+self.action_class_dim))

				
		completion_score = self.hidden2completion_score(completion_classifier_input)
		# completion_score: (batch_size x seq_len x1)
		completion_score = completion_score.squeeze(2)
		# completion_score: (batch_size x seq_len)
		
		if self.use_mean:
			video_level_complete_pred = torch.sigmoid(torch.mean(completion_score, dim=1)) # == torch.sum(score*uniform_dist, dim=1)
		else:
			# consider last step only
			video_level_complete_pred = torch.sigmoid(completion_score[:,-1]) # == torch.sum(score*uniform_dist, dim=1)
		final_score = video_level_complete_pred
	
		return final_score




class BaseLinePlusAttentionClassifier(nn.Module):
	"""
	classifier seq(emb(frame)) - > completeness	
	# # <weakly-supervised setting>
	input: seq of embedd_feature
	output : scalar 0~1: non-complete / complete
	
	"""
	def __init__(self, action_class_num=1, embed_dim=128, hidden_dim=32, use_mean=False, one_layer=False):
		super(BaseLinePlusAttentionClassifier, self).__init__()
		self.action_class_dim = action_class_num
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		self.use_mean = use_mean
		self.one_layer = one_layer

		
		lstm_input_dim = self.embed_dim
		self.completionLSTM = nn.LSTM(lstm_input_dim, hidden_dim, CONFIG.MODEL.DECODER.RNN_LAYER)
		self.attentionLSTM = nn.LSTM(lstm_input_dim, hidden_dim, CONFIG.MODEL.DECODER.RNN_LAYER)		


		mlp_input_dim = hidden_dim + self.action_class_dim
		self.hidden2completion_score = MLP(mlp_input_dim, [hidden_dim, 1])
		self.hidden2attenion_score = MLP(mlp_input_dim, [hidden_dim, 1])
		
		# print(self.completionLSTM)
		# print(self.hidden2completion_score)

	def forward(self, seq_emb_frame, action_class_one_hot):
		# seq_emb_frame - [batch_size x seq_len x embed_dim]
		# action_class_one_hot - [batch_size x 1 x action_class_dim]
		
		batch_size = seq_emb_frame.shape[0]
		seq_len = seq_emb_frame.shape[1]






		attention_lstm_out, (ht, ct) = self.attentionLSTM(seq_emb_frame) 
		# lstm_out : (batch_size x seq_len x hidden_dim)	
		
		lstm_out, (ht,ct) = self.completionLSTM(seq_emb_frame)
		# lstm_out : (batch_size x seq_len x hidden_dim)	

		# action_class_one_hot = F.one_hot(action_class, self.action_class_dim)
		# action_class_one_hot : (batch_size x self.action_class_dim)
		action_class_one_hot_seq = action_class_one_hot.repeat(1,seq_len,1).float()
		# action_class_one_hot_seq : (batch_size x seq_len x self.action_class_dim)

		
		
		


		attention_classifier_input = torch.cat([attention_lstm_out, action_class_one_hot_seq], dim=2)
		attention_score = self.hidden2attenion_score(attention_classifier_input)
		attention_score = attention_score.squeeze(2)
		attention_score_softmax = F.softmax(attention_score, dim=1)

		# append 'action class one hot vector'
		completion_classifier_input = torch.cat([lstm_out, action_class_one_hot_seq], dim=2)
		# completion_classifier_input : (batch_size x seq_len x (hidden_dim+self.action_class_dim))		
		completion_score = self.hidden2completion_score(completion_classifier_input)
		# completion_score: (batch_size x seq_len x1)
		completion_score = completion_score.squeeze(2)
		# completion_score: (batch_size x seq_len)
		
		# if self.use_mean:
		# 	video_level_complete_pred = torch.sigmoid(torch.mean(completion_score, dim=1)) # == torch.sum(score*uniform_dist, dim=1)
		# else:
		# 	# consider last step only
		# 	video_level_complete_pred = torch.sigmoid(completion_score[:,-1]) # == torch.sum(score*uniform_dist, dim=1)
		video_level_complete_pred = torch.sigmoid(torch.sum(attention_score_softmax*completion_score, dim=1))
		final_score = video_level_complete_pred
	
		return final_score



class DeprecatedBaseLineClassifier(nn.Module):
	"""
	classifier seq(emb(frame)) - > completeness	
	# # <weakly-supervised setting>
	input: seq of embedd_feature
	output : scalar 0~1: non-complete / complete
	
	"""
	def __init__(self, embed_dim=128, hidden_dim=32, use_mean=False, one_layer=False):
		super(DeprecatedBaseLineClassifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.use_mean = use_mean
		self.one_layer = one_layer

		
		
		self.completionLSTM = nn.LSTM(embed_dim, hidden_dim, CONFIG.MODEL.DECODER.RNN_LAYER)
		self.hidden2completion_score = MLP(hidden_dim, [hidden_dim, 1])
		# self.hidden2completion_score = MLP(hidden_dim, [1])
		
		
		# if self.one_layer:
		# 	self.completionLSTM = nn.LSTM(embed_dim, hidden_dim)
		# 	# self.hidden2completion_score = nn.Linear(hidden_dim,1)
		# else:
		# 	# self.completionLSTM = RNN_STACK(embed_dim, [hidden_dim, hidden_dim, hidden_dim])
		# 	# self.completionLSTM = RNN_STACK(embed_dim, [hidden_dim, hidden_dim])
            
  #           self.hidden2completion_score = MLP(hidden_dim, [1])
		# 	# self.hidden2completion_score = MLP(hidden_dim, [ 1])

		print(self.completionLSTM)
		print(self.hidden2completion_score)

	def forward(self, seq_emb_frame):
		# seq_emb_frame - [batch_size x seq_len x embed_dim]
		# print(seq_emb_frame.shape)
		completion_list = []
		
		# for emb_frame in seq_emb_frame:
		batch_size = seq_emb_frame.shape[0]
		seq_len = seq_emb_frame.shape[1]
		
		
		lstm_out, (ht,ct) = self.completionLSTM(seq_emb_frame)
		#else:
		#	lstm_out = self.completionLSTM(seq_emb_frame)
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



def activation_layer(act_name):
    if act_name.lower()=='sigmoid':
        return nn.Sigmoid()
    elif act_name.lower()=='relu':
        return nn.ReLU(inplace=True)
    return None

class RNN_STACK(nn.Module):
	def __init__(self, input_dim, hidden_dims):
		super(RNN_STACK, self).__init__()
		dim_list = [input_dim]+hidden_dims
		self.rnn_layers = nn.ModuleList(
			[nn.LSTM(dim_list[i], dim_list[i+1]) for i in range(len(dim_list)-1)]
		)
		
	def forward(self, x):
		rnn_out = x
		for i in range(len(self.rnn_layers)):
			rnn_out, _ = self.rnn_layers[i](rnn_out)
		return rnn_out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, activation='relu',  dropout_rate=0, use_bn=True, init_std=1e-3):
        super(MLP, self).__init__()
        # self.dropout_rate = dropout_rate
        # self.dropout = nn.Dropout(self.dropout_rate)
        
        # self.use_bn = use_bn
        if len(hidden_units)==0:
            raise ValueError("hidden_units is empty")

        hidden_units = [input_dim]+hidden_units

        self.linears = nn.ModuleList(   
            [nn.Linear(hidden_units[i],hidden_units[i+1]) for i in range(len(hidden_units)-1)]
        )

        # if self.use_bn:
        #     self.bn = nn.ModuleList(
        #         [nn.BatchNorm1d(hidden_units[i+1]) for i in range(len(hidden_units)-1)]
        #     )
        # self.activation_layers = nn.ModuleList(
        #     [activation_layer(activation) for i in range(len(hidden_units)-1) ]
        # )

        # initialize weight
        # for name, tensor in self.linears.named_parameters():
        #     if 'weight' in name:
        #         nn.init.normal_(tensor, mean=0, std=init_std)
    
    def forward(self, x):
        cur_layer = x
        for i in range(len(self.linears)):
            cur_layer = self.linears[i](cur_layer)
            # if self.use_bn:
            #     cur_layer = self.bn[i](cur_layer)
            # cur_layer = self.activation_layers[i](cur_layer)
            # cur_layer = self.dropout(cur_layer)
        return cur_layer


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

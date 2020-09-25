from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim

from config import CONFIG


from models.encoder import FrameSequenceEncoder
from models.alignment import batch_get_alignment
from models.memory_module import MemoryModule
# from baseline import BaseLinePlusAttentionClassifier
from models.decoder import BaseLinePlusAttentionClassifier




import pickle


class BASE(nn.Module):
	def __init__(self, enc_conv_embedder_freeze):
		super(BASE, self).__init__()

		model_name  = CONFIG.MODEL.BASE_MODEL.NETWORK
		cnn_embed_dim =CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
		action_class_num = CONFIG.DATA.ACTION_CLASS_NUM
		self.enc = FrameSequenceEncoder(model_name=model_name, embedder_freeze=enc_conv_embedder_freeze)
		self.dec = BaseLinePlusAttentionClassifier(action_class_num=action_class_num, embed_dim=cnn_embed_dim)

	def forward(self, seq_emb_frame, class_index):
		feat_seq = self.enc(seq_emb_frame)
		result_on_query = self.dec(feat_seq, class_index)
		# -> shape: batch_size

		return result_on_query

	def pre_processing_at_training_epoch(self):
		pass

	def post_processing_save(self, params):
		pass
	

	def post_processing_load(self, params):
		pass

class BASE_MEMORY(nn.Module):
	def __init__(self, enc_conv_embedder_freeze, loader, memory_capacity_per_class):
		'''
		memory_class_set: action class set 
		memory_capacity_per_class: number of referecnes per action class
		'''
		super(BASE_MEMORY, self).__init__()

		model_name  = CONFIG.MODEL.BASE_MODEL.NETWORK
		cnn_embed_dim =CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE
		action_class_num = CONFIG.DATA.ACTION_CLASS_NUM

		self.loader = loader
		self.memory_class_set = loader.dataset.class_set
		

		self.enc = FrameSequenceEncoder(model_name=model_name, embedder_freeze=enc_conv_embedder_freeze)
		self.dec = BaseLinePlusAttentionClassifier(action_class_num=action_class_num, embed_dim=cnn_embed_dim)

	
		self.memory = MemoryModule(class_set=self.memory_class_set, capacity_per_class=memory_capacity_per_class)
			
		self.nb_memory = self.memory.capacity_per_class

	def forward(self, seq_emb_frame, class_index):
		feat_seq = self.enc(seq_emb_frame)
		
		result_on_query = self.dec(feat_seq, class_index) # forward(x)		
		# -> shape: (batch_size) 1D
		result_with_memory = self.forward_with_memory(feat_seq, class_index) # forward(x,memory)
		# -> shape: (batch_size*nb_memory) 1D

		result_aggregated = self.aggregate_results(result_on_query, result_with_memory)
		return result_aggregated

	
	def forward_with_memory(self, cnn_feat_seq, class_one_hot):
		seq_len = cnn_feat_seq.size(1)
		feat_dim = cnn_feat_seq.size(2)
		
		# TODO - broadcasting cnn_feat_seq 	# current impl : copy
		# 1) expand cnn_feat_seq by (nb_memory) times
		# expanded_cnn_feat_seq = cnn_feat_seq.unsqueeze(1).repeat(1,self.nb_memory,1,1) 
		# expanded_cnn_feat_seq = expanded_cnn_feat_seq.reshape(-1,seq_len,feat_dim)
		expanded_cnn_feat_seq = torch.repeat_interleave(cnn_feat_seq, self.nb_memory, dim=0)
		# -> shape: (batch_size*nb_memory) x seq_len x dim

		# 2) expand class_one_hot by (nb_memory) times
		# expanded_class_one_hot = class_one_hot.unsqueeze(1).repeat(1,self.nb_memory).reshape(-1)
		expanded_class_one_hot = torch.repeat_interleave(class_one_hot, self.nb_memory, dim=0)
		# -> shape: (batch_size*nb_memory*1)

		# 3) get memory (action_class_idx)
		# extract corresponing memory cnn_feat_seq given query (cnn_feat_seq)
		reference_memory_cnn_feat_seq = self.memory.get_batch_memory(class_one_hot,self.nb_memory).cuda()
		# -> shape: batch_size x nb_memory x seq_len x dim
		reference_memory_cnn_feat_seq = reference_memory_cnn_feat_seq.reshape(-1,seq_len,feat_dim)
		# -> shape:  (batch_size*nb_memory) x seq_len x dim 
			
		# 4) get alinged seq by mapping query to reference
		aligned_cnn_feat_seq_by_memory= batch_get_alignment(expanded_cnn_feat_seq, reference_memory_cnn_feat_seq.detach())
		# 5) get prediction
		aligned_pred = self.dec(aligned_cnn_feat_seq_by_memory, expanded_class_one_hot)
		# -> shape : (batch_size x nb_memory) x 1 
		return aligned_pred

	
	def aggregate_results(self, result_on_query, result_with_memory):		
		# output(query)
		result_on_query = result_on_query.reshape(-1,1)
		#  output(query~reference)
		result_with_memory = result_with_memory.reshape(-1, self.nb_memory)
		result_with_memory_aggregated = torch.mean(result_with_memory, dim=1, keepdim=True)
		
		result_aggregated = torch.cat([result_on_query, result_with_memory_aggregated], dim=1)
		# -> shape: batch_size x (1+1)

		# average (implicitly reshape)		
		result_aggregated = torch.mean(result_aggregated, dim=1)
		# -> shape: batch_size
		return result_aggregated



	def update_memory(self):
		''' 
		loader should be train_dataloader
		'''
		self.enc.eval()
		self.memory.construct_positive_memory_fast(self.loader, self.enc)
	
	def pre_processing_at_training_epoch(self):
		self.update_memory()




	def post_processing_save(self, params):
		# save memory
		with open(params.memory_ckpt_save_path, 'wb') as output:
			pickle.dump(self.memory, output)
			print("saving ... memory ", params.memory_ckpt_save_path)

	def post_processing_load(self, params):
		# load memory
		with open(params.memory_ckpt_load_path, 'rb') as _input:
			self.memory = pickle.load(_input)
			print("loading ... memory ", params.memory_ckpt_load_path)

		

		




def get_model(params):
	print('get model params ', params)
	if params.model_type=='base':
		return BASE(
			enc_conv_embedder_freeze=params.enc_conv_embedder_freeze
			)
	elif params.model_type=='base_memory':
		return BASE_MEMORY(
			enc_conv_embedder_freeze=params.enc_conv_embedder_freeze, 
			loader=params.loader, 
			memory_capacity_per_class=params.memory_capacity_per_class
			)
	else:
		raise ValueError("[get_model] specify model_type")



def get_model_loss_optim(params):
    # 1 model
    model = get_model(params)

    # 2 define loss_fn
    loss_fn = nn.BCELoss()
    # loss_fn = loss_fn.cuda()

    # 3 define optim
    params_to_update = list(model.parameters())
    # list(frame_sequence_encoder.parameters())+list(completion_classifier.parameters())
    if CONFIG.OPTIMIZER.TYPE =='adam':
    	optimizer = optim.Adam(params_to_update, lr=CONFIG.OPTIMIZER.LR, weight_decay=CONFIG.OPTIMIZER.WD)
    else:
    	optimizer = optim.SGD(params_to_update, lr=CONFIG.OPTIMIZER.LR, momentum=CONFIG.OPTIMIZER.MOMENTUM, weight_decay=CONFIG.OPTIMIZER.WD)

    return model,loss_fn, optimizer





def save_checkpoint_distributed(rank, params, model, optimizer, epoch, performance):

    if rank == 0  and params.model_ckpt_save_path:
        print(f"Running save_checkpoint_distributed() on rank {rank}. epoch {epoch}")
        print("Saving..", params.model_ckpt_save_path)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'performance': performance,
            'epoch' : epoch,
            'rng_state' : torch.get_rng_state()
        }
        torch.save(state, params.model_ckpt_save_path)

        model.module.post_processing_save(params)
        # # use a barrier() to prevent other processes loading the model !before! process 0 saves it
        # dist.barrier()


def load_checkpoint_distributed(rank, params, ddp_model, optimizer):
    print(f"Running load_checkpoint_distributed() on rank {rank}.")
    print("==> Resuming from ckpt ", params.model_ckpt_load_path)

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}  # ddp

    ckpt = torch.load(params.model_ckpt_load_path, map_location=map_location) # ddp
    
    ddp_model.load_state_dict(ckpt['model'])

    optimizer.load_state_dict(ckpt['optimizer'])

    start_epoch = ckpt['epoch']
    ckpt_performance = ckpt['performance']
    ckpt_rng_state = ckpt['rng_state']
    torch.set_rng_state(ckpt_rng_state)

    ddp_model.module.post_processing_load(params)


    return ddp_model, optimizer, start_epoch, ckpt_performance, ckpt_rng_state





# def save_checkpoint(ckpt_path, model, optimizer, epoch, performance):
#     print("Saving..", ckpt_path)
#     # cnn_encoder, rnn_decoder = model
#     state = {
#         # 'config': config.samples,
#         'model': model.state_dict(),
#         # 'encoder': cnn_encoder.state_dict(),
#         # 'decoder': rnn_decoder.state_dict(),
#         'optimizer': optimizer.state_dict(),
#         'performance': performance,
#         'epoch' : epoch,
#         'rng_state' : torch.get_rng_state()
#     }
#     torch.save(state, ckpt_path)

# def load_checkpoint(ckpt_path, model, optimizer):
#     print("==> Resuming from ckpt ", ckpt_path)
#     ckpt = torch.load(ckpt_path)
    
#     # cnn_encoder, rnn_decoder = model
#     # cnn_encoder.load_state_dict(ckpt['encoder'])
#     # rnn_decoder.load_state_dict(ckpt['decoder'])
#     model.load_state_dict(ckpt['model'])

#     optimizer.load_state_dict(ckpt['optimizer'])
#     start_epoch = ckpt['epoch']
#     ckpt_performance = ckpt['performance']
#     ckpt_rng_state = ckpt['rng_state']
#     torch.set_rng_state(ckpt_rng_state)
#     return model, optimizer, start_epoch, ckpt_performance, ckpt_rng_state

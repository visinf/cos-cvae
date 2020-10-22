from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools
from tqdm import tqdm
import pickle
import argparse
import sys
import json
sys.path.insert(0,'./data')
from build_vocab_coco import Vocabulary
sys.path.insert(0,'./')
from modules.data_loader_noc import COCO_Dataset
from modules.textmodules import TextEncoderMask, TextEncoderObj, GaussPrior, TextDecoder
from modules.utils import sequence_cross_entropy_with_logits
import inflect

inflect = inflect.engine()

torch.multiprocessing.set_sharing_strategy('file_system')


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def get_properseq(bottom_up_features, bottom_up_classes, seq, mask_seq, obj_seq, seq_lengths):
	sorted_bottom_up_features = torch.zeros(bottom_up_features.size())
	if bottom_up_classes is not None:
		sorted_bottom_up_classes = torch.zeros(bottom_up_classes.size())
	else:
		sorted_bottom_up_classes = None
	sort_order = np.argsort(seq_lengths)[::-1]
	seq_sorted = np.zeros((len(seq),maxlength))
	mask_seq_sorted = np.zeros((len(seq),maxlength))
	if obj_seq is not None:
		obj_seq_sorted = np.zeros((len(seq),maxlength))
	else:
		obj_seq_sorted = None
	for idx in range(len(seq)):
		a = sort_order[idx]
		seq_sorted[idx,0:seq_lengths[a]] = seq[a,:seq_lengths[a]]
		mask_seq_sorted[idx,0:seq_lengths[a]] = mask_seq[a,:seq_lengths[a]]
		if obj_seq is not None:
			obj_seq_sorted[idx,0:5] = obj_seq[a,:5]
		sorted_bottom_up_features[idx] = bottom_up_features[a]
		if bottom_up_classes is not None:
			sorted_bottom_up_classes[idx] = bottom_up_classes[a]

	seq_sorted = seq_sorted.astype(np.float64)
	mask_seq_sorted = mask_seq_sorted.astype(np.float64)
	if obj_seq is not None:
		obj_seq_sorted = obj_seq_sorted.astype(np.float64)
	seq_lengths = seq_lengths[sort_order].astype(np.int32)
	seq_sorted = torch.LongTensor(seq_sorted.astype(np.float64)).to(device)
	mask_seq_sorted = torch.LongTensor(mask_seq_sorted.astype(np.float64)).to(device)
	if obj_seq is not None:
		obj_seq_sorted = torch.LongTensor(obj_seq_sorted.astype(np.float64)).to(device)
	seq_lengths = torch.LongTensor(seq_lengths.astype(np.int32)).to(device) 
	return sorted_bottom_up_features,sorted_bottom_up_classes,seq_sorted,mask_seq_sorted,obj_seq_sorted, seq_lengths, sort_order

def get_next_batch(dataloader, dataloader_iterator):
	try:
		data = next(dataloader_iterator)
	except StopIteration:
		dataloader_iterator = iter(dataloader)  
		data = next(dataloader_iterator)	
	return data, dataloader_iterator	


def get_loss( logits, targets, target_mask ):
	target_lengths = torch.sum(target_mask, dim=-1).float()

		# shape: (batch_size, )
	return target_lengths * sequence_cross_entropy_with_logits(
		logits, targets, target_mask, average=None
	)	


def fill_slot_with_class(mask_seq, alphas, bottom_up_classes):
	
	for j in range(1,mask_seq.size(1)):
		slot_idxs = torch.nonzero(mask_seq[:,j] == 4)[:,0].cpu().numpy().tolist()
		slot_idxs_p = torch.nonzero(mask_seq[:,j] == 3)[:,0].cpu().numpy().tolist()

		
		if len(slot_idxs) > 0:
			max_attn_idx = torch.argmax(alphas[slot_idxs,j-1,:],dim=1).cpu().numpy().tolist()
			max_attn_class = bottom_up_classes[slot_idxs,max_attn_idx]
			mask_seq[slot_idxs,j] = max_attn_class.long().cuda()

		if len(slot_idxs_p) > 0:
			max_attn_idx = torch.argmax(alphas[slot_idxs_p,j-1,:],dim=1).cpu().numpy().tolist()
			max_attn_class = bottom_up_classes[slot_idxs_p, [idx + 100 for idx in max_attn_idx]]
			mask_seq[slot_idxs_p,j] = max_attn_class.long().cuda()	
	return mask_seq



def load_dataset(data_path,vocab_path,coco_class,maxlength):

	image_data_path = str(data_path)+'/coco_train_2014_noc_adaptive_withclasses.h5'
	text_data_path = str(data_path)+'/annotations/captions_train2014.json'
	coco_train = COCO_Dataset(text_data_path,image_data_path,vocab_path,coco_class,None,maxlength)

	image_data_path = str(data_path)+'/coco_train_extra_2014_noc_adaptive_withclasses.h5'
	text_data_path = str(data_path)+'/annotations/captions_train2014.json'
	coco_unpaired = COCO_Dataset(text_data_path,image_data_path,vocab_path,coco_class,None,maxlength)
	return coco_train, coco_unpaired


if __name__== "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='params', help='Experiment settings.')
	args = parser.parse_args()

	config_setting = args.config
	config = json.loads(open('./params_noc.json', 'r').read())
	config = config[config_setting]
	data_path = config['pathToData']
	vocab_path = config['vocab_path']
	coco_class = config['coco_class']
	nn_dict_path = config['nn_dict_path']
	batch_size = int(config['batch_size'])
	maxlength = int(config['maxlength'])  
	latent_dim_tx  = int(config['latent_dim_tx'])
	meanimfeats_size = int(config['meanimfeats_size'])
	word_dim = int(config['word_dim'])
	mask_hidden_size = int(config['mask_hidden_size'])
	max_iterations = int(config['max_iterations'])


	nn_dict = pickle.load( open( str(nn_dict_path), "rb" ) )
	coco_idx_nn_map = nn_dict['coco_idx_vse_map']
	mask_seq_hash_map = nn_dict['mask_seq_hash_map']
	last_key = list(mask_seq_hash_map.keys())[0]

	vocab_class = pickle.load( open( str(vocab_path),'rb'))
	vocab = vocab_class.idx2word
	vocab_w2i = vocab_class.word2idx
	vocab_word2vec = []
	vocab_wordlist = [];
	for w_id,word in vocab.items():
		vocab_wordlist.append(word)
	vocab_size = len(vocab_wordlist)
	device = torch.device("cuda:0")


	coco_dataset_train, coco_dataset_unpaired = load_dataset(data_path,vocab_path,coco_class,maxlength) 
	coco_dataloader_train = torch.utils.data.DataLoader(coco_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
	coco_unpaired_dataloader = torch.utils.data.DataLoader(coco_dataset_unpaired, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
	coco_unpaired_iterator = iter(coco_unpaired_dataloader)		

	texEncMask = TextEncoderMask(word_dim,latent_dim_tx,batch_size, maxlength,meanimfeats_size,vocab_size,mask_hidden_size).cuda()
	texEncObj = TextEncoderObj(word_dim,latent_dim_tx,batch_size,maxlength, meanimfeats_size,vocab_size, mask_hidden_size).cuda()

	texDec= TextDecoder(batch_size, 2*latent_dim_tx, vocab_size).cuda()
	g_prior = GaussPrior(meanimfeats_size=meanimfeats_size, vocab_size=vocab_size, sent_emd_size=word_dim, sent_enc_size=latent_dim_tx, max_length=maxlength).cuda()

	optimizer = optim.SGD( itertools.chain(texEncMask.parameters(), texEncObj.parameters(), texDec.parameters(), g_prior.parameters() ), 
		lr=0.015, momentum=0.9, weight_decay=0.001)
	lr_scheduler = optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=lambda iteration: 1 - iteration / max_iterations )

	iteration = 0
	init_epoch = True
	epoch = 0

	while iteration <max_iterations:
		neighbors_list = []
		neighbors_caps = []
		neighbors_caps_masks = []
		oi_pred_sent_emb = []
		if epoch == 0:
			kl_rate = 0.2
		else:
			kl_rate = 1.0
		train_bar = tqdm(coco_dataloader_train)
		for data in train_bar:
			optimizer.zero_grad()
			caps = data[0]
			mask_cap = data[1]
			obj_caps = data[2]
			bottom_up_features = data[3]
			bottom_up_classes = data[4]
			seq_lengths = data[6]
			im_idx = data[-1]

			cap_idx = random.randint(0,4)
			seq  = torch.reshape(caps[0],(batch_size*5,maxlength))
			seq = seq[cap_idx::5,:]
			mask_seq  = torch.reshape(mask_cap[0],(batch_size*5,maxlength))
			mask_seq = mask_seq[cap_idx::5,:]
			seq_lengths = seq_lengths[:,cap_idx]
			obj_seq  = torch.reshape(obj_caps[0],(batch_size*5,maxlength))
			obj_seq = obj_seq[cap_idx::5,:]
			
			if not init_epoch:
				bottom_up_features_nn = bottom_up_features.clone()
				bottom_up_classes_nn = bottom_up_classes.clone()
				seq_nn = []
				mask_seq_nn = []
				seq_lengths_nn = []

				data_unpaired, coco_unpaired_iterator = get_next_batch(coco_unpaired_dataloader, coco_unpaired_iterator)

				bottom_up_features_nn = data_unpaired[3]
				bottom_up_classes_nn = data_unpaired[4]

				obj_seq_nn = data_unpaired[5]
				obj_seq_nn  = torch.reshape(obj_seq_nn,(batch_size,5))

				im_idx_nn = data_unpaired[-1]

				for j in range(batch_size):
				
					nn_mask_idx = random.randint(0,3)
					if int(im_idx_nn[j]) in coco_idx_nn_map:
						hash_map_idx = int(coco_idx_nn_map[int(im_idx_nn[j])][nn_mask_idx])
						if hash_map_idx in mask_seq_hash_map:
							seq_nn.append( mask_seq_hash_map[ hash_map_idx ][0]  )
							mask_seq_nn.append( mask_seq_hash_map[ hash_map_idx ][1]  )
							seq_lengths_nn.append( mask_seq_hash_map[ hash_map_idx ][2] )
						else:
							seq_nn.append( mask_seq_hash_map[ last_key ][0]  )
							mask_seq_nn.append( mask_seq_hash_map[ last_key ][1]  )
							seq_lengths_nn.append( mask_seq_hash_map[ last_key ][2] )
					else:
						seq_nn.append( mask_seq_hash_map[ last_key ][0]  )
						mask_seq_nn.append( mask_seq_hash_map[ last_key ][1]  )
						seq_lengths_nn.append( mask_seq_hash_map[ last_key ][2] )
				seq_nn = torch.tensor( np.array(seq_nn) ).cuda()
				mask_seq_nn = torch.tensor( np.array(mask_seq_nn) ).cuda()
				seq_lengths_nn = torch.tensor( np.array( seq_lengths_nn) )
			

			bottom_up_features,_,seq,mask_seq,obj_seq, seq_lengths, sort_order = get_properseq(bottom_up_features, None, seq.cpu().numpy(), mask_seq.cpu().numpy(), obj_seq.cpu().numpy(), seq_lengths.cpu().numpy())
			list_v_k = bottom_up_features.float()
			list_v_k = list_v_k[:,:,:meanimfeats_size]
			mask_q_means, mask_q_logs, mask_q_z, mask_hidden = texEncMask(mask_seq[:,1:].cuda(),(seq_lengths-1).cuda())

			obj_seq = obj_seq[:,:5] #maximum top-5 objects 
			obj_seq_len = (torch.ones(obj_seq.size(0),)*5).long().cuda()
			obj_q_means, obj_q_logs, obj_q_z = texEncObj(obj_seq.cuda(), list_v_k.cuda(), mask_hidden, mask_q_z, obj_seq_len )
			
			kl_loss =  g_prior(obj_enc=list_v_k.cuda(), x=seq.cuda(), p_z_t_1=None, hiddens=None, 
				q_means_mask=mask_q_means, q_logs_mask=mask_q_logs, q_z_mask=mask_q_z, 
				q_means_obj=obj_q_means, q_logs_obj=obj_q_logs, q_z_obj=obj_q_z, train=True, reverse=False)

			seq_lengths = seq_lengths - 1;
			q_z = torch.cat([mask_q_z,obj_q_z],dim=2)
			logp, alphas, attn_ent = texDec(seq.long().cuda(),None,q_z.cuda(),list_v_k.cuda(),seq_lengths.cuda())#

			tokens_mask = seq != 0 # Loss for valid sequences ignoring pads
			
			nll_tex = get_loss(logp, seq[:, 1:].contiguous(), tokens_mask[:, 1:].contiguous())
			nll_tex = torch.mean(nll_tex)
		
			kl_loss = torch.mean(kl_loss)
			


			if not init_epoch:
				bottom_up_features_nn,bottom_up_classes_nn,seq_nn,mask_seq_nn, obj_seq_nn, seq_lengths_nn, sort_order = get_properseq( bottom_up_features_nn, bottom_up_classes_nn, seq_nn.cpu().numpy(), mask_seq_nn.cpu().numpy(), obj_seq_nn.cpu().numpy(), seq_lengths_nn.cpu().numpy())

				mask_dec_in_nn = bottom_up_features_nn.float() 
				mask_dec_in_nn = mask_dec_in_nn[:,:,:2048]

				mask_q_means_nn, mask_q_logs_nn, mask_q_z_nn, mask_hidden_nn = texEncMask(mask_seq_nn[:,1:].clone().cuda(),(seq_lengths_nn - 1).cuda())

				obj_seq_nn = obj_seq_nn[:,:5].long()
				obj_seq_len_nn = (torch.ones(obj_seq_nn.size(0),)*5).long().cuda()
				obj_q_means_nn, obj_q_logs_nn, obj_q_z_nn = texEncObj(obj_seq_nn.cuda(), mask_dec_in_nn.cuda(), mask_hidden_nn, mask_q_z_nn, obj_seq_len_nn )

				kl_loss_nn =  g_prior(obj_enc=mask_dec_in_nn.cuda(), x=seq_nn.cuda(), p_z_t_1=None, hiddens=None, 
					q_means_mask=mask_q_means_nn, q_logs_mask=mask_q_logs_nn, q_z_mask=mask_q_z_nn, 
					q_means_obj=obj_q_means_nn, q_logs_obj=obj_q_logs_nn, q_z_obj=obj_q_z_nn, train=True, reverse=False)
			
				q_z_nn = torch.cat([mask_q_z_nn,obj_q_z_nn],dim=2)

				logp, alphas, _ = texDec(seq_nn.long().cuda(),None,q_z_nn.cuda(),mask_dec_in_nn.cuda(),(seq_lengths_nn-1).cuda())

				mask_seq_nn_org = mask_seq_nn.clone()
				mask_seq_nn = fill_slot_with_class(mask_seq_nn, alphas.clone().detach(), bottom_up_classes_nn)

				tokens_mask = mask_seq_nn != 0
				nll_tex_nn = get_loss(logp, mask_seq_nn[:, 1:].clone().contiguous(), tokens_mask[:, 1:].contiguous())
				nll_tex_nn = torch.mean(nll_tex_nn)
				kl_loss_nn = torch.mean(kl_loss_nn)
				loss = 0.8*nll_tex.cuda() + 0.8*kl_rate*kl_loss.cuda() + 0.15*kl_rate*kl_loss_nn.cuda() + 0.2*nll_tex_nn.cuda()

			else:
				nll_tex_nn = torch.tensor(0.)
				kl_loss_nn = torch.tensor(0.)
				loss = 1.0*nll_tex.cuda() + 1.0*kl_rate*kl_loss.cuda()


			loss.backward()
			nn.utils.clip_grad_norm_(texEncMask.parameters(), 12.5)
			nn.utils.clip_grad_norm_(texEncObj.parameters(), 12.5)
			nn.utils.clip_grad_norm_(g_prior.parameters(), 12.5)
			nn.utils.clip_grad_norm_(texDec.parameters(), 12.5)
			
			optimizer.step()
			lr_scheduler.step(iteration)
			iteration += 1

			train_bar.set_description('Train loss %.2f | Epoch %d -- Iteration ' % (loss.item(),epoch))

		init_epoch = False
		epoch += 1

		torch.save({
			'texEncoder_Mask_sd': texEncMask.state_dict(),
			'texEncoder_Obj_sd': texEncObj.state_dict(),
			'txtDecoder_sd': texDec.state_dict(),
			'g_prior_sd': g_prior.state_dict(),
			'optimizer': optimizer.state_dict(),
			}, './model_checkpoint_noc.pt')

	
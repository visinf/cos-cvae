from __future__ import print_function
import os
import random
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import itertools
from tqdm import tqdm
import pickle
import json
import sys
sys.path.insert(0,'./data')
from build_vocab_coco import Vocabulary
sys.path.insert(0,'./')
from modules.data_loader import COCO_Dataset
from modules.textmodules import TextDecoder, GaussPrior
from modules.utils import masked_mean
from GoogleConceptualCaptioning.eval_multi import eval_oracle, eval_spice_n, eval_div_stats

torch.multiprocessing.set_sharing_strategy('file_system')


def loglikehihoods_to_str( loglikelihood_seq, length ):
	curr_pred = [vocab_wordlist[int(p)] for p in loglikelihood_seq][0:length]
	curr_pred_str = []
	for curr_word in curr_pred:
		if (curr_word == "."):
			curr_pred_str += [curr_word]
			break;
		elif curr_word == "<end>":
			break;
		elif (curr_word != "<start>") and (curr_word != "<end>") and (curr_word != "<pad>") and (curr_word != "<ukn>"):
			curr_pred_str += [curr_word] #+ ' ';	

	return curr_pred_str

def load_dataset(data_path,vocab_path,coco_class,maxlength):
	image_data_path = str(data_path)+'/coco_val_2014_adaptive_withclasses.h5'
	text_data_path = str(data_path)+'/annotations/captions_val2014.json'
	coco_test_idxs_files = [str(data_path)+'/coco_test_mRNN.txt']
	return COCO_Dataset(text_data_path,image_data_path,vocab_path,coco_class,coco_test_idxs_files,True,maxlength)

if __name__== "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='params', help='Experiment settings.')
	args = parser.parse_args()

	config_setting = args.config
	config = json.loads(open('./params.json', 'r').read())
	config = config[config_setting]
	data_path = config['pathToData']
	vocab_path = config['vocab_path']
	coco_class = config['coco_class']
	batch_size = 1000#int(config['batch_size'])
	maxlength = int(config['maxlength']) 
	latent_dim_tx  = int(config['latent_dim_tx'])
	meanimfeats_size = int(config['meanimfeats_size'])
	word_dim = int(config['word_dim'])
	mask_hidden_size = int(config['mask_hidden_size'])
	max_iterations = int(config['max_iterations'])

	vocab_class = pickle.load( open( str(vocab_path),'rb'))
	vocab = vocab_class.idx2word
	vocab_w2i = vocab_class.word2idx
	vocab_word2vec = []
	vocab_wordlist = [];
	for w_id,word in vocab.items():
		vocab_wordlist.append(word)
	vocab_size = len(vocab_wordlist)
	device = torch.device("cuda:0")

	coco_dataset = load_dataset(data_path,vocab_path,coco_class,maxlength)
	coco_dataloader = torch.utils.data.DataLoader(coco_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

	texDec= TextDecoder(batch_size, 2*latent_dim_tx, vocab_size).cuda()
	g_prior = GaussPrior(meanimfeats_size=meanimfeats_size, vocab_size=vocab_size, sent_emd_size=word_dim, sent_enc_size=latent_dim_tx, max_length=maxlength).cuda()
	
	test_samples = 100
	beam_width = 2

	annFile = str(data_path)+'/annotations/captions_val2014.json'

	try:
		model = torch.load('./ckpts/model_checkpoint_release.pt')
		g_prior_st = model['g_prior_sd']
		g_prior.load_my_state_dict(g_prior_st)
		g_prior = g_prior.cuda()

		txtDecoder_st = model['txtDecoder_sd']
		texDec.load_my_state_dict(txtDecoder_st)
		texDec = texDec.cuda()
		file_load_success = True


	except Exception:
		print('ckpt not found')
		sys.exit(0)
				
	predictions = []
	results_json = []
	for i,data in tqdm(enumerate(coco_dataloader)):
		with torch.no_grad():
			caps = data[0]
			bottom_up_features = data[3]
			seq_lengths = data[4]
			im_idx = data[-1].cpu().numpy()

			seq  = torch.reshape(caps[0],(batch_size,5,maxlength))
			


			seq_in = None
			seq_lengths = maxlength

			mask_dec_in = bottom_up_features.float()
			mask_dec_in = mask_dec_in[:,:,:2048]
			#print('bottom_up_classes ',bottom_up_classes[0].view(-1))
			obj_enc_mask = torch.sum(torch.abs(mask_dec_in), dim=-1) > 0
			#print('obj_enc_mask ',obj_enc_mask[0])
			meanimfeats = masked_mean( mask_dec_in, obj_enc_mask.unsqueeze(-1) ,dim=1)
			all_preds_inf_latent = []
			for t_idx in range(test_samples):
				preds, _ = texDec(seq_in,None, None,mask_dec_in.cuda(),seq_lengths, prior_model=g_prior, train = False)
				for b in range(beam_width):
					all_preds_inf_latent.append(preds[:,:,b])

			all_preds_inf_latent = np.array(all_preds_inf_latent)
			all_preds_inf_latent = np.transpose(all_preds_inf_latent,(1,0,2))

			count =0
			for j in range(batch_size):
				for test_sample_idx in range(beam_width*test_samples):
					curr_pred_str = loglikehihoods_to_str(all_preds_inf_latent[j,test_sample_idx], maxlength-1)#.split(' ')[:-2]all_preds_inf_latent[j,test_sample_idx]#
					curr_pred_str = ' '.join(curr_pred_str)
					results_json += [{"image_id":int(im_idx[j]), "caption": curr_pred_str}]

	out = eval_oracle( annFile, results_json, 'ours', 'val')
	print(out['overall'])
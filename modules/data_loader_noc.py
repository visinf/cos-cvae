from __future__ import print_function, division
import json
import h5py
import numpy as np
import sys
import os
import csv
import torch
import base64
import copy

from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import pickle
sys.path.insert(0,'./data')
from build_vocab_coco import Vocabulary
import nltk
from nltk.tokenize import word_tokenize as tokenize
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import inflect

inflect = inflect.engine() # for handling plural forms in the captions for pseudo-supervision
csv.field_size_limit(sys.maxsize)


class COCO_Dataset(Dataset):
	"""COCO dataset."""

	def __init__(self, text_data_path,image_data_path, vocab_path,coco_class,coco_test_idxs_file,maxlength=20):
		"""
		Args:
			text_data_path (string): Path to the json file with captions or annotations.
			image_data_path (string): tsv file with image features
			vocab_path (string): Path to the vocab pickle file.
			coco_class (coco_class): list of coco classes removed from captions to get contexual descriptions
		"""
		"""
		Returns:
		captions: The ground-truth captions 
		bottom_up_features: Features from bounding boxes extracted from Faster-RCNN [4]
		bottom_up_classes: Classes from bounding boxes corresponding to bottom_up_features
		x_m_caps: Contextual descriptions after removing COCO classes from captions
		caption_length: Caption lengths
		x_o_caps: Object descriptions of COCO classes from captions
		image_idx: image-id  in the annotations file



		"""

		self.image_data_path = image_data_path
		self.vocab = pickle.load(open(str(vocab_path),'rb'))
		self.word2idx = self.vocab.word2idx
		self.idx2workd = self.vocab.idx2word
		self.maxlength = maxlength
		coco_class_all = []
		coco_class_name = open(coco_class, 'r')
		for line in coco_class_name:
			coco_class = line.rstrip("\n").split(', ')
			coco_class_all.append(coco_class)
		self.wtod = {}
		for i in range(len(coco_class_all)):
			for w in coco_class_all[i]:
				self.wtod[w] = i
		self.wtol = {}
		lemmatizer = WordNetLemmatizer()
		for w in self.word2idx:
			tok = tokenize(w)[0]
			self.wtol[w] = lemmatizer.lemmatize(tok)
		self.dtoi = {w:i+1 for i,w in enumerate(self.wtod.keys())}

		self.imagefeatures  = {}
		self.imagefeatures_h5 = h5py.File(open(self.image_data_path, 'rb'), 'r')
		if 'val' in self.image_data_path:
			imkeyfile_ = open(str(coco_test_idxs_file),'r').read().split('\n')[:5000]
			imkeylist = []
			for images in imkeyfile_:
				imagekey = os.path.splitext(os.path.basename(images))[0]
				imagekey  = int(imagekey.split('_')[-1])
				imkeylist.append(imagekey)
		for k in self.imagefeatures_h5.keys():
			self.imagefeatures[k] = self.imagefeatures_h5[k][()]#.astype(np.float32)
		
		if 'val' in self.image_data_path:
			test_idx = [int(np.where(self.imagefeatures["image_id"]==_key)[0]) for _key in imkeylist]
			for k in self.imagefeatures_h5.keys():
				self.imagefeatures[k] = np.delete(self.imagefeatures[k],test_idx,axis=0)
		self.imagefeatures_h5.close()

		self.inv_annotations = {}
		self.image_filenames = {}

		annotations = json.load(open(text_data_path))

		for c in annotations["annotations"]:
			if str(c["image_id"]) in self.inv_annotations.keys():
				self.inv_annotations[str(c["image_id"])].append(c["caption"])
			else:
				self.inv_annotations[str(c["image_id"])] = []
				self.inv_annotations[str(c["image_id"])].append(c["caption"])
		
		for c in annotations["images"]:
			if str(c["id"]) not in self.image_filenames.keys():
				self.image_filenames[str(c["id"])] = c["file_name"]
		
		self.blacklist_classes = {
				"auto part":'vehicle', "bathroom accessory":'furniture', "bicycle wheel":'bicycle', "boy":'boy',
				"door handle":'door', "fashion accessory":'clothing', "footwear":'shoes', "human arm":'person',
				"human beard":'person', "human body":'person', "human ear":'person', "human eye":'person', "human face":'person', "human foot":'person',
				"human hair":'person', "human hand":'person', "human head":'person', "human leg":'person', "human mouth":'person', "human nose":'person',
				"land vehicle":'vehicle', "plumbing fixture":'toilet',
				"seat belt":'vehicle', "vehicle registration plate":'vehicle',
				"face":'person',"hair":'person',"head":'person',"ear":'person',"tail":'giraffe',"neck":'giraffe',
				"hat":'person',"helmet":'person',"nose":'person',"tire":'bus',"tour":'bus',"hand":'person',"shadow":'person'
			}

		self.punctuations = [
			"''", "'", "``", "`", "(", ")", "{", "}",
			".", "?", "!", ",", ":", "-", "--", "...", ";"
		]

		self.vg_classes_to_vocab = {}
		self.vg_classes_to_vocab[0] = 0
		self.vg_classes_to_vocab_p = {}
		self.vg_classes_to_vocab_p[0] = 0
		classes = ['__background__']
		vg_obj_counter = 1
		with open('./data/visual_genome_classes.txt') as f:
			for _object in f.readlines():
				#classes.append(object.split(',')[0].lower().strip())
				_object = _object.split(',')[0].lower().strip()
				if _object in self.word2idx:
					if _object in self.blacklist_classes:
						self.vg_classes_to_vocab[vg_obj_counter] = self.word2idx[self.blacklist_classes[_object]]
						self.vg_classes_to_vocab_p[vg_obj_counter] = self.word2idx[self.blacklist_classes[_object]]
					else:
						self.vg_classes_to_vocab[vg_obj_counter] = self.word2idx[_object]
						if inflect.singular_noun( _object ) == False:
							_object_p = inflect.plural(_object)
						else:
							_object_p = _object
						if _object_p in self.word2idx:
							self.vg_classes_to_vocab_p[vg_obj_counter] = self.word2idx[_object_p]
						else:
							self.vg_classes_to_vocab_p[vg_obj_counter] = self.word2idx[_object]
				else:
					self.vg_classes_to_vocab[vg_obj_counter] = 0
					self.vg_classes_to_vocab_p[vg_obj_counter] = 0
				vg_obj_counter += 1

	def get_det_word(self,captions, ngram=2):
		# get the present category. taken from NBT []
		indicator = []
		stem_caption = []
		for s in captions:
			tmp = []
			for w in s:
				if w in self.wtol.keys():	
					tmp.append(self.wtol[w])
			stem_caption.append(tmp)
			indicator.append([(-1, -1, -1)]*len(s)) # category class, binary class, fine-grain class.

		ngram_indicator = {i+1:copy.deepcopy(indicator) for i in range(ngram)}
		# get the 2 gram of the caption.
		
		for i, s in enumerate(stem_caption):
			for n in range(ngram,0,-1):
				#print('stem_caption ', s)
				for j in range(len(s)-n+1):
					ng = ' '.join(s[j:j+n])
					#print('ng ', ng)
					# if the n-gram exist in word_to_detection dictionary.
					if ng in self.wtod and indicator[i][j][0] == -1: #and self.wtod[ng] in pcats: # make sure that larger gram not overwright with lower gram.
						bn = (ng != ' '.join(captions[i][j:j+n])) + 1
						fg = self.dtoi[ng]
						#print('fg ',fg)
						ngram_indicator[n][i][j] = (int(self.wtod[ng]), int(bn), int(fg))
						indicator[i][j:j+n] = [(int(self.wtod[ng]), int(bn), int(fg))] * n
			#sys.exit(0)
		return ngram_indicator

	def get_caption_seq(self,captions):
		cap_seq = np.zeros([len(captions), self.maxlength])
		masked_cap_seq = np.zeros([len(captions), self.maxlength])
		object_cap_seq = np.zeros([len(captions), self.maxlength])
		det_indicator = self.get_det_word(captions, ngram=2)
		for i, caption in enumerate(captions):
			j = 0
			k = 0
			o = 0
			while j < len(caption) and j < self.maxlength:
				is_det = False
				for n in range(2, 0, -1):
					if det_indicator[n][i][j][0] != -1:
						cap_seq[i,k] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
						if inflect.singular_noun( caption[j] ) == False:
							masked_cap_seq[i,k] = 4 #placeholder in vocab for singular visual genome class object
						else:
							masked_cap_seq[i,k] = 3 #placeholder in vocab for plural visual genome class object
						object_cap_seq[i,o] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
						is_det = True
						j += n # skip the ngram.
						o += 1
						break
				if is_det == False:
					cap_seq[i,k] = int(self.word2idx[caption[j]] if caption[j] in self.word2idx.keys() else 0)
					masked_cap_seq[i,k] = cap_seq[i,k]
					j += 1
				k += 1
		return cap_seq, masked_cap_seq, object_cap_seq


	def __len__(self):
		return len(self.imagefeatures["image_id"])

	def close(self):
		self.imagefeatures.close()


	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		captions = []
		masked_images = []
		x_m_caps = []
		x_o_caps = []
		image_idx = self.imagefeatures["image_id"][idx]
		if str(image_idx) in self.image_filenames.keys():  
			i=idx
		else:
			image_idx = self.imagefeatures["image_id"][1]
			i=1
		bottom_up_features = []
		bottom_up_classes = []
		bottom_up_classes_p = []
		bottom_up_classes_top = []
		total_boxes = 0
		nms_boxes = []
		nms_class_names = []
		for j in range(self.imagefeatures["num_boxes"][i]):
			bottom_up_features.append( self.imagefeatures["features"][i][(j)*2048:((j)+1)*2048] )
			bottom_up_classes.append(self.vg_classes_to_vocab[ self.imagefeatures["classes"][i][j] ])
			bottom_up_classes_p.append(self.vg_classes_to_vocab_p[ self.imagefeatures["classes"][i][j] ])
		bottom_up_features = np.array(bottom_up_features)
		bottom_up_classes = np.array(bottom_up_classes)
		bottom_up_classes_p = np.array(bottom_up_classes_p)
		if bottom_up_features.shape[0] < 100:
			bottom_up_features_pad = np.zeros((100 - bottom_up_features.shape[0], 2048))
			bottom_up_classes_pad = np.zeros((100 - bottom_up_classes.shape[0],))
			bottom_up_features = np.concatenate([bottom_up_features,bottom_up_features_pad],axis=0) 
			bottom_up_classes = np.concatenate([bottom_up_classes,bottom_up_classes_pad],axis=0)
			bottom_up_classes_p = np.concatenate([bottom_up_classes_p,bottom_up_classes_pad],axis=0)
		bottom_up_classes = np.concatenate([bottom_up_classes,bottom_up_classes_p], axis=0)
		sorted_score_idx = np.argsort(self.imagefeatures["scores"][i])[::-1]
		for j in sorted_score_idx[:5]:
			if self.imagefeatures["scores"][i][j] > 0.6:
				bottom_up_classes_top.append(self.vg_classes_to_vocab[ self.imagefeatures["classes"][i][j] ])
		bottom_up_classes_top = np.array(bottom_up_classes_top).astype(np.float32)
		if bottom_up_classes_top.shape[0] < 5:
			bottom_up_classes_top_pad = np.zeros((5 - bottom_up_classes_top.shape[0],)).astype(np.float32)
			bottom_up_classes_top = np.concatenate([bottom_up_classes_top,bottom_up_classes_top_pad]).astype(np.float32)
		caps = self.inv_annotations[str(image_idx)]
		targets = []
		caption_length = []
		for c in caps:
			caption_tokens = nltk.tokenize.word_tokenize(c.lower().strip())
			caption_tokens = [ct for ct in caption_tokens if ct not in self.punctuations]
			caption = []
			caption.append('<start>')
			caption.extend(caption_tokens)
			caption = caption[0:(self.maxlength-1)]
			caption.append('<end>')
			targets.append(caption)
			caption_length.append(len(caption))

		gt_cap, mask_cap, obj_cap = self.get_caption_seq([targets[i] for i in range(5)]) #5 human annotations
		caption_length = [caption_length[i] for i in range(5)]
		captions.append(gt_cap)
		x_m_caps.append(mask_cap)
		x_o_caps.append(obj_cap)
		return captions, x_m_caps, x_o_caps, bottom_up_features, bottom_up_classes, bottom_up_classes_top, np.array(caption_length), image_idx


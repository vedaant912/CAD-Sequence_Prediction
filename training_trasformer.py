import time
from models.c3d import c3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import tqdm
import sys
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import os
from decoder import CaptionDecoder
from utils.utils import save_checkpoint, log_gradient_norm, set_up_causal_mask
from dataloader import CadDataset
from utils.decoding_utils import greedy_decoding
from nltk.translate.bleu_score import corpus_bleu
import json
import pandas as pd

label_encoding_dictionary = {'Sägen': 2,
 'Fräsen': 3,
 'Härten/Oberfläche': 4,
 'Flachschleifen': 5,
 'Koordinatenschleifen': 6,
 'HSC-Fräsen': 7,
 'Messen': 8,
 'Laserbeschriftung': 9,
 'Drehen': 10,
 'Rundschleifen': 11,
 'Drahterodieren': 12,
 'Polieren': 13,
 'Senkerodieren': 14,
 'Fremdvergabe': 15,
 'Startlochbohren': 16,
 '<start>': 0,
 '<end>': 1}

label_encoding_dictionary = {value:key for key, value in label_encoding_dictionary.items()}


def evaluate(subset, encoder, decoder, config, device):
	"""Evaluates (BLEU score) caption generation model on a given subset."""

	batch_size = config["batch_size"]["eval"]
	max_len = config["max_len"]
	bleu_w = config["bleu_weights"]

	# Ids for special tokens
	sos_id = subset._start_idx
	eos_id = subset._end_idx
	pad_id = subset._pad_idx

	references_total = []
	predictions_total = []

	print("Evaluating model.")
	for x_img, y_caption in subset.inference_batch(batch_size):
		x_img = x_img.to(device)

		x_img = x_img.view(x_img.shape[0],1,256,256,256)
		x_img = x_img.to(device)

		# Extract image features
		img_features = encoder(x_img)
		img_features = img_features.view(img_features.size(0), img_features.size(2)*img_features.size(3)*img_features.size(4), img_features.size(1))
		img_features = img_features.detach()

		# Get the caption prediction for each image in the mini-batch
		predictions = greedy_decoding(decoder, img_features, sos_id, eos_id, pad_id, label_encoding_dictionary, max_len, device)

		y_caption_new = list()
		for i in y_caption:
			temp = json.loads(i[0])
			temp.remove(0)
			temp.remove(1)
			y_caption_new.append(temp)

		y_caption_final = list()
		for i in y_caption_new:
			temp=list()
			for j in i:
				temp.append(label_encoding_dictionary[j])
			y_caption_final.append(temp)

		y_new_caption = list()
		for i in y_caption_final:
			temp = [i]
			y_new_caption.append(temp)

		print('Captions :', y_new_caption)
		print('Prediction :', predictions)
		print('\n')
		
		references_total += y_new_caption
		predictions_total += predictions

	print('Captions : ')
	print(references_total)
	print('Predictions')
	print(predictions_total)
	print('\n')

	# Evaluate BLEU score of the generated captions
	bleu_1 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-1"]) * 100
	bleu_2 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-2"]) * 100
	bleu_3 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-3"]) * 100
	bleu_4 = corpus_bleu(references_total, predictions_total, weights=bleu_w["bleu-4"]) * 100
	bleu = [bleu_1, bleu_2, bleu_3, bleu_4]
	return bleu

def main(args):

	config_path = "config.json"
	with open(config_path, "r", encoding="utf8") as f:
		config = json.load(f)

	train(config, args)

def train(config, args):

	######################### GETTING DATA AND SETTING PARAMETERS #################################################
	# Pytorch train and test sets
	data_path = './Voxels_transformers/training_files.csv'
	data_path_val = './Voxels_transformers/validation_list.csv'

	train_set = CadDataset(config, data_path, True)
	val_set = CadDataset(config, data_path_val, False)

	train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle=False, drop_last=True)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	criterion = nn.CrossEntropyLoss()
	
	#################################################################################################################
	causal_mask = set_up_causal_mask(config["max_len"], device)
		
	encoder = c3d()
	decoder = CaptionDecoder(config)

	encoder = encoder.to(device)
	decoder = decoder.to(device)

	optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
	train_config = config['train_config']

	start_time = time.strftime("%b-%d_%H-%M-%S")
	train_step = 0
	losses = list()
	train_bleu_value = list()
	val_bleu_value = list()

	for epoch in tqdm.trange(train_config['num_of_epochs'], position=0, leave=True):
		print("Epoch: ", epoch)

		encoder.train()
		decoder.train()

		running_losses = 0

		for i, (voxels, x_words, y, tgt_padding_mask) in enumerate(train_loader):

			optimizer.zero_grad()
			train_step += 1

			# make the captions for targets and teacher forcer
			y = y.type(torch.LongTensor)
			y = y.to(device)

			tgt_padding_mask = tgt_padding_mask.to(device)
			x_words = x_words.to(device)

			voxels = voxels.view(voxels.shape[0],1,256,256,256)
			voxels = voxels.to(device)

			img_features = encoder(voxels)
			img_features = img_features.view(img_features.size(0), img_features.size(2)*img_features.size(3)*img_features.size(4), img_features.size(1))
			img_features = img_features.detach()

			y_pred =  decoder(x_words, img_features, tgt_padding_mask, causal_mask)
			tgt_padding_mask = torch.logical_not(tgt_padding_mask)
			y_pred = y_pred[tgt_padding_mask]

			y = y[tgt_padding_mask]
			
			loss = criterion(y_pred, y.long())
			
			running_losses += loss.item()

			loss.backward()
			optimizer.step()


		losses.append(running_losses / len(train_loader))
		save_checkpoint(encoder, decoder, optimizer, start_time, epoch)

		with torch.no_grad():
			encoder.eval()
			decoder.eval()

			# Evaluate model performance on subsets
			train_bleu = evaluate(train_set, encoder, decoder, config, device)
			valid_bleu = evaluate(val_set, encoder, decoder, config, device)
			
			print('BLUE : ')
			print(train_bleu)
			print(valid_bleu)

			train_bleu_value.append(train_bleu)
			val_bleu_value.append(valid_bleu)

			decoder.train()

		np.save('./results/losses',np.array(losses))
		np.save('./results/train_bleu',np.array(train_bleu_value))
		np.save('./results/val_bleu', np.array(val_bleu_value))

		stats = 'Epoch [%d/%d], Loss: %.4f' % (epoch, train_config['num_of_epochs'], running_losses/len(train_loader))
		print('\r' + stats, end="")

	torch.save(encoder.state_dict(), './trained_models/encoder.dict')
	torch.save(decoder.state_dict(), './trained_models/decoder.dict')

	return 0


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--model_path', type=str, default='trained_models/',help='Path for saving trained models')
	parser.add_argument('--hidden_size', type=int, default=512, help='dimension for lstm hidden states')
	parser.add_argument('--batch_size', type=int, default=2)
	parser.add_argument('--learning_rate', type=float, default=0.001)


	args = parser.parse_args()

	main(args)

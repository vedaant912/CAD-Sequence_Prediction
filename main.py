import os
import sys
import json
import numpy as np
from util.get_data import get_data
import torch
from torch import nn
from torch import optim
import argparse
import tqdm
from model import generate_model
from train_2 import train_epoch

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('model', type=str, default='basic_model', help='Choose the model to be trained. (basic_model, c3d')
	parser.add_argument('--model_path', type=str, default='trained_models/',help='Path for saving trained models')
	parser.add_argument('--voxels_data_path', type=str, default='./data/three_ops/voxel_representations/voxel_' , help='Path for voxel representations.')
	parser.add_argument('--voxels_label_path', type=str, default='./data/three_ops/voxel_data.csv', help='Path for voxel data labels')
	parser.add_argument('--embed_size', type=int, default=256, help='dimensions for word embedding vectors')
	parser.add_argument('--hidden_size', type=int, default=512, help='dimension for lstm hidden states')
	parser.add_argument('--num_of_layers', type=int, default=1, help='number of layers in lstm')
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	parser.add_argument('--save_step', type=int, default=500)
	parser.add_argument('--forward_approach', type=str, default='teacher_forcer', help='Choose the forward method. simple/teacher_forcer')
	parser.add_argument('--proportion', type=float, default=0.1, help='Choose the proportion of data for training between 1 and 100 percent.')

	args = parser.parse_args()

	if torch.cuda.is_available():
	    device = torch.device('cuda')
	else:
	    device = torch.device('cpu')

	encoder, decoder = generate_model(args)

	encoder.to(device)
	decoder.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)

	# Get data.
	train_x, train_y, test_x, test_y, val_x, val_y = get_data(args.voxels_data_path, args.voxels_label_path, args.proportion)

	train = torch.utils.data.TensorDataset(train_x,train_y)
	test = torch.utils.data.TensorDataset(test_x,test_y)
	val = torch.utils.data.TensorDataset(val_x,val_y)

	train_loader = torch.utils.data.DataLoader(train, batch_size = args.batch_size, shuffle = False)
	test_loader = torch.utils.data.DataLoader(test, batch_size = args.batch_size, shuffle = False)
	val_loader = torch.utils.data.DataLoader(val, batch_size = args.batch_size, shuffle = False)

	vocab_size = 5

	train_epoch(train_loader,val_loader, encoder, decoder, criterion, optimizer, args, device, vocab_size)
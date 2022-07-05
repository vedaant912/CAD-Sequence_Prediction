import torch
from torch import nn

from models import basic_model, c3d

def generate_model(opt):

	if opt.model == 'basic_model':
		print('BASIC MODEL')
		vocab_size = 5
		encoder, decoder =	basic_model.get_model(opt.embed_size, opt.hidden_size, vocab_size, opt.batch_size)

	elif opt.model == 'c3d':
		print('c3d')
		vocab_size = 5
		encoder, decoder = c3d.get_model(opt.embed_size, opt.hidden_size, vocab_size, opt.batch_size)


	return encoder, decoder
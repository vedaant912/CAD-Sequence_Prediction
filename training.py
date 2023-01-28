from models.basic_model import CNNModel, DecoderRNN
from util.get_data import get_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import tqdm
import sys
import numpy as np
import argparse
import os
from torchmetrics.functional import bleu_score

label_encoding_dictionary = {0:'<start>',2:'sp',3:'cu',4:'cy',1:'<end>'}

def calculate_bleu_score(outputs, labels):

    actual = list()
    predicted = list()
    
    list_predicted_caption=[]
    list_labels=[]
    
    for j,i in enumerate(outputs):
        predicted = i.argmax(1)
        
        predicted_caption = ''
        predicted = predicted.cpu().detach().numpy()
        for k in predicted:
            word = label_encoding_dictionary[k]
            predicted_caption += word + ' '
            
        label_caption=''
        labelsss = labels[j].cpu().detach().numpy()
        for k in labelsss:
            word = label_encoding_dictionary[k]
            label_caption += word + ' '
        
        list_predicted_caption.append(predicted_caption.rstrip())
        list_labels.append(label_caption.rstrip())
        
    return bleu_score(list_predicted_caption,list_labels,n_gram=3)

def main(args):

	# label_encoding_dictionary = {'<start>':0,'sp':2,'cu':3,'cy':4,'<end>':1}

	# Getting the Training/Testing Data.
	train_x, train_y, test_x, test_y, val_x, val_y = get_data(args.voxels_data_path, args.voxels_label_path, args.proportion)

	# Pytorch train and test sets
	train = torch.utils.data.TensorDataset(train_x,train_y)
	test = torch.utils.data.TensorDataset(test_x,test_y)
	val = torch.utils.data.TensorDataset(val_x,val_y)

	train_loader = torch.utils.data.DataLoader(train, batch_size = args.batch_size, shuffle = False)
	test_loader = torch.utils.data.DataLoader(test, batch_size = args.batch_size, shuffle = False)
	val_loader = torch.utils.data.DataLoader(val, batch_size = args.batch_size, shuffle = False)

	if torch.cuda.is_available():
	    device = torch.device('cuda')
	else:
	    device = torch.device('cpu')

	encoder = CNNModel()

	# Input related to RNN
	vocab_size = len(label_encoding_dictionary)
	criterion = nn.CrossEntropyLoss()
	decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size)

	optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)

	encoder.to(device)
	decoder.to(device)

	losses = list()
	val_losses = list()
	bleu_score_values = list()

	for epoch in tqdm.trange(args.epochs, position=0, leave=True):

	    for i, (voxels, labels) in enumerate(train_loader):
	        
	        # zero the gradients
	        decoder.zero_grad()
	        encoder.zero_grad()
	        
	        # set decoder and encoder into train mode
	        encoder.train()
	        decoder.train()
	        
	        # make the captions for targets and teacher forcer
	        labels = labels.to(device)

	        # Make the voxels ready as the input to the encoder (CNN)
	        voxels = voxels.view(args.batch_size,1,64,64,64)
	        voxels = voxels.to(device)
	        
	        # Passing the inputs through the CNN-RNN / Encoder-Decoder model.
	        features = encoder(voxels)
	        outputs = decoder(features, labels, args.forward_approach)
	                
	        # Calculate the loss
	        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

	        # Backward pass
	        loss.backward()
	        
	        # Update the parameters in the optimizer
	        optimizer.step()	        

	        with torch.no_grad():
            
	            # set the evaluation mode
	            encoder.eval()
	            decoder.eval()

	            for voxels, labels in val_loader:

	            	labels = labels.to(device)
	            	voxels = voxels.view(args.batch_size,1,64,64,64).to(device)
	            	
	            	features = encoder(voxels)
	            	outputs = decoder(features, labels, args.forward_approach)

            		val_loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

	        losses.append(loss.item())
	        val_losses.append(val_loss.item())
	        bleuScore=calculate_bleu_score(outputs, labels)
	        bleu_score_values.append(bleuScore.item())

	        np.save('losses', np.array(losses))
       		np.save('val_losses', np.array(val_losses))
       		np.save('bleu_score_values', np.array(bleu_score_values))

	        # Get training statistics.
	        stats = 'Epoch [%d/%d], Loss: %.4f, Validation Loss: %.4f, BLEU SCORE: %.4f' % (epoch, args.epochs, loss.item(), val_loss.item(),bleuScore.item())
	        
	        # Print training statistics (on same line).
	        print('\r' + stats, end="")

	        if (i+1) % args.save_step == 0:
	        	torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
	        	torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
	        sys.stdout.flush()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

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
	parser.add_argument('forward_approach', type=str, help='Choose the forward method. simple/teacher_forcer')
	parser.add_argument('--proportion', type=float, default=0.1, help='Choose the proportion of data for training between 1 and 100 percent.')

	args = parser.parse_args()

	main(args)

	
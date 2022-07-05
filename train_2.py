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

def train_epoch(train_loader, val_loader, encoder, decoder, criterion, optimizer, args, device, vocab_size):

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
	        outputs = decoder(features, labels)
	                
	        # Calculate the loss
	        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

	        # Backward pass
	        loss.backward()
	        
	        # Update the parameters in the optimizer
	        optimizer.step()

	        bleuScore=calculate_bleu_score(outputs, labels)
	        bleu_score_values.append(bleuScore.item())
	        
	        with torch.no_grad():
            
	            # set the evaluation mode
	            encoder.eval()
	            decoder.eval()

	            val_loss = 0

	            for j, (voxels, labels) in enumerate(val_loader):

	            	labels = labels.to(device)
	            	voxels = voxels.view(args.batch_size,1,64,64,64).to(device)
	            	
	            	features = encoder(voxels)
	            	outputs = decoder(features, labels)

            		val_loss += criterion(outputs.view(-1, vocab_size), labels.view(-1))

	        losses.append(loss.item())
	        val_losses.append(val_loss.item() / j)
	        

	        np.save('losses', np.array(losses))
       		np.save('val_losses', np.array(val_losses))
       		np.save('bleu_score_values', np.array(bleu_score_values))

	        # Get training statistics.
	        stats = 'Epoch [%d/%d], Loss: %.4f, Validation Loss: %.4f, BLEU SCORE: %.4f' % (epoch, args.epochs, loss.item(), val_loss.item() / j,bleuScore.item())
	        
	        # Print training statistics (on same line).
	        print('\r' + stats, end="")

	        if (i+1) % args.save_step == 0:
	        	torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
	        	torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
	        sys.stdout.flush()



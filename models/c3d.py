import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

class c3d(nn.Module):

	def __init__(self):

		super(c3d, self).__init__()

		self.group1 = nn.Sequential(
			nn.Conv3d(1, 64, kernel_size=3, padding=1),
			nn.BatchNorm3d(64),
			nn.ReLU(),
			nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
		self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
		self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
		self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))


		self.fc1 = nn.Linear(112*512, 128)
		self.fc2 = nn.Linear(128, 256)
		self.relu = nn.LeakyReLU()
		self.batch=nn.BatchNorm1d(128)
		self.drop=nn.Dropout(p=0.15)


	def forward(self, x):
		out = self.group1(x)
		out = self.group2(out)
		out = self.group3(out)
		out = self.group4(out)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=6):
        super(DecoderRNN, self).__init__()
        
        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
    
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
    
        # activations
        self.softmax = nn.Softmax(dim=1)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)

        self.fc_2 = nn.Linear(in_features=self.hidden_size, out_features=self.embed_size)
    
    def forward(self, features, captions, forward_approach):
        
        # batch size
        batch_size = features.size(0)
        
        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
    
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()
    
        captions_embed = self.embed(captions)
        for t in range(captions.size(1)):
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
            else:
                hidden_state, cell_state = self.lstm_cell(captions_embed[:, t-1, :], (hidden_state, cell_state))
            
            out = self.fc_out(hidden_state)
            
            # build the output tensor
            outputs[:, t, :] = out
        
        return outputs

def get_model(embed_size, hidden_size, vocab_size, batch_size):
    """
    Returns the model.
    """
    print('Inside c3d model')
    
    encoder = c3d()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    return encoder, decoder
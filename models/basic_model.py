import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(1, 16)
        self.conv_layer2 = self._conv_layer_set(16, 32)
        self.fc1 = nn.Linear(14**3*32, 128)
        self.fc2 = nn.Linear(128, 256)
        self.relu = nn.LeakyReLU()
        self.batch=nn.BatchNorm1d(128)
        self.drop=nn.Dropout(p=0.15)
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3),stride=[1,1,1],padding=[0,0,0]),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer
    

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        
        return self.drop(self.relu(out))


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
    
    def forward(self, features, captions):
        
        # batch size
        batch_size = features.size(0)
        
        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
    
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()

        # pass each steps output to next steps input.
        # if forward_approach == 'simple':

        #     for t in range(captions.size(1)):
        #         if t == 0:
        #             hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
        #         else:
        #             # output = self.fc_2(hidden_state)
        #             most_probable = out.argmax(1)
        #             output = self.embed(most_probable)
        #             hidden_state, cell_state = self.lstm_cell(output, (hidden_state, cell_state))
                
        #         out = self.fc_out(hidden_state)

        #         outputs[:, t, :] = out

        # elif forward_approach == 'teacher_forcer':

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
    encoder = CNNModel()
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    return encoder, decoder
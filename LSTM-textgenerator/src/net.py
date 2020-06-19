###This is the network class
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class LSTMNetwork(nn.Module):
    
    def __init__(self, input_size, seq_size, lstm_size, embedding_size, lstm_layers=1):
        
        #Call the father function
        super(LSTMNetwork, self).__init__()
        
        # size of the words sequence
        self.seq_size = seq_size
        
        # size of hidden layer 
        self.lstm_size = lstm_size
        
        # vocab_size
        self.input_size = input_size
        
        # embedding layer; In an embedding, words are represented by dense vectors
        # where a vector represents the projection of the word into a continuous vector space.
        # see https://pytorch.org/docs/master/nn.html#torch.nn.Embedding
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence
        # batch_first=True causes input/output tensors to be of shape (batch_dim, seq_dim, feature_dim)
        # see (https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            lstm_layers,
                            batch_first=True)
        
        # Define dense layer
        self.out = nn.Linear(lstm_size, input_size)
        
        # Define a softmax layer 
        self.prob = nn.LogSoftmax(dim=1)

        # Initialize weights
        self.init_weights()
        
    #Initialize weights
    def init_weights(self):
        initrange = 0.1      
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)
         
        
    def forward(self, x, prev_state, batch_size):
            
        # embedding the input
        x = self.embedding(x)
        
        # LSTM
        x, state = self.lstm(x, prev_state) 
        
        # Stack up lstm outputs
        x = x.contiguous().view(-1,self.lstm_size)
        
        # dense layer + logsoftmax
        x = self.prob(self.out(x))
        
        # reshape into (batch_size, seq_length, vocab_size)
        x = x.view(batch_size, -1, self.input_size)

        # return new cell_state and new hidden_state to propagate 
        return x, state

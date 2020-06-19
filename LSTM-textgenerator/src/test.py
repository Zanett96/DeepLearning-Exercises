import torch
import os
import torch.nn as nn
import numpy as np
import re
import torch.nn.functional as F
from random import choice
from net import LSTMNetwork
import argparse

## based on some seed text, generate the following words
def predict(model, phrase, input_size, lenght, word2idx, idx2word, temperature, device):
    
    # tell the net we're testing
    model.eval()

    # reset states and move to GPU
    state_h, state_c = torch.zeros(1, 1, lstm_size), torch.zeros(1, 1, lstm_size)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    
    # pass trough the input phrase 
    for w in phrase:
        word = torch.LongTensor([[word2idx[w]]]).to(device)
        _, (state_h, state_c) = model(word, (state_h, state_c), 1)
    top_i = word
    
    for p in range(lenght):
        
        word = torch.LongTensor([[int(top_i)]]).to(device)
        
         # forward the word trough the LSTM
        output, (state_h, state_c) = model(word, (state_h, state_c), 1)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add translated word to phrase
        phrase.append(idx2word[int(top_i)][0])
   
    return phrase

## retrieve vocabularies 
v1=np.load("word2idx.npy", allow_pickle = True)
v2=np.load("idx2word.npy", allow_pickle = True)

word2idx = dict()
idx2word = dict()
for w in v1.item():
    word2idx[w] = v1.item().get(w)  
for i in v2.item():
    idx2word[i] = v2.item().get(i)
    

lstm_size = 512
embedding_size = 356
seq_size = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Retrieve network model
net = LSTMNetwork(len(word2idx), seq_size, lstm_size, embedding_size)
net.load_state_dict(torch.load('.\\' + 'netModel.txt'))

# Parse the text seed and the generated text's lenght from command line
parser = argparse.ArgumentParser(description='Generate text from a certain seed')

parser.add_argument('--seed', type=str, default='darcy was', help='Initial text of the generation')
parser.add_argument('--length',   type=int, default=100, help='How many words to generate')


### Parse input arguments
args = parser.parse_args()
phrase = args.seed.lower()
phrase = phrase.split()
lenght = args.length

# Set the temperature.
temperature = 0.7


# Generate and print text
text_generated = predict(net, phrase, len(word2idx), lenght, word2idx, idx2word, temperature, device)
print(' '.join(text_generated))







### Import libraries
import torch
from collections import Counter
import os
import torch.nn as nn
from torch import optim
import numpy as np
import re
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from random import choice
from sklearn.manifold import TSNE

from net import LSTMNetwork


## Produce the corpus of the training data
def corpus(train_file):
      
    # Read the text file 
    with open(train_file, 'r') as f:
        clean_txt = f.read().lower().replace('\n', ' \n ')

    # clean up the text (lower the uppercases, remove all the digits, special characters, and extra spaces from the text)
    #clean_txt = re.sub('[^A-Za-z]+', ' ', clean_txt)  ##not really needed when embedding
    clean_txt = re.sub(r'\s+', ' ', clean_txt) # Eliminate duplicate whitespaces

    # Tokenize the text
    words = clean_txt.split()

    return words
    
# Remove words that are not frequent -- not really needed when embedding
def rmv_less_freq(words, frequency):
    
    # Count the occurency of the words
    vocab_cnt = Counter()
    vocab_cnt.update(words)
    vocab_cnt = Counter({w:c for w,c in vocab_cnt.items() if c > frequency})
    
        
    vocab = set()
    unigram_dist = list()
    word2idx = dict()
    for i, (w, c) in enumerate(vocab_cnt.most_common()):
        vocab.add(w)
        unigram_dist.append(c)
        word2idx[w] = i

    unigram_dist = np.array(unigram_dist)
    word_freq = unigram_dist / unigram_dist.sum()
    
    #Generate word frequencies to use with negative sampling
    w_freq_neg_samp = unigram_dist ** 0.75
    w_freq_neg_samp /= w_freq_neg_samp.sum() #normalize
    
    #Get words drop prob
    w_drop_p = 1 - np.sqrt(0.00001/word_freq)

    #Generate train corpus dropping less common words
    train_corpus = [w for w in words if w in vocab]
    
    return train_corpus

   
## Generate the vocabularies 
def vocabs(words):
    word2idx = dict()
    idx2word = dict()
    
    sorted_vocab = sorted(Counter(words), key=Counter(words).get, reverse=True)
    
    ## Generate the vocabularies from word to int and vice versa
    for i, word in enumerate(sorted_vocab):
            word2idx[word] = i
    for i, word in enumerate(word2idx.items()):
        idx2word[i] = word
        
    return word2idx, idx2word
        
def prepare_datas(word2idx, idx2word, words, batch_size, seq_size):
     
    # Process the data and prepare the input and the target 
    int_text = [word2idx[w] for w in words]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    ## input[...t] -> output [..t+1]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))

    return in_text, out_text
    
    
# Put the text into batches
def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]
        
# Plot the losses over the number of epochs
def training_loss_plot(loss):

    plt.close('all')
    plt.figure(figsize=(8,6))
    plt.semilogy(loss, label='Test loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# PLot the embeddings values for the top_k words of the embedding layer    
def plot_embedding(model, top_k, idx2word):
    # load the embedding weights
    emb = model.embedding.weight.cpu().data.numpy()
    
    #calculate the distance
    tsne = TSNE(metric='cosine', random_state=123)
    embed_tsne = tsne.fit_transform(emb[:top_k, :])
    
    # PLot the embeddings and the corresponding words
    fig, ax = plt.subplots(figsize=(22, 22))
    for idx in range(top_k):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(idx2word[idx][0], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
    plt.savefig('embeddings_map.png')


# Train the network 
def training(model, in_text, out_text, batch_size, seq_size):
    
    losses = []
    iteration = 0
    
    for epoch in range(epochs):

        batches = get_batches(in_text, out_text, batch_size, seq_size)
        
        ## Wraps hidden states in new Variables, to detach them from their history
        state_h, state_c = torch.zeros(1, batch_size, lstm_size), torch.zeros(1, batch_size, lstm_size)
        
        # Move to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        
        total_loss = []
        
        for context, target in batches:
            
            iteration += 1

            # Tell it we are in training mode
            model.train()
            
            # reshape and move to GPU
            context = torch.LongTensor(context).to(device)
            target = torch.LongTensor(target).to(device)
            
    
            # zero the parameter gradients
            model.zero_grad()
            
            # forward propagation
            logits, (state_h, state_c) = model(context, (state_h, state_c), batch_size)

            # loss calculation
            loss = loss_fn(logits.transpose(1,2), target)

            
            state_h = state_h.detach()
            state_c = state_c.detach()

            # backpropagation
            loss.backward()
            
            #gradient clipping
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            # weight optimization
            optimizer.step()

            # loss update 
            total_loss.append(loss.item())
                            
        print('Epoch: {}/{}'.format(epoch, epochs),
                    #'Iteration: {}'.format(iterations),
                    'Train loss: {}'.format(np.mean(total_loss)))
                            
        losses.append(np.mean(total_loss))
    print(' Training ended!!')
    return losses


### Initialize hyperparameters

# Training data (free books to use can be found on http://www.gutenberg.org/browse/scores/top)
#train_file = '.\\'+'wonderland.txt'
train_file = '.\\'+'pride.txt'

# size of the embedding of the word
embedding_size = 356
# Size of the batch for the batch learning
batch_size = 64
# Sequence of words to consider prior the prediction
seq_size = 10
# Learning rate of the network
lr = 0.00015
# Number of epochs
epochs = 80
# Size of the hidden layer of LSTM
lstm_size = 512

## Produce the corpus of the training data
words = corpus(train_file)

#Remove worde that are not frequent
#words = rmv_less_freq(words, 5)

# Generate vocabulary from word to int and vice versa
v1, v2 = vocabs(words)

# Generate the vocabularies and the data to give in input and output
intt, outt = prepare_datas(v1, v2, words, batch_size, seq_size)

# Use GPU if avaliable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# istance of a network
model = LSTMNetwork(len(v1), seq_size, lstm_size, embedding_size)
model = model.to(device)

#Set the LR
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#Set the loss function (CrossEntropyLoss is a good choice too)
loss_fn = nn.NLLLoss()

# Batchify the datas
batches = get_batches(intt, outt, batch_size, seq_size)

# Train our network, return the mean of the losses over the epochs
loss = training(model, intt, outt, batch_size, seq_size)

## Save the model of the network in the folder of the data
torch.save(model.state_dict(), '.\\'+ 'netModel.txt')

#Save the vocabularies
np.save("word2idx.npy", v1)
np.save("idx2word.npy", v2)

# plot the training loss
training_loss_plot(loss)

# plot the embeddings
plot_embedding(model, 300, v2)


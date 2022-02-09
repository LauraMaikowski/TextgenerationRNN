import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import glob
import scipy.io
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import datetime 
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def text_to_torch(text, raw_text):
    ''' converts string input into torchtensor (used for text generation)'''
    vocab = sorted(set(raw_text))
    char_to_int = dict((c, i) for i, c in enumerate(vocab))

    vec_out = np.array([char_to_int[char] for char in text])
    X = vec_out
    #torch_X = torch.tensor(X, dtype= torch.float32) 
    #torch_X = torch.tensor(np.reshape(X,(len(X),1, 1)), dtype= torch.float32) 
    torch_X = torch.tensor(np.reshape(X,(len(X), 1, 1)), dtype= torch.float32)
    return torch_X
    
def torch_to_text(torch_vec, raw_text):
    '''converts torch to text'''
    vocab = sorted(set(raw_text))
    int_to_char = dict((i, c) for i, c in enumerate(vocab))
    vec_out = [int_to_char[i] for i in np.array([torch_vec])]
    return vec_out


def tokenize(raw_text, seq_len=100):
    """ splits text into chunks of seq_len+1
    and slides seq_len window to create input and target:
    input data (dataX) of text[i:seq_len]
    and target (dataY) text[i+1:] for target_seq=True and text[i + seq_length+1]
    Gives posibility to return as string (text=True) or integer vector (text=False)
    
    """
    vocab = sorted(set(raw_text))
    n_vocab = len(vocab)
    char_to_int = dict((c, i) for i, c in enumerate(vocab))
    
    dataX = []
    dataY = []

    #steps = seq_len + 1
    steps = 1
    for i in range(0, len(raw_text)-seq_len, steps):
        seq_in = raw_text[i:i + seq_len]
        seq_out = raw_text[i + 1:i + seq_len + 1]
            
        X_int = [char_to_int[char] for char in seq_in] # transform list of string into list of numbers
        Y_int = [char_to_int[char] for char in seq_out]
        dataX.append(X_int) # (1,)
        dataY.append(Y_int)
                
    return dataX,dataY,n_vocab


def pytorch_dataloader(dataX, dataY, n_vocab, seq_len, batch_size=27):
    """ prepares data and load it into dataloader, puts it in batches"""
    dataX = np.array(dataX) 
    dataY = np.array(dataY)
    
    # having same batchsize over whole data (alternative to batch-padding)
    lenx = len(dataX)
    rest = lenx%batch_size
    X = dataX[rest::]
    Y = dataY[rest::]
    

    # reshaping into [num_sequences, seq_length, num_features]:
    torch_X = torch.tensor(np.reshape(X,(len(X), seq_len)), dtype= torch.float32) 
    torch_Y = torch.tensor(np.reshape(Y,(len(Y), seq_len)), dtype=torch.float32)

    dataset = TensorDataset(torch_X, torch_Y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # getting data by iterating data_loader
    
    return data_loader


def save_model( save, train, device_name, hidden_size, batch_size, 
               epochs, learning_rate, 
               num_embeddings, seq_len, dropout, name_prefix = 'models/gru_'):
    save = True
    if save:
        save_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
    
        save_name= name_prefix+device_name+save_time+'_hs%sb%sepoch%stime3p23.pt'%(hidden_size, batch_size, epochs)
        # save_name model_anzahl der Neuronen in layer
        # _aktivirungsfunktion in jeder layer_learning rate_anzahl der epochen
        torch.save({
            'epoch_loss': train[2],
            'model_state_dict': train[0],
            'optimizer_state_dict': train[1],
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'embedding_dimension': num_embeddings,
            'sequence_length': seq_len,
            'hidden_dimension': hidden_size,
            'dropout': dropout
            }, save_name)
    else:
        pass

class Model(nn.Module):
    def __init__(self, vocab_size, num_embeddings, hidden_size, 
                  lstm_layers=1, dropout=0.2):
        super(Model,self).__init__()
        
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers  # refers to gru layer
        self.dropout = dropout
        self.num_embeddings = num_embeddings
        
        self.embedding = nn.Embedding(vocab_size, self.num_embeddings)
        
        self.lstm = nn.GRU(
            input_size=self.num_embeddings,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers, batch_first=True, dropout=self.dropout)
        
        #self.drop = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(self.hidden_size, vocab_size)
        #self.relu = nn.ReLU()
        
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, h = self.lstm(embed, prev_state)
        #output = self.drop(output)
        #output = self.fc(self.relu(output[:,-1]))
        output = self.fc(output)
        return output, h
        
    def init_state(self, batch_size):
        hidden = torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(device)
        return hidden
    
    def network_structure(self):
        self.network = nn.ModuleList([self.embedding]+[self.lstm]+[self.fc])
        return self.network


def training(train_data, epochs, batch_size, learning_rate, model):
    loss = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
    epoch_loss = []

    for epoch in tqdm(range(epochs)):
        batch_loss = []
        state_h = model.init_state(batch_size)
        for data in train_data:
            X0 = data[0].to(device)
            y0 = data[1].to(device)

            optimizer.zero_grad()  # optimizer history auf 0 
            y_pred, state_h = model(X0.long(), state_h)
            
            error = loss(y_pred.transpose(1,2),y0.long())  # berechnung des Fehlers (hier via Logloss)
            
            state_h = state_h.detach()
            
            error.backward()  # backward propagation wird definiert
            optimizer.step()  # backward propagation wird durchgeführt 
            
            batch_loss.append(error.item())
        epoch_loss.append(np.mean(batch_loss))
        #writer.add_scalar("Loss/train", np.mean(batch_loss), epoch)
    
    plt.plot(epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    return model.state_dict(), optimizer.state_dict(), epoch_loss

def mulitnomial_prediction(seed, raw_text, model, temperature=0.5, predict_len=500):
    gru = model.eval()
    initial_str = seed
    #### word generator: from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Projects/text_generation_babynames/generating_names.py 19.01.22
    hidden = gru.init_state(1)
    
    initial_input = text_to_torch(initial_str, raw_text).long()
    predicted = initial_str

    # predicted hidden state for first characters
    for p in range(len(initial_str)-1):
        _, hidden = gru(initial_input[p].to(device), hidden)
        
    last_char = initial_input[-1]
    for p in range(predict_len):
        output, hidden = gru(last_char.to(device), hidden)
        output_dist = output.view(-1).div(temperature).exp()
        top_char = torch.multinomial(output_dist, 1)[0]

        predicted_char = torch_to_text(top_char.cpu(), raw_text)[0]
        predicted += predicted_char
        last_char = text_to_torch(predicted_char, raw_text).long()[0]

    return predicted
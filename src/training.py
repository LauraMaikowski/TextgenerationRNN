import functions as fn
import torch

# cuda or cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('this computation is running on {}'.format(device))
device_name = str(device)[:3]

seq_len = int(input('sequence length:'))
hidden_size = int(input('hidden size:'))
num_embeddings = int(input('size of embeddings:'))
learning_rate = float(input('learning rate:'))
epochs = int(input('number of epochs:'))
dropout = float(input('dropout rate:'))
gru_layer = int(input('number of GRU layer:'))
batch_size = int(input('batch size:'))
temperature = float(input('temperature between 0 and 1:'))
predict_len = int(input('prediction length:'))
seed = str(input('prediction seed'))

# open ascii txt:
filename = "data/rl_stanza.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()

# loading data and model
dataX,dataY,n_vocab = fn.tokenize(raw_text, seq_len=seq_len)
data_loader = fn.pytorch_dataloader(dataX,dataY, n_vocab, seq_len, batch_size=batch_size)
# model initiation
gru = fn.Model(vocab_size=n_vocab, num_embeddings = num_embeddings, 
        hidden_size=hidden_size,lstm_layers=gru_layer).to(device)
# model training
train = fn.training(train_data=data_loader, epochs=epochs, 
    batch_size= batch_size, learning_rate=learning_rate, model=gru)
# saving model

name_prefix = 'models/project_data/gru%iseqlen%i_'%(gru_layer, seq_len)
fn.save_model(save=True, train=train, device_name=device_name, 
                   hidden_size=hidden_size, batch_size=batch_size, 
                   epochs=epochs, learning_rate=learning_rate, 
                   num_embeddings=num_embeddings, seq_len=seq_len, dropout=dropout, name_prefix=name_prefix)

# text prediction:
prediction = fn.mulitnomial_prediction(seed, model=gru, temperature=temperature, predict_len=predict_len)

# saving text prediction:
text_save = 'data/txtbatch'+'%iseqlen%ihs%ib%iepoch%itemperature%i.txt'%(gru_layer, 
                                                                        seq_len, hidden_size, 
                                                                        batch_size, epochs, 
                                                                        temperature*100)
text_file = open(text_save, "w")
n = text_file.write('#seed:'+seed+'\n\n'+prediction)
text_file.close()
print(text_save)
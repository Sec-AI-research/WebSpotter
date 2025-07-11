import torch
import torch.nn as nn
import torch.nn.functional as F


Num_filter = 128
Filter_sizes = [3, 4, 5, 6]
m = 256

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, dropout, n_class):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embedding.weight.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=Num_filter, kernel_size=(size, emb_dim)) for size in Filter_sizes])
        self.fc1 = nn.Linear(Num_filter * len(Filter_sizes), m)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(m, n_class)
        self.n_class = n_class
        #self.sf = nn.LogSoftmax(1)


    def forward(self, x):  # x.shape(batch, max_len)  mini-batch=4
        x = self.embedding(x)  # shape(batch, max_len, 50)
        #print(x.shape)
        x = x.unsqueeze(1)  # shape(batch, 1, max_len, 50)  
        #print(x.shape)
        x = [(conv(x)).squeeze(3) for conv in self.convs]  # shape(batch , Num_filter,50-size+1,1).squeeze(3)
        #print(x[0].shape)
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x] #shape (batch,Num_filter,1 ).squeeze(2)
        #print(x[0].shape)
        x = torch.cat(x, 1) # shape (  batch, 4*NUm_filter)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        #print(x.shape)
        #x = self.sf(x)
        return x
    



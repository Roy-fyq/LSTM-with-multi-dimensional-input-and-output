import torch
import torch.nn as nn 


class LSTM1(nn.Module):
    def __init__(self, inputsize = 1, hiddenlayer = 100, outputsize = 1):
        super().__init__() 
        self.hiddenlayer = hiddenlayer
        self.lstm = nn.LSTM(inputsize, hiddenlayer, batch_first=False) # batch_first=True，输入的数据为Batch*seq_len*inputsize;batch_first=False,输入的数据为seq__len*Batch*inputsize
        self.liner = nn.Linear(hiddenlayer, outputsize)
    
    def forward(self, seq):
        lstm_out, (hn, cn) = self.lstm(seq.contiguous().view(seq.size()[1], 1, -1))
        predictions = self.liner(lstm_out.view(seq.size()[1], -1))
        return predictions[-1]
from DataProcess import data_2_seq
from model import LSTM1
from train import train
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

all_data = pd.read_csv("flights.csv")
data = all_data['passengers'].values.astype(float)
train_data = data[:-12]
train_data = np.vstack((train_data, train_data))
train_data = train_data.reshape(2, -1)
train_data = torch.tensor(train_data, dtype=torch.float32)
train_seq = data_2_seq(train_data, 12, single=False, feature_index=[0,1], out_index=[0,1])

net = LSTM1(inputsize=2, hiddenlayer=100, outputsize=2)
lr = 1e-3
optim = torch.optim.Adam(net.parameters(),lr = lr)
epochs = 5
loss_func = nn.MSELoss()
trained_model, loss_hist = train(train_seq, net, epochs, optim, loss_func)
torch.save(trained_model, "/home/code/lstm-free/a.pth")

# plt.plot(loss_hist)
# plt.show()
import pandas as pd 
import numpy as np 

def data_2_seq(data, length, single=True, feature_index=[0,1], out_index = [2]):
    res = []
    if single:
        l = len(data)
        for i in range(l-length):
            seq = data[i: i+length]
            label = data[i+length: i+1+length]
            res.append((seq, label))
        return res
    else:
        l = data.size(1)
        for i in range(l-length):
            seq = data[feature_index,i: i+length]
            label = data[out_index,i+length: i+1+length]
            res.append((seq, label))
        return res
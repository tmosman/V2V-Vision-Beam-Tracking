# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 02:46:40 2023

@author: Tawfik Osman
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
import ast
import torch.nn as nn


from keras.models import Sequential
from keras.layers import LSTM,Dense, Dropout,TimeDistributed,RepeatVector


class MultipleRegression(nn.Module):
    def __init__(self, num_features,node,out):
        super(MultipleRegression, self).__init__()
        self.layer_1 = nn.Linear(num_features, node)
        self.layer_2 = nn.Linear(node, node)
        self.layer_3 = nn.Linear(node, node)
        self.layer_out = nn.Linear(node, out)
        self.relu = nn.ReLU()
            
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)
    
    def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)
    

###### Create data sample list ########

class DataFeed(Dataset):
    '''
    A class retrieving a tuple of (image,label). It can handle the case
    of empty classes (empty folders).
    '''
    def __init__(self,root_dir, nat_sort = False, transform=None, init_shuflle = True):
        self.root = root_dir
        self.samples = self.create_samples(self.root,shuffle=init_shuflle,nat_sort=nat_sort)
        self.transform = transform


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pos_val = sample[:1]
        pos_val = ast.literal_eval(str(pos_val[0]))
        pos_val = np.asarray(pos_val)
        pos_centers = sample[1:]
        pos_centers = np.asarray(pos_centers)
        return (pos_val, pos_centers)
    
    def create_samples(self, start_idx,end_idx):
        f = pd.read_csv(self.root)
        data_samples = []
        for idx, row in f.iterrows():
            #data = list(row.values[10:12])
            data = list(row.values[start_idx:end_idx])
            data_samples.append(data)
        return data_samples
   
    
if __name__ == '__main__':
    print('file run !')
    
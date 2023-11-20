'''
A data feeding class. It generates a list of data samples, each of which is
a tuple of a string (image path) and an integer (beam index), and it defines
a data-fetching method.
Author: Muhammad Alrabeiah
Aug. 2019
'''

import os
import random
import pandas as pd
import torch
import numpy as np
import random
#from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ast


############### Create data sample list #################
def create_samples(root,start_index, stop_index,shuffle=False, nat_sort=False):
    f = pd.read_csv(root)
    data_samples = []
    pred_val = []
    #print(f.iterrows())
    for idx, row in f.iterrows():
        #data = list(row.values[5:9])
        data = list(row.values[start_index:stop_index])
        #print(data)
        data_samples.append(data)
    

    
   
    #print(data_samples)
    return data_samples
#############################################################


#############################################################
class DataFeed(Dataset):
    '''
    A class retrieving a tuple of (image,label). It can handle the case
    of empty classes (empty folders).
    '''
    def __init__(self,root_dir,start_index, stop_index,nat_sort = False, transform=None, init_shuflle = True):
        self.root = root_dir
        self.samples = create_samples(self.root,start_index, stop_index,shuffle=init_shuflle,nat_sort=nat_sort)
        self.transform = transform


    def __len__(self):
        return len( self.samples )

    def __getitem__(self, idx):
        sample = self.samples[idx]
        #print(sample)
        pos_val = sample[:1]
        
        #print(pos_val[0])
        
        pos_val = ast.literal_eval(str(pos_val[0]))
        pos_val = np.asarray(pos_val)
        
        #print('Power : ',pos_val)
        # = ast.literal_eval(sample)
        #print(sample)
        pos_centers = ast.literal_eval(str(sample[0]))
       # print(pos_centers)
        pos_centers = sample[1:]
        pos_centers = np.asarray(pos_centers)

        return (pos_val, pos_centers)
   
 
   

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:21:55 2023

@author: osman
"""

import pandas as pd
import numpy as np
import ast

dataset = pd.read_csv('./scenario36_updated_new.csv')
print(dataset.shape)

## Load Useful data
index_set = dataset['abs_index'].values
sequences = dataset['seq_index'].values
rgb1 = dataset['unit1_rgb1'].values
rgb2 = dataset['unit1_rgb2'].values
pwr_vec_path = dataset['pwr_vec_256'].values
beams = dataset['optimal_beam_256'].values

bbox1_path ='./pwr_256/'

for j in range(1,14):
    seq =[]
    abs_ind =[]
    rgb1_path = []
    rgb2_path = []
    pwr_256_path = []
    pwr_vec = []
    beam_256 = []
    for k, val in enumerate(index_set):
        if sequences[k] in [j]:
            pwr_name = pwr_vec_path[k].split('/')[2]
            pwr_value = np.loadtxt(f'{bbox1_path}{pwr_name}').tolist()
            pwr_vec.append(pwr_value)
            #print(pwr_value.shape)
            abs_ind.append(index_set[k])
            seq.append(sequences[k])
            rgb1_path.append(rgb1[k])
            rgb2_path.append(rgb2[k])
            pwr_256_path.append(pwr_vec_path[k])
            beam_256.append(beams[k])
    
    df1 = pd.DataFrame()
    df1['abs_index'] = abs_ind
    df1['seq_index'] = seq
    df1['unit1_rgb1'] = rgb1_path
    df1['unit1_rgb2'] = rgb2_path
    df1['pwr_vec_path'] = pwr_256_path
    df1['pwr_vec'] = pwr_vec
    df1['gt_beam'] = beam_256
    df1.to_csv(f'./Datasets_Seqs/scenario36_seq_{j}.csv')



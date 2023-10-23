# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:05 2021

@author: osman
"""
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense, Dropout
# from keras.layers import TimeDistributed
# from keras.layers import RepeatVector
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split

from functools import reduce
from operator import concat
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score
# import tensorflow as tf
# from keras.utils import np_utils

dataset = pd.read_csv( r'./Outputs/Tx_Tracking_output_Seq_1_April20th_Exp1.csv')





dists = dataset['selected_dist'].values
angles = dataset['selected_angle'].values
beams = dataset['gt_beams'].values

for j in range(6):
    exec(f'pred_dist_{j}= []')
    exec(f'pred_angle_{j}= []')
    exec(f'gt_beam_{j}= []')

for val in range(len(dists)):
    dist_val = ast.literal_eval(dists[val])
    angle_val = ast.literal_eval(angles[val])
    beam_val =  ast.literal_eval(beams[val])
    
    for k in range(6):
        globals()[f'pred_dist_{k}'].append(dist_val[k])
        globals()[f'pred_angle_{k}'].append( angle_val[k])
        globals()[f'gt_beam_{k}'].append(beam_val[k])

dist_all =[]
angle_all =[]
for pr in range(6):
    dist_all.append(globals()[f'pred_dist_{k}'])
    angle_all.append(globals()[f'pred_angle_{k}'])
    
dist_all = reduce(concat, dist_all)
angle_all = reduce(concat, angle_all)

#dist_all



for j in range(6):
    dataset[f'pred_dist_{j}'] = np.array(globals()[f'pred_dist_{j}']) 
    dataset[f'pred_dist_{j}_norm'] = (np.array(globals()[f'pred_dist_{j}'])-min(dist_all))/(max(dist_all)-min(dist_all))
    dataset[f'pred_angle_{j}'] =  np.array(globals()[f'pred_angle_{j}'])
    dataset[f'pred_angle_{j}_norm'] =  (np.array(globals()[f'pred_angle_{j}'])-min(angle_all))/(max(angle_all)-min(angle_all))
    dataset[f'gt_beam_{j}'] =  np.array(globals()[f'gt_beam_{j}'])

dataset.to_csv( r'./Outputs/Tx_Tracking_output_Seq_1_Exp1_minmax.csv',index=False)

#%%
#for j in range(6):
# pred_dist_0_norm,pred_angle_0_norm = dataset[f'pred_dist_{0}_norm'],dataset[f'pred_angle_{0}_norm']
# pred_dist_1_norm,pred_angle_1_norm = dataset[f'pred_dist_{1}_norm'],dataset[f'pred_angle_{1}_norm']
# pred_dist_2_norm,pred_angle_2_norm = dataset[f'pred_dist_{2}_norm'],dataset[f'pred_angle_{2}_norm']
# pred_dist_3_norm,pred_angle_3_norm = dataset[f'pred_dist_{3}_norm'],dataset[f'pred_angle_{3}_norm']
# pred_dist_4_norm,pred_angle_4_norm = dataset[f'pred_dist_{4}_norm'],dataset[f'pred_angle_{4}_norm']
# pred_dist_5_norm,pred_angle_5_norm = dataset[f'pred_dist_{5}_norm'],dataset[f'pred_angle_{5}_norm']

# gt_beam_0 = dataset['gt_beam_0']
# gt_beam_1 = dataset['gt_beam_1']
# gt_beam_2 = dataset['gt_beam_2']
# gt_beam_3 = dataset['gt_beam_3']
# gt_beam_4 = dataset['gt_beam_4']
# #gt_beam_5 = dataset['gt_beam_5']
#%%
for q in range(6):
    exec(f"gt_beam_{q}= dataset['gt_beam_{q}']")
    exec(f"pred_dist_{q}_norm= dataset['pred_dist_{q}_norm']")
    exec(f"pred_angle_{q}_norm= dataset['pred_angle_{q}_norm']")



# %% Create Train & Test Datasets
X = np.arange(1, dataset.shape[0])
y = np.arange(1, dataset.shape[0])
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=11)


df1 = pd.DataFrame()
df2 = pd.DataFrame()
csv_to_save_train = r'./Outputs/Final/BT' + '_train_seq_1_exp1' + '.csv'
csv_to_save_test = r'./Outputs/Final/BT' + '_test_seq_1_exp1' + '.csv'


for j in range(6):
    exec(f'pred_dist_{j}_train= []')
    exec(f'pred_angle_{j}_train= []')
    exec(f'gt_beam_{j}_train= []')
    exec(f'pred_dist_{j}_test= []')
    exec(f'pred_angle_{j}_test= []')
    exec(f'gt_beam_{j}_test= []')



for j in range(len(beams)):
    if j in X_train:
        
        
        for k in range(6):
            dist = globals()[f'pred_dist_{k}_norm'][j]
            angle = globals()[f'pred_angle_{k}_norm'][j]
            beam = globals()[f'gt_beam_{k}'][j]
            globals()[f'pred_dist_{k}_train'].append(dist)
            globals()[f'pred_angle_{k}_train'].append(angle)
            globals()[f'gt_beam_{k}_train'].append(beam)
    else:
        for k in range(6):
            dist = globals()[f'pred_dist_{k}_norm'][j]
            angle = globals()[f'pred_angle_{k}_norm'][j]
            beam = globals()[f'gt_beam_{k}'][j]
            globals()[f'pred_dist_{k}_test'].append(dist)
            globals()[f'pred_angle_{k}_test'].append(angle)
            globals()[f'gt_beam_{k}_test'].append(beam)
            
  
# Train
for j in range(6):
    df1[f'pred_dist_{j}_norm'] = np.array(globals()[f'pred_dist_{j}_train']) 
    df1[f'pred_angle_{j}_norm'] =  np.array(globals()[f'pred_angle_{j}_train']) 
    df1[f'gt_beam_{j}'] =  np.array(globals()[f'gt_beam_{j}_train'])

#Test 
for j in range(6):
    df2[f'pred_dist_{j}_norm'] = np.array(globals()[f'pred_dist_{j}_test']) 
    df2[f'pred_angle_{j}_norm'] =  np.array(globals()[f'pred_angle_{j}_test']) 
    df2[f'gt_beam_{j}'] =  np.array(globals()[f'gt_beam_{j}_test'])
    

dataset.to_csv( r'./Outputs/Final/Tx_Tracking_output_Seq_1_Exp1.csv',index=False)

df1.to_csv(csv_to_save_train, index=False)
df2.to_csv(csv_to_save_test, index=False)




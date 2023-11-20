# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:05 2021

@author: osman
"""

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from functools import reduce
from operator import concat



seq_no = 1
dataset = pd.read_csv( f'./TX_Tracking/Tx_Tracking_output_Seq_{seq_no}_Nov_run1_updated.csv')


abs_index = dataset['abs_index'].values
dists = dataset['selected_dist'].values
angles = dataset['selected_angle'].values
beams = dataset['gt_beams'].values
objs = dataset['no of objs'].values

timestamp = dataset['timestamp'].values
tx_lat =dataset['unit1_gps1_lat'].values
tx_lon =dataset['unit1_gps1_lon'].values
rx_lat =dataset['unit2_gps1_lat'].values
rx_lon =dataset['unit2_gps1_lon'].values


for j in range(6):
    exec(f'pred_dist_{j}= []')
    exec(f'pred_angle_{j}= []')
    exec(f'gt_beam_{j}= []')
objs_list =[]
abs_ind =[]
timestamp_ls = []
tx_lat_ls = []
tx_lon_ls = []
rx_lat_ls = []
rx_lon_ls = []

for val in range(len(dists)):
    dist_val = ast.literal_eval(dists[val])
    angle_val = ast.literal_eval(angles[val])
    beam_val =  ast.literal_eval(beams[val])
    objs_values = ast.literal_eval(objs[val])
    objs_list.append(np.floor(np.mean(objs_values)))
    abs_ind.append(abs_index[val])
    print(timestamp[val])
    timestamp_ls.append(timestamp[val])
    tx_lat_ls.append(tx_lat[val])
    tx_lon_ls.append(tx_lon[val])
    rx_lat_ls.append(rx_lat[val])
    rx_lon_ls.append(rx_lon[val])
    '''
    timestamp_ls.append(ast.literal_eval(timestamp[val]))
    tx_lat_ls.append(ast.literal_eval(tx_lat[val]))
    tx_lon_ls.append(ast.literal_eval(tx_lon[val]))
    rx_lat_ls.append(ast.literal_eval(rx_lat[val]))
    rx_lon_ls.append(ast.literal_eval(rx_lon[val]))
    '''
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


dataset['objs'] = objs_list
dataset['abs_index']= abs_ind
dataset['timestamp']= timestamp_ls
dataset['unit1_gps1_lat'] = tx_lat_ls
dataset['unit1_gps1_lon'] = tx_lon_ls
dataset['unit2_gps1_lat'] = rx_lat_ls
dataset['unit2_gps1_lon'] = rx_lon_ls
for j in range(6):
    dataset[f'pred_dist_{j}'] = np.array(globals()[f'pred_dist_{j}']) 
    dataset[f'pred_dist_{j}_norm'] = (np.array(globals()[f'pred_dist_{j}'])-min(dist_all))/(max(dist_all)-min(dist_all))
    dataset[f'pred_angle_{j}'] =  np.array(globals()[f'pred_angle_{j}'])
    dataset[f'pred_angle_{j}_norm'] =  (np.array(globals()[f'pred_angle_{j}'])-min(angle_all))/(max(angle_all)-min(angle_all))
    dataset[f'gt_beam_{j}'] =  np.array(globals()[f'gt_beam_{j}'])


#dataset.to_csv( f'./Outputs/New_Tracked_Data/Tx_Tracking_output_Seq_{seq_no}_April20th_Exp2_final_updated_norm.csv',index=False)

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
csv_to_save_train = f'./LSTM_datasets/BT_train_seq_{seq_no}_updated.csv'
csv_to_save_test = f'./LSTM_datasets/BT_test_seq_{seq_no}_updated.csv'


for j in range(6):
    exec(f'pred_dist_{j}_train= []')
    exec(f'pred_angle_{j}_train= []')
    exec(f'gt_beam_{j}_train= []')
    exec(f'pred_dist_{j}_test= []')
    exec(f'pred_angle_{j}_test= []')
    exec(f'gt_beam_{j}_test= []')


objs_train =[]
objs_test =[]
index_train =[]
index_test =[]
for j in range(len(beams)):
    if j in X_train:
        objs_values = ast.literal_eval(objs[j])
        objs_train.append(np.floor(np.mean(objs_values)))
        index_train.append(abs_index[j])
        for k in range(6):
            dist = globals()[f'pred_dist_{k}_norm'][j]
            angle = globals()[f'pred_angle_{k}_norm'][j]
            beam = globals()[f'gt_beam_{k}'][j]
            globals()[f'pred_dist_{k}_train'].append(dist)
            globals()[f'pred_angle_{k}_train'].append(angle)
            globals()[f'gt_beam_{k}_train'].append(beam)
    else:
        objs_values = ast.literal_eval(objs[j])
        objs_test.append(np.floor(np.mean(objs_values)))
        index_test.append(abs_index[j])
        for k in range(6):
            dist = globals()[f'pred_dist_{k}_norm'][j]
            angle = globals()[f'pred_angle_{k}_norm'][j]
            beam = globals()[f'gt_beam_{k}'][j]
            globals()[f'pred_dist_{k}_test'].append(dist)
            globals()[f'pred_angle_{k}_test'].append(angle)
            globals()[f'gt_beam_{k}_test'].append(beam)
            
  
# Train
df1['abs_index'] = index_train
df1['Ave. objs'] = objs_train

for j in range(6):
    df1[f'pred_dist_{j}_norm'] = np.array(globals()[f'pred_dist_{j}_train']) 
    df1[f'pred_angle_{j}_norm'] =  np.array(globals()[f'pred_angle_{j}_train']) 
    df1[f'gt_beam_{j}'] =  np.array(globals()[f'gt_beam_{j}_train'])

#Test 
df2['abs_index'] = index_test
df2['Ave. objs'] = objs_test

for j in range(6):
    df2[f'pred_dist_{j}_norm'] = np.array(globals()[f'pred_dist_{j}_test']) 
    df2[f'pred_angle_{j}_norm'] =  np.array(globals()[f'pred_angle_{j}_test']) 
    df2[f'gt_beam_{j}'] =  np.array(globals()[f'gt_beam_{j}_test'])
    

#dataset.to_csv( r'./Outputs/Final/Tx_Tracking_output_Seq_1_Exp1.csv',index=False)

df1.to_csv(csv_to_save_train, index=False)
df2.to_csv(csv_to_save_test, index=False)




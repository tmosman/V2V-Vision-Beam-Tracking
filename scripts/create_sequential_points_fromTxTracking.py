# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 03:22:15 2023

@author: osman
"""


import os
import random
import pandas as pd
import numpy as np
import math
import ast
import matplotlib.pyplot as plt

df = pd.read_csv(r'./Outputs/scenario36_teston_seq_1_cleaned_April_15th_Pred_Dsit_Angle_Seq_best.csv')

csv_to_save = r'./Outputs/Tx_Tracking_output_Seq_1_April20th_Exp1.csv'
index_abs = df['abs_index'].values
gc_dist_objs = df['gc_dist'].values
gc_angles_objs = df['gc_angle'].values
bbox_dist = df['actual_dist'].values
bbox_angle = df['actual_angle'].values
gt_beam = df['gc_beam_gt'].values



sel_dist = []
sel_angle = []
orig_dist =[]
orig_angle =[]
new_dist_objs =[]
new_angle_objs =[]
beam = []
new_index =[]
no_objs =[]

pred_dist_0 =[]
pred_angle_0 =[]


pred_dist_1 =[]
pred_angle_1 =[]
pred_dist_angle_1 =[]



pred_dist_2 =[]
pred_angle_2 =[]
pred_dist_angle_2 =[]


pred_dist_3 =[]
pred_angle_3 =[]
pred_dist_angle_3 =[]



pred_dist_4 =[]
pred_angle_4 =[]
pred_dist_angle_4 =[]

pred_dist_5 =[]
pred_angle_5 =[]
pred_dist_angle_5 =[]
gt_beam_5 =[]

for val in range(len(index_abs)-6):
    pred_dist,pred_angle = bbox_dist[val], bbox_angle[val]
    print('Predict: ', pred_dist)
 
    future_dist = []
    future_angle = []
    beam_list = []
    
    gt_dist_list = []
    gt_angle_list = []
    no_objs_list =[]
    
    for iteration in range(0,6):
          # val is from the for loop
        dist,angle = gc_dist_objs[val+iteration],  gc_angles_objs[val+iteration]
        dist,angle = ast.literal_eval(dist),ast.literal_eval(angle)
        beam_list.append(gt_beam[val+iteration])
        no_objs_list.append(len(angle))
    
        print(f'GC dists: {dist}')
        if len(dist)>= 1:
            new_dist_objs.append(dist)
            new_angle_objs.append(angle)
            
            check_min =[]
            for q,(dis,ang) in enumerate(zip(dist,angle)):
                min_d = math.dist([pred_dist,pred_angle],[dis,ang])
                check_min.append(min_d)
        
            dist_pos = np.argmin(check_min)
            final_dist,final_angle = dist[dist_pos],angle[dist_pos]
            future_dist.append(final_dist)
            future_angle.append(final_angle)
            print('Final dist:',future_dist)
            
            pred_dist,pred_angle = final_dist,final_angle
            print('Update dist:',pred_dist)
            
        else:
            if len(future_dist)>0:
                future_dist.append(future_dist[-1])
                future_angle.append(future_angle[-1])

            else:
                future_dist.append(pred_dist)
                future_angle.append(pred_angle)
    new_index.append(index_abs[val])
            
    no_objs.append(no_objs_list)      
    sel_dist.append(future_dist)
    sel_angle.append(future_angle) 
    beam.append(beam_list)
    
    pred_dist_0.append(future_dist[0])
    pred_angle_0.append(future_angle[0])
    
    pred_dist_1.append(future_dist[1])
    pred_angle_1.append(future_angle[1])
    pred_dist_angle_1.append([future_dist[1],future_angle[1]])

    pred_dist_2.append(future_dist[2])
    pred_angle_2.append(future_angle[2])
    pred_dist_angle_2.append([future_dist[2],future_angle[2]])

    pred_dist_3.append(future_dist[3])
    pred_angle_3.append(future_angle[3])
    pred_dist_angle_3.append([future_dist[3],future_angle[3]])
    
    pred_dist_4.append(future_dist[4])
    pred_angle_4.append(future_angle[4])
    pred_dist_angle_4.append([future_dist[4],future_angle[4]])
    
    pred_dist_5.append(future_dist[5])
    pred_angle_5.append(future_angle[5])
    pred_dist_angle_5.append([future_dist[5],future_angle[5]])
    gt_beam_5.append(beam_list[5])
    
    
df1 = pd.DataFrame()
df1['abs_index'] = new_index
df1['no of objs'] = no_objs
df1['selected_dist'] =  sel_dist
df1['selected_angle'] =  sel_angle
df1['gt_beams'] = beam


df1['pred_dist_5'] = pred_dist_5
df1['pred_angle_5'] = pred_angle_5

df1['pred_dist_angle_5'] = pred_dist_angle_5

# dist_angle= []
# for j in range(len(dist)):
#     dist_angle.append([dist[j], angle[j]])
# df1['dist_angle_norm']  = dist_angle 


df1.to_csv(csv_to_save, index=False) 

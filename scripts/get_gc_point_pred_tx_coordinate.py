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

df = pd.read_csv(r'Annotated_data_samples.csv')
df1 = pd.read_csv('./Outputs/scenario36_teston_seq_1_cleaned_April_15th_val_Angle_dist_best.csv')
csv_to_save = 'check_test_data1.csv'
index_abs = df['abs_index'].values
index_abs1 =  df1['abs_index'].values
gc_dist_objs = df['gc_dists_output'].values
gc_angles_objs = df['gc_angles_output'].values
gt_beam = df['gt_beam'].values

pred_dist = df1['actual_dist'].values
pred_angle = df1['actual_angle'].values

gc_gt_dist = df['gc_tx_bbox_dist'].values
gc_gt_angle = df['gc_tx_bbox_angle'].values




beam = []
new_index =[]


pred_dist_0 =[]
pred_angle_0 =[]
gt_dist_0= []
gt_angle_0=[]
orig_dist =[]
orig_angle =[]


for val in range(len(index_abs)):
    if index_abs[val] in index_abs1:
        print(index_abs[val])
        pred_dist_0.append(gc_dist_objs[val])
        pred_angle_0.append(gc_angles_objs[val])
        new_index.append(index_abs[val])
        gt_dist_0.append(gc_gt_dist[val])
        gt_angle_0.append(gc_gt_angle[val])
        
df1 = pd.DataFrame()
df1['abs_index'] = new_index
df1['gc_dists'] = pred_dist_0
df1['gc_angles'] = pred_angle_0 
df1['gt_dist'] =  gt_dist_0
df1['gt_angle']  = gt_angle_0
df1.to_csv(csv_to_save, index=False)     
#     gt_dist,gt_angle = gc_gt_dist[val], gc_gt_angle[val]  # val is from the for loop
#     beam_list = []
#     gt_dist_list = []
#     gt_angle_list = []
    
#     for iteration in range(0,1):
#         #pred_dist,pred_angle = bbox_dist[val], bbox_angle[val]
#           # val is from the for loop
#         dist,angle = gc_dist_objs[val+iteration],  gc_angles_objs[val+iteration]
#         dist,angle = ast.literal_eval(dist),ast.literal_eval(angle)
#         beam_list.append(gt_beam[val+iteration])
#         gt_dist,gt_angle = gc_gt_dist[val+iteration], gc_gt_angle[val+iteration]  # pick the annotated gc_gt point at future
#         print(f'GC dists: {dist}')
#         if len(dist)>= 1:
    
            
#             check_min1 =[]
#             for q,(dis,ang) in enumerate(zip(dist,angle)):
#                 min_d1 = math.dist([gt_dist,gt_angle],[dis,ang])
#                 #print(min_d)
#                 check_min1.append(min_d1)
#             dist_pos1 = np.argmin(check_min1)
#             final_gt_dist,final_gt_angle = dist[dist_pos1],angle[dist_pos1]
#             #gt_dist,gt_angle = final_gt_dist,final_gt_angle
#             gt_dist_list.append(final_gt_dist)
#             gt_angle_list.append(final_gt_angle)
            
#         # else:
#         #     if len(final_gt_dist)>0:
#         #         gt_dist_list.append(gt_dist_list[-1])
#         #         gt_angle_list.append(gt_angle_list[-1])
#         #     else:
#         #         gt_dist_list.append(final_gt_dist)
#         #         gt_angle_list.append(final_gt_angle)
        
        
#         orig_dist.append(gt_dist)
#         orig_angle.append(gt_angle)
#         #gt_x,gt_y = gt_dist*np.cos(np.array(gt_angle)*np.pi/180), gt_dist*np.sin(np.array(gt_angle)*np.pi/180)
#         #if len(gt_dist)>= 1:
            
            
    
#     beam.append(beam_list)
#     gt_dist_0.append(gt_dist_list[0])
#     gt_angle_0.append(gt_angle_list[0])
    
    
# df1 = pd.DataFrame()
# # df1['gc_dists'] = new_dist_objs
# # df1['gc_angles'] = new_angle_objs
# df1['gc_tx_bbox_dist'] = orig_dist
# df1['gc_tx_bbox_angle'] = orig_angle
# df1['gt_dist_0'] = gt_dist_0
# df1['gt_angle_0'] = gt_angle_0
# df1['gt_beam']  = beam 
# # df1['gt_dist_5_norm'] = (np.array(gt_dist_5)-0.38152)/(0.497861-0.38152)

# # df1['gt_angle_5_norm'] = (np.array(gt_angle_5)-1)/(360-1)
# # dist = df1['gt_dist_5_norm'].values
# # angle = df1['gt_angle_5_norm'].values


# # dist_angle= []
# # for j in range(len(dist)):
# #     dist_angle.append([dist[j], angle[j]])
# # df1['dist_angle_norm']  = dist_angle 
df1.to_csv(csv_to_save, index=False) 



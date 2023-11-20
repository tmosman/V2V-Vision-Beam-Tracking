# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 23:03:49 2023

@author: osman
"""


import pandas as pd 
import numpy as np
df = pd.read_csv('./BT_outputs_new/test_results_fold_3_updated.csv')

objs = df['avg_objs'].values
pred = df['pred_beam_5'].values
actual = df['gt_beam_5'].values
abs_index = df['abs_index'].values
acc = df['Acc_diff'].values
beam = df['beam_diff'].values
# acc = []
# for val in range(len(abs_index)):
#     acc.append(np.abs(pred[val]- actual[val]))
    
# df['Acc_diff'] = acc
# df.to_csv('./BT_output/CV_output_1/test_results_fold_0_updated.csv',index=False)

first = 0
second = 0
third =  0
forth =  0
five =0

first_all = 0
second_all = 0
third_all =  0
forth_all =  0
five_all =0

first_5 = 0
second_5 = 0
third_5 =  0
forth_5 =  0
five_5 =0
other_5 =0

six = 0
other =0
other_all =0
for val in range(len(abs_index)):
    if beam[val] ==0 and acc[val] == 0:
        first_all+=1
    elif beam[val] in range(1,2) and acc[val] ==0:
        second_all+=1
    elif beam[val] in range(2,3) and acc[val] == 0:
        third_all+=1 
    elif beam[val] in range(3,11) and acc[val] == 0:
        forth_all+=1 
    # elif beam[val] in range(5,10) and acc[val] == 0:
    #     five_all+=1
#     elif beam[val] in range(10,16) and acc[val] == 0:
#         six+=1
    elif beam[val] in range(11,np.max(beam) )and acc[val] == 0:
        other_all+=1
        
        
for val in range(len(abs_index)):
    if beam[val] ==0 and acc[val] <=5:
        first_5+=1
    elif beam[val] in range(1,2) and acc[val] <=5:
        second_5+=1
    elif beam[val] in range(2,3) and acc[val] <= 5:
        third_5+=1 
    elif beam[val] in range(3,11) and acc[val] <= 5:
        forth_5+=1 
    # elif beam[val] in range(5,10) and acc[val] <= 5:
    #     five_5+=1
#     elif beam[val] in range(10,16) and acc[val] == 0:
#         six+=1
    elif beam[val] in range(11,np.max(beam) )and acc[val] <= 5:
        other_5+=1
        
        
        
for val in range(len(abs_index)):
    if beam[val] ==0:
        first+=1
    elif beam[val] in range(1,2) :
        second+=1
    elif beam[val] in range(2,3) :
        third+=1 
    elif beam[val] in range(3,11) :
        forth+=1 
    # elif beam[val] in range(5,10) :
    #     five+=1
    # elif beam[val] in range(10,16) :
    #     six+=1
    elif beam[val] in range(11,np.max(beam)+1):
        other+=1

# for val in range(len(abs_index)):
#      if objs[val] in range(6):
#          first+=1
#      elif objs[val] in range(5,11) :
#          second+=1
#      elif objs[val] in range(11,16) :
#          third+=1 
#      elif objs[val] in range(16,21) :
#          forth+=1 
#      elif objs[val] in range(21,28) :
#          five+=1
#      else:
#          other+=1

df1 = pd.DataFrame()
df1['classes'] = ['first','second','third','forth','other']
df1['beam_acc_count'] = [first_all,second_all,third_all,forth_all,other_all]
df1['beam_acc_count_5'] = [first_5,second_5,third_5,forth_5,other_5]
df1['beam_count'] = [first,second,third,forth,other]
df1.to_csv('./BT_outputs_new/test_results_fold_3_updated_plot_beam_count_1.csv',index=False)        

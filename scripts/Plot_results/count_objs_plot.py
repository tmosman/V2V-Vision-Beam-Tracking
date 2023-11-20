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
other =0
first_1 = 0
second_1 = 0
third_1 =  0
forth_1 =  0
five_1 =0
other_1 =0
first_5 = 0
second_5 = 0
third_5 =  0
forth_5 =  0
five_5 =0
other_5 =0
for val in range(len(abs_index)):
    if objs[val] in range(6) and acc[val] == 0:
        first_1+=1
    elif objs[val] in range(6,11) and acc[val] ==0:
        second_1+=1
    elif objs[val] in range(11,16) and acc[val] == 0:
        third_1+=1 
    elif objs[val] in range(16,21) and acc[val] == 0:
        forth_1+=1 
    elif objs[val] in range(21,28) and acc[val] == 0:
        five_1+=1
    else:
        other_1+=1

for val in range(len(abs_index)):
    if objs[val] in range(6) and acc[val] <= 5:
        first_5+=1
    elif objs[val] in range(6,11) and acc[val] <=5:
        second_5+=1
    elif objs[val] in range(11,16) and acc[val] <= 5:
        third_5+=1 
    elif objs[val] in range(16,21) and acc[val] <= 5:
        forth_5+=1 
    elif objs[val] in range(21,28) and acc[val] <= 5:
        five_5+=1
    else:
        other_5+=1

for val in range(len(abs_index)):
      if objs[val] in range(6):
          first+=1
      elif objs[val] in range(6,11) :
          second+=1
      elif objs[val] in range(11,16) :
          third+=1 
      elif objs[val] in range(16,21) :
          forth+=1 
      elif objs[val] in range(21,28) :
          five+=1
      else:
          other+=1

df1 = pd.DataFrame()
df1['classes'] = ['first','second','third','forth','five','other']
df1['Acc_count_1'] = [first_1,second_1,third_1,forth_1,five_1,other_1]
df1['Acc_count_5'] = [first_5,second_5,third_5,forth_5,five_5,other_5]
df1['Count'] = [first,second,third,forth,five,other]
df1.to_csv('./BT_outputs_new/test_results_fold_3_updated_plot_objs_1.csv')        

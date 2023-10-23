# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 00:34:45 2023

@author: Gouranga
"""
import os
import math
import shutil 
import ast
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'./Data/scenario36_updated_new.csv')
#df1 = pd.read_csv(r'scenario36_updated_subset_front_back_plus_xy_front_back.csv')


exp = 2
img1_path = f'sub_dataset_analysis/Seq_Dataset_{exp}/rgb1'
img2_path = f'sub_dataset_analysis/Seq_Dataset_{exp}/rgb2'

bbox1_img = f'sub_dataset_analysis/Seq_Dataset_{exp}/bbox1_img'
bbox2_img = f'sub_dataset_analysis/Seq_Dataset_{exp}/bbox2_img'

bbox1_path_new = f'sub_dataset_analysis/Seq_Dataset_{exp}/bbox1'
bbox2_path_new = f'sub_dataset_analysis/Seq_Dataset_{exp}/bbox2'  

plot_save_path = f'sub_dataset_analysis/Seq_Dataset_{exp}/plot'

folder_paths = [img1_path, img2_path, bbox1_img, bbox2_img, bbox1_path_new, bbox2_path_new, plot_save_path]
for folder_path in folder_paths:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    
def angle_and_dist(pos1, pos2):
    hyp_dist = math.dist(pos1, pos2)
    x = pos2[0] - pos1[0]
    y = pos2[1] - pos1[1]
    rad_angle = math.atan2(y, x)
    deg_angle = math.degrees(rad_angle)
    data = [hyp_dist, rad_angle, deg_angle]
    return data
def generate_final_bbox_coord_seq1(bbox1_lst, bbox2_lst):
    '''
    generate the linear regression coeeficients for bbox2
    '''
    x2 = np.arange(30, 46,1).reshape(-1,1)
    y2 = np.arange(0,46,3)

    # Create a linear regression object and fit the data
    regressor2 = LinearRegression()
    regressor2.fit(x2, y2)  
    
    x1 = np.arange(305, 325,1).reshape(-1,1)
    y1 = np.arange(302,360,3)

    # Create a linear regression object and fit the data
    regressor1 = LinearRegression()
    regressor1.fit(x1, y1)
    
    final_bbox_dist = []
    final_bbox_angle = []
    for val in bbox2_lst:
        dist, angle = val[0], val[2]
        if dist > 0.25 and angle > 25:
            if angle > 45:
                final_bbox_dist.append(dist)
                final_bbox_angle.append(angle)
            else:
                angle_updated = regressor2.predict(np.array([[angle]]))
                final_bbox_dist.append(dist)
                final_bbox_angle.append(angle_updated[0])
                
    for val in bbox1_lst:
        dist, angle = val[0], (180+ val[2])
        if dist > 0.25 and angle < 325:
            if angle < 305:
                final_bbox_dist.append(dist)
                final_bbox_angle.append(angle)
            else:
                angle_updated = regressor1.predict(np.array([[angle]]))
                final_bbox_dist.append(dist)
                final_bbox_angle.append(angle_updated[0])
                
    return final_bbox_dist, final_bbox_angle
def generate_final_bbox_coord_seq2(bbox1_lst, bbox2_lst):

    #generate the linear regression coeeficients for bbox2

    x2 = np.arange(134, 150,1).reshape(-1,1)
    y2 = np.arange(134,180,3)

    # Create a linear regression object and fit the data
    regressor2 = LinearRegression()
    regressor2.fit(x2, y2)  
    
    x1 = np.arange(215, 235,1).reshape(-1,1)
    y1 = np.arange(180,238,3)

    # Create a linear regression object and fit the data
    regressor1 = LinearRegression()
    regressor1.fit(x1, y1)
    
    final_bbox_dist = []
    final_bbox_angle = []
    for val in bbox2_lst:
        dist, angle = val[0], val[2]
        if dist > 0.25 and angle < 155:
            if angle < 135:
                final_bbox_dist.append(dist)
                final_bbox_angle.append(angle)
            else:
                angle_updated = regressor2.predict(np.array([[angle]]))
                final_bbox_dist.append(dist)
                final_bbox_angle.append(angle_updated[0])
                
    for val in bbox1_lst:
        dist, angle = val[0], (180+ val[2])
        if dist > 0.25 and angle > 215:
            if angle > 235:
                final_bbox_dist.append(dist)
                final_bbox_angle.append(angle)
            else:
                angle_updated = regressor1.predict(np.array([[angle]]))
                final_bbox_dist.append(dist)
                final_bbox_angle.append(angle_updated[0])
                
    return final_bbox_dist, final_bbox_angle        


          
bbox1_path = 'C:/Users/osman/OneDrive/Documents/V2V_Project_SP2023/ML/LSTM/bbox_labels_tmp/rgb1/'
bbox2_path = 'C:/Users/osman/OneDrive/Documents/V2V_Project_SP2023/ML/LSTM/bbox_labels_tmp/rgb2/'
#r'C:/Users/osman/OneDrive/Documents/V2V_Project_SP2023/ML/LSTM/Datasets_Seqs/'
#C:\Users\osman\OneDrive\Documents\V2V_Project_SP2023\ML\LSTM\bbox_labels_tmp
bbox1_img_path = 'C:/Users/osman/OneDrive/Documents/V2V_Project_SP2023/ML/LSTM/bbox_images_out_tmp/rgb1/'
bbox2_img_path = 'C:/Users/osman/OneDrive/Documents/V2V_Project_SP2023/ML/LSTM/bbox_images_out_tmp/rgb2/'


fig = plt.figure()
for seq_dataset in ['scenario36_seq_1_cleaned']:
    df1 = pd.read_csv(f'./Data/{seq_dataset}.csv')

    rgb1 = df1['unit1_rgb1'].values
    rgb2 = df1['unit1_rgb2'].values
    pwr_256_beam = df1['gt_beam'].values
    abs_index = df1['abs_index'].values
    
    pwr_vec_256 =df1['pwr_vec'].values
    #tx_user_dist_1 =  df1['user_dist_1'].values
    #tx_user_angle_1 =  df1['user_angle_1'].values
    #tx_user_dist_2 =  df1['user_dist_2'].values
    #tx_user_angle_2 =  df1['user_angle_2'].values
    
    

    
    abs_index_ext = []
    pwr_vec_gt = []
    gc_output_dist =[]
    gc_output_angle =[]
    pwr_beam_gt =[]
    gc_users_x =[]
    gc_users_y =[]
    gt_dist_norm =[]
    gt_angle_norm =[]
    
    user_x =[]
    user_y =[]
    
    
    tx_gc_ouput_dist = []
    tx_gc_ouput_angle = []
    
    tx_gc_ouput_x = []
    tx_gc_ouput_y = []
    pwr_norm =[]
    
    #for val in range(2552, 2722): 
    for val in range(len(abs_index)): 
        #if val not in [7804,8067]: # Skip samples in seq 5
        #if val not in [24]: # skip sample in seq 6
        if True:
            img1_name = rgb1[val].split('/')[2]
            img2_name = rgb2[val].split('/')[2]
            
            bbox1 = img1_name.split('.')[0:2]
            bbox2 = img2_name.split('.')[0:2]
            
            #print(img1_name)
            #print(img2_name)
            
            bbox1_name = f'{bbox1[0]}.{bbox1[1]}.txt'
            bbox2_name = f'{bbox2[0]}.{bbox2[1]}.txt'        
            
            img1_name_out = f'{bbox1[0]}.{bbox1[1]}_out.jpg'
            img2_name_out = f'{bbox2[0]}.{bbox2[1]}_out.jpg'   
            
            bbox1_data = np.loadtxt(f'{bbox1_path}{bbox1_name}').tolist()
            bbox2_data = np.loadtxt(f'{bbox2_path}{bbox2_name}').tolist()
            
            # shutil.copy(rgb1[val], f'{img1_path}/{img1_name}')
            # shutil.copy(rgb2[val], f'{img2_path}/{img2_name}')
            
            # shutil.copy(f'{bbox1_path}{bbox1_name}', f'{bbox1_path_new}/{bbox1_name}')
            # shutil.copy(f'{bbox2_path}{bbox2_name}', f'{bbox2_path_new}/{bbox2_name}')    
            
            # shutil.copy(f'{bbox1_img_path}{img1_name_out}', f'{bbox1_img}/{img1_name_out}')
            # shutil.copy(f'{bbox2_img_path}{img2_name_out}', f'{bbox2_img}/{img2_name_out}')    
               
            #print("bbox2")
            
            bbox1_angle_dist = []
            bbox2_angle_dist = []
            
            if any(isinstance(i, list) for i in bbox2_data):
                for data in bbox2_data:
                    if data[0] in [2.0,5.0,7.0]:
                        #print(data)
                        pos1 = [0.5, 0]
                        pos2 = [data[1], (1 - data[2])]
                        angle_dist = angle_and_dist(pos1,pos2 )
                        #print(angle_dist)
                        bbox2_angle_dist.append(angle_dist)
            else:
                if data[0] in [2.0,5.0,7.0]:
                    #print(bbox2_data)
                    pos1 = [0.5, 0]
                    pos2 = [bbox2_data[1], (1 - bbox2_data[2])]
                    angle_dist = angle_and_dist(pos1,pos2 )
                    #print(angle_dist) 
                    bbox2_angle_dist.append(angle_dist)
               
                
                
            #print("bbox1")    
            if any(isinstance(i, list) for i in bbox1_data):
                for data in bbox1_data:
                    if data[0] in [2.0,5.0,7.0]:
                    
                        #print(data)
                        pos1 = [0.5, 0]
                        pos2 = [data[1], (1 - data[2])]
                        angle_dist = angle_and_dist(pos1,pos2 )
                        #print(angle_dist)
                        bbox1_angle_dist.append(angle_dist)
            else:
                if data[0] in [2.0,5.0,7.0]:
                
                    #print(bbox1_data)
                    pos1 = [0.5, 0]
                    pos2 = [bbox1_data[1], (1 - bbox1_data[2])]
                    angle_dist = angle_and_dist(pos1,pos2 )
                    #print(angle_dist) 
                    bbox1_angle_dist.append(angle_dist)
            
         
           
            if pwr_256_beam[val] in range(0,32) or pwr_256_beam[val] in range(160,256):  
                bbox_dist, bbox_angle = generate_final_bbox_coord_seq1(bbox1_angle_dist, bbox2_angle_dist)
                
            elif pwr_256_beam[val] in range(33,160):
                bbox_dist, bbox_angle = generate_final_bbox_coord_seq1(bbox1_angle_dist, bbox2_angle_dist)
               
                    
            print(f'Iternation index: {val}')
 
            gc_output_dist.append(bbox_dist)
            gc_output_angle.append(bbox_angle)
            user_x.append(bbox_dist*np.cos(np.array(bbox_angle)*np.pi/180))
            user_y.append(bbox_dist*np.sin(np.array(bbox_angle)*np.pi/180))
            abs_index_ext.append(abs_index[val])
            pwr_beam_gt.append(pwr_256_beam[val])
            pwr_vec_gt.append(pwr_vec_256[val])
            pwr_vec =  ast.literal_eval(pwr_vec_256[val])
            pwr_vec = list((pwr_vec-np.min(pwr_vec))/(np.max(pwr_vec)-np.min(pwr_vec)))
            pwr_norm.append(pwr_vec)
          
            # plt.cla()
            # ax = plt.subplot(111, projection='polar')
            # ax.set_rlim(0,1)
            # ax.scatter(np.array(bbox_angle)*np.pi/180, np.array(bbox_dist),   alpha=0.9)
            
            # ax.set_theta_zero_location('E')
            # ax.set_theta_direction(1) # clockwise
            # ax.grid(True)
            
            # # ax.set_ylabel('Time', color='crimson')
            # ax.tick_params(axis='y', colors='crimson')
            
            # #plt.show() 
            # plot_name = f'{bbox1[0]}.{bbox1[1]}.png'
            # # save plot to file
            # fig.savefig(f'{plot_save_path}/{plot_name}')       
                
    csv_to_save = './Outputs/'+seq_dataset+'April_20th_run1.csv'
    df2 = pd.DataFrame()
    df2['abs_index'] = abs_index_ext  
    
    df2['gc_bbox_centers'] = user_x
    df2['gc_bbox_centers'] = user_y 
    #df2['gc_tx_bbox_x'] = tx_gc_ouput_x
    #df2['gc_tx_bbox_y'] = tx_gc_ouput_y
         
    df2['gc_dist'] = gc_output_dist      
    df2['gc_angle'] = gc_output_angle
    
    
    df2['pwr_vec_256'] = pwr_vec_gt
    df2['pwr_vec_256_norm'] = pwr_norm
    df2['gc_beam_gt'] = pwr_beam_gt
   
    df2.to_csv(csv_to_save, index=False)



   



    
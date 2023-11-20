# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:36:27 2023

@author: osman
"""

import torch
from torch.utils.data import DataLoader
from load_datasets import DataFeed
import pandas as pd
import numpy as np


for q in [1,5,6,9,12]:
    test_dir = f'../Outputs_polar_cord/scenario36_seq_{q}_cleanedNov_run1.csv'
    dataset = pd.read_csv(test_dir)
    val_batch_size = 1
    
    dist_PATH =r'../TX_ID_Problem/Saved_FC_Models/saved_FC_model_Dist_run1.pt'
    Angle_PATH =r'../TX_ID_Problem/Saved_FC_Models/saved_FC_model_Angle_run1.pt'

    val_save_path = f'./TX_ID_Outputs/scenario36_teston_seq_{q}_cleaned_Nov_Pred_dist_Angle_run1.csv'
    
    test_loader = DataLoader(DataFeed(test_dir,start_index=5,stop_index=9),
                              batch_size=1,
                              #num_workers=8,
                              shuffle=False)
    
    # Loaded Model 1
    model = torch.jit.load(dist_PATH)
    #model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # df = pd.read_csv(val_dir)
    for e in  range(1):   
               
        # VALIDATION    
        with torch.no_grad():
            
            val_epoch_loss = 0
            
            model.eval()
            
            pred_val = []
            
            for tr_count, (pwr_data, center_data) in enumerate(test_loader):
                data = pwr_data.type(torch.Tensor)   
                #print(pwr_data, center_data)               
                #label = center_data[:, 0].type(torch.Tensor)                      
                #x_val, y_val = data.to(device), label.to(device)
                x_val= data.to(device)
                print(x_val.shape)
    
                
                y_val_pred = model(x_val)
                y_val_pred = torch.squeeze(y_val_pred,1)
                #print("y_val_pred", y_val_pred.shape)
                pred_val.append(y_val_pred)
        
        pred_val = [a.squeeze().tolist() for a in pred_val]   
    
    dataset['pred_dist_norm'] =  pred_val
    
    # Loaded Model 2
    model = torch.jit.load(Angle_PATH)
    
    for e in  range(1):   
        # VALIDATION    
        with torch.no_grad():
            
            val_epoch_loss = 0
            
            model.eval()
            
            pred_val = []
            
            for tr_count, (pwr_data, center_data) in enumerate(test_loader):
                data = pwr_data.type(torch.Tensor)   
                #print(pwr_data, center_data)               
                #label = center_data[:, 0].type(torch.Tensor)                      
                #x_val, y_val = data.to(device), label.to(device)
                x_val= data.to(device)
                #print(x_val.shape)
    
                
                y_val_pred = model(x_val)
                y_val_pred = torch.squeeze(y_val_pred,1)
                #print("y_val_pred", y_val_pred.shape)
                pred_val.append(y_val_pred)
        
        pred_val = [a.squeeze().tolist() for a in pred_val]   
    
    dataset['pred_angle_norm'] =  pred_val
    dist_pred = dataset['pred_dist_norm'].values
    angle_pred = dataset['pred_angle_norm'].values
    dataset['actual_dist'] = np.array(dist_pred)*(np.sqrt(0.5**2+1)-0.25)+0.25
    dataset['actual_angle'] = np.array(angle_pred)*(362 -(-14.85))-14.85
    dataset.to_csv(val_save_path, index=False)
    print(f'Done with {q}')
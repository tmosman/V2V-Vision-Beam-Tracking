# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 02:14:07 2023

@author: osman
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 01:08:56 2023

@author: osman
"""

import pandas as pd 
import numpy as np
from datetime import datetime
import utm
import math



for k in [1,5,6,9,12]:
    
    #val_save_path = f'./TX_ID_Outputs/scenario36_teston_seq_{q}_cleaned_Nov_Pred_dist_Angle_run1.csv'
    #df = pd.read_csv(f'./TX_ID_Outputs/scenario36_teston_seq_{k}_cleaned_Nov_Pred_dist_Angle_Seq_Exp2_final.csv')
    #Tx_Tracking_output_Seq_1_Nov_run1
    df = pd.read_csv(f'./TX_Tracking/Tx_Tracking_output_Seq_{k}_Nov_run1.csv')
    df1 =  pd.read_csv('../Raw_datasets/scenario36_updated_new.csv')
    
    index_orig = df1['abs_index'].values
    index_test = df['abs_index'].values
    timestamp = df1['timestamp'].values
    tx_gps = df1['unit1_gps1'].values
    tx_lat =df1['unit1_gps1_lat'].values
    tx_lon =df1['unit1_gps1_lon'].values
    
    rx_gps = df1['unit2_gps1'].values
    rx_lat =df1['unit2_gps1_lat'].values
    rx_lon =df1['unit2_gps1_lon'].values
    
    
    
    timestamp_list = []
    tx_gps_list = []
    tx_lat_list = []
    tx_lon_list = []
    
    rx_gps_list =  []
    rx_lat_list = []
    rx_lon_list = []
    
    
    count = 0
    
    for val in range(len(index_orig)):
        if index_orig[val] in  index_test:
           timestamp_list.append(timestamp[val])
           #tx_gps_list.append(tx_gps[val])
           tx_lat_list.append(tx_lat[val])
           tx_lon_list.append(tx_lon[val])
           rx_lat_list.append(rx_lat[val])
           rx_lon_list.append(rx_lon[val])
    
    
    df['timestamp'] = timestamp_list
    df['unit1_gps1_lat'] = tx_lat_list
    df['unit1_gps1_lon'] = tx_lon_list
    df['unit2_gps1_lat'] = rx_lat_list
    df['unit2_gps1_lon'] = rx_lon_list
    
    df.to_csv(f'./TX_Tracking/Tx_Tracking_output_Seq_{k}_Nov_run1_updated.csv', index=False)
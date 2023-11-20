# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:01:32 2023

@author: Tawfik Osman
"""

import torch
import torch.optim as optimizer
import torch.nn as nn
from load_datasets import DataFeed
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ML_Model import MultipleRegression


def Train_model(datasets_dir,output_dir,model_dir,header_index, nn, optimizer):
    
    # Hyper-parameters
    batch_size = 32
    val_batch_size = 1
    decay = 1e-4
    num_epochs = 20
    
    LEARNING_RATE = 0.01
    NUM_FEATURES = 256
    node = 128
    out = 1
    
    # Loading datasets
    train_dir = f'./{datasets_dir}/scenario36_train_3_final.csv'
    val_dir = f'./{datasets_dir}/scenario36_test_3_final.csv'
    val_save_path = f'./{output_dir}/scenario36_teston_seq_1_cleaned_April_20th_val_Angle.csv'
    
    train_loader = DataLoader(DataFeed(train_dir,start_index=5,stop_index=8),
                              batch_size=batch_size,
                              #num_workers=8,
                              shuffle=False)
    
    val_loader = DataLoader(DataFeed(val_dir,start_index=5,stop_index=8),
                            batch_size=val_batch_size,
                            #num_workers=8,
                            shuffle=False)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = MultipleRegression(NUM_FEATURES,node,out)
    model.to(device)
    
    #print(model)
    criterion = nn.MSELoss()
    optimizer = optimizer.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-3, weight_decay=decay, amsgrad=True) # 0.9934
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    loss_stats = {"train": [],"val": [],"mse": [],"R2square":[]}    

    df = pd.read_csv(val_dir)
    if header_index == 1:
        y_orig = df['gc_angle_norm']
        save_model_path = f'./{model_dir}/saved_FC_model_Angle_run1.pt'
    else:
        y_orig = df['gc_dist_norm']
        save_model_path = f'./{model_dir}/saved_FC_model_Dist_run1.pt'
    
    print("Orig data", y_orig.shape,y_orig)
    epoch_list = []
    MSE_list = []
    r2_list =[]
    
  
    
    print("Begin training.")
    for e in  range(1, num_epochs+1):   
        
        # TRAINING
        train_epoch_loss = 0
        model.train()
       
        for tr_count, (pwr_data, center_data) in enumerate(train_loader):
            data = pwr_data.type(torch.Tensor) 
            label = center_data[:,header_index].type(torch.Tensor)  
            #print("label shape", label.shape)
                        
            x, label = data.to(device), label.to(device)
            #print("Label",label , "x:input ",x)
            optimizer.zero_grad()
            pred = model(x)
            pred = torch.squeeze(pred, 1)
            #print("Predicted", pred)
           
            train_loss = criterion(pred, label)
            
            train_loss.backward()
            optimizer.step()
            
            train_epoch_loss += train_loss.item()
            
            
        # VALIDATION    
        with torch.no_grad():
            
            val_epoch_loss = 0
            
            model.eval()
            
            pred_val = []
            
            for tr_count, (pwr_data, center_data) in enumerate(val_loader):
                data = pwr_data.type(torch.Tensor)   
                #print(pwr_data, center_data)               
                label = center_data[:, header_index].type(torch.Tensor)                      
                x_val, y_val = data.to(device), label.to(device)
                
                #print(x_val.shape, y_val.shape)

                
                y_val_pred = model(x_val)
                y_val_pred = torch.squeeze(y_val_pred,1)
                #print("y_val_pred", y_val_pred.shape)
                pred_val.append(y_val_pred)
                            
                val_loss = criterion(y_val_pred, y_val)
                
                val_epoch_loss += val_loss.item()
        scheduler.step(val_loss)
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        
        
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')    
        
        pred_val = [a.squeeze().tolist() for a in pred_val]   
        #print("pred_val", len(pred_val))
        mse = mean_squared_error(y_orig, pred_val)
        loss_stats['mse'].append(mse)
        r_square = r2_score(y_orig, pred_val)
        loss_stats['R2square'].append(r_square)
      
        r2_list.append(r_square)
        MSE_list.append(mse)
        epoch_list.append(e)     
            
            
    #df[['pred_dist_norm','pred_angle_norm']] =  pred_val
    #df['pred_dist_norm'] =  pred_val
    if header_index == 1:
        df['pred_angle_norm'] =  pred_val
    else:
        df['pred_dist_norm'] =  pred_val
    df.to_csv(val_save_path, index=False)   
    plt.plot(epoch_list,MSE_list)    
    plt.savefig('MSE_history_plot.png')  
    
    
    print(f'Mean R2Square:  {np.max(np.array(r2_list))}')
    print(f'Mean MSE:  {np.min(np.array(MSE_list))}') 
    
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(save_model_path) # Save
    
    return model

def Test_model(model, datasets_dir,save_dir):
    combined_dir = f'./{datasets_dir}/scenario36_seq_1_cleanedApril_7th_run1.csv'
    save_combined = f'./{datasets_dir}/scenario36_teston_seq_1_cleaned_April_20th_Pred_Angle_Seq_Exp3.csv'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(save_combined)
    test_loader = DataLoader(DataFeed(combined_dir,start_index=5,stop_index=9),
                              batch_size=1,
                              #num_workers=8,
                              shuffle=False)
    
    obj_tx = []
    
    with torch.no_grad():
         model.eval()
         pred_val = []
         for tr_count, (pwr_data, center_data) in enumerate(test_loader):
             #pwr_data = pwr_data/max(pwr_data)
             #print(pwr_data)
             data = pwr_data.type(torch.Tensor)                     
             label = center_data[:, 0].type(torch.Tensor)                      
             x_val, y_val = data.to(device), label.to(device)
             #print(x_val, y_val)
             #print(x_val.shape, y_val.shape)
             #print(x_val.shape)
             y_test_pred = model(x_val)
             y_test_pred = torch.squeeze(y_test_pred,1)
             pred = y_test_pred.detach().cpu().numpy()
             #print("count", tr_count,pred)
             obj_tx.append(pred[0])
             pred_val.append(y_test_pred)
    pred_val = [a.squeeze().tolist() for a in pred_val] 
    obj_tx = [a.squeeze().tolist() for a in obj_tx] 
    df['pred_angle_norm'] =  pred_val 
    # df[['actual_dist','actual_angle']] = (pred_val-min(pred_val))/(max(pred_val)-min(pred_val))
    df['tx_obj'] =  obj_tx 
    df.to_csv(save_combined, index=False)
    
    return 1


def main(IND):
    trained_model = Train_model(datasets_dir='Annotated_Datasets',output_dir='Output_Datasets',model_dir='Saved_FC_Models', header_index=IND, nn=nn,optimizer=optimizer)
    
    ## Uncomment Test_model, if you want to test the trained model on Test-set
    #Test_model(trained_model,datasets_dir='Outputs',save_dir='FC_output_new')
    
   
    


if __name__ == "__main__":
    #run()
    IND = 0 # if IND is 1, the model maps pwr <--> polar angle, IND is 0 the model maps pwr <--> polar distance
    main(IND)

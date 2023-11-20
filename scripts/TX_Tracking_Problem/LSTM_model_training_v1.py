# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:05 2021

@author: osman
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import r2_score
from all_model import LSTM_class
import csv


seq_no = '1'

train_dir = f'./LSTM_datasets/BT_train_seq_{seq_no}_updated.csv'
val_dir = f'./LSTM_datasets/BT_test_seq_{seq_no}_updated.csv'


#%%## Loading Trainset and Testset ##

# Train-set
df_beam_train=pd.read_csv(train_dir, sep=',',header='infer')

dist_X = df_beam_train.loc[:,['pred_dist_0_norm','pred_dist_1_norm',
                              'pred_dist_2_norm',"pred_dist_3_norm",
                              "pred_dist_4_norm","pred_dist_5_norm"]].values
angle_X = df_beam_train.loc[:,['pred_angle_0_norm','pred_angle_1_norm',
                               'pred_angle_2_norm','pred_angle_3_norm',
                         'pred_angle_4_norm','pred_angle_5_norm']].values

Beams_y = df_beam_train.loc[:,['gt_beam_5']].values

dist_X= dist_X.reshape((dist_X.shape[0], dist_X.shape[1], 1))
angle_X = angle_X.reshape((angle_X.shape[0],  dist_X.shape[1], 1))


Beams_y = Beams_y.astype(int)


train_X = np.stack((dist_X,angle_X), axis=-1).reshape((angle_X.shape[0], dist_X.shape[1], 2))
train_y = Beams_y.reshape((Beams_y.shape[0], Beams_y.shape[1], 1))

print(train_X.shape)
print(train_y.shape)

# Test-set
df_beam_test=pd.read_csv(val_dir, sep=',',header='infer')

dist_X = df_beam_test.loc[:,['pred_dist_0_norm','pred_dist_1_norm',
                             'pred_dist_2_norm',"pred_dist_3_norm",
                             "pred_dist_4_norm","pred_dist_5_norm"]].values
angle_X = df_beam_test.loc[:,['pred_angle_0_norm','pred_angle_1_norm',
                              'pred_angle_2_norm','pred_angle_3_norm', 
                              'pred_angle_4_norm','pred_angle_5_norm']].values

Beams_y = df_beam_test.loc[:,['gt_beam_5']].values
Beams_y = Beams_y.astype(int)

dist_X= dist_X.reshape((dist_X.shape[0], dist_X.shape[1], 1))
angle_X = angle_X.reshape((angle_X.shape[0],  dist_X.shape[1], 1))

test_X = np.stack((dist_X,angle_X), axis=-1).reshape((angle_X.shape[0], dist_X.shape[1], 2))
test_y = Beams_y.reshape((Beams_y.shape[0], Beams_y.shape[1], 1))

print(test_X.shape)
print(test_y.shape)

#%%



###### Creation of LSTM Model ######
# configure problem
n_features = 256
n_timesteps_in = 6
n_timesteps_out = 1
input_dim = test_X.shape[2]
num_hidden_nodes = 150
batch_size= 16
num_epochs = 20

number_of_classes = n_features # number of unique digits
train_y = to_categorical(train_y,number_of_classes)
test_y = to_categorical(test_y,number_of_classes)


#%%  define model and train it 
model_class = LSTM_class()
model= model_class.create_structure(input_dim, n_timesteps_in, n_timesteps_out, num_hidden_nodes, n_features)
print(f'The model structure: \n {model.summary()}')

SelectedOptimizer = tf.optimizers.Adam(learning_rate=0.001)  ### Optimizer used
model.compile(loss='categorical_crossentropy', optimizer=SelectedOptimizer, metrics=['accuracy'])

###### Model Training ######
history = model.fit(train_X,train_y,
                    batch_size=batch_size, 
                    validation_data=(test_X, test_y)
                    ,epochs=num_epochs, verbose=1)



#%%

#%%##  Evaluation of model ###
pred = model.predict(test_X)
pred_y = np.argmax(pred,axis=2)
y_test = np.argmax(test_y,axis=2)


r1 = r2_score(y_test[:,0], pred_y[:,0]),

#out_lst =  np.hstack((pred_y,y_test))
print(f'R-square for Beam 5:  {(r1[0]):.4f}')
print(f'Top-1 : {((pred_y[:,0] == y_test[:,0]).sum()/y_test[:,0].shape[0]):.4f}')
print(f'Top-5 : {((abs(pred_y[:,0]- y_test[:,0])<5).sum()/y_test[:,0].shape[0]):.4f}')

#%% 

'''

with open("./analysis/Results_Seq_Beam.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(out_lst)
    
'''
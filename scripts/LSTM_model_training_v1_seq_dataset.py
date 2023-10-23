# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:05 2021

@author: osman
"""
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.utils import np_utils
import os


seq_no = '1'
train_dir = f'./Outputs/Final/BT_train_seq_{seq_no}_exp2.csv'
val_dir = f'./Outputs/Final/BT_test_seq_{seq_no}_exp2.csv'

###### Loading Trainset and Testset ######
#Train
df_beam_train=pd.read_csv(train_dir, sep=',',header='infer')

dist_X = df_beam_train.loc[:,['pred_dist_0_norm','pred_dist_1_norm','pred_dist_2_norm',"pred_dist_3_norm",
                        "pred_dist_4_norm","pred_dist_5_norm"]].values
angle_X = df_beam_train.loc[:,['pred_angle_0_norm','pred_angle_1_norm','pred_angle_2_norm','pred_angle_3_norm',
                         'pred_angle_4_norm','pred_angle_5_norm']].values

#Beams_y = df_beam.loc[:,['gt_beam_0','gt_beam_1','gt_beam_2','gt_beam_3','gt_beam_4','gt_beam_5']].values

Beams_y = df_beam_train.loc[:,['gt_beam_5']].values
Beams_y = Beams_y.astype(int)

dist_X= dist_X.reshape((dist_X.shape[0], dist_X.shape[1], 1))
angle_X = angle_X.reshape((angle_X.shape[0],  dist_X.shape[1], 1))
train_X = np.stack((dist_X,angle_X), axis=-1).reshape((angle_X.shape[0], dist_X.shape[1], 2))
#train_X = angle_X
train_y = Beams_y.reshape((Beams_y.shape[0], Beams_y.shape[1], 1))

print(train_X.shape)
print(train_y.shape)

#Test
df_beam_test=pd.read_csv(val_dir, sep=',',header='infer')

dist_X = df_beam_test.loc[:,['pred_dist_0_norm','pred_dist_1_norm','pred_dist_2_norm',"pred_dist_3_norm",
                        "pred_dist_4_norm","pred_dist_5_norm"]].values
angle_X = df_beam_test.loc[:,['pred_angle_0_norm','pred_angle_1_norm','pred_angle_2_norm','pred_angle_3_norm',
                         'pred_angle_4_norm','pred_angle_5_norm']].values

#Beams_y = df_beam.loc[:,['gt_beam_0','gt_beam_1','gt_beam_2','gt_beam_3','gt_beam_4','gt_beam_5']].values

Beams_y = df_beam_test.loc[:,['gt_beam_5']].values
Beams_y = Beams_y.astype(int)

dist_X= dist_X.reshape((dist_X.shape[0], dist_X.shape[1], 1))
angle_X = angle_X.reshape((angle_X.shape[0],  dist_X.shape[1], 1))
test_X = np.stack((dist_X,angle_X), axis=-1).reshape((angle_X.shape[0], dist_X.shape[1], 2))
#test_X = angle_X
test_y = Beams_y.reshape((Beams_y.shape[0], Beams_y.shape[1], 1))

print(test_X.shape)
print(test_y.shape)


##############################################################


###### Convert Target Beams to one hot code ######
nb_classes = 256 # number of unique digits
train_y = np_utils.to_categorical(train_y,nb_classes)
test_y = np_utils.to_categorical(test_y,nb_classes)


###### Creation of LSTM Model ######
# configure problem
n_features = 256
n_timesteps_in = 6
n_timesteps_out = 1
input_dim = test_X.shape[2]
# define model
model = Sequential()
model.add(LSTM(150, input_shape=(n_timesteps_in,input_dim)))
model.add(RepeatVector(n_timesteps_out))
model.add(Dropout(0.5))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(n_features, activation='softmax')))


pt = tf.optimizers.Adam(learning_rate=0.001)  ### Optimizer used
model.compile(loss='categorical_crossentropy', optimizer=pt, metrics=['accuracy'])
print(model.summary())

checkpoint_path = f"lstm_training/cp_seq{seq_no}_Dist_Angle_1_dataset.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(os.listdir(checkpoint_dir))

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

###### Model Training ######
history = model.fit(train_X,train_y,
                    batch_size=16, 
                    validation_data=(test_X, test_y)
                    ,epochs=200, verbose=1)


###### Visualizing MSE error ###### 
#plt.figure(0)
# plt.plot(history.history['val_accuracy'],'r-', label='Val Accuracy Curve')
# #plt.plot(history.history['accuracy'], label='Accuracy')
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.savefig('seq_model_training_history')
# plt.show()


###### Evaluation of model ######
pred = model.predict(test_X)
pred_y = np.argmax(pred,axis=2)
y_test = np.argmax(test_y,axis=2)


r1 = r2_score(y_test[:,0], pred_y[:,0]),

#out_lst =  np.hstack((pred_y,y_test))
print("R-square for Beam 5: ", 
      r1[0])
print(f'Top-1 : {(pred_y[:,0] == y_test[:,0]).sum()/y_test[:,0].shape[0]}')
print(f'Top-5 : {(abs(pred_y[:,0]- y_test[:,0])<5).sum()/y_test[:,0].shape[0]}')

model.save(f'lstm_models/my_model_Dist_Angle_Seq{seq_no}_dataset_{1}')

df_beam_test['pred_beam_5'] = pred_y[:,0]

df_beam_test.to_csv(f'./Outputs/Final/BT_test_seq{seq_no}_pred_exp2.csv')
# #Test
# df_beam_test=pd.read_csv('BeamTracking_Dataset_Seq1_all.csv', sep=',',header='infer')

# dist_X = df_beam_test.loc[:,['pred_dist_0_norm','pred_dist_1_norm','pred_dist_2_norm',"pred_dist_3_norm",
#                         "pred_dist_4_norm","pred_dist_5_norm"]].values
# angle_X = df_beam_test.loc[:,['pred_angle_0_norm','pred_angle_1_norm','pred_angle_2_norm','pred_angle_3_norm',
#                           'pred_angle_4_norm','pred_angle_5_norm']].values

# #Beams_y = df_beam.loc[:,['gt_beam_0','gt_beam_1','gt_beam_2','gt_beam_3','gt_beam_4','gt_beam_5']].values

# Beams_y = df_beam_test.loc[:,['gt_beam_5']].values
# Beams_y = Beams_y.astype(int)

# dist_X= dist_X.reshape((dist_X.shape[0], dist_X.shape[1], 1))
# angle_X = angle_X.reshape((angle_X.shape[0],  dist_X.shape[1], 1))
# test_X_seq = np.stack((dist_X,angle_X), axis=-1).reshape((angle_X.shape[0], dist_X.shape[1], 2))
# #test_X_seq = angle_X
# test_y_seq = Beams_y.reshape((Beams_y.shape[0], Beams_y.shape[1], 1))

# print(test_X_seq.shape)
# print(test_y_seq.shape)


##############################################################


# ###### Convert Target Beams to one hot code ######
# nb_classes = 256 # number of unique digits
# #train_y = np_utils.to_categorical(train_y,nb_classes)
# test_y_seq = np_utils.to_categorical(test_y_seq,nb_classes)

pred_seq = model.predict(test_X)
pred_seq1 = np.argmax(pred_seq[:,0],axis=-1)
test_seq1 = np.argmax(test_y[:,0],axis=-1)

plt.figure(0)
plt.plot(test_seq1,'-*r', label='Ground Truth')
plt.plot(pred_seq1,'-*b',label='Predicted Index',alpha=0.5)


plt.ylabel('Beam_5 Index')
plt.xlabel('Sample_index')
plt.legend()

r1 = r2_score(test_seq1, pred_seq1)
print("R-square for Beam 5 on Sequence 1: ", 
      r1)


'''
new_model = tf.keras.models.load_model('lstm_models/my_model')
new_model.summary()
predx = new_model.predict(test_X)
(pred_y[:,0] == y_test[:,0]).sum()/y_test[:,0].shape[0
])
(abs(pred_y[:,0]- y_test[:,0])<5).sum()/y_test[:,0].shape[0])

(abs(pred_y[:,0]- y_test[:,0])<10).sum()/y_test[:,0].shape[0]
)

pred_seq = model.predict(test_X_seq)
pred_seq1 = np.argmax(pred_seq[:,0],axis=-1)
test_seq1 = np.argmax(test_y_seq[:,0],axis=-1)

plt.plot(test_seq1,'-*r', label='Ground Truth')
plt.plot(pred_seq1,'-*b',label='Predicted Index')
plt.ylabel('Beam_5 Index')
plt.xlabel('Sample_index')
plt.legend()


import csv
with open("./analysis/Results_Seq_Beam.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(out_lst)
    
'''
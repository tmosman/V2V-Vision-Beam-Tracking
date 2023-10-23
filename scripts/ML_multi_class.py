# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:01:32 2023

@author: osman
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from model_script import MultipleRegression
from ignite.engine import Engine
from ignite.metrics import TopKCategoricalAccuracy
from sklearn.metrics import r2_score


def process_function(engine, batch):
    y_pred, y = batch
    return y_pred, y

def one_hot_to_binary_output_transform(output):
    y_pred, y = output
    y = torch.argmax(y, dim=1)  # one-hot vector to label index vector
    return y_pred, y

# read data and apply one-hot encoding
train_data = pd.read_csv("scenario36_train_4_new.csv")
test_data = pd.read_csv("scenario36_test_4_new.csv")

X = train_data.iloc[:, 6:8]
y = train_data.iloc[:, 12].values.reshape(-1, 1)

X1 = test_data.iloc[:, 6:8]
y1 = test_data.iloc[:, 12].values.reshape(-1, 1)
#print(y1)
y_all = np.vstack((y,y1))
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_all)
y = ohe.transform(y)
y1 = ohe.transform(y1)


#inverse_transform
#print(np.argmax(y[1]))


# convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
#X = np.array(np.array(X.values),dtype=np.float16)
#X = torch.from_numpy(X.values)
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
X1 = torch.tensor(X1.values, dtype=torch.float32)
y1 = torch.tensor(y1, dtype=torch.float32)
# split
X_train, _, y_train, _ = train_test_split(X, y, train_size=0.99, shuffle=True)
_, X_test, _, y_test = train_test_split(X1, y1, test_size=0.99, shuffle=True)
engine = Engine(process_function)
metric = TopKCategoricalAccuracy(
    k=5, output_transform=one_hot_to_binary_output_transform)
metric.attach(engine, 'top_k_accuracy')





NUM_FEATURES = 2
node = 32
out = y.shape[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MultipleRegression(NUM_FEATURES,node,out)

# loss metric and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 200
batch_size = 5
batches_per_epoch = len(X_train) // batch_size

best_acc = - np.inf   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []
predict_list =[]
# training loop

for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    # set model in training mode and run through each batch
    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
    # set model in evaluation mode and run through the test set
    model.eval()
    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    state = engine.run([[y_pred, y_test]])
    print(state.metrics['top_k_accuracy'])
    predict_list.append(torch.argmax(y_pred, 1))
    ce = float(ce)
    acc = float(acc)
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(ce)
    test_acc_hist.append(state.metrics['top_k_accuracy'])
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

    



# Restore best model
model.load_state_dict(best_weights)
plt.figure(1)
# Plot the loss and accuracy
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.show()
plt.figure(0)
plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()



y_t = ohe.inverse_transform(y_test.detach().numpy())
y_p = ohe.inverse_transform(y_pred.detach().numpy())
r1 = r2_score(y_t, y_p)
print(f'R-2 Square: {r1}')
plt.figure(2)
plt.plot(y_t,'-*r', label='Ground Truth')
plt.plot(y_p,'-*b',label='Predicted Index')
plt.ylabel('Beam_5 Index')
plt.xlabel('Sample_index')
plt.legend()
plt.show()
print(f'Top-1 Accuracy: {(abs(y_t- y_p)<1).sum()/y_t.shape[0]}')
print(f'Top-5 Accuracy: {(abs(y_t- y_p)<5).sum()/y_t.shape[0]}')





test_data1 = pd.read_csv("Seq1_selected_GT_BeamTracking_April_5th_Exp1.csv")

X11 = test_data1.loc[:, ['pred_dist_5_norm','pred_angle_5_norm']]
y11 = test_data1.loc[:, 'pred_beam_5'].values.reshape(-1, 1)
y11 = ohe.transform(y11)

X11 = torch.tensor(X11.values, dtype=torch.float32)
y11 = torch.tensor(y11, dtype=torch.float32)

print(y11)
y_pred1 = model(X11)


y_t = ohe.inverse_transform(y11)
y_p = ohe.inverse_transform(y_pred1.detach().numpy())

# y_t_1  = []
# for t in y_t[:,0]:
#     if t is not None:
#         y_t_1.append(t)
        

# y_p_1  = []
# for t in y_p[:,0]:
#     if t is not None:
#         y_p_1.append(t)
y_t_1  = []
y_p_1  = []
for t,t1 in zip(y_t[:,0],y_p[:,0]):
    if t is not None:
        y_t_1.append(t)
        y_p_1.append(t1)


# for t in y_p[:,0]:
#     if t is not None:
#         y_p_1.append(t)
r1 = r2_score(y_t_1, y_p_1)
y_t_1, y_p_1 = np.array(y_t_1),np.array(y_p_1)
print(f'R-2 Square: {r1}')
plt.figure(2)
plt.plot(y_t,'-*r', label='Ground Truth')
plt.plot(y_p,'-*b',label='Predicted Index')
plt.ylabel('Beam_5 Index')
plt.xlabel('Sample_index')
plt.legend()
plt.show()
print(f'Top-1 Accuracy: {(abs(y_t_1- y_p_1)<1).sum()/y_t_1.shape[0]}')
print(f'Top-5 Accuracy: {(abs(y_t_1- y_p_1)<5).sum()/y_t_1.shape[0]}')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")
kfold = StratifiedKFold(n_splits=10, random_state=seed)

X = datasetTrain.values[:, 0:8]
y = datasetTrain.values[:, 8]

#%%

k_max = 18

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=k_max, 
                              random_state=seed)

acc_train = []
acc_val = []
mcc_train = []
mcc_val = []
f1_train = []
f1_val = []

i = 10

for train_index, val_index in rkf.split(X, y):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    knn = KNeighborsClassifier(n_neighbors=i//10)
    
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_val = knn.predict(X_val)
    
    acc_train.append(accuracy_score(y_train, y_pred_train))
    acc_val.append(accuracy_score(y_val, y_pred_val))

    f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
    f1_val.append(f1_score(y_val, y_pred_val, average='macro')) 
    
    mcc_train.append(matthews_corrcoef(y_train, y_pred_train))
    mcc_val.append(matthews_corrcoef(y_val, y_pred_val))
    
#    print('MCC score train for k=%d: %.5f'%(i//10, mcc_train[i-10]))
#    print('MCC score val for k=%d: %.5f'%(i//10, mcc_val[i-10]))
    
    i += 1
    
acc_train = np.asarray([ np.mean(acc_train[i:i+10])
                        for i in range(0, 10*k_max, 10) ])

acc_val = np.asarray([ np.mean(acc_val[i:i+10])
                        for i in range(0, 10*k_max, 10) ])

mcc_train = np.asarray([ np.mean(mcc_train[i:i+10])
                        for i in range(0, 10*k_max, 10) ])

mcc_val = np.asarray([ np.mean(mcc_val[i:i+10])
                        for i in range(0, 10*k_max, 10) ])

f1_train = np.asarray([ np.mean(f1_train[i:i+10])     
                        for i in range(0, 10*k_max, 10) ])

f1_val = np.asarray([ np.mean(f1_val[i:i+10]) 
                    for i in range(0, 10*k_max, 10) ])

#%%
print('Range n_neighbors: from %d to %d'% (1, k_max))

print('accuracy train: %s \n'%acc_train)
print('accuracy validation: %s \n'%acc_val)

print('f1 score train: %s \n'%f1_train)
print('f1 score validation: %s \n'%f1_val)

print('MCC train: %s \n'%mcc_train)
print('MCC validation: %s \n'%mcc_val)    

#%% Create learning curves

x = range(1, k_max+1)

plt.figure(1)
plt.plot(x, acc_train, 'b', label='Train score')
plt.plot(x, acc_val, 'r', label='Validation score')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend(loc='best')
plt.xticks(np.arange(min(x), max(x), 2))
plt.grid()
plt.show()

plt.figure(2)
plt.plot(x, f1_train, 'b', label='Train score')
plt.plot(x, f1_val, 'r', label='Validation score')
plt.ylabel('F1 score')
plt.xlabel('n_neighbors')
plt.legend(loc='best')
plt.xticks(np.arange(min(x), max(x), 2))
plt.grid()
plt.show()

plt.figure(3)
plt.plot(x, mcc_train, 'b', label='Train score')
plt.plot(x, mcc_val, 'r', label='Validation score')
plt.ylabel('MCC score')
plt.xlabel('n_neighbors')
plt.legend(loc='best')
plt.xticks(np.arange(min(x), max(x), 2))
plt.grid()
plt.show()

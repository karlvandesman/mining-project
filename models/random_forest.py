#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 9 18:29:39 2019

@author: note
"""

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RepeatedStratifiedKFold

import matplotlib.pyplot as plt

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X = datasetTrain.values[:, 0:8]
y = datasetTrain.values[:, 8]

#%% Testing the max_depth parameter

dp_max = 20

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=dp_max,
                              random_state=seed)

rf_dp_acc_train = []
rf_dp_acc_val = []
rf_dp_mcc_train = []
rf_dp_mcc_val = []
rf_dp_f1_train = []
rf_dp_f1_val = []

i = 10

for train_index, val_index in rkf.split(X, y):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    rfc = RandomForestClassifier(criterion='entropy', max_depth=4+i//10, 
                                 n_estimators=10, random_state=seed)

    rfc.fit(X_train, y_train)
    y_pred_train = rfc.predict(X_train)
    y_pred_val = rfc.predict(X_val)
    
    rf_dp_acc_train.append(accuracy_score(y_train, y_pred_train))
    rf_dp_acc_val.append(accuracy_score(y_val, y_pred_val))

    rf_dp_f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
    rf_dp_f1_val.append(f1_score(y_val, y_pred_val, average='macro')) 
    
    rf_dp_mcc_train.append(matthews_corrcoef(y_train, y_pred_train))
    rf_dp_mcc_val.append(matthews_corrcoef(y_val, y_pred_val))
    
    print('MCC score train for max_depth=%d: %.5f'%(4+i//10, rf_dp_mcc_train[i-10]))
    print('MCC score val for max_depth=%d: %.5f'%(4+i//10, rf_dp_mcc_val[i-10]))
    
    i += 1

rf_dp_acc_train = np.asarray([ np.mean(rf_dp_acc_train[i:i+10])
                        for i in range(0, 10*dp_max, 10) ])

rf_dp_acc_val = np.asarray([ np.mean(rf_dp_acc_val[i:i+10])
                        for i in range(0, 10*dp_max, 10) ])

rf_dp_mcc_train = np.asarray([ np.mean(rf_dp_mcc_train[i:i+10])
                        for i in range(0, 10*dp_max, 10) ])

rf_dp_mcc_val = np.asarray([ np.mean(rf_dp_mcc_val[i:i+10])
                        for i in range(0, 10*dp_max, 10) ])

rf_dp_f1_train = np.asarray([ np.mean(rf_dp_f1_train[i:i+10])     
                        for i in range(0, 10*dp_max, 10) ])

rf_dp_f1_val = np.asarray([ np.mean(rf_dp_f1_val[i:i+10]) 
                    for i in range(0, 10*dp_max, 10) ])

#%%
print('accuracy train: %s \n'%rf_dp_acc_train)
print('accuracy validation: %s \n'%rf_dp_acc_val)

print('f1 score train: %s \n'%rf_dp_f1_train)
print('f1 score validation: %s \n'%rf_dp_f1_val)

print('MCC train: %s \n'%rf_dp_mcc_train)
print('MCC validation: %s \n'%rf_dp_mcc_val)

#%% Create learning curves

x = range(5, dp_max+5)

plt.figure(1)
plt.plot(x, rf_dp_acc_train, 'b', label='Train score')
plt.plot(x, rf_dp_acc_val, 'r', label='Validation score')
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.plot(x, rf_dp_f1_train, 'b', label='Train score')
plt.plot(x, rf_dp_f1_val, 'r', label='Validation score')
plt.ylabel('F1 score')
plt.xlabel('max_depth')
plt.legend(loc='best')
plt.show()

plt.figure(3)
plt.plot(x, rf_dp_mcc_train, 'b', label='Train score')
plt.plot(x, rf_dp_mcc_val, 'r', label='Validation score')
plt.ylabel('MCC score')
plt.xlabel('max_depth')
plt.legend(loc='best')
plt.show()

#%% Testing the number of estimators parameter

max_depth = 8
n_estimators_max = 30

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_estimators_max, 
                              random_state=seed)    

rf_est_acc_train = []
rf_est_acc_val = []
rf_est_mcc_train = []
rf_est_mcc_val = []
rf_est_f1_train = []
rf_est_f1_val = []

i = 10

for train_index, val_index in rkf.split(X, y):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    rfc = RandomForestClassifier(criterion='entropy', max_depth=max_depth, 
                                 n_estimators=i//10, random_state=seed)

    rfc.fit(X_train, y_train)
    y_pred_train = rfc.predict(X_train)
    y_pred_val = rfc.predict(X_val)
    
    rf_est_acc_train.append(accuracy_score(y_train, y_pred_train))
    rf_est_acc_val.append(accuracy_score(y_val, y_pred_val))

    rf_est_f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
    rf_est_f1_val.append(f1_score(y_val, y_pred_val, average='macro')) 
    
    rf_est_mcc_train.append(matthews_corrcoef(y_train, y_pred_train))
    rf_est_mcc_val.append(matthews_corrcoef(y_val, y_pred_val))
    
    print('MCC score train for n_estimators=%d: %.5f'%(i//10, rf_est_mcc_train[i-10]))
    print('MCC score val for n_estimators=%d: %.5f'%(i//10, rf_est_mcc_val[i-10]))
    
    i += 1

rf_est_acc_train = np.asarray([ np.mean(rf_est_acc_train[i:i+10])
                        for i in range(0, 10*n_estimators_max, 10) ])

rf_est_acc_val = np.asarray([ np.mean(rf_est_acc_val[i:i+10])
                        for i in range(0, 10*n_estimators_max, 10) ])

rf_est_mcc_train = np.asarray([ np.mean(rf_est_mcc_train[i:i+10])
                        for i in range(0, 10*n_estimators_max, 10) ])

rf_est_mcc_val = np.asarray([ np.mean(rf_est_mcc_val[i:i+10])
                        for i in range(0, 10*n_estimators_max, 10) ])

rf_est_f1_train = np.asarray([ np.mean(rf_est_f1_train[i:i+10])     
                        for i in range(0, 10*n_estimators_max, 10) ])

rf_est_f1_val = np.asarray([ np.mean(rf_est_f1_val[i:i+10]) 
                    for i in range(0, 10*n_estimators_max, 10) ])

#%% 

print('For max_depth=%d, range n_estimators from 1 to %d'%(max_depth, n_estimators_max))    

print('accuracy train: %s \n'%rf_est_acc_train)
print('accuracy validation: %s \n'%rf_est_acc_val)

print('f1 score train: %s \n'%rf_est_f1_train)
print('f1 score validation: %s \n'%rf_est_f1_val)

print('MCC train: %s \n'%rf_est_mcc_train)
print('MCC validation: %s \n'%rf_est_mcc_val)    

#%% Create learning curves

x = range(1, n_estimators_max+1)

plt.figure(1)
plt.title('Train and validation accuracy x n_estimators for max_depth=8')
plt.plot(x, rf_est_acc_train, 'b', label='Train score')
plt.plot(x, rf_est_acc_val, 'r', label='Validation score')
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.legend(loc='best')
plt.xticks(np.arange(min(x), max(x), 3))
plt.grid()
plt.show()

plt.figure(2)
plt.title('Train and validation f1 score x n_estimators for max_depth=8')
plt.plot(x, rf_est_f1_train, 'b', label='Train score')
plt.plot(x, rf_est_f1_val, 'r', label='Validation score')
plt.ylabel('F1 score')
plt.xlabel('n_estimators')
plt.legend(loc='best')
plt.xticks(np.arange(min(x), max(x), 3))
plt.grid()
plt.show()

plt.figure(3)
plt.title('Train and validation MCC x n_estimators for max_depth=8')
plt.plot(x, rf_est_mcc_train, 'b', label='Train score')
plt.plot(x, rf_est_mcc_val, 'r', label='Validation score')
plt.ylabel('MCC score')
plt.xlabel('n_estimators')
plt.legend(loc='best')
plt.xticks(np.arange(min(x), max(x), 3))
plt.grid()
plt.show()

#%% Using the best combination of parameters to predict

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.2, 
                                                  random_state=seed)

rfc = RandomForestClassifier(max_depth=12, n_estimators=20, random_state=seed)

rfc.fit(X_train, y_train)
y_pred_val = rfc.predict(X_val)

print("Clasification report:\n", classification_report(y_val, y_pred_val))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred_val))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")
kfold = StratifiedKFold(n_splits=10, random_state=seed)

X = datasetTrain.values[:, 0:8]
y = datasetTrain.values[:, 8]

dp_init = 4
dp_max = 28

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=dp_max-dp_init+1, 
                              random_state=seed)

dt_acc_train = []
dt_acc_val = []
dt_mcc_train = []
dt_mcc_val = []
dt_f1_train = []
dt_f1_val = []

i = 0

for train_index, val_index in rkf.split(X, y):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    clf = tree.DecisionTreeClassifier(max_depth=dp_init+i//10, random_state=seed)
    
    clf.fit(X_train, y_train)
        
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)

    dt_acc_train.append(accuracy_score(y_train, y_pred_train))
    dt_acc_val.append(accuracy_score(y_val, y_pred_val))

    dt_f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
    dt_f1_val.append(f1_score(y_val, y_pred_val, average='macro')) 
    
    dt_mcc_train.append(matthews_corrcoef(y_train, y_pred_train))
    dt_mcc_val.append(matthews_corrcoef(y_val, y_pred_val))
    
    #print('MCC score train for max depth=%d: %.5f'%(i//10, performance_train[i-10]))
    #print('MCC score val for max depth=%d: %.5f'%(i//10, performance_val[i-10]))
    
    i += 1

dt_acc_train = np.asarray([ np.mean(dt_acc_train[i:i+10])
                        for i in range(0, 10*(dp_max-dp_init+1), 10) ])

dt_acc_val = np.asarray([ np.mean(dt_acc_val[i:i+10])
                        for i in range(0, 10*(dp_max-dp_init+1), 10) ])

dt_f1_train = np.asarray([ np.mean(dt_f1_train[i:i+10])     
                        for i in range(0, 10*(dp_max-dp_init+1), 10) ])

dt_f1_val = np.asarray([ np.mean(dt_f1_val[i:i+10]) 
                    for i in range(0, 10*(dp_max-dp_init+1), 10) ])

dt_mcc_train = np.asarray([ np.mean(dt_mcc_train[i:i+10])
                        for i in range(0, 10*(dp_max-dp_init+1), 10) ])

dt_mcc_val = np.asarray([ np.mean(dt_mcc_val[i:i+10])
                        for i in range(0, 10*(dp_max-dp_init+1), 10) ])
    
#%%
print('Range max_depth: from %d to %d'% (dp_init, dp_max))

print('accuracy train: %s \n'%dt_acc_train)
print('accuracy validation: %s \n'%dt_acc_val)

print('f1 score train: %s \n'%dt_f1_train)
print('f1 score validation: %s \n'%dt_f1_val)

print('MCC train: %s \n'%dt_mcc_train)
print('MCC validation: %s \n'%dt_mcc_val)

#%% Create learning curves

x = range(dp_init, dp_max+1)

plt.figure(1)
plt.plot(x, dt_acc_train, 'b', label='Train score')
plt.plot(x, dt_acc_val, 'r', label='Validation score')
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
plt.legend(loc='best')
plt.xticks(np.arange(min(x), max(x), 2))
plt.grid()
plt.show()

plt.figure(2)
plt.plot(x, dt_f1_train, 'b', label='Train score')
plt.plot(x, dt_f1_val, 'r', label='Validation score')
plt.ylabel('F1 score')
plt.xlabel('max_depth')
plt.xticks(np.arange(min(x), max(x), 2))
plt.grid()
plt.legend(loc='best')
plt.show()

plt.figure(3)
plt.plot(x, dt_mcc_train, 'b', label='Train score')
plt.plot(x, dt_mcc_val, 'r', label='Validation score')
plt.ylabel('MCC score')
plt.xlabel('max_depth')
plt.xticks(np.arange(min(x), max(x), 2))
plt.grid()
plt.legend(loc='best')
plt.show()

#%%    

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.3, 
                                                  random_state=seed)


clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)

y_pred_val = clf.predict(X_val)

print("Classification report:\n", classification_report(y_val, y_pred_val))
print("Confussion matrix:\n", confusion_matrix(y_val, y_pred_val))

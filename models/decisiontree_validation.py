#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
from sklearn.model_selection import RepeatedStratifiedKFold

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")
kfold = StratifiedKFold(n_splits=10, random_state=seed)

X = datasetTrain.values[:, 0:8]
y = datasetTrain.values[:, 8]

dp_max = 20

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=dp_max, 
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
    
    clf = tree.DecisionTreeClassifier(max_depth=i//10)
    
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)

    acc_train.append(accuracy_score(y_train, y_pred_train))
    acc_val.append(accuracy_score(y_val, y_pred_val))
    
    mcc_train.append(matthews_corrcoef(y_train, y_pred_train))
    mcc_val.append(matthews_corrcoef(y_val, y_pred_val))
    
    f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
    f1_val.append(f1_score(y_val, y_pred_val, average='macro')) 
    
    #print('MCC score train for max depth=%d: %.5f'%(i//10, performance_train[i-10]))
    #print('MCC score val for max depth=%d: %.5f'%(i//10, performance_val[i-10]))
    
    i += 1

acc_train = np.asarray([ np.mean(acc_train[i:i+10])
                        for i in range(0, 10*dp_max, 10) ])

acc_val = np.asarray([ np.mean(acc_val[i:i+10])
                        for i in range(0, 10*dp_max, 10) ])

mcc_train = np.asarray([ np.mean(mcc_train[i:i+10])
                        for i in range(0, 10*dp_max, 10) ])

mcc_val = np.asarray([ np.mean(mcc_val[i:i+10])
                        for i in range(0, 10*dp_max, 10) ])

f1_train = np.asarray([ np.mean(f1_train[i:i+10])     
                        for i in range(0, 10*dp_max, 10) ])

f1_val = np.asarray([ np.mean(f1_val[i:i+10]) 
                    for i in range(0, 10*dp_max, 10) ])

#%%

print('accuracy train: %s \n'%acc_train)
print('accuracy validation: %s \n'%acc_val)

print('f1 score train: %s \n'%f1_train)
print('f1 score validation: %s \n'%f1_val)

print('MCC train: %s \n'%mcc_train)
print('MCC validation: %s \n'%mcc_val)

#%%
    
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(range(1, dp_max+1), acc_train, 'b', label='Train score')
line2, = plt.plot(range(1, dp_max+1), acc_val, 'r', label='Validation score')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
plt.xticks(np.arange(1, dp_max+1, step=2))
plt.show()

#%%    

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.3, 
                                                  random_state=seed)


clf = tree.DecisionTreeClassifier(max_depth=1)

clf = clf.fit(X_train, y_train)

y_pred_val = clf.predict(X_val)

print("Classification report:\n", classification_report(y_val, y_pred_val))
print("Confussion matrix:\n", confusion_matrix(y_val, y_pred_val))

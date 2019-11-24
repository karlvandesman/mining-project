#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:29:39 2019

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
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RepeatedStratifiedKFold

import matplotlib.pyplot as plt

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X = datasetTrain.values[:, 0:8]
y = datasetTrain.values[:, 8]

dp_max = 20

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=dp_max,
                              random_state=seed)

#%% Testing the max_depth parameter

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

    rfc = RandomForestClassifier(criterion='entropy', max_depth=4+i//10, 
                                 n_estimators=10, random_state=seed)

    rfc.fit(X_train, y_train)
    y_pred_train = rfc.predict(X_train)
    y_pred_val = rfc.predict(X_val)
    
    acc_train.append(accuracy_score(y_train, y_pred_train))
    acc_val.append(accuracy_score(y_val, y_pred_val))

    f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
    f1_val.append(f1_score(y_val, y_pred_val, average='macro')) 
    
    mcc_train.append(matthews_corrcoef(y_train, y_pred_train))
    mcc_val.append(matthews_corrcoef(y_val, y_pred_val))
    
    print('MCC score train for max_depth=%d: %.5f'%(4+i//10, mcc_train[i-10]))
    print('MCC score val for max_depth=%d: %.5f'%(4+i//10, mcc_val[i-10]))
    
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

#%% Create learning curves

x = range(5, dp_max+5)

plt.figure(1)
plt.plot(x, acc_train, 'b', label='Train score')
plt.plot(x, acc_val, 'r', label='Validation score')
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.plot(x, f1_train, 'b', label='Train score')
plt.plot(x, f1_val, 'r', label='Validation score')
plt.ylabel('F1 score')
plt.xlabel('max_depth')
plt.legend(loc='best')
plt.show()

plt.figure(3)
plt.plot(x, mcc_train, 'b', label='Train score')
plt.plot(x, mcc_val, 'r', label='Validation score')
plt.ylabel('MCC score')
plt.xlabel('max_depth')
plt.legend(loc='best')
plt.show()

#%% Testing the number of estimators parameter

n_estimators_max = 35

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_estimators_max, 
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

    rfc = RandomForestClassifier(criterion='entropy', max_depth=12, 
                                 n_estimators=i//10, random_state=seed)

    rfc.fit(X_train, y_train)
    y_pred_train = rfc.predict(X_train)
    y_pred_val = rfc.predict(X_val)
    
    acc_train.append(accuracy_score(y_train, y_pred_train))
    acc_val.append(accuracy_score(y_val, y_pred_val))

    f1_train.append(f1_score(y_train, y_pred_train, average='macro'))
    f1_val.append(f1_score(y_val, y_pred_val, average='macro')) 
    
    mcc_train.append(matthews_corrcoef(y_train, y_pred_train))
    mcc_val.append(matthews_corrcoef(y_val, y_pred_val))
    
    print('MCC score train for n_estimators=%d: %.5f'%(i//10, mcc_train[i-10]))
    print('MCC score val for n_estimators=%d: %.5f'%(i//10, mcc_val[i-10]))
    
    i += 1

acc_train = np.asarray([ np.mean(acc_train[i:i+10])
                        for i in range(0, 10*n_estimators_max, 10) ])

acc_val = np.asarray([ np.mean(acc_val[i:i+10])
                        for i in range(0, 10*n_estimators_max, 10) ])

mcc_train = np.asarray([ np.mean(mcc_train[i:i+10])
                        for i in range(0, 10*n_estimators_max, 10) ])

mcc_val = np.asarray([ np.mean(mcc_val[i:i+10])
                        for i in range(0, 10*n_estimators_max, 10) ])

f1_train = np.asarray([ np.mean(f1_train[i:i+10])     
                        for i in range(0, 10*n_estimators_max, 10) ])

f1_val = np.asarray([ np.mean(f1_val[i:i+10]) 
                    for i in range(0, 10*n_estimators_max, 10) ])

#%% 
    
print('accuracy train: %s \n'%acc_train)
print('accuracy validation: %s \n'%acc_val)

print('f1 score train: %s \n'%f1_train)
print('f1 score validation: %s \n'%f1_val)

print('MCC train: %s \n'%mcc_train)
print('MCC validation: %s \n'%mcc_val)    

#%% Create learning curves

x = range(4, n_estimators_max+4)

plt.figure(1)
plt.plot(x, acc_train, 'b', label='Train score')
plt.plot(x, acc_val, 'r', label='Validation score')
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.plot(x, f1_train, 'b', label='Train score')
plt.plot(x, f1_val, 'r', label='Validation score')
plt.ylabel('F1 score')
plt.xlabel('n_estimators')
plt.legend(loc='best')
plt.show()

plt.figure(3)
plt.plot(x, mcc_train, 'b', label='Train score')
plt.plot(x, mcc_val, 'r', label='Validation score')
plt.ylabel('MCC score')
plt.xlabel('n_estimators')
plt.legend(loc='best')
plt.show()


#%%
for i, k in enumerate(max_d):
    print(" ----------> max_depth =", k)
    rfc = RandomForestClassifier(criterion='entropy', max_depth=k, 
                                 n_estimators=15, random_state=seed)
    rfc = rfc.fit(X_train, Y_train)
    
    Y_pred_train = rfc.predict(X_train)
    
    mcc_train = matthews_corrcoef(Y_train, Y_pred_train)
    mcc_train.append(mcc_train)
    print("MCC train: %0.3f" %  mcc_train[i])
    
    resultsRFC = model_selection.cross_val_score(rfc, X_train, Y_train, 
                                                 cv=kfold, scoring=\
                                                 make_scorer(matthews_corrcoef))

    print('MCC k-fold mean:', resultsRFC.mean())
    
    mcc_val.append(resultsRFC.mean())

#%%
    
line1, = plt.plot(max_d, mcc_train, 'b', label='Train score')
line2, = plt.plot(max_d, mcc_val, 'r', label='Validation score')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('MCC score')
plt.xlabel('max_depth')
plt.show()

#%% Using the best combination of parameters to predict

X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, 
                                                  test_size=0.2, 
                                                  random_state=seed)

rfc = RandomForestClassifier(max_depth=11, n_estimators=21, random_state=seed)

rfc.fit(X_train, y_train)
y_pred_val = rfc.predict(X_val)

print("Clasification report:\n", classification_report(y_val, y_pred_val))
print("Confusion matrix:\n", confusion_matrix(y_val, y_pred_val))
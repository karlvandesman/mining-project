#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn

from sklearn.model_selection import RepeatedStratifiedKFold

seed = 10

datasetTrain = pd.read_csv("Dataset_processado/dataset_treino_processado.csv")
kfold = StratifiedKFold(n_splits=10, random_state=seed)

X = datasetTrain.values[:, 0:8]
y = datasetTrain.values[:, 8]

k_max = 15

rkf = RepeatedStratifiedKFold(n_splits=10, n_repeats=k_max, 
                              random_state=seed)

performance_train = []
performance_val = []
i = 10

for train_index, val_index in rkf.split(X, y):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    knn = KNeighborsClassifier(n_neighbors=i//10)
    
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_val = knn.predict(X_val)
    
    mcc_train = matthews_corrcoef(y_train, y_pred_train)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    
    performance_train.append(mcc_train)
    performance_val.append(mcc_val)
    
    print('MCC score train for k=%d: %.5f'%(i//10, performance_train[i-10]))
    print('MCC score val for k=%d: %.5f'%(i//10, performance_val[i-10]))
    
    i += 1
    
performance_train = np.asarray([ np.mean(performance_train[i:i+10]) 
                                    for i in range(0, 10*k_max, 10) ])

performance_val = np.asarray([ np.mean(performance_val[i:i+10]) 
                                for i in range(0, 10*k_max, 10) ])

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(range(1, k_max+1), performance_train, 'b', label='Train score')
line2, = plt.plot(range(1, k_max+1), performance_val, 'r', label='Validation score')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('MCC score')
plt.xlabel('n_neighbors')
plt.show()

#%%    

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.3, 
                                                  random_state=seed)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred_val = knn.predict(X_val)

print("Classification report:\n", classification_report(y_val, y_pred_val))
print("Confussion matrix:\n", confusion_matrix(y_val, y_pred_val))
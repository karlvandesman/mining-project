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
from sklearn.metrics import matthews_corrcoef
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X_train = datasetTrain.values[:, 0:8]
Y_train = datasetTrain.values[:, 8]

#%% Testing the max_depth parameter

max_d = np.arange(1, 25)
rfc_mcc_train = []
rfc_mcc_val = []

for i, k in enumerate(max_d):
    print(" ----------> max_depth =", k)
    rfc = RandomForestClassifier(criterion='entropy', max_depth=k, 
                                 n_estimators=21, random_state=seed)
    rfc = rfc.fit(X_train, Y_train)
    
    Y_pred_train = rfc.predict(X_train)
    
    mcc_train = matthews_corrcoef(Y_train, Y_pred_train)
    rfc_mcc_train.append(mcc_train)
    print("MCC train: %0.3f" %  rfc_mcc_train[i])
    
    resultsRFC = model_selection.cross_val_score(rfc, X_train, Y_train, 
                                                 cv=kfold, scoring=\
                                                 make_scorer(matthews_corrcoef))

    print('MCC k-fold mean:', resultsRFC.mean())
    
    rfc_mcc_val.append(resultsRFC.mean())

line1, = plt.plot(max_d, rfc_mcc_train, 'b', label='Train score')
line2, = plt.plot(max_d, rfc_mcc_val, 'r', label='Validation score')

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
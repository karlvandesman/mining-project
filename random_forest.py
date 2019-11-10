#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:29:39 2019

@author: note
"""

import pandas
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


seed = 10

datasetTrain = pandas.read_csv("Dataset_processado/dataset_treino_processado.csv")
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X_train = datasetTrain.values[:, 0:8]
Y_train = datasetTrain.values[:, 8]

max_d = np.arange(1, 11)
for i, k in enumerate(max_d):
    print(" ----------> max_depth =", k)
    rfc = RandomForestClassifier(criterion='entropy', max_depth=k, n_estimators=21, random_state=seed)
    rfc = rfc.fit(X_train, Y_train)
    print("Acuracia: %0.3f" %  rfc.score(X_train, Y_train))
    
    resultsRFC = model_selection.cross_val_score(rfc, X_train, Y_train, cv=kfold)
    print(resultsRFC)
    print(resultsRFC.mean())
    Y_prediction = rfc.predict(X_train)
    print("Clasification report:\n", classification_report(Y_train, Y_prediction))
    print("Confussion matrix:\n", confusion_matrix(Y_train, Y_prediction))
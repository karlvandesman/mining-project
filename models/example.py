#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 2019

@author: karlvandesman
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

import numpy as np
import matplotlib.pyplot as plt

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")

X_train = datasetTrain.values[:, 0:8]
y_train = datasetTrain.values[:, 8]

seed = 10
n_folds = 10
n_repeats = 30

mcc_cv = []
f1_cv = []

kfold = StratifiedKFold(n_splits=n_folds, random_state=seed)

for i in range(n_repeats):
    # Cria classificador
    clf = tree.DecisionTreeClassifier(max_depth=i+2)
    
    # Separa os dados pelos k-folds
    
    # Calcula métricas de desempenho para os k-folds
    mcc_cv.append(np.mean(cross_val_score(clf, X_train, y_train, cv=kfold, 
                          scoring=make_scorer(matthews_corrcoef))))
    
    f1_cv.append(np.mean(cross_val_score(clf, X_train, y_train, cv=kfold, 
                          scoring=make_scorer(f1_score, average='macro'))))

print('MCC mean scores for k-fold CV:', mcc_cv)
print('f1 mean scores for k-fold CV:', f1_cv)

plt.plot(np.arange(n_repeats)+2, mcc_cv)
plt.ylabel('MCC mean score for 10-fold')
plt.xlabel('Max_depth')

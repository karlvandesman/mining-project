#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karlvandesman
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer

seed=10

datasetTrain = pd.read_csv("Dataset_processado/dataset_treino_processado.csv")
kfold = StratifiedKFold(n_splits=10, random_state=seed)

X_train = datasetTrain.values[:, 0:8]
Y_train = datasetTrain.values[:, 8]

# Create classifiers
dtc = DecisionTreeClassifier(max_depth=11, random_state=seed)
knn = KNeighborsClassifier(n_neighbors=2)
mlp = MLPClassifier(solver='lbfgs', activation='tanh', alpha=1e-5, 
                    hidden_layer_sizes=(20, 20), random_state=seed)

#Fitting single models
results = cross_val_score(dtc, X_train, Y_train, cv=kfold, 
                          scoring=make_scorer(matthews_corrcoef))
print('Decision Tree: ', results.mean())

results = cross_val_score(knn, X_train, Y_train, cv=kfold,
                          scoring=make_scorer(matthews_corrcoef))
print('KNN: ', results.mean())

results = cross_val_score(mlp, X_train, Y_train, cv=kfold,
                          scoring=make_scorer(matthews_corrcoef))
print('MLP: ', results.mean())

estimators = [('dtc', dtc), ('knn', knn), ('mlp', mlp)]

#%%

ensemble = VotingClassifier(estimators, voting='soft')
scores = cross_val_score(ensemble, X_train, Y_train, cv=kfold,
                          scoring=make_scorer(matthews_corrcoef))

print(scores.mean())
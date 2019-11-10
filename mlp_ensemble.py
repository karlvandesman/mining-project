#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:53:37 2019

@author: note
"""

import pandas
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier

seed = 10

datasetTrain = pandas.read_csv("Dataset_processado/dataset_treino_processado.csv")
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X_train = datasetTrain.values[:, 0:8]
Y_train = datasetTrain.values[:, 8]

mlp = MLPClassifier(
        hidden_layer_sizes=(20, 20),
        activation="tanh",
        random_state=seed,
        max_iter=400,
        solver="sgd",
        )

baggingMlp = BaggingClassifier(mlp, oob_score=True, max_samples=0.8, n_estimators=21, random_state=seed)
baggingMlp = baggingMlp.fit(X_train,Y_train)
Y_prediction = baggingMlp.predict(X_train)
print("Acuracia de treinamento: %0.3f" %  baggingMlp.score(X_train, Y_train))

resultsMLPE = model_selection.cross_val_score(baggingMlp, X_train, Y_train, cv=kfold)
print(resultsMLPE)
print(resultsMLPE.mean())
print("Clasification report:\n", classification_report(Y_train, Y_prediction))
print("Confussion matrix:\n", confusion_matrix(Y_train, Y_prediction))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:24:49 2019

@author: note
"""
import pandas
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

seed = 10

datasetTrain = pandas.read_csv("Dataset_processado/dataset_treino_processado.csv")
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X_train = datasetTrain.values[:, 0:8]
Y_train = datasetTrain.values[:, 8]

mlp = MLPClassifier(
        hidden_layer_sizes=(10,10),
        activation="tanh",
        random_state=seed,
        max_iter=400,
        solver="lbfgs",
        )

mlp = mlp.fit(X_train,Y_train)
Y_prediction = mlp.predict(X_train)
print("Acurácia de treinamento: %0.3f" %  mlp.score(X_train, Y_train))

resultsDTC = model_selection.cross_val_score(mlp, X_train, Y_train, cv=kfold)
print(resultsDTC)
print(resultsDTC.mean())
print("Clasification report:\n", classification_report(Y_train, Y_prediction))
print("Confussion matrix:\n", confusion_matrix(Y_train, Y_prediction))

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
datasetTest = pandas.read_csv("Dataset_processado/dataset_teste_processado.csv")
fullDataset = datasetTest.append(datasetTrain)

kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X_train = datasetTrain.values[:, 0:8]
Y_train = datasetTrain.values[:, 8]

X_test = datasetTest.values[:, 0:8]
Y_test = datasetTest.values[:, 8]

X = fullDataset.values[:, 0:8]
Y = fullDataset.values[:, 8]

mlp = MLPClassifier(
        hidden_layer_sizes=(10,10),
        activation="tanh",
        random_state=seed,
        max_iter=400,
        solver="lbfgs",
        )

mlp = mlp.fit(X_train,Y_train)
Y_test_prediction = mlp.predict(X_test)
print("Acurácia de treinamento: %0.3f" %  mlp.score(X_train, Y_train))
print("Acurácia de teste: %0.3f" %  mlp.score(X_test, Y_test))

resultsDTC = model_selection.cross_val_score(mlp, X, Y, cv=kfold)
print(resultsDTC)
print(resultsDTC.mean())
print("Clasification report:\n", classification_report(Y_test, Y_test_prediction))
print("Confussion matrix:\n", confusion_matrix(Y_test, Y_test_prediction))

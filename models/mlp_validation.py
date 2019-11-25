#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")

X = datasetTrain.values[:, 0:8]
y = datasetTrain.values[:, 8]

hl_max = 12
fold = 5

rkf = RepeatedStratifiedKFold(n_splits=fold, n_repeats=hl_max, random_state=seed)

performance_train = []
performance_val = []
i = fold

#parameter_space = {
#    'hidden_layer_sizes': [(2, 2), (4, 4), (8, 4)],
#    'activation': ['tanh', 'relu'],
#    'solver': ['sgd', 'adam'],
#    'alpha': [0.01, 0.1],
#    'learning_rate': ['constant','adaptive'],
#}

for train_index, val_index in rkf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(i//fold, 2),
        activation="tanh",
        random_state=seed,
        max_iter=400,
        solver="sgd",
        )
    
    mlp.fit(X_train, y_train)
    y_pred_train = mlp.predict(X_train)
    y_pred_val = mlp.predict(X_val)
    
    mcc_train = matthews_corrcoef(y_train, y_pred_train)
    mcc_val = matthews_corrcoef(y_val, y_pred_val)
    
    performance_train.append(mcc_train)
    performance_val.append(mcc_val)
    
    print('MCC score train for hidden layer sizes=%d: %.5f'%(i//fold, performance_train[i-fold]))
    print('MCC score val for hidden layer sizes=%d: %.5f'%(i//fold, performance_val[i-fold]))
    
    i += 1

#%%

mcc_train_mean = np.asarray([ np.mean(performance_train[i:i+fold])     
                                    for i in range(0, fold*hl_max, fold) ])

mcc_val_mean = np.asarray([ np.mean(performance_val[i:i+fold]) 
                                for i in range(0, fold*hl_max, fold) ])

#%%
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(range(2, hl_max+1), mcc_train_mean[1:], 'b', label='Train score')
line2, = plt.plot(range(2, hl_max+1), mcc_val_mean[1:], 'r', label='Validation score')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('MCC score')
plt.xlabel('hidden_layers_size')
plt.show()

#%%    

X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                  test_size=0.3, 
                                                  random_state=seed)

mlp = MLPClassifier(
        hidden_layer_sizes=(1,20),
        activation="tanh",
        random_state=seed,
        max_iter=400,
        solver="sgd",
        )

mlp = mlp.fit(X_train, y_train)

y_pred_val = mlp.predict(X_val)

print("Classification report:\n", classification_report(y_val, y_pred_val))
print("Confussion matrix:\n", confusion_matrix(y_val, y_pred_val))

#%%
# Para cv kfold realizado com algumas configurações de hiperparâmetros do MLP,
# foram obtidos os seguintes valores de acurácia:

acc_mlp = np.array([ 0.902432021343565, 0.9258210522381619, 0.9258210522381619,
                0.9257830221916785, 0.9318491478591031, 0.9738730228579309,
                0.9806232834413395, 0.981093893012223, 0.9486395891789468,
                0.9792923005878288, 0.9816310724445086, 0.9832474759156822])

x = np.arange(1, len(acc_mlp)+1)

plt.xlabel('Configurações de hiperparâmetros')
plt.ylabel('Acurácia')
plt.plot(x, acc_mlp)
plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.grid()
plt.show()
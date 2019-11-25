#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:50:35 2019

@author: karlvandesman
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt

from scipy.stats import kruskal

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")

X = datasetTrain.values[:, 0:8]
y = datasetTrain.values[:, 8]

dt_acc_val = []
dt_f1_val = []
dt_mcc_val = []

knn_acc_val = []
knn_f1_val = []
knn_mcc_val = []

mlp_acc_val = []
mlp_f1_val = []
mlp_mcc_val = []

rfc_acc_val = []
rfc_f1_val = []
rfc_mcc_val = []

mlpE_acc_val = []
mlpE_f1_val = []
mlpE_mcc_val = []

hetEns_acc_val = []
hetEns_f1_val = []
hetEns_mcc_val = []

fold = 10
n_repeats = 1

rkf = RepeatedStratifiedKFold(n_splits=fold, n_repeats=n_repeats, random_state=seed)

#%%

for train_index, val_index in rkf.split(X, y):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    ### Decision Tree
    print('Running DT')
    dt = tree.DecisionTreeClassifier(max_depth=12, random_state=seed)
    
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_val)

    dt_acc_val.append(accuracy_score(y_val, y_pred_dt))
    dt_f1_val.append(f1_score(y_val, y_pred_dt, average='macro'))     
    dt_mcc_val.append(matthews_corrcoef(y_val, y_pred_dt))
    
    ### K-Nearest Neighbors
    print('Running knn')
    knn = KNeighborsClassifier(n_neighbors=2)
    
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_val)
   
    knn_acc_val.append(accuracy_score(y_val, y_pred_knn))
    knn_f1_val.append(accuracy_score(y_val, y_pred_knn))
    knn_mcc_val.append(accuracy_score(y_val, y_pred_knn))
    
    ### MLP
    print('Running MLP')
    mlp = MLPClassifier(hidden_layer_sizes=(20, 20), activation="tanh", 
                        solver="lbfgs", max_iter=200, random_state=seed)
    
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_val)
    
    mlp_acc_val.append(accuracy_score(y_val, y_pred_mlp))
    mlp_f1_val.append(f1_score(y_val, y_pred_mlp, average='macro'))     
    mlp_mcc_val.append(matthews_corrcoef(y_val, y_pred_mlp))
    
    ### Random Forest
    print('Running Random Forest')

    rfc = RandomForestClassifier(criterion='entropy', max_depth=12,
                                 n_estimators=20, random_state=seed)
    
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_val)
    
    rfc_acc_val.append(accuracy_score(y_val, y_pred_rfc))
    rfc_f1_val.append(f1_score(y_val, y_pred_rfc, average='macro'))     
    rfc_mcc_val.append(matthews_corrcoef(y_val, y_pred_rfc))
    
    ### MLP Ensemble
    print('Running MLP Ensemble')

    baggingMLP = BaggingClassifier(mlp, oob_score=True, max_samples=0.8, 
                                   n_estimators=5, random_state=seed)
    
    baggingMLP = baggingMLP.fit(X_train, y_train)
    
    y_pred_mlpe = baggingMLP.predict(X_val)
    
    mlpE_acc_val.append(accuracy_score(y_val, y_pred_mlpe))
    mlpE_f1_val.append(f1_score(y_val, y_pred_mlpe, average='macro'))     
    mlpE_mcc_val.append(matthews_corrcoef(y_val, y_pred_mlpe))

    ### Heterogeneous Ensemble
    print('Running Heterogeneous Ensemble')
    
    estimators = [('dt', dt), ('knn', knn), ('mlp', mlp)]
    hetEns = VotingClassifier(estimators, voting='soft')
    
    hetEns.fit(X_train, y_train)
    
    y_pred_hetEns = hetEns.predict(X_val)

    hetEns_acc_val.append(accuracy_score(y_val, y_pred_hetEns))
    hetEns_f1_val.append(f1_score(y_val, y_pred_hetEns, average='macro'))     
    hetEns_mcc_val.append(matthews_corrcoef(y_val, y_pred_hetEns))
    
    print('Round finished')

#%%

clf_acc = np.array([np.asarray(dt_acc_val), np.asarray(knn_acc_val), np.asarray(mlp_acc_val), np.asarray(rfc_acc_val), 
                 np.asarray(mlpE_acc_val), np.asarray(hetEns_acc_val)])

clf_f1 = np.array([dt_f1_val, knn_f1_val, mlp_f1_val, rfc_f1_val, 
                 mlpE_f1_val, hetEns_f1_val])

clf_mcc = np.array([dt_mcc_val, knn_mcc_val, mlp_mcc_val, rfc_mcc_val, 
                 mlpE_mcc_val, hetEns_mcc_val])

plt.figure(1)
labels = ['DecTree','KNN', 'MLP', 'RandForest', 'MLPEns', 'HetEns']
plt.boxplot(clf_acc, patch_artist=True, labels=labels)
plt.ylabel('Acurácia')
plt.show()    

plt.figure(2)
labels = ['DecTree','KNN', 'MLP', 'RandForest', 'MLPEns', 'HetEns']
plt.boxplot(clf_f1, patch_artist=True, labels=labels)
plt.ylabel('F1 score')
plt.show()    

plt.figure(3)
labels = ['DecTree','KNN', 'MLP', 'RandForest', 'MLPEns', 'HetEns']
plt.ylabel('MCC score')
plt.boxplot(clf_mcc, patch_artist=True, labels=labels)
plt.show()

#%%
clf_acc_mean = clf_acc.mean(axis=0)


#%%
stat_acc, p_acc = kruskal(dt_acc_val, knn_acc_val, mlp_acc_val, rfc_acc_val, 
                  mlpE_acc_val, hetEns_acc_val)

stat_f1, p_f1 = kruskal(dt_f1_val, knn_f1_val, mlp_f1_val, rfc_f1_val, 
                  mlpE_f1_val, hetEns_f1_val)

stat_mcc, p_mcc = kruskal(dt_mcc_val, knn_mcc_val, mlp_mcc_val, rfc_mcc_val, 
                  mlpE_mcc_val, hetEns_mcc_val)

print('Kruskal-Wallis Statistics=%.3f, p=%.10f' % (stat_acc, p_acc))
print('Kruskal-Wallis Statistics=%.3f, p=%.10f' % (stat_f1, p_f1))
print('Kruskal-Wallis Statistics=%.3f, p=%.10f' % (stat_mcc, p_mcc))

# interpret
alpha = 0.05
if p_acc > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')


#%% Calculating the mean values for the k-fold
    
### Decision Tree
dt_acc_val = np.asarray([ np.mean(dt_acc_val[i:i+fold])
                        for i in range(0, fold*n_repeats, fold) ])

dt_f1_val = np.asarray([ np.mean(dt_f1_val[i:i+fold])     
                        for i in range(0, fold*n_repeats, fold) ])

dt_mcc_val = np.asarray([ np.mean(dt_mcc_val[i:i+fold]) 
                        for i in range(0, fold*n_repeats, fold) ])

### K-Nearest Neighbors
knn_acc_val = np.asarray([ np.mean(knn_acc_val[i:i+fold])
                        for i in range(0, fold*n_repeats, fold) ])

knn_f1_val = np.asarray([ np.mean(knn_f1_val[i:i+fold])     
                        for i in range(0, fold*n_repeats, fold) ])

knn_mcc_val = np.asarray([ np.mean(knn_mcc_val[i:i+fold]) 
                        for i in range(0, fold*n_repeats, fold) ])

### MLP
mlp_acc_val = np.asarray([ np.mean(mlp_acc_val[i:i+fold])
                        for i in range(0, fold*n_repeats, fold) ])

mlp_f1_val = np.asarray([ np.mean(mlp_f1_val[i:i+fold])     
                        for i in range(0, fold*n_repeats, fold) ])

mlp_mcc_val = np.asarray([ np.mean(mlp_mcc_val[i:i+fold]) 
                        for i in range(0, fold*n_repeats, fold) ])

### Random Forest
rfc_acc_val = np.asarray([ np.mean(rfc_acc_val[i:i+fold])
                        for i in range(0, fold*n_repeats, fold) ])

rfc_f1_val = np.asarray([ np.mean(rfc_f1_val[i:i+fold])     
                        for i in range(0, fold*n_repeats, fold) ])

rfc_mcc_val = np.asarray([ np.mean(rfc_mcc_val[i:i+fold]) 
                        for i in range(0, fold*n_repeats, fold) ])

### MLP Ensemble
mlpE_acc_val = np.asarray([ np.mean(mlpE_acc_val[i:i+fold])
                        for i in range(0, fold*n_repeats, fold) ])

mlpE_f1_val = np.asarray([ np.mean(mlpE_f1_val[i:i+fold])     
                        for i in range(0, fold*n_repeats, fold) ])

mlpE_mcc_val = np.asarray([ np.mean(mlpE_mcc_val[i:i+fold]) 
                        for i in range(0, fold*n_repeats, fold) ])

### Heterogeneous Ensemble
hetEns_acc_val = np.asarray([ np.mean(hetEns_acc_val[i:i+fold])
                        for i in range(0, fold*n_repeats, fold) ])

hetEns_f1_val = np.asarray([ np.mean(hetEns_f1_val[i:i+fold])
                        for i in range(0, fold*n_repeats, fold) ])

hetEns_mcc_val = np.asarray([ np.mean(hetEns_mcc_val[i:i+fold])
                        for i in range(0, fold*n_repeats, fold) ])


    
#%%
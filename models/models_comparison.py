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

from scikit_posthocs import posthoc_nemenyi
from scipy.stats import kruskal

from sklearn.metrics import confusion_matrix, classification_report

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

fold = 20
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

clf_acc = [dt_acc_val, knn_acc_val, mlp_acc_val, rfc_acc_val, 
           mlpE_acc_val, hetEns_acc_val]
    
clf_f1 = [dt_f1_val, knn_f1_val, mlp_f1_val, rfc_f1_val, 
          mlpE_f1_val, hetEns_f1_val]

clf_mcc = [dt_mcc_val, knn_mcc_val, mlp_mcc_val, rfc_mcc_val, 
           mlpE_mcc_val, hetEns_mcc_val]

#%% Mean and std (for 1 repeated k-fold)

x = ['DecTree','KNN', 'MLP', 'RandForest', 'MLPEns', 'HetEns']

clf_acc_mean = np.mean(clf_acc, axis=1)
clf_f1_mean = np.mean(clf_f1, axis=1)
clf_mcc_mean = np.mean(clf_mcc, axis=1)

clf_acc_std = np.std(clf_acc, axis=1)
clf_f1_std = np.std(clf_f1, axis=1)
clf_mcc_std = np.std(clf_mcc, axis=1)

plt.figure(1)
plt.errorbar(x, clf_acc_mean, yerr=clf_acc_std, fmt='.b', ecolor='r',
             marker='o')
plt.grid()
plt.ylabel('Acurácia')
plt.show()

plt.figure(2)
plt.errorbar(x, clf_f1_mean, yerr=clf_f1_std, fmt='.b', ecolor='r',
             marker='o')
plt.grid()
plt.ylabel('F1 score')
plt.show()

plt.figure(3)
plt.errorbar(x, clf_mcc_mean, yerr=clf_mcc_std, fmt='.b', ecolor='r',
             marker='o')
plt.grid()
plt.ylabel('MCC')
plt.show()

#%% Boxplot

plt.figure(4)
labels = ['DecTree','KNN', 'MLP', 'RandForest', 'MLPEns', 'HetEns']
plt.boxplot(clf_acc, patch_artist=True, labels=labels)
plt.ylabel('Acurácia')
plt.show()    

plt.figure(5)
labels = ['DecTree','KNN', 'MLP', 'RandForest', 'MLPEns', 'HetEns']
plt.boxplot(clf_f1, patch_artist=True, labels=labels)
plt.ylabel('F1 score')
plt.show()    

plt.figure(6)
labels = ['DecTree','KNN', 'MLP', 'RandForest', 'MLPEns', 'HetEns']
plt.ylabel('MCC score')
plt.boxplot(clf_mcc, patch_artist=True, labels=labels)
plt.show()


#%%
stat_acc, p_acc = kruskal(dt_acc_val, knn_acc_val, mlp_acc_val, rfc_acc_val, 
                  mlpE_acc_val, hetEns_acc_val)

stat_f1, p_f1 = kruskal(dt_f1_val, knn_f1_val, mlp_f1_val, rfc_f1_val, 
                  mlpE_f1_val, hetEns_f1_val)

stat_mcc, p_mcc = kruskal(dt_mcc_val, knn_mcc_val, mlp_mcc_val, rfc_mcc_val, 
                  mlpE_mcc_val, hetEns_mcc_val)

print('Kruskal-Wallis test for cross validation with k=%d'%fold)
print()
print('Accuracy: statistics=%.3f, p=%.20f' % (stat_acc, p_acc))
print('F1 score: statistics=%.3f, p=%.20f' % (stat_f1, p_f1))
print('MCC: statistics=%.3f, p=%.20f' % (stat_mcc, p_mcc))
print()

# interpret
alpha = 0.05
if p_acc > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')
print()

posthoc_acc = posthoc_nemenyi(clf_acc)
posthoc_f1 = posthoc_nemenyi(clf_f1)
posthoc_mcc = posthoc_nemenyi(clf_mcc)

print('Posthoc Nemenyi for accuracy\n', posthoc_acc)
print()

print('Posthoc Nemenyi for F1 score\n', posthoc_f1)
print()

print('Posthoc Nemenyi for MCC\n', posthoc_mcc)
print()

#%% Application for the final test

datasetTest = pd.read_csv("../Dataset_processado/dataset_teste_processado.csv")

X_train = datasetTrain.values[:, 0:8]
y_train = datasetTrain.values[:, 8]

X_test = datasetTrain.values[:, 0:8]
y_test = datasetTrain.values[:, 8]

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

y_pred_test = knn.predict(X_test)

acc_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average='macro')
mcc_test = matthews_corrcoef(y_test, y_pred_test)

#%%
print('Final result for KNN with k=2 with test data set')

print("Classification report:\n", classification_report(y_test, y_pred_test))
print("Confussion matrix:\n", confusion_matrix(y_test, y_pred_test))
print()

print('Accuracy for test: %.5f%%'%(100*acc_test))
print('F1 score for test: %.5f%%'%(100*f1_test))
print('MCC score for test: %.5f'%(mcc_test))

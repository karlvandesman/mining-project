# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:55:44 2019

@author: eugeniap
"""
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree

seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")

kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X_train = datasetTrain.values[:, 0:8]
Y_train = datasetTrain.values[:, 8]


# Setup arrays to store train and test accuracies
#max_d = np.arange(1, 11)
#train_accuracy = np.empty(len(max_d))

# Loop over different values of max_depth
#for i, k in enumerate(max_d):
    #clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    #clf.fit(X_train, Y_train)
    
    #Compute accuracy on the training set
    #train_accuracy[i] = clf.score(X_train, Y_train)
   # print (train_accuracy)
   
#criando arvore
clf = tree.DecisionTreeClassifier(max_depth=12)

clf = clf.fit(X_train, Y_train)

print("Acuracia de treinamento clf: %0.3f" %  clf.score(X_train, Y_train))

print("Profundidade da arvore criada")
print(clf.tree_.max_depth)


Y_prediction = clf.predict(X_train)

#create a new tree model
clf_cv = tree.DecisionTreeClassifier(max_depth=10)

#train model  
cv_scores = cross_val_score(clf_cv, X_train, Y_train, cv=kfold)

#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


print("Classification report:\n", classification_report(Y_train, Y_prediction))
print("Confussion matrix:\n", confusion_matrix(Y_train, Y_prediction))
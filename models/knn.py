# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:55:44 2019

@author: eugeniap
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans
import seaborn


seed = 10

datasetTrain = pd.read_csv("../Dataset_processado/dataset_treino_processado.csv")
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

X_train = datasetTrain.values[:, 0:8]
Y_train = datasetTrain.values[:, 8]

#Nc = range(1, 10)
#kmeans = [KMeans(n_clusters=i) for i in Nc]
#kmeans
#score = [kmeans[i].fit(X_train).score(X_train) for i in range(len(kmeans))]
#score
#pl.plot(Nc,score)
#pl.xlabel('Number of Clusters')
#pl.ylabel('Score')
#pl.title('Elbow Curve')
#pl.show()

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 6, metric='euclidean')

# Fit the classifier to the data
knn.fit(X_train,Y_train)

print("Acuracia de treinamento: %0.3f" %  knn.score(X_train, Y_train))

Y_prediction = knn.predict(X_train)

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors = 6, metric='euclidean')

#train model  
cv_scores = model_selection.cross_val_score(knn_cv, X_train, Y_train, cv=kfold)

#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

#cm= confusion_matrix(Y_train, Y_prediction)
#seaborn.heatmap(cm)
#pl.show()

print("Classification report:\n", classification_report(Y_train, Y_prediction))
print("Confussion matrix:\n", confusion_matrix(Y_train, Y_prediction))
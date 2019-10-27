#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------
# Projeto Soluções de Mineração de Dados
# ----------------------------------------
# ********** Pré-processamento **********

# Base de dados: 
# Activity recognition with healthy older people using a batteryless
# wearable sensor Data Set

#%%********************************
# *** Importação de bibliotecas ***
# *********************************

import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#%%*******************************
# *** Leitura da base de dados ***
# ********************************

# Salvar todos os data paths dos arquivos
data_path_arquivos = sorted(glob.glob('S*_Dataset/d*'))

df = pd.DataFrame()

# Nome dos atributos dos arquivos da base de dados
colunas = ['tempo', 'frontal', 'vertical', 'lateral', 'antena', 'rssi', 
           'fase', 'frequencia', 'atividade']

for data_path in data_path_arquivos:
    pasta = data_path[0:11]         # Salva o nome da pasta, 'S1_Dataset/'
    nome_arquivo = data_path[11:]   # Salva o nome do arquivo, ex.: 'd1p01M'
    
    if nome_arquivo != 'README.txt':
        # Leitura do arquivo
        data = pd.read_csv(data_path, header=None, names=colunas)
        
        # Substituir diretamente os caracteres por valores numéricos, para
        # criação das colunas 'sala' e 'sexo':
        data['sala'] = (0, 1)[nome_arquivo.startswith('d2')] # S1: 0 / S2:1
        data['sexo'] = (0, 1)[nome_arquivo.endswith('F')] # 'M':0 / 'F':1
        
        # Juntando todos os arquivos lidos em um mesmo dataframe
        df = df.append(data, ignore_index=True)

# Reordenamento das colunas
df = df[['sala', 'sexo', 'tempo', 'frontal', 'vertical', 'lateral', 'antena', 
         'rssi', 'fase', 'frequencia', 'atividade']]

# Separação dos dados em atributos e classe
X = df.values[:, 0:-1]
y = df.values[:, -1]

#%% **************************************
# *** Separação dos dados treino/teste ***
# ****************************************

seed = 75128
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, 
                                                        random_state=seed)

# Fixando as dimensões das classes como (n_exemplos x 1)
y_treino = y_treino.reshape(-1, 1)
y_teste = y_teste.reshape(-1, 1)

#%% ****************************
# *** Normalização dos dados ***
# ******************************

MinMax = MinMaxScaler(feature_range=(0, 1))

# É importante que o fit seja realizado somente nos dados de treino
MinMax.fit(X_treino)

# Depois de determinados os parâmetros da normalização, aplicar nos
# dados de treino e teste
X_treino_escalonado = MinMax.transform(X_treino)
X_teste_escalonado = MinMax.transform(X_teste)

#%% **************************
# *** Seleção de Atributos ***
# ****************************

# Usar o RandomForest para ver importância de features na classificação

clf = RandomForestClassifier(n_estimators=1000, random_state=seed, n_jobs=-1)

# Treinando o classificador, com dados de treino
clf.fit(X_treino_escalonado, np.ravel(y_treino))

importancia_RF = []

# Lista com níveis de importância
for feature in zip(df.columns, clf.feature_importances_):
    plt.bar(feature[0], feature[1], color='blue', edgecolor='k')
    importancia_RF.append(feature[1])

plt.rcParams['figure.figsize'] = 12, 8

plt.bar(df.columns.values[0:-1], importancia_RF, color='b', edgecolor='k')
plt.title('Importância dos atributos usando Random Forest', size=16)
plt.xlabel('Atributos')
plt.ylabel('Nível de importância')
plt.show()

# Usar o teste chi2 (X²) e selecionar os K melhores atributos
teste_chi2 = SelectKBest(chi2, k=8).fit(X_treino_escalonado, y_treino)

plt.bar(df.columns.values[0:-1], teste_chi2.scores_, color='g', edgecolor='k')
plt.title('Teste de independência usando qui-quadrado', size=16)
plt.xlabel('Atributos')
plt.ylabel('Valor qui-quadrado')
plt.show()

# Os valores vão ser substituídos porque o teste_chi2 e o nível de importância
# com o Random Forest estavam de acordo com as 8 melhores features
X_treino_escalonado = teste_chi2.transform(X_treino_escalonado)
X_teste_escalonado = teste_chi2.transform(X_teste_escalonado)

#%% ************************
# *** Salvar os arquivos ***
# **************************

# Especificando os tipos de dados de cada coluna
tipos_dados = ['%d', '%d', '%10.9f', '%10.9f', '%10.9f', '%10.9f', '%10.9f', 
               '%10.9f', '%d']

# Passando o string de cabeçalho para ser lido da maneira correta, 
# com os atributos mais significantes
colunas_cabecalho = ','.join(df.columns[0:8]) + ','+ df.columns[-1]

# Salvar os dados de treino e teste em arquivos .csv
np.savetxt("Dataset_processado/dataset_treino_processado.csv", 
           np.hstack((X_treino_escalonado, y_treino)), comments='',
           fmt=tipos_dados, delimiter=",", header=colunas_cabecalho)
np.savetxt("Dataset_processado/dataset_teste_processado.csv", 
           np.hstack((X_teste_escalonado, y_teste)), comments='',
           fmt=tipos_dados, delimiter=",", header=colunas_cabecalho)

#%% ***************************************
# *** Análise de Componentes Principais ***
# *****************************************

pca = PCA().fit(X_treino_escalonado)
plt.plot(np.arange(1, 9), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de componentes')
plt.ylabel('Variância explicada cumulativa')
plt.title('Análise da variância explicada com PCA\n(normalização MinMax)', 
          size=16)
plt.show()
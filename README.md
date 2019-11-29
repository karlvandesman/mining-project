# Activity Recognition

## Grupo:
- Karl Sousa (kvms)
- Maria Eugênia (meps)
- Mateus Silva (mmps)

Esse projeto descreve o processo de mineração de dados da base *Activity recognition with healthy older people using a batteryless wearable sensor*, disponível no [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Activity+recognition+with+healthy+older+people+using+a+batteryless+wearable+sensor). A base é composta de oito atributos coletados por meio de um sensor passivo, que mede a aceleração em três eixos. A partir de antenas fixadas em uma sala, são captados esses
sinais, como apresentado na a seguir.

![image](https://user-images.githubusercontent.com/31048109/69872467-3876f000-1294-11ea-9b64-855630dc3f80.png)

Como parte final para o processo de predição da classe dos exemplos, são utilizados os seguintes algoritmos de classificação:
- Árvore de decisão
- k-Nearest Neighbors (kNN)
- Multilayer Perceptron (MLP)
- Random Forest (comitê de árvores de decisão)
- Comitê de MLP
- Comitê heterogêneo, formado por árvore de decisão, kNN e MLP

## Resultados
Após pré-processamento dos dados, ajuste de hiper-parâmetros e avaliação dos diferentes classificadores (com testes estatísticos de Kruskal-Wallis e Nemenyi), foram obtidas as seguintes métricas nos dados de teste:
- Acurácia: 99,47%
- F1 score: 97,51% 
- MCC: 0,98898

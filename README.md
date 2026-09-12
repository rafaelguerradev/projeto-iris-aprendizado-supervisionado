# Classificação do Dataset Iris 🌸

Este projeto é uma aplicação de Machine Learning para classificação multiclasse utilizando o clássico dataset Iris. O objetivo é classificar a espécie de uma flor (Setosa, Versicolor ou Virginica) com base nas medidas de suas sépalas e pétalas.
A seguir o pipeline de implementação do Machine Learning

## 1. Entendimento do Problema
O problema consiste em classificar 150 amostras de flores em três espécies distintas. Trata-se de uma tarefa de **Aprendizado Supervisionado** voltada para **Classificação Multiclasse Nominal** (as classes não possuem uma ordem de grandeza entre si).

## 2. Coleta dos Dados (ETL)
Os dados foram extraídos diretamente da fonte oficial, o [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris). 
* **Atributos numéricos:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
* **Alvo (Target):** `class` (Iris-setosa, Iris-versicolor, Iris-virginica).
* O dataset encontra-se perfeitamente balanceado, contendo exatamente 50 amostras de cada classe.

## 3. Visualização e Análise de Dados
Para entender o comportamento das *features*, foram realizadas análises gráficas:
* **Gráficos Univariados:** Boxplots, histogramas e gráficos de densidade para observar a distribuição individual de cada atributo.
* **Gráficos Multivariados:** Matrizes de dispersão (Scatter Matrix) e Mapa de Calor (Heatmap) da correlação de Pearson. Foi possível notar uma altíssima correlação positiva entre o tamanho e a largura da pétala (`0.96`).

## 4. Divisão do Dataset
Para garantir a confiabilidade do modelo e evitar *overfitting*, os dados foram divididos da seguinte forma:
* **Conjunto de Validação (Hold-out):** 20% dos dados (30 amostras) foram separados e mantidos intocáveis para o teste final.
* **Conjunto de Treinamento:** 80% dos dados (120 amostras) foram utilizados para treinamento. Devido ao tamanho reduzido do dataset, aplicou-se a técnica de **Validação Cruzada Estratificada (10 fatias)** durante a fase de treino para otimizar os resultados e garantir estabilidade.

## 5. Treinamento e Avaliação de Modelos
Foram testados seis algoritmos clássicos de classificação:
1. Regressão Logística (LR)
2. Análise Linear Discriminante (LDA)
3. K-Vizinhos Mais Próximos (KNN)
4. Árvore de Decisão (CART)
5. Naive Bayes (NB)
6. Support Vector Machine (SVM)

Na etapa de validação final com os 20% de dados inéditos, os algoritmos foram comparados usando métricas de **Acurácia**, **Matriz de Confusão** e **F1-Score**.

## 6. Modelo Final: LDA
O **Linear Discriminant Analysis (LDA)** apresentou o melhor desempenho geral nos dados de validação, alcançando **96.67% de acurácia** e excelente F1-Score para todas as classes.

O modelo final foi treinado com 100% dos dados disponíveis para maximizar seu aprendizado e exportado como um arquivo `.pkl` utilizando a biblioteca `joblib`, estando pronto para uso em produção.

## Tecnologias Utilizadas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, SciPy

## Como usar o modelo treinado
Você pode usar o modelo salvo neste repositório para fazer novas previsões sem precisar treinar os algoritmos novamente.

1. Instale as dependências:
```python
pip install pandas scikit-learn joblib
```

2. Faça a previsão em um arquivo Python:
```python
import joblib

# Carrega o modelo treinado salvo
modelo = joblib.load('modelo_lda_iris.pkl')

# Insira as medidas da nova flor na ordem: 
# [sepal_length, sepal_width, petal_length, petal_width]
flor_inedita = [[5.7, 3.0, 1.2, 0.2]]

# Realiza a previsão
previsao = modelo.predict(flor_inedita)
print(f"A espécie detectada é: {previsao[0]}")
# Saída esperada: Iris-setosa
```

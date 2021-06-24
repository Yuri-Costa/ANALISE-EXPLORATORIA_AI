#!/usr/bin/env python
# coding: utf-8

# # Projeto Ciência de Dados - Previsão de Vendas
# 
# - Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio

# ### Passo a Passo de um Projeto de Ciência de Dados
# 
# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# #### Importar a Base de dados

# In[2]:


import pandas as pd

tabela = pd.read_csv("advertising.csv")
display(tabela)


# #### Análise Exploratória
# - Vamos tentar visualizar como as informações de cada item estão distribuídas
# - Vamos ver a correlação entre cada um dos itens

# In[9]:


#pacotes para inserir graficos
import seaborn as sns
import matplotlib.pyplot as plt

#para exibir o grafico
sns.heatmap(tabela.corr(), annot=True)
plt.show()

##para exibir o grafico pear->melhor
sns.pairplot(tabela)
plt.show(tabela)


# #### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning
# 
# - Separando em dados de treino e dados de teste

# In[15]:


from sklearn.model_selection import train_test_split

y = tabela["Vendas"]
x = tabela.drop("Vendas", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)


# #### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:
# 
# - Regressão Linear
# - RandomForest (Árvore de Decisão)

# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#cria a inteligencia artificial
modelo_regressaoLinear = LinearRegression()
modelo_arvoreDecicao = RandomForestRegressor()

#treina a inteligencia artificial
modelo_regressaoLinear.fit(x_treino, y_treino)
modelo_arvoreDecicao.fit(x_treino, y_treino)



# #### Teste da AI e Avaliação do Melhor Modelo
# 
# - Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece

# In[21]:


from sklearn import metrics

#criar previsões
previsao_regressaoLinear = modelo_regressaoLinear.predict(x_teste)
previsao_arvoreDecisao = modelo_arvoreDecicao.predict(x_teste)

#comparar os modelos
print(metrics.r2_score(y_teste, previsao_regressaoLinear))
print(metrics.r2_score(y_teste, previsao_arvoreDecisao)) 


# #### Visualização Gráfica das Previsões

# In[26]:


tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_test"] = y_teste
tabela_auxiliar["Previsões ArvoreDecisao"] = previsao_arvoreDecisao
tabela_auxiliar["Previsões Regressão Linear"] = previsao_regressaoLinear
plt.figure(figsize=(15, 5))
sns.lineplot(data=tabela_auxiliar)
plt.show()


# #### Qual a importância de cada variável para as vendas?

# In[27]:


# no eixo x=as colunas e eixo y + a importancia de cada uma das caracteristicas
sns.barplot(x = x_treino.columns, y=modelo_arvoreDecicao.feature_importances_ )


# #### Será que estamos investindo certo?

# In[ ]:


Deve investir cada vez mais em TV e menos em Radio


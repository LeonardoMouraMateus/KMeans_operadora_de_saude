import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from sklearn.preprocessing import StandardScaler

#importação de dados
estados=pd.read_csv('estados_br.csv',delimiter=';', encoding='latin-1')
clientes=pd.read_excel('clientes_operadora_saude.xlsx')
idade=pd.read_csv('idade_clientes.csv',delimiter=';')

#verificica se o Dtype da variável é coerente
clientes.info()
estados.info()
idade.info()

#verifica se temos valores faltantes em alguma das tabelas
clientes.isna().sum()
estados.isna().sum()
idade.isna().sum()

#preenchimento de dados de peso com a media 
clientes.peso.fillna(round(clientes.peso.mean(),2), inplace=True)

#preenchimento de dados de pais com a moda
estados.pais.fillna(estados.pais.mode()[0], inplace = True)


def hist_grafico_clientes(titulo,data,x):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="white",palette="deep",rc=custom_params)
    plt.figure(figsize=(10,5))
    plt.title(titulo)
    plt.ylabel("Ocorrências")
    sns.histplot(data=data,x=x,kde=True)
    plt.show()
    
def boxplot_clientes(titulo,data,x):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="dark",palette="muted",rc=custom_params)
    plt.figure(figsize=(10,5))
    plt.title(titulo)
    plt.ylabel("Gênero")
    sns.boxplot(data=data,x=x,y='genero',hue='genero',orient="h")
    plt.show()

def linearidade(data,x,y,hue):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style='darkgrid',rc=custom_params,palette='muted')
    sns.lmplot(data=data, x=x, y=y,hue=hue)

hist_grafico_clientes("Peso dos Clientes",clientes,clientes.peso)
hist_grafico_clientes("Colesterol dos Clientes",clientes,clientes.colesterol)

boxplot_clientes("Colesterol dos Clientes",clientes,clientes.colesterol)
boxplot_clientes("Peso dos Clientes",clientes,clientes.peso)

linearidade(clientes,"peso","colesterol","genero")


plt.figure(figsize=(10,5))
plt.title('Correlação entre colesterol e peso')
dados_correlacao=clientes[['colesterol','peso']]
ax=sns.heatmap(dados_correlacao.corr(), cmap='viridis', linewidths=0.1, linecolor='white', annot=True,cbar=False)
ax.set(xlabel="", ylabel="")
ax.xaxis.tick_top()
plt.show()

#codificando o genero
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
clientes['id_genero'] = LE.fit_transform(clientes['genero'])

#join entre as tabelas de cliente e estados atráves do id_estado
clientes_estados=pd.merge(clientes, estados, on='id_estado')
clientes_estados.sort_values('id_cliente')

#join para agregar os dados da tabela de idade
clientes_operadora=pd.merge(clientes_estados, idade, on='id_cliente').sort_values('id_cliente')
clientes_operadora.reset_index(drop=True, inplace=True)

#renomeando colunas
clientes_operadora.rename(columns={'id_cliente':'cod_cliente', 'id_genero':'cod_genero'}, inplace=True)

#passa a sigla dos estados para maiusculo 
clientes_operadora['sigla_estado']=clientes_operadora.sigla_estado.str.upper()

#ordenando as colunas 
ordem_colunas = ['cod_cliente', 'genero', 'idade', 'peso', 'colesterol','id_estado','estado','sigla_estado']
clientes_operadora=clientes_operadora[ordem_colunas].reset_index()
clientes_operadora.drop(columns=['index'],inplace=True)



from sklearn.cluster import KMeans

# calculo da curva do cotovelo para definir o melhor número de clusters 
def calcular_wcss(dados_cliente):
    wcss = []
    for k in range(1,11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(dados_clientes)
        wcss.append(kmeans.inertia_)
    return wcss

dados_clientes=clientes_operadora[['peso','colesterol','idade']]
wcss_clientes = calcular_wcss(dados_clientes)

#plot da curva do cotovelo
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style='darkgrid',palette="deep",rc=custom_params)
plt.figure(figsize=(10,5))
plt.title("Calculando o WCSS")
plt.ylabel("Valor do WCSS")
sns.lineplot(data=wcss_clientes)
plt.show()

#aplicação do KMeans
kmeans_clientes = KMeans(n_clusters = 3, random_state = 0)
clientes_operadora['cluster'] = kmeans_clientes.fit_predict(dados_clientes)

#classificação
clientes_operadora.loc[clientes_operadora['cluster']==1, 'nome_cluster'] = 'Baixo Risco'
clientes_operadora.loc[clientes_operadora['cluster']==2, 'nome_cluster'] = 'Moderado Risco'
clientes_operadora.loc[clientes_operadora['cluster']==0, 'nome_cluster'] = 'Alto Risco'

#localização dos centroides
centroides_clusters = kmeans_clientes.cluster_centers_
print(centroides_clusters)
#grafico de dispersão que combina os clusters e seus centroides 
def gera_graficos(data,x,y,hue):
    markers = {"Baixo Risco": "s", "Moderado Risco": "X","Alto Risco":"o"}
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style='darkgrid',palette="muted",rc=custom_params)
    plt.legend(loc=1,facecolor='white')
    sns.scatterplot(data=data, x=x, y=y, hue=hue,style=hue,palette='deep',markers=markers)
    sns.scatterplot(x = centroides_clusters[:,0],y = centroides_clusters[:,1],style=[0,1,2],markers={1:'D',2:'D',0:'D'},c=[1,1,1],legend=False)
    plt.show()

x = "peso"
y = "colesterol"
hue='nome_cluster'
gera_graficos(clientes_operadora,x,y,hue)


#boxplot para analises 
def boxplot_analise(data,titulo,x,y):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="dark",palette="muted",rc=custom_params)
    plt.figure(figsize=(10,5))
    plt.title(titulo)
    sns.boxplot(data=data.sort_values(x),x=x,y=y,orient="h",hue=y)
    plt.show()


x = "peso"
y = "estado"
titulo="Boxplot por "+y+" ordenado por "+x

boxplot_analise(clientes_operadora,titulo,x,y)

x = "colesterol"
y = "nome_cluster"
titulo="Boxplot por "+y+" ordenado por "+x

boxplot_analise(clientes_operadora,titulo,x,y)

x = "idade"
y = "nome_cluster"
titulo="Boxplot por "+y+" ordenado por "+x

boxplot_analise(clientes_operadora,titulo,x,y)

#possiveis arquivos de saida
df_cluster_genero=clientes_operadora.groupby(['nome_cluster','genero'], as_index=False).agg(total=('genero','count'))
df_cluster_estado=clientes_operadora.query('nome_cluster=="Alto Risco" and estado =="Distrito Federal"')

#alguns exemplos de consultas 
clientes_operadora.groupby('nome_cluster')['idade'].describe()
clientes_operadora.groupby('nome_cluster')['estado'].describe()
clientes_operadora.groupby(['nome_cluster', 'genero'])['peso'].describe()
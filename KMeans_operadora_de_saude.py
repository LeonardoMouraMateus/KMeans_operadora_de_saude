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

#verificica se Dtype da variável é coerente
clientes.info()
estados.info()
idade.info()

#verifica se temos valores faltantes em alguma das tabelas
clientes.isna().sum()
estados.isna().sum()
idade.isna().sum()
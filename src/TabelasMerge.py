from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Definir caminhos das pastas com base na localização exata do script (src/)
pasta_script = Path(__file__).resolve().parent  # Pasta src/
pasta_raiz = pasta_script.parent  # Raiz vacinacao-covid-bahia-ml
pasta_data = pasta_raiz / 'data'  # Pasta data/

print(f'-> Lendo dados brutos de: {pasta_data}')

# Carregar os dois ficheiros originais na pasta data/
df_doses = pd.read_excel(
    pasta_data / 'Doses distribuídas e aplicadas 26.06.2023.xlsx'
)
df_consolidado = pd.read_excel(
    pasta_data / 'Consolidado por município 26.06.2023_15h.xlsx'
)

# Limpeza e padronização das chaves antes do Merge
df_doses_clean = df_doses.dropna(subset=['Municipio']).copy()
df_doses_clean = df_doses_clean[
    df_doses_clean['Municipio'].str.strip().str.upper() != 'BAHIA'
]
df_doses_clean['Municipio'] = (
    df_doses_clean['Municipio'].str.strip().str.upper()
)

df_cons_clean = df_consolidado.dropna(subset=['Cód. IBGE']).copy()
df_cons_clean['Município'] = (
    df_cons_clean['Município'].str.strip().str.upper()
)

# Execução do Merge
df_merged = pd.merge(
    df_cons_clean,
    df_doses_clean,
    left_on='Município',
    right_on='Municipio',
    how='inner',
)

# Salvar a planilha Excel unificada do merge na pasta data/
df_merged.to_excel(
    pasta_data / 'dataset_vacinacao_bahia_merged.xlsx', index=False
)
print(
    f'[1/3] Merge realizado com sucesso! ({len(df_merged)} municípios integrados)'
)

# Criar a Variável Alvo (Target) -> 1 para < 85%, 0 para >= 85%
df_merged['pct_aplicadas'] = df_merged['% de doses Aplicadas'] * 100
df_merged['target_baixa_eficiencia'] = (
    df_merged['pct_aplicadas'] < 85
).astype(int)

# Remover Colunas Redundantes, Identificadores e Data Leakage
colunas_para_remover = [
    'Cód. IBGE',
    'Município',
    'Municipio',
    'Dires',
    '% de doses Aplicadas',
    'pct_aplicadas',
    'Total de Doses Aplicadas 22062023',
    'Total de Doses Distribuídas (Adulto)',
]

df_model = df_merged.drop(columns=colunas_para_remover)

# Separar Variáveis Explicativas (X) e Target (y)
X = df_model.drop(columns=['target_baixa_eficiencia'])
y = df_model['target_baixa_eficiencia']

# Encoding da variável categórica NRS
X_encoded = pd.get_dummies(
    X, columns=['Núcleo Regional de Saúde (NRS)'], drop_first=True
)

# Divisão Treino (80%) e Teste (20%) Estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

# Salvar todas as tabelas na pasta data/
X_train.to_excel(pasta_data / 'X_train.xlsx', index=False)
X_test.to_excel(pasta_data / 'X_test.xlsx', index=False)
y_train.to_excel(pasta_data / 'y_train.xlsx', index=False)
y_test.to_excel(pasta_data / 'y_test.xlsx', index=False)

print('[2/3] Remoção de colunas e One-Hot Encoding concluídos!')
print(
    '[3/3] Bases X_train, X_test, y_train e y_test salvas com sucesso em data/!'
)
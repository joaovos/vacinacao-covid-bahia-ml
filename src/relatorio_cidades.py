from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Mapeamento de Caminhos
pasta_script = Path(__file__).resolve().parent  # Pasta src/
pasta_raiz = pasta_script.parent  # Raiz do projeto
pasta_data = pasta_raiz / 'data'
pasta_docs = pasta_raiz / 'docs'

pasta_docs.mkdir(parents=True, exist_ok=True)

print('-> Carregando dados e integrando nomes dos municípios...')

# Carregar e tratar as planilhas brutas para recuperar o nome das cidades
df_doses = pd.read_excel(
    pasta_data / 'Doses distribuídas e aplicadas 26.06.2023.xlsx'
)
df_consolidado = pd.read_excel(
    pasta_data / 'Consolidado por município 26.06.2023_15h.xlsx'
)

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

df_merged = pd.merge(
    df_cons_clean,
    df_doses_clean,
    left_on='Município',
    right_on='Municipio',
    how='inner',
)

# Carregar conjuntos de Treino e Teste do Machine Learning
X_train = pd.read_excel(pasta_data / 'X_train.xlsx')
X_test = pd.read_excel(pasta_data / 'X_test.xlsx')
y_train = pd.read_excel(pasta_data / 'y_train.xlsx').values.ravel()
y_test = pd.read_excel(pasta_data / 'y_test.xlsx').values.ravel()

# Treinar a Regressão Logística (Modelo Campeão) e realizar previsões
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)
y_probs = model_lr.predict_proba(X_test)[
    :, 1
]  # Probabilidade de ser Baixa Eficiência

# Recuperar os nomes das cidades e a região sanitária de forma segura
nomes_cidades_teste = df_merged.loc[X_test.index, 'Município'].values

# Busca flexível pela coluna de região (NRS ou Dires)
col_regiao = [
    c
    for c in df_merged.columns
    if 'NRS' in c.upper() or 'DIRES' in c.upper() or 'REGIONAL' in c.upper()
]

if col_regiao:
    regiao_sanitaria = df_merged.loc[X_test.index, col_regiao[0]].values
else:
    regiao_sanitaria = ['N/A'] * len(X_test)

# Montar a tabela de diagnóstico final
df_relatorio = pd.DataFrame({
    'Município': nomes_cidades_teste,
    'Região Sanitária / Dires': regiao_sanitaria,
    'Status Real': [
        'Baixa Eficiência (<85%)' if v == 1 else 'Boa Eficiência (>=85%)'
        for v in y_test
    ],
    'Previsão do Modelo': [
        'Baixa Eficiência (<85%)' if v == 1 else 'Boa Eficiência (>=85%)'
        for v in y_pred
    ],
    'Probabilidade de Risco (%)': np.round(y_probs * 100, 2),
    'Resultado Diagnóstico': np.where(
        y_test == y_pred, '✅ Acerto', '❌ Erro do Modelo'
    ),
    'Alerta SESAB': np.where(
        y_pred == 1,
        '⚠️ Intervenção Necessária',
        '🟢 Operação Normal',
    ),
})

# Ordenar colocando os municípios em situação de alerta no topo
df_relatorio = df_relatorio.sort_values(
    by=['Alerta SESAB', 'Probabilidade de Risco (%)'], ascending=[True, False]
)

# Salvar em formato Excel na pasta docs/
caminho_excel = pasta_docs / 'relatorio_cidades_testadas.xlsx'
df_relatorio.to_excel(caminho_excel, index=False)

print(
    f'\n[Sucesso] Relatório gerado com sucesso para {len(df_relatorio)} municípios!'
)
print(f'Arquivo salvo em: {caminho_excel}')

# Exibir resumo das cidades em alerta no terminal
cidades_alerta = df_relatorio[
    df_relatorio['Alerta SESAB'] == '⚠️ Intervenção Necessária'
]
print(
    f'\nTotal de cidades sinalizadas com Baixa Eficiência: {len(cidades_alerta)}'
)
print(
    cidades_alerta[
        ['Município', 'Região Sanitária / Dires', 'Probabilidade de Risco (%)']
    ]
    .head(10)
    .to_string(index=False)
)
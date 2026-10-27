
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


# Script dentro de 'src/', .parent.parent pega o diretório raiz do repositório
pasta_script = Path(__file__).parent
pasta_raiz = pasta_script.parent

pasta_data = pasta_raiz / 'data'
pasta_docs = pasta_raiz / 'docs'

# Garantir que a pasta 'docs' existe para salvar a imagem
pasta_docs.mkdir(exist_ok=True)

# Carrega os datasets brutos disponibilizados para a modelagem
df_doses = pd.read_excel(pasta_data / 'Doses distribuídas e aplicadas 26.06.2023.xlsx')
df_consolidado = pd.read_excel(pasta_data / 'Consolidado por município 26.06.2023_15h.xlsx')

# Trata a base de Doses: remove o totalizador estadual ('BAHIA') e padroniza a chave de junção
df_doses_clean = df_doses.dropna(subset=['Municipio']).copy()
df_doses_clean = df_doses_clean[df_doses_clean['Municipio'].str.strip().str.upper() != 'BAHIA']
df_doses_clean['Municipio'] = df_doses_clean['Municipio'].str.strip().str.upper()

# Trata a base Consolidada: garante a integridade da chave primária (Cód. IBGE/Município)
df_cons_clean = df_consolidado.dropna(subset=['Cód. IBGE']).copy()
df_cons_clean['Município'] = df_cons_clean['Município'].str.strip().str.upper()

# Integração das bases (Inner Join) para consolidar a visão única do município
df_eda = pd.merge(df_cons_clean, df_doses_clean, left_on='Município', right_on='Municipio', how='inner')

# Transforma a taxa de aplicação em percentual padronizado (escala 0-100)
df_eda['pct_aplicadas'] = df_eda['% de doses Aplicadas'] * 100

# Mapeamento da Variável Alvo (Target) para Classificação Binária (Limiar de Decisão: 85%)
# Classe 1: Baixa Eficiência (< 85%) | Classe 0: Boa Eficiência (>= 85%)
df_eda['target_baixa_eficiencia'] = (df_eda['pct_aplicadas'] < 85).astype(int)

# Configura o padrão visual do painel gráfico para o relatório técnico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Figura 1: Distribuição da Variável de Eficiência (Detecção de Outliers) 
sns.histplot(df_eda['pct_aplicadas'], kde=True, ax=axes[0, 0], color='#2b5c8f', bins=30)
axes[0, 0].axvline(85, color='#e74c3c', linestyle='--', linewidth=2, label='Limiar de Eficiência (85%)')
axes[0, 0].set_title('Figura 1: Histograma da Taxa de Utilização (% Aplicado)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Taxa de Utilização (%)')
axes[0, 0].set_ylabel('Frequência (Nº de Instâncias)')
axes[0, 0].legend()

# Figura 2: Balanceamento das Classes da Variável Alvo 
counts = df_eda['target_baixa_eficiencia'].value_counts()
bars = axes[0, 1].bar(['Boa Eficiência (0)', 'Baixa Eficiência (1)'], [counts.get(0, 0), counts.get(1, 0)], color=['#2ecc71', '#e74c3c'])
axes[0, 1].set_title('Figura 2: Proporção das Classes da Variável Target', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Número de Municípios')
for bar in bars:
    yval = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval} ({yval/len(df_eda)*100:.1f}%)", ha='center', fontweight='bold')

# Figura 3: Composição do Estoque de Insumos
doses = [
    df_eda['PFIZER'].sum() / 1e6,
    df_eda['FIOCRUZ/ASTRAZENECA'].sum() / 1e6,
    df_eda['BUTANTAN/CORONAVAC'].sum() / 1e6,
    df_eda['JANSSEN'].sum() / 1e6
]
axes[1, 0].barh(['Pfizer', 'AstraZeneca', 'Coronavac', 'Janssen'], doses, color='#3498db')
axes[1, 0].set_title('Figura 3: Doses Distribuídas por Fabricante (Milhões)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Volume (Milhões de Doses)')

# Figura 4: Matriz de Correlação Linear (Multicolinearidade)
features_corr = ['Total de Doses Aplicadas - D1', 'Total de Doses Aplicadas - D2', 'Total de Doses Aplicadas - D3', 'pct_aplicadas']
sns.heatmap(df_eda[features_corr].corr(), annot=True, fmt='.2f', cmap='Blues', ax=axes[1, 1],
            xticklabels=['D1', 'D2', 'D3', '% Utilização'],
            yticklabels=['D1', 'D2', 'D3', '% Utilização'])
axes[1, 1].set_title('Figura 4: Matriz de Correlação de Pearson', fontsize=12, fontweight='bold')

# Exportação do artefato visual para a pasta docs/
plt.tight_layout()
plt.savefig(pasta_docs / 'painel_eda_bahia.png', dpi=300)
plt.show()
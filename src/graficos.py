"""
Script Unificado de Geração de Gráficos e Visualizações do Projeto (SESAB/BA)
Gera os artefatos visuais de EDA e de Modelagem na pasta docs/.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.tree import DecisionTreeClassifier

# Definir caminhos das pastas usando pathlib
pasta_script = Path(__file__).resolve().parent  # Pasta src/
pasta_raiz = pasta_script.parent  # Raiz do projeto
pasta_data = pasta_raiz / 'data'
pasta_docs = pasta_raiz / 'docs'

# Garantir que a pasta 'docs' existe para salvar as imagens
pasta_docs.mkdir(parents=True, exist_ok=True)

# Configura o padrão visual do painel gráfico
plt.style.use(
    'seaborn-v0_8-whitegrid'
    if 'seaborn-v0_8-whitegrid' in plt.style.available
    else 'default'
)

# 1. PAINEL DE ANÁLISE EXPLORATÓRIA DE DADOS (EDA)

def gerar_painel_eda():
    print('-> Gerando Painel de Análise Exploratória (EDA)...')

    # Carrega os datasets brutos disponibilizados para a modelagem
    df_doses = pd.read_excel(
        pasta_data / 'Doses distribuídas e aplicadas 26.06.2023.xlsx'
    )
    df_consolidado = pd.read_excel(
        pasta_data / 'Consolidado por município 26.06.2023_15h.xlsx'
    )

    # Trata a base de Doses: remove o totalizador estadual ('BAHIA') e padroniza a chave
    df_doses_clean = df_doses.dropna(subset=['Municipio']).copy()
    df_doses_clean = df_doses_clean[
        df_doses_clean['Municipio'].str.strip().str.upper() != 'BAHIA'
    ]
    df_doses_clean['Municipio'] = (
        df_doses_clean['Municipio'].str.strip().str.upper()
    )

    # Trata a base Consolidada: garante a integridade da chave primária
    df_cons_clean = df_consolidado.dropna(subset=['Cód. IBGE']).copy()
    df_cons_clean['Município'] = (
        df_cons_clean['Município'].str.strip().str.upper()
    )

    # Integração das bases (Inner Join)
    df_eda = pd.merge(
        df_cons_clean,
        df_doses_clean,
        left_on='Município',
        right_on='Municipio',
        how='inner',
    )

    # Transforma a taxa de aplicação em percentual padronizado (escala 0-100)
    df_eda['pct_aplicadas'] = df_eda['% de doses Aplicadas'] * 100

    # Mapeamento da Variável Alvo (Target) -> Classe 1: Baixa Eficiência (< 85%)
    df_eda['target_baixa_eficiencia'] = (df_eda['pct_aplicadas'] < 85).astype(
        int
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Figura 1: Distribuição da Variável de Eficiência (Detecção de Outliers)
    sns.histplot(
        df_eda['pct_aplicadas'],
        kde=True,
        ax=axes[0, 0],
        color='#2b5c8f',
        bins=30,
    )
    axes[0, 0].axvline(
        85,
        color='#e74c3c',
        linestyle='--',
        linewidth=2,
        label='Limiar de Eficiência (85%)',
    )
    axes[0, 0].set_title(
        'Figura 1: Histograma da Taxa de Utilização (% Aplicado)',
        fontsize=12,
        fontweight='bold',
    )
    axes[0, 0].set_xlabel('Taxa de Utilização (%)')
    axes[0, 0].set_ylabel('Frequência (Nº de Instâncias)')
    axes[0, 0].legend()

    # Figura 2: Balanceamento das Classes da Variável Alvo
    counts = df_eda['target_baixa_eficiencia'].value_counts()
    bars = axes[0, 1].bar(
        ['Boa Eficiência (0)', 'Baixa Eficiência (1)'],
        [counts.get(0, 0), counts.get(1, 0)],
        color=['#2ecc71', '#e74c3c'],
    )
    axes[0, 1].set_title(
        'Figura 2: Proporção das Classes da Variável Target',
        fontsize=12,
        fontweight='bold',
    )
    axes[0, 1].set_ylabel('Número de Municípios')
    for bar in bars:
        yval = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            yval + 5,
            f'{yval} ({yval/len(df_eda)*100:.1f}%)',
            ha='center',
            fontweight='bold',
        )

    # Figura 3: Composição do Estoque de Insumos
    doses = [
        df_eda['PFIZER'].sum() / 1e6,
        df_eda['FIOCRUZ/ASTRAZENECA'].sum() / 1e6,
        df_eda['BUTANTAN/CORONAVAC'].sum() / 1e6,
        df_eda['JANSSEN'].sum() / 1e6,
    ]
    axes[1, 0].barh(
        ['Pfizer', 'AstraZeneca', 'Coronavac', 'Janssen'],
        doses,
        color='#3498db',
    )
    axes[1, 0].set_title(
        'Figura 3: Doses Distribuídas por Fabricante (Milhões)',
        fontsize=12,
        fontweight='bold',
    )
    axes[1, 0].set_xlabel('Volume (Milhões de Doses)')

    # Figura 4: Matriz de Correlação Linear (Multicolinearidade)
    features_corr = [
        'Total de Doses Aplicadas - D1',
        'Total de Doses Aplicadas - D2',
        'Total de Doses Aplicadas - D3',
        'pct_aplicadas',
    ]
    sns.heatmap(
        df_eda[features_corr].corr(),
        annot=True,
        fmt='.2f',
        cmap='Blues',
        ax=axes[1, 1],
        xticklabels=['D1', 'D2', 'D3', '% Utilização'],
        yticklabels=['D1', 'D2', 'D3', '% Utilização'],
    )
    axes[1, 1].set_title(
        'Figura 4: Matriz de Correlação de Pearson',
        fontsize=12,
        fontweight='bold',
    )

    # Exportação do artefato visual para a pasta docs/
    plt.tight_layout()
    plt.savefig(pasta_docs / 'painel_eda_bahia.png', dpi=300)
    plt.close()
    print('  [✓] painel_eda_bahia.png gerado com sucesso!')



#  GRÁFICOS DE AVALIAÇÃO DOS MODELOS DE MACHINE LEARNING

def gerar_graficos_modelagem():
    print('-> Gerando gráficos de avaliação dos modelos...')

    # Carregar dados pré-processados
    X_train = pd.read_excel(pasta_data / 'X_train.xlsx')
    X_test = pd.read_excel(pasta_data / 'X_test.xlsx')
    y_train = pd.read_excel(pasta_data / 'y_train.xlsx').values.ravel()
    y_test = pd.read_excel(pasta_data / 'y_test.xlsx').values.ravel()

    # Treinar os 3 modelos
    model_lr = LogisticRegression(max_iter=1000, random_state=42).fit(
        X_train, y_train
    )
    model_dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(
        X_train, y_train
    )

    
    # Matriz de Confusão do Modelo Campeão (Regressão Logística)

    y_pred_lr = model_lr.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_lr)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=['Boa Eficiência (0)', 'Baixa Eficiência (1)'],
        yticklabels=['Boa Eficiência (0)', 'Baixa Eficiência (1)'],
    )
    plt.title(
        'Matriz de Confusão - Regressão Logística (Campeão)',
        pad=15,
        fontweight='bold',
    )
    plt.xlabel('Classe Preditada pelo Modelo')
    plt.ylabel('Classe Real do Município')

    plt.tight_layout()
    plt.savefig(pasta_docs / 'matriz_confusao_campeao.png', dpi=300)
    plt.close()
    print('  [✓] matriz_confusao_campeao.png gerado com sucesso!')

   
    # Curva ROC Comparativa dos 3 Modelos
  
    plt.figure(figsize=(8, 6))
    modelos = {
        'Regressão Logística': model_lr,
        'Árvore de Decisão': model_dt,
        'Random Forest': model_rf,
    }

    for nome, model in modelos.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc:.2f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Classificador Aleatório (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)')
    plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade / Recall)')
    plt.title('Comparação da Curva ROC entre os Modelos', pad=15, fontweight='bold')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(pasta_docs / 'curva_roc_comparativa.png', dpi=300)
    plt.close()
    print('  [✓] curva_roc_comparativa.png gerado com sucesso!')
   
    # Importância das Variáveis (Top 10 Coeficientes)

    coefs = model_lr.coef_[0]
    df_importance = (
        pd.DataFrame({'Atributo': X_train.columns, 'Coeficiente': coefs})
        .assign(abs_coef=lambda x: x['Coeficiente'].abs())
        .sort_values(by='abs_coef', ascending=False)
        .head(10)
    )

    plt.figure(figsize=(9, 5))
    colors = ['#d9534f' if c > 0 else '#5bc0de' for c in df_importance['Coeficiente']]
    sns.barplot(
        data=df_importance,
        x='Coeficiente',
        y='Atributo',
        palette=colors,
        hue='Atributo',
        legend=False,
    )
    plt.title(
        'Top 10 Atributos de Maior Impacto - Regressão Logística',
        pad=15,
        fontweight='bold',
    )
    plt.xlabel('Coeficiente (Vermelho = Aumenta Risco de Baixa Eficiência)')
    plt.ylabel('Atributo')

    plt.tight_layout()
    plt.savefig(pasta_docs / 'feature_importance_top10.png', dpi=300)
    plt.close()
    print('  [✓] feature_importance_top10.png gerado com sucesso!')

# EXECUÇÃO PRINCIPAL

if __name__ == '__main__':
    gerar_painel_eda()
    gerar_graficos_modelagem()
    print(
        '\n[Sucesso] Todos os 4 arquivos de imagem foram salvos com sucesso na pasta docs/!'
    )
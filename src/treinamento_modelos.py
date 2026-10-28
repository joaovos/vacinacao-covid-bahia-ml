from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

# Definir caminhos das pastas
pasta_script = Path(__file__).resolve().parent  # Pasta src/
pasta_raiz = pasta_script.parent  # Raiz vacinacao-covid-bahia-ml
pasta_data = pasta_raiz / 'data'

print('-> Lendo bases de dados de treino e teste...')

# Carregar as tabelas pre-processadas
X_train = pd.read_excel(pasta_data / 'X_train.xlsx')
X_test = pd.read_excel(pasta_data / 'X_test.xlsx')
y_train = pd.read_excel(pasta_data / 'y_train.xlsx').values.ravel()
y_test = pd.read_excel(pasta_data / 'y_test.xlsx').values.ravel()

# Dicionário com os três modelos selecionados
models = {
    'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
    'Árvore de Decisão': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

# Loop de Treinamento e Avaliação
resultados = []

print('\n======================================================')
print('        INICIANDO TREINAMENTO E AVALIAÇÃO             ')
print('======================================================\n')

for nome, model in models.items():
    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer predições no conjunto de teste
    y_pred = model.predict(X_test)

    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    resultados.append({
        'Modelo': nome,
        'Acurácia': f'{acc:.2%}',
        'Precisão': f'{prec:.2%}',
        'Recall': f'{rec:.2%}',
        'F1-Score': f'{f1:.2%}',
    })

    print(f'>>> Matriz de Confusão - {nome}:')
    print(confusion_matrix(y_test, y_pred))
    print('-' * 50)

# Exibir Tabela Comparativa Final
df_res = pd.DataFrame(resultados)
print('\n======================================================')
print('           TABELA COMPARATIVA DE DESEMPENHO           ')
print('======================================================')
print(df_res.to_string(index=False))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Carregar os dados
df = pd.read_excel('dados/StudentsPrepared.xlsx')

# Filtrar apenas registros de interesse (Desistente e Graduado)
df = df[df['Target'].isin(['Desistente', 'Graduado'])].copy()

# PREPARAÇÃO DAS VARIÁVEIS PARA MODELAGEM
# --------------------------------------
# Definir variáveis:
# - y: Variável TARGET (resposta/dependente) que queremos prever
#       Mapeada para valores binários:
#       - 1 = 'Desistente' (caso positivo de evasão)
#       - 0 = 'Graduado' (caso negativo)
# - X: Variáveis PREDITORAS (features/independentes) usadas para prever y
#       Contém todas as colunas exceto 'Target'
y = df['Target'].map({'Desistente': 1, 'Graduado': 0})  
X = df.drop('Target', axis=1)

# IDENTIFICAÇÃO DOS TIPOS DE VARIÁVEIS
# ------------------------------------
# Separa as colunas preditoras em:
# - Categóricas: Variáveis qualitativas (texto/object)
# - Numéricas: Variáveis quantitativas (int/float)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# PRÉ-PROCESSAMENTO DOS DADOS
# ------------------------------------
# ColumnTransformer permite aplicar transformações diferentes para diferentes colunas
# Cada transformador é definido como uma tupla com:
# - Nome: identificador único
# - Transformador: técnica a ser aplicada
# - Colunas: lista de colunas alvo
preprocessor = ColumnTransformer([
    # Transformação para variáveis NUMÉRICAS:
    # - StandardScaler: normaliza os dados (média=0, desvio padrão=1)
    # - Importante para algoritmos sensíveis à escala como Random Forest
    ('num', StandardScaler(), numeric_cols),
    
    # Transformação para variáveis CATEGÓRICAS:
    # - OneHotEncoder: converte categorias em colunas binárias (0/1)
    # - handle_unknown='ignore': evita erros se encontrar novas categorias no teste
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# CONFIGURAÇÃO DO MODELO
# -------------------------------
# RandomForestClassifier: algoritmo baseado em árvores de decisão

model = RandomForestClassifier(random_state=42, n_estimators=100)

# 1. CONFIGURAÇÃO DO PIPELINE DE MODELO
# -------------------------------
# Cria um pipeline que primeiro pré-processa os dados e depois aplica o modelo
clf = Pipeline([
    ('preprocessor', preprocessor),  # Etapa de pré-processamento (normalização numérica + one-hot encoding)
    ('classifier', model)            # Modelo Random Forest para classificação
])

# 2. DIVISÃO DOS DADOS
# -------------------------------
# Separa os dados em conjuntos de treino (80%) e teste (20%)
# stratify=y garante que a proporção de classes seja mantida
# random_state=42 para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. VALIDAÇÃO CRUZADA
# -------------------------------
# Usa StratifiedKFold para manter proporção de classes em cada fold
# shuffle=True para embaralhar os dados antes da divisão
# cross_val_score calcula AUC ROC para cada fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"AUC ROC médio (validação cruzada): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 4. TREINAMENTO FINAL E AVALIAÇÃO
# -------------------------------
# Treina o modelo com todos os dados de treino
clf.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = clf.predict(X_test)          # Classes preditas (0 ou 1)
y_proba = clf.predict_proba(X_test)[:,1]  # Probabilidades para classe positiva

# 5. MÉTRICAS DE DESEMPENHO
# -------------------------------
print("\nRelatório de Classificação (Teste):")
print(classification_report(y_test, y_pred))  # Precisão, recall, f1-score
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))       # VP, FP, FN, VN
print(f"AUC ROC (Teste): {roc_auc_score(y_test, y_proba):.4f}")  # Área sob curva ROC

# 6. ANÁLISE DE OVERFITTING/UNDERFITTING
# -------------------------------
# Cálculo do AUC ROC (Area Under the ROC Curve):
# Métrica que avalia a capacidade do modelo de distinguir entre classes (0 e 1)
# Varia de 0.5 (pior) a 1.0 (melhor)

# AUC no conjunto de TESTE:
# - y_test: valores reais das classes
# - y_proba: probabilidades preditas para a classe positiva (1=Desistente)
# - Mostra o desempenho em dados não vistos durante o treino
test_auc = roc_auc_score(y_test, y_proba)

# AUC no conjunto de TREINO:
# - clf.predict_proba(X_train)[:,1]: probabilidades para o próprio conjunto de treino
# - Idealmente similar ao teste para indicar boa generalização
train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])

# Exibição dos resultados formatados com 4 casas decimais
print(f"\nAUC ROC (Treino): {train_auc:.4f}")  # Desempenho nos dados de treino
print(f"AUC ROC (Teste): {test_auc:.4f}")    # Desempenho nos dados de teste

# Critérios:
# - Overfitting: grande diferença (>0.1) entre AUC treino e teste
# - Underfitting: AUC teste baixo (<0.7) indica modelo pouco capaz
# - Equilibrado: desempenho similar e aceitável
if train_auc - test_auc > 0.1:
    print("AVISO: Possível overfitting detectado - modelo muito especializado nos dados de treino")
elif test_auc < 0.7:
    print("AVISO: Possível underfitting detectado - modelo não está aprendendo padrões suficientes")
else:
    print("RESULTADO: Modelo equilibrado - bom desempenho em ambos conjuntos")

# 7. PERSISTÊNCIA DO MODELO
# -------------------------------
# Salva o modelo treinado para uso futuro
joblib.dump(clf, 'modelo_evasao.joblib')
print("\nModelo salvo com sucesso como 'modelo_evasao.joblib'")

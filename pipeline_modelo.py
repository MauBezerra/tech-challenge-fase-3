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

# Definir X e y
y = df['Target'].map({'Desistente': 1, 'Graduado': 0})  # 1 = evasão, 0 = não evasão
X = df.drop('Target', axis=1)

# Identificar colunas categóricas e numéricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pipeline de pré-processamento
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Modelo
model = RandomForestClassifier(random_state=42, n_estimators=100)

# Pipeline completa
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"AUC ROC médio (validação cruzada): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Treinar no treino completo e avaliar no teste
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("\nRelatório de Classificação (Teste):")
print(classification_report(y_test, y_pred))
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))
print(f"AUC ROC (Teste): {roc_auc_score(y_test, y_proba):.4f}")

# Análise de overfitting/underfitting
test_auc = roc_auc_score(y_test, y_proba)
train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
print(f"AUC ROC (Treino): {train_auc:.4f}")
if train_auc - test_auc > 0.1:
    print("Possível overfitting detectado.")
elif test_auc < 0.7:
    print("Possível underfitting detectado.")
else:
    print("Modelo equilibrado.")

# Salvar o modelo
joblib.dump(clf, 'modelo_evasao.joblib')

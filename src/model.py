"""
Loan Approval Classification – Pipeline Completo
Autor: EleazarCV
Modelo: Regresión Logística
Dataset: Loan Approval Dataset (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 1. CARGA DE DATOS
# ──────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset cargado: {df.shape[0]:,} registros, {df.shape[1]} columnas")
    return df


# ──────────────────────────────────────────────
# 2. EXPLORACIÓN BÁSICA
# ──────────────────────────────────────────────
def explore_data(df: pd.DataFrame) -> None:
    """Imprime resumen exploratorio del dataset."""
    print("\n=== EXPLORACIÓN DE DATOS ===")
    print(f"Shape: {df.shape}")
    print(f"\nTipos de variables:\n{df.dtypes}")
    print(f"\nValores nulos:\n{df.isnull().sum()}")
    print(f"\nEstadísticas descriptivas:\n{df.describe()}")

    if 'loan_approved' in df.columns:
        print(f"\nDistribución del target:\n{df['loan_approved'].value_counts()}")


# ──────────────────────────────────────────────
# 3. PREPROCESAMIENTO
# ──────────────────────────────────────────────
def preprocess(df: pd.DataFrame, target_col: str = 'loan_approved') -> tuple:
    """
    Limpia y prepara el dataset para el modelo.

    Returns:
        X (pd.DataFrame): Features procesadas
        y (pd.Series): Variable objetivo
    """
    df_clean = df.copy()

    # Convertir target a binario si es string
    if df_clean[target_col].dtype == object:
        mapping = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'Approved': 1, 'Rejected': 0}
        df_clean[target_col] = df_clean[target_col].map(mapping)

    # Eliminar columnas con más del 80% de nulos
    null_pct = df_clean.isnull().mean()
    cols_to_drop = null_pct[null_pct > 0.8].index.tolist()
    if cols_to_drop:
        print(f"⚠️  Eliminando columnas con >80% nulos: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)

    # Imputar nulos numéricos con la mediana
    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

    # One-hot encoding de variables categóricas
    cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        print(f"📌 Aplicando one-hot encoding a: {cat_cols}")
        df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

    # Separar X e y
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    print(f"\n✅ Preprocesamiento completo: X={X.shape}, y={y.shape}")
    return X, y


# ──────────────────────────────────────────────
# 4. ENTRENAMIENTO
# ──────────────────────────────────────────────
def train_model(X_train, y_train) -> tuple:
    """
    Escala los features y entrena el modelo de Regresión Logística.

    Returns:
        model: Modelo entrenado
        scaler: Scaler ajustado (necesario para predicciones futuras)
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        C=1.0,
        random_state=42
    )
    model.fit(X_train_s, y_train)

    print("✅ Modelo entrenado correctamente")
    return model, scaler


# ──────────────────────────────────────────────
# 5. EVALUACIÓN
# ──────────────────────────────────────────────
def evaluate_model(model, scaler, X_test, y_test) -> dict:
    """
    Evalúa el modelo y genera visualizaciones.

    Returns:
        dict con las métricas principales
    """
    X_test_s = scaler.transform(X_test)
    y_pred   = model.predict(X_test_s)
    y_proba  = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
        'roc_auc'  : roc_auc_score(y_test, y_proba)
    }

    print("\n" + "="*45)
    print("      MÉTRICAS DEL MODELO (Test Set)")
    print("="*45)
    for k, v in metrics.items():
        print(f"  {k.upper():12s}: {v:.4f}")
    print("="*45)
    print()
    print(classification_report(y_test, y_pred, target_names=['Rechazado', 'Aprobado']))

    # Matriz de confusión
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Rechazado', 'Aprobado'],
                yticklabels=['Rechazado', 'Aprobado'])
    axes[0].set_title('Matriz de Confusión')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicción')

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, color='steelblue', lw=2,
                 label=f"AUC = {metrics['roc_auc']:.4f}")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_title('Curva ROC')
    axes[1].set_xlabel('Tasa de Falsos Positivos')
    axes[1].set_ylabel('Tasa de Verdaderos Positivos')
    axes[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=120)
    plt.show()

    return metrics


# ──────────────────────────────────────────────
# 6. IMPORTANCIA DE VARIABLES
# ──────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> None:
    """Visualiza los coeficientes más importantes del modelo."""
    coef_df = pd.DataFrame({
        'variable'   : feature_names,
        'coeficiente': model.coef_[0]
    }).sort_values('coeficiente', key=abs, ascending=False).head(top_n)

    colors = ['#d9534f' if c < 0 else '#5bc0de' for c in coef_df['coeficiente']]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, x='coeficiente', y='variable', palette=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f'Top {top_n} Variables más Importantes (Coeficientes)')
    plt.xlabel('Coeficiente')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=120)
    plt.show()

    print(f"\nTop {top_n} variables:\n")
    print(coef_df.to_string(index=False))


# ──────────────────────────────────────────────
# 7. PREDICCIÓN INDIVIDUAL
# ──────────────────────────────────────────────
def predict_applicant(model, scaler, feature_names: list, applicant: dict) -> None:
    """
    Predice la probabilidad de aprobación para un solicitante.

    Args:
        applicant: dict con los valores del solicitante
    """
    X_new = pd.DataFrame([{col: applicant.get(col, 0) for col in feature_names}])
    X_new_s = scaler.transform(X_new)

    prob     = model.predict_proba(X_new_s)[0, 1]
    decision = 'APROBADO ✅' if prob >= 0.5 else 'RECHAZADO ❌'

    print(f"\n{'='*40}")
    print(f"  Probabilidad de aprobación: {prob:.2%}")
    print(f"  Decisión: {decision}")
    print(f"{'='*40}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == '__main__':

    # 1. Cargar datos
    df = load_data('../data/raw/loan_approval_dataset.csv')

    # 2. Explorar
    explore_data(df)

    # 3. Preprocesar
    X, y = preprocess(df)

    # 4. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Entrenar
    model, scaler = train_model(X_train, y_train)

    # 6. Evaluar
    metrics = evaluate_model(model, scaler, X_test, y_test)

    # 7. Importancia de variables
    plot_feature_importance(model, list(X.columns), top_n=15)

    # 8. Ejemplo de predicción individual
    nuevo = {
        'income'        : 55000,
        'credit_score'  : 720,
        'loan_amount'   : 15000,
        'years_employed': 5,
        'points'        : 80
    }
    predict_applicant(model, scaler, list(X.columns), nuevo)

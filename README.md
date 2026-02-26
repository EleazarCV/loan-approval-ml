# Loan Approval Classification – Modelo de Regresión Logística

Este proyecto analiza una base de datos de solicitudes de préstamo y construye un modelo de clasificación para predecir si un préstamo será aprobado o no.  
El flujo completo incluye: exploración de datos, limpieza, codificación, modelado, evaluación y análisis de las variables más importantes.

El dataset utilizado proviene de Kaggle: *Loan Approval Dataset*.

---

## Objetivo del proyecto

El objetivo principal es estimar la probabilidad de aprobación `P(loan_approved = 1 | X)`,
donde `X` representa variables como:

- income  
- credit_score  
- loan_amount  
- years_employed  
- points  
- name y city (codificadas mediante one-hot encoding)

Este modelo ayuda a entender qué factores influyen más en la aprobación de un préstamo y permite predecir el resultado de nuevas solicitudes.

---

## Estructura del proyecto
```
loan-approval-ml/
│
├── data/
│   └── raw/                        # Archivo original .csv del dataset
│
├── notebooks/
│   └── loan_approval_model.ipynb   # Notebook principal con todo el análisis
│
├── src/
│   └── model.py                    # Pipeline completo del modelo
│
├── README.md                       # Este archivo
└── requirements.txt                # Librerías necesarias
```

---

## Exploración y preparación de datos

- Se revisaron tipos de variables, valores faltantes y estadísticas básicas.  
- Se identificaron columnas numéricas (`income`, `credit_score`, `loan_amount`, `years_employed`, `points`) y categóricas (`name`, `city`).  
- Las variables `name` y `city` se codificaron con **one-hot encoding**.  
- La variable objetivo `loan_approved` originalmente es booleana (`True/False`) y se trató directamente como categórica binaria.
- Se aplicó **StandardScaler** para normalizar las variables numéricas antes del entrenamiento.

---

## Modelo utilizado: Regresión Logística

La Regresión Logística se eligió porque:

- Permite predecir probabilidades entre 0 y 1.  
- Es fácil de interpretar mediante coeficientes.  
- Funciona bien con datos categóricos codificados.  
- Es un modelo rápido, estable y apropiado como baseline.

El modelo aprende una combinación lineal:

$$z = b_0 + b_1 x_1 + \dots + b_n x_n$$

y aplica la función sigmoide:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

para transformar ese valor en una probabilidad entre 0 y 1.

---

## Resultados del modelo

El modelo fue evaluado en un conjunto de prueba de 400 registros (split 80/20 estratificado):

| Métrica    | Valor  |
|------------|--------|
| Accuracy   | 0.9300 |
| Precision  | 0.9744 |
| Recall     | 0.8636 |
| F1-Score   | 0.9157 |
| **ROC-AUC**| **0.9888** |

La curva ROC con AUC = 0.9888 indica que el modelo discrimina muy bien entre préstamos aprobados y rechazados.

---

## Importancia de las variables

Los coeficientes de la regresión logística revelan qué variables tienen mayor influencia:

| Variable      | Coeficiente | Efecto |
|---------------|-------------|--------|
| `points`      | +1.51       | Mayor score → más probabilidad de aprobación |
| `credit_score`| +1.27       | Mejor historial crediticio → favorece la aprobación |
| `income`      | +0.33       | Mayor ingreso → efecto positivo |
| `loan_amount` | -0.26       | Mayor monto solicitado → reduce probabilidad |
| `years_employed` | +0.11    | Más antigüedad laboral → efecto positivo |

### 🔍 Análisis sin la variable `points`

Al eliminar `points` (que actúa como score interno derivado del target), el modelo con solo variables financieras reales mantiene métricas sólidas:

| Métrica   | Con `points` | Sin `points` |
|-----------|-------------|--------------|
| Accuracy  | 0.9300      | 0.8775       |
| F1-Score  | 0.9157      | 0.8444       |
| ROC-AUC   | 0.9888      | 0.9611       |

Esto confirma que **`credit_score`, `income` y `loan_amount` tienen poder predictivo real**, independientemente del score interno.

---

##Limitaciones del dataset

- La variable `points` contiene información casi determinística sobre la aprobación.  
- Las columnas `name` y `city` generan cientos de variables dummy con poco valor predictivo real.  
- El dataset es sintético, lo cual no refleja completamente un escenario productivo.

Aun así, funciona bien para demostrar un flujo completo de clasificación crediticia.

---

##Posibles mejoras futuras

- Entrenar modelos como **Random Forest**, **XGBoost** o **Árboles de Decisión**.  
- Aplicar **regularización L1 (Lasso)** para selección automática de variables.  
- Reducir dimensionalidad agrupando ciudades con baja frecuencia.  
- Evaluar con **validación cruzada k-fold** para mayor robustez.  
- Eliminar variables de baja utilidad (`name`, `city`) y reentrenar.

---

## 🛠 Librerías utilizadas
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

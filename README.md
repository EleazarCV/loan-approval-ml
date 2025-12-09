# Loan Approval Classification – Modelo de Regresión Logística

Este proyecto tiene como objetivo analizar una base de datos de solicitudes de préstamo y construir un modelo de clasificación que nos permita predecir si un préstamo será aprobado o no.  
El flujo completo incluye: exploración de datos, limpieza, codificación, modelado, evaluación y análisis de las variables más importantes.

El dataset utilizado proviene de Kaggle: *Loan Approval Dataset*.

---

## 📌 Objetivo del proyecto

El objetivo principal es estimar la probabilidad:

\[
P(\text{loan\_approved} = 1 \mid X)
\]

donde \( X \) representa variables como:

- income  
- credit_score  
- loan_amount  
- years_employed  
- points  
- city (codificada mediante one-hot encoding)

Este modelo nos ayuda a entender qué factores influyen más en la aprobación de un préstamo y nos permite predecir el resultado de nuevas solicitudes.

---

## 📂 Estructura del proyecto

loan-approval-ml/
│
├── data/
│ └── raw/ # Archivo original .csv del dataset
│
├── notebooks/
│ └── loan_approval_model.ipynb # Notebook principal con todo el análisis
│
├── src/
│ └── utils.py # Funciones auxiliares (si se requieren)
│
├── README.md # Este archivo
└── requirements.txt # Librerías necesarias


---

## 🔍 Exploración y preparación de datos

- Se revisaron tipos de variables, valores faltantes y estadísticas básicas.  
- Se identificaron columnas numéricas y categóricas.  
- La variable `city` se convirtió a variables dummy usando **one-hot encoding**, generando más de 1800 columnas.  
- La variable objetivo `loan_approved` se convirtió a 0/1 para ser compatible con el modelo.

---

## 🤖 Modelo utilizado: Regresión Logística

La Regresión Logística se eligió porque:

- Nos permite predecir probabilidades.  
- Es fácil de interpretar mediante coeficientes.  
- Funciona bien con datos categóricos codificados.  
- Es un modelo rápido y estable.

El modelo aprende una combinación lineal:

\[
z = b_0 + b_1 x_1 + \dots + b_n x_n
\]

y luego aplica la función sigmoide:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

para transformar ese valor en una probabilidad entre 0 y 1.

---

## 📊 Resultados del modelo

El modelo alcanzó métricas perfectas en el conjunto de prueba:

- **Accuracy : 1.0000**  
- **Precision: 1.0000**  
- **Recall   : 1.0000**  
- **F1-score : 1.0000**

Además, la matriz de confusión muestra que el modelo clasificó correctamente el 100 % de los casos.

Estos resultados indican que la variable `points` tiene un poder predictivo muy fuerte y prácticamente determina la aprobación del préstamo.

---

## 📈 Importancia de las variables

Se analizaron los coeficientes de la Regresión Logística:

- **`points`** es la variable más influyente del modelo.  
- Las variables `city_...` (resultado del one-hot encoding) muestran coeficientes muy pequeños.  
- Variables numéricas como income y credit_score tienen influencia, pero menor comparada con `points`.

---

## ⚠️ Limitaciones del dataset

- La variable `points` contiene información casi determinística sobre la aprobación.
- La columna `city` genera cientos de variables dummy con poco valor.
- El problema es demasiado “perfecto”, lo cual no refleja un escenario real.

Aun así, el dataset funciona bien para aprender el flujo completo de un proyecto de clasificación.

---

## 🚀 Posibles mejoras futuras

- Entrenar modelos como Random Forest, XGBoost o Árboles de Decisión.  
- Analizar el modelo eliminando `points` para ver si las variables financieras predicen por sí solas.  
- Reducir dimensionalidad eliminando ciudades raras con baja frecuencia.  
- Probar técnicas de regularización como L1/Lasso.

---

## 🛠 Librerías utilizadas

pandas
numpy
matplotlib
seaborn
scikit-learn

---

Este proyecto forma parte de mi portafolio personal, mostrando un flujo completo de análisis de datos y modelado aplicado a problemas de clasificación.

Home Credit Default Risk — Machine Learning

1. Descripción general

Este proyecto aborda un problema de clasificación binaria orientado a la predicción de incumplimiento crediticio (default) utilizando el dataset Home Credit Default Risk.
El objetivo es construir una solución completa de Machine Learning, desde el análisis de datos hasta el despliegue del modelo, siguiendo la metodología CRISP-DM.

El proyecto integra múltiples fuentes de información histórica, aplica ingeniería de características, compara modelos predictivos y expone el modelo final mediante una API REST.

2. Dataset

Se utilizan las siguientes tablas proporcionadas por Home Credit:

application

bureau

bureau_balance

previous_application

POS_CASH_balance

installments_payments

credit_card_balance

Las tablas secundarias fueron agregadas y consolidadas por cliente (SK_ID_CURR) para evitar la explosión de filas y construir una vista única por observación.

Dimensión final del dataset de modelado:

Registros: 307.511

Variables totales: 171

Variables creadas mediante feature engineering: 50

3. Metodología (CRISP-DM)
3.1 Business & Data Understanding

Comprensión del problema de riesgo crediticio

Análisis de granularidad y claves entre tablas

Validación de relaciones uno-a-muchos

Exploración del desbalance del target

Identificación de variables numéricas y categóricas

3.2 Data Preparation

Agregaciones jerárquicas (mensual → crédito → cliente)

Generación de variables estadísticas (sumas, promedios, conteos)

Manejo explícito de valores nulos e infinitos

Consolidación de features en una única tabla de modelado

Split estratificado en conjuntos de entrenamiento, validación y test

3.3 Modeling

Modelo baseline: Regresión Logística con imputación y escalado

Modelo campeón: Histogram Gradient Boosting utilizando variables numéricas

Manejo del desbalance mediante ponderación de clases (class_weight)

3.4 Evaluation

Métrica principal: ROC-AUC

Comparación directa entre baseline y modelo campeón

Evaluación final realizada exclusivamente sobre el conjunto de test

Análisis de precisión, recall y capacidad discriminativa

3.5 Deployment

Persistencia del modelo campeón entrenado

Exposición del modelo mediante una API REST desarrollada con FastAPI

Endpoint de predicción con documentación automática vía Swagger

4. Resultados
Baseline — Regresión Logística

ROC-AUC en validación: ~ 0.76

Buen recall para la clase minoritaria (default)

Precisión limitada debido al fuerte desbalance

Modelo campeón — Histogram Gradient Boosting

ROC-AUC en validación: ~ 0.77

ROC-AUC en test: 0.773

Mejor equilibrio entre precisión y recall

Buen nivel de generalización, sin evidencia de overfitting

5. Despliegue del modelo (API)

El modelo campeón fue desplegado mediante una API REST utilizando FastAPI.

Endpoints disponibles

GET /health → Verificación del estado del servicio

POST /predict → Predicción de probabilidad de default

Ejecución de la API

Desde la raíz del proyecto:

uvicorn 05_deployment.app:app --host 127.0.0.1 --port 8000

Documentación interactiva

Una vez levantado el servicio, la documentación Swagger está disponible en:

http://127.0.0.1:8000/docs

6. Estructura del proyecto
home-credit-risk
├── 01_data_understanding
├── 02_data_preparation
├── 03_modeling
├── 05_deployment
├── artifacts
├── data
│   ├── raw
│   └── processed
├── src
├── requirements.txt
└── README.md

7. Reproducibilidad
Requisitos

Python 3.11 o superior

Instalación de dependencias

Ejecutar en la raíz del proyecto:

pip install -r requirements.txt

Ejecución del pipeline

Los scripts están diseñados para ejecutarse en el siguiente orden:

Data Understanding

Data Preparation

Modeling y Evaluación

Deployment (API)

8. Limitaciones y mejoras futuras

Ajuste del umbral de decisión según costos de negocio

Optimización de hiperparámetros

Uso de modelos especializados como LightGBM o CatBoost

Análisis de importancia de variables (SHAP)

Calibración de probabilidades

9. Autores

Daniel Fernández

Giovanni Ortiz

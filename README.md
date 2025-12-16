# Home Credit Default Risk — Machine Learning Project

## 1. Descripción general
Este proyecto aborda un problema de **clasificación binaria** orientado a la predicción de **incumplimiento crediticio (default)** utilizando el dataset *Home Credit Default Risk*.  
El objetivo es construir un pipeline completo de Machine Learning siguiendo la metodología **CRISP-DM**, integrando múltiples fuentes de datos y evaluando distintos modelos predictivos.

---

## 2. Dataset
Se utilizan las siguientes tablas proporcionadas por Home Credit:

- application  
- bureau  
- bureau_balance  
- previous_application  
- POS_CASH_balance  
- installments_payments  
- credit_card_balance  

Las tablas secundarias fueron agregadas y consolidadas por cliente (`SK_ID_CURR`) para evitar explosión de filas.

**Dimensión final del dataset de modelado:**
- Registros: **307.511**
- Variables totales: **171**
- Variables creadas mediante feature engineering: **50**

---

## 3. Metodología (CRISP-DM)

### 3.1 Data Understanding
- Análisis de granularidad y claves entre tablas
- Validación de relaciones (uno-a-muchos)
- Identificación de variables numéricas y categóricas
- Exploración de valores nulos y distribución del target

### 3.2 Data Preparation
- Agregaciones jerárquicas (mensual → crédito → cliente)
- Manejo explícito de valores infinitos generados por divisiones
- Consolidación de features en una única tabla de modelado
- Split estratificado en conjuntos de entrenamiento, validación y test

### 3.3 Modeling
- **Modelo baseline:** Regresión Logística con preprocesamiento e imputación
- **Modelo campeón:** Histogram Gradient Boosting usando variables numéricas
- Manejo del desbalance mediante ponderación de clases

### 3.4 Evaluation
- Métrica principal: **ROC-AUC**
- Comparación entre baseline y modelo campeón
- Evaluación final realizada exclusivamente sobre el conjunto de test

---

## 4. Resultados

### Baseline (Regresión Logística)
- ROC-AUC en validación: ~ **0.76**
- Alto recall para la clase positiva (default), con baja precisión debido al desbalance

### Modelo campeón (Histogram Gradient Boosting)
- ROC-AUC en validación: ~ **0.77**
- ROC-AUC en test: **0.773**
- Mejor equilibrio entre precisión y recall
- Buen nivel de generalización (sin evidencia de overfitting)

---

## 5. Estructura del proyecto

home-credit-risk  
├── 01_data_understanding  
├── 02_data_preparation  
├── 03_modeling  
├── artifacts  
├── data  
│   ├── raw  
│   └── processed  
├── src  
├── requirements.txt  
└── README.md  

---

## 6. Reproducibilidad

### Requisitos
- Python 3.11 o superior

### Instalación de dependencias
Ejecutar el siguiente comando en la raíz del proyecto:

pip install -r requirements.txt

### Ejecución
Los scripts están diseñados para ejecutarse en el siguiente orden:
1. Data understanding  
2. Data preparation  
3. Modeling y evaluación  

---

## 7. Limitaciones y mejoras futuras
- Ajuste del umbral de decisión según costos de negocio
- Búsqueda de hiperparámetros (tuning)
- Uso de modelos especializados como LightGBM o CatBoost
- Análisis de importancia de variables (SHAP)
- Calibración de probabilidades

---

## 8. Autor
Daniel Fernandez y Giovanni Ortiz


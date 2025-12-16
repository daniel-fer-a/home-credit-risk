from pathlib import Path

# Root del proyecto = carpeta donde está este archivo /src/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Nombres de archivos esperados (ajusta solo si tus nombres difieren)
FILES = {
    "application": "application_.parquet",
    "bureau": "bureau.parquet",
    "bureau_balance": "bureau_balance.parquet",
    "previous_application": "previous_application.parquet",
    "pos_cash_balance": "POS_CASH_balance.parquet",
    "installments_payments": "installments_payments.parquet",
    "credit_card_balance": "credit_card_balance.parquet",
    "columns_description": "HomeCredit_columns_description.parquet",
}

# Llaves típicas del dataset
KEYS = {
    "SK_ID_CURR": "SK_ID_CURR",
    "SK_ID_PREV": "SK_ID_PREV",
    "SK_ID_BUREAU": "SK_ID_BUREAU",
}

# Target (si tu application lo trae; si no, se ignora)
TARGET_COL = "TARGET"
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import TARGET_COL

RANDOM_STATE = 42

def main():
    data_dir = PROJECT_ROOT / "data" / "processed"
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar train + valid
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")[TARGET_COL]

    X_valid = pd.read_parquet(data_dir / "X_valid.parquet")
    y_valid = pd.read_parquet(data_dir / "y_valid.parquet")[TARGET_COL]

    # Solo numéricas
    X_train = X_train.select_dtypes(include=["number"])
    X_valid = X_valid.select_dtypes(include=["number"])

    # Limpiar infinitos
    for df in (X_train, X_valid):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    X_full = pd.concat([X_train, X_valid], axis=0)
    y_full = pd.concat([y_train, y_valid], axis=0)

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
        class_weight={0: 1.0, 1: 5.0},
    )

    model.fit(X_full, y_full)

    # Guardar modelo + lista de columnas numéricas usadas
    model_path = out_dir / "champion_model.joblib"
    cols_path = out_dir / "champion_numeric_cols.joblib"

    joblib.dump(model, model_path)
    joblib.dump(list(X_full.columns), cols_path)

    print(f"[OK] Saved model to: {model_path}")
    print(f"[OK] Saved numeric columns to: {cols_path}")
    print(f"n_features={len(X_full.columns)}")

if __name__ == "__main__":
    main()

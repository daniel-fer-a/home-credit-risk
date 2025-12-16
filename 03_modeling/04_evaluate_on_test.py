import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

# ---- FIX IMPORT PATH ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
# -------------------------

from src.config import TARGET_COL


RANDOM_STATE = 42


def main():
    data_dir = PROJECT_ROOT / "data" / "processed"

    # Cargar splits
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")[TARGET_COL]

    X_valid = pd.read_parquet(data_dir / "X_valid.parquet")
    y_valid = pd.read_parquet(data_dir / "y_valid.parquet")[TARGET_COL]

    X_test = pd.read_parquet(data_dir / "X_test.parquet")
    y_test = pd.read_parquet(data_dir / "y_test.parquet")[TARGET_COL]

    # Usar solo numéricas
    X_train = X_train.select_dtypes(include=["number"])
    X_valid = X_valid.select_dtypes(include=["number"])
    X_test = X_test.select_dtypes(include=["number"])

    # Limpiar infinitos
    for df in [X_train, X_valid, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Entrenar con train + valid
    X_full = pd.concat([X_train, X_valid])
    y_full = pd.concat([y_train, y_valid])

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
        class_weight={0: 1.0, 1: 5.0},
    )

    model.fit(X_full, y_full)

    # Evaluar en test
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_test_proba)

    print(f"\nFINAL TEST ROC-AUC: {auc:.4f}")
    print("\nClassification report (TEST):")
    print(classification_report(y_test, y_test_pred, digits=4))


if __name__ == "__main__":
    main()

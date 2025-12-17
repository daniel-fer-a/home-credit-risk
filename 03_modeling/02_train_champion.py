import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

from src.config import TARGET_COL


RANDOM_STATE = 42


def main():
    data_dir = PROJECT_ROOT / "data" / "processed"
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    
    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet")[TARGET_COL]

    X_valid = pd.read_parquet(data_dir / "X_valid.parquet")
    y_valid = pd.read_parquet(data_dir / "y_valid.parquet")[TARGET_COL]

    
    X_train = X_train.select_dtypes(include=["number"])
    X_valid = X_valid.select_dtypes(include=["number"])

    print(f"Train shape (numeric only): {X_train.shape}")
    print(f"Valid shape (numeric only): {X_valid.shape}")

    
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_valid = X_valid.replace([np.inf, -np.inf], np.nan)

    
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
        class_weight={0: 1.0, 1: 5.0},  
    )

    
    model.fit(X_train, y_train)

    
    y_valid_proba = model.predict_proba(X_valid)[:, 1]
    y_valid_pred = (y_valid_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_valid, y_valid_proba)

    print(f"\nChampion ROC-AUC (valid): {auc:.4f}")
    print("\nClassification report (valid):")
    print(classification_report(y_valid, y_valid_pred, digits=4))

    
    metrics = {
        "model": "hist_gradient_boosting_numeric_only",
        "roc_auc_valid": float(auc),
        "class_weight": {0: 1.0, 1: 5.0},
        "n_features": int(X_train.shape[1]),
    }

    metrics_path = artifacts_dir / "metrics_champion.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Champion metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

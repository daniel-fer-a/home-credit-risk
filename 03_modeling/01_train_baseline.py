import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

    print(f"Train shape: {X_train.shape}")
    print(f"Valid shape: {X_valid.shape}")

    import numpy as np

    
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_valid = X_valid.replace([np.inf, -np.inf], np.nan)
	

    
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"Numeric features: {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")

    
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    
    pipe.fit(X_train, y_train)

    
    y_valid_proba = pipe.predict_proba(X_valid)[:, 1]
    y_valid_pred = pipe.predict(X_valid)

    auc = roc_auc_score(y_valid, y_valid_proba)

    print(f"\nBaseline ROC-AUC (valid): {auc:.4f}")
    print("\nClassification report (valid):")
    print(classification_report(y_valid, y_valid_pred, digits=4))

    
    metrics = {
        "model": "logistic_regression_baseline",
        "roc_auc_valid": float(auc),
        "class_weight": "balanced",
        "n_numeric_features": len(num_cols),
        "n_categorical_features": len(cat_cols),
    }

    metrics_path = artifacts_dir / "metrics_baseline.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[OK] Baseline metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

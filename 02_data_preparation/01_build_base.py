import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, require_columns
from src.config import KEYS, TARGET_COL


def main():
    # 1) Cargar application
    res = load_parquet("application")
    df = res.df.copy()

    print(f"Loaded application: shape={df.shape}")

    # 2) Verificar llave principal
    require_columns(df, [KEYS["SK_ID_CURR"]], df_name="application")

    # 3) Separar target si existe
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].astype("int8")
        X = df.drop(columns=[TARGET_COL])
        print("TARGET column found and separated.")
    else:
        y = None
        X = df
        print("No TARGET column found (test set).")

    # 4) Identificar columnas ID (no para modelar)
    id_cols = [c for c in X.columns if c.startswith("SK_ID")]
    print(f"ID columns detected: {id_cols}")

    # 5) Guardar base
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_path = out_dir / "base_X.parquet"
    X.to_parquet(X_path)

    print(f"[OK] Base features saved to: {X_path}")

    if y is not None:
        y_path = out_dir / "base_y.parquet"
        y.to_frame("TARGET").to_parquet(y_path)
        print(f"[OK] Target saved to: {y_path}")

    # 6) Guardar metadata simple
    meta = {
        "rows": X.shape[0],
        "n_features": X.shape[1],
        "id_columns": id_cols,
        "has_target": y is not None,
    }

    meta_path = out_dir / "base_metadata.json"
    pd.Series(meta).to_json(meta_path, indent=2)
    print(f"[OK] Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.config import KEYS
from src.io import load_parquet, require_columns


def load_processed(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed file: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {name}: shape={df.shape}")
    return df


def main():
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # 1) Cargar base
    base_X = load_processed(processed_dir / "base_X.parquet", "base_X")
    base_y_path = processed_dir / "base_y.parquet"
    base_y = load_processed(base_y_path, "base_y") if base_y_path.exists() else None

    require_columns(base_X, [KEYS["SK_ID_CURR"]], df_name="base_X")

    n_base = base_X.shape[0]

    # 2) Cargar features agregadas (todas son 1 fila por SK_ID_CURR)
    feat_files = [
        ("feat_bureau", processed_dir / "feat_bureau.parquet"),
        ("feat_bureau_balance", processed_dir / "feat_bureau_balance.parquet"),
        ("feat_previous_application", processed_dir / "feat_previous_application.parquet"),
        ("feat_pos_cash", processed_dir / "feat_pos_cash.parquet"),
        ("feat_installments", processed_dir / "feat_installments.parquet"),
        ("feat_credit_card", processed_dir / "feat_credit_card.parquet"),
    ]

    feats = []
    for name, path in feat_files:
        df = load_processed(path, name)
        require_columns(df, [KEYS["SK_ID_CURR"]], df_name=name)

        # sanity: debe ser único por cliente
        dups = int(df.duplicated(subset=[KEYS["SK_ID_CURR"]]).sum())
        if dups > 0:
            raise ValueError(f"[{name}] tiene {dups} SK_ID_CURR duplicados (debería ser 1 fila por cliente).")

        feats.append(df)

    # 3) Merge secuencial (left join para no perder filas)
    merged = base_X
    for df_feat in feats:
        merged = merged.merge(df_feat, on=KEYS["SK_ID_CURR"], how="left")
        print(f"After merge -> shape={merged.shape}")

    # 4) Check filas (no debe cambiar)
    if merged.shape[0] != n_base:
        raise ValueError(f"Row count changed after merges! base={n_base}, merged={merged.shape[0]}")

    # 5) Guardar tabla final
    out_dir = processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    X_out = out_dir / "model_X.parquet"
    merged.to_parquet(X_out)
    print(f"[OK] Model features saved to: {X_out}")

    if base_y is not None and "TARGET" in base_y.columns:
        y_out = out_dir / "model_y.parquet"
        base_y.to_parquet(y_out)
        print(f"[OK] Model target saved to: {y_out}")

    # 6) Guardar metadata merge
    meta = {
        "rows": int(merged.shape[0]),
        "n_features": int(merged.shape[1]),
        "n_added_features": int(merged.shape[1] - base_X.shape[1]),
    }

    meta_path = out_dir / "model_metadata.json"
    pd.Series(meta).to_json(meta_path, indent=2)
    print(f"[OK] Model metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()

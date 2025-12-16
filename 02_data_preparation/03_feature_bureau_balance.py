import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, require_columns
from src.config import KEYS


def main():
    # 1) Cargar bureau_balance
    res_bb = load_parquet("bureau_balance")
    bb = res_bb.df.copy()

    print(f"Loaded bureau_balance: shape={bb.shape}")

    require_columns(bb, [KEYS["SK_ID_BUREAU"], "MONTHS_BALANCE"], df_name="bureau_balance")

    # 2) Mapear STATUS a severidad numérica (si existe)
    status_map = {
        "C": 0,  # cerrado
        "X": 0,  # sin info
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
    }

    if "STATUS" in bb.columns:
        bb["status_severity"] = bb["STATUS"].map(status_map).fillna(0).astype("int8")
    else:
        bb["status_severity"] = 0

    # 3) Agregar por crédito (SK_ID_BUREAU)
    bb_agg_credit = (
        bb
        .groupby(KEYS["SK_ID_BUREAU"])
        .agg(
            bb_months_count=("MONTHS_BALANCE", "count"),
            bb_months_min=("MONTHS_BALANCE", "min"),
            bb_months_max=("MONTHS_BALANCE", "max"),
            bb_status_max=("status_severity", "max"),
            bb_status_mean=("status_severity", "mean"),
        )
        .reset_index()
    )

    print(f"bureau_balance aggregated per credit: shape={bb_agg_credit.shape}")

    # 4) Cargar bureau y unir
    res_bureau = load_parquet("bureau")
    bureau = res_bureau.df[[KEYS["SK_ID_BUREAU"], KEYS["SK_ID_CURR"]]].copy()

    require_columns(bureau, [KEYS["SK_ID_BUREAU"], KEYS["SK_ID_CURR"]], df_name="bureau")

    merged = bureau.merge(bb_agg_credit, on=KEYS["SK_ID_BUREAU"], how="left")

    print(f"Merged bureau + bureau_balance: shape={merged.shape}")

    # 5) Agregar por cliente
    cust_agg = (
        merged
        .groupby(KEYS["SK_ID_CURR"])
        .agg(
            bb_credits_count=("bb_months_count", "count"),
            bb_status_max=("bb_status_max", "max"),
            bb_status_mean=("bb_status_mean", "mean"),
            bb_months_min=("bb_months_min", "min"),
            bb_months_max=("bb_months_max", "max"),
        )
        .reset_index()
    )

    print(f"Final bureau_balance features per customer: shape={cust_agg.shape}")

    # 6) Guardar
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "feat_bureau_balance.parquet"
    cust_agg.to_parquet(out_path)

    print(f"[OK] Bureau balance features saved to: {out_path}")


if __name__ == "__main__":
    main()

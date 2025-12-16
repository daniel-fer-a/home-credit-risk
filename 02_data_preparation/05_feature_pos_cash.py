import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, require_columns
from src.config import KEYS


def main():
    # 1) Cargar POS_CASH_balance
    res_pos = load_parquet("pos_cash_balance")
    pos = res_pos.df.copy()

    print(f"Loaded POS_CASH_balance: shape={pos.shape}")

    require_columns(
        pos,
        [KEYS["SK_ID_PREV"], "MONTHS_BALANCE"],
        df_name="POS_CASH_balance"
    )

    # 2) Flags de atraso (si existe columna)
    if "SK_DPD" in pos.columns:
        pos["is_late"] = (pos["SK_DPD"] > 0).astype("int8")
        pos["late_days"] = pos["SK_DPD"]
    else:
        pos["is_late"] = 0
        pos["late_days"] = 0

    # 3) Agregación por solicitud previa
    pos_prev_agg = (
        pos
        .groupby(KEYS["SK_ID_PREV"])
        .agg(
            pos_months_count=("MONTHS_BALANCE", "count"),
            pos_months_min=("MONTHS_BALANCE", "min"),
            pos_months_max=("MONTHS_BALANCE", "max"),
            pos_late_ratio=("is_late", "mean"),
            pos_late_days_max=("late_days", "max"),
        )
        .reset_index()
    )

    print(f"POS aggregated per SK_ID_PREV: shape={pos_prev_agg.shape}")

    # 4) Cargar previous_application para mapear a cliente
    res_prev = load_parquet("previous_application")
    prev = res_prev.df[[KEYS["SK_ID_PREV"], KEYS["SK_ID_CURR"]]].copy()

    require_columns(prev, [KEYS["SK_ID_PREV"], KEYS["SK_ID_CURR"]], df_name="previous_application")

    merged = prev.merge(pos_prev_agg, on=KEYS["SK_ID_PREV"], how="left")

    print(f"Merged previous_application + POS: shape={merged.shape}")

    # 5) Agregar por cliente
    cust_agg = (
        merged
        .groupby(KEYS["SK_ID_CURR"])
        .agg(
            pos_prev_count=("pos_months_count", "count"),
            pos_late_ratio=("pos_late_ratio", "mean"),
            pos_late_days_max=("pos_late_days_max", "max"),
            pos_months_min=("pos_months_min", "min"),
            pos_months_max=("pos_months_max", "max"),
        )
        .reset_index()
    )

    print(f"Final POS features per customer: shape={cust_agg.shape}")

    # 6) Guardar
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "feat_pos_cash.parquet"
    cust_agg.to_parquet(out_path)

    print(f"[OK] POS_CASH features saved to: {out_path}")


if __name__ == "__main__":
    main()

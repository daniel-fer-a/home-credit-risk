import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, require_columns
from src.config import KEYS


def main():
    # 1) Cargar installments_payments
    res_inst = load_parquet("installments_payments")
    inst = res_inst.df.copy()

    print(f"Loaded installments_payments: shape={inst.shape}")

    require_columns(
        inst,
        [KEYS["SK_ID_PREV"], "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"],
        df_name="installments_payments"
    )

    # 2) Features a nivel cuota
    inst["days_delay"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["is_late"] = (inst["days_delay"] > 0).astype("int8")
    inst["is_early"] = (inst["days_delay"] < 0).astype("int8")

    if "AMT_INSTALMENT" in inst.columns and "AMT_PAYMENT" in inst.columns:
        inst["payment_diff"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]
    else:
        inst["payment_diff"] = 0.0

    # 3) Agregación por solicitud previa
    inst_prev_agg = (
        inst
        .groupby(KEYS["SK_ID_PREV"])
        .agg(
            inst_count=("days_delay", "count"),
            inst_late_ratio=("is_late", "mean"),
            inst_late_days_max=("days_delay", "max"),
            inst_late_days_mean=("days_delay", "mean"),
            inst_payment_diff_mean=("payment_diff", "mean"),
        )
        .reset_index()
    )

    print(f"Installments aggregated per SK_ID_PREV: shape={inst_prev_agg.shape}")

    # 4) Mapear a cliente vía previous_application
    res_prev = load_parquet("previous_application")
    prev = res_prev.df[[KEYS["SK_ID_PREV"], KEYS["SK_ID_CURR"]]].copy()

    require_columns(prev, [KEYS["SK_ID_PREV"], KEYS["SK_ID_CURR"]], df_name="previous_application")

    merged = prev.merge(inst_prev_agg, on=KEYS["SK_ID_PREV"], how="left")

    print(f"Merged previous_application + installments: shape={merged.shape}")

    # 5) Agregar por cliente
    cust_agg = (
        merged
        .groupby(KEYS["SK_ID_CURR"])
        .agg(
            inst_prev_count=("inst_count", "count"),
            inst_late_ratio=("inst_late_ratio", "mean"),
            inst_late_days_max=("inst_late_days_max", "max"),
            inst_late_days_mean=("inst_late_days_mean", "mean"),
            inst_payment_diff_mean=("inst_payment_diff_mean", "mean"),
        )
        .reset_index()
    )

    print(f"Final installments features per customer: shape={cust_agg.shape}")

    # 6) Guardar
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "feat_installments.parquet"
    cust_agg.to_parquet(out_path)

    print(f"[OK] Installments features saved to: {out_path}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, require_columns
from src.config import KEYS


def main():
    
    res_cc = load_parquet("credit_card_balance")
    cc = res_cc.df.copy()

    print(f"Loaded credit_card_balance: shape={cc.shape}")

    require_columns(
        cc,
        [KEYS["SK_ID_PREV"], "MONTHS_BALANCE"],
        df_name="credit_card_balance"
    )

    
    if "AMT_BALANCE" in cc.columns and "AMT_CREDIT_LIMIT_ACTUAL" in cc.columns:
        cc["utilization"] = (
            cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
        )
    else:
        cc["utilization"] = None

    if "SK_DPD" in cc.columns:
        cc["is_late"] = (cc["SK_DPD"] > 0).astype("int8")
        cc["late_days"] = cc["SK_DPD"]
    else:
        cc["is_late"] = 0
        cc["late_days"] = 0

    
    cc_prev_agg = (
        cc
        .groupby(KEYS["SK_ID_PREV"])
        .agg(
            cc_months_count=("MONTHS_BALANCE", "count"),
            cc_utilization_mean=("utilization", "mean"),
            cc_utilization_max=("utilization", "max"),
            cc_late_ratio=("is_late", "mean"),
            cc_late_days_max=("late_days", "max"),
        )
        .reset_index()
    )

    print(f"Credit card aggregated per SK_ID_PREV: shape={cc_prev_agg.shape}")

    
    res_prev = load_parquet("previous_application")
    prev = res_prev.df[[KEYS["SK_ID_PREV"], KEYS["SK_ID_CURR"]]].copy()

    require_columns(prev, [KEYS["SK_ID_PREV"], KEYS["SK_ID_CURR"]], df_name="previous_application")

    merged = prev.merge(cc_prev_agg, on=KEYS["SK_ID_PREV"], how="left")

    print(f"Merged previous_application + credit_card: shape={merged.shape}")

    
    cust_agg = (
        merged
        .groupby(KEYS["SK_ID_CURR"])
        .agg(
            cc_prev_count=("cc_months_count", "count"),
            cc_utilization_mean=("cc_utilization_mean", "mean"),
            cc_utilization_max=("cc_utilization_max", "max"),
            cc_late_ratio=("cc_late_ratio", "mean"),
            cc_late_days_max=("cc_late_days_max", "max"),
        )
        .reset_index()
    )

    print(f"Final credit card features per customer: shape={cust_agg.shape}")

    
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "feat_credit_card.parquet"
    cust_agg.to_parquet(out_path)

    print(f"[OK] Credit card features saved to: {out_path}")


if __name__ == "__main__":
    main()

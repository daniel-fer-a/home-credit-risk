import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, require_columns
from src.config import KEYS


def main():
    
    res = load_parquet("bureau")
    df = res.df.copy()

    print(f"Loaded bureau: shape={df.shape}")

    
    require_columns(df, [KEYS["SK_ID_CURR"]], df_name="bureau")

    
    numeric_cols = [
        c for c in [
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
            "AMT_CREDIT_MAX_OVERDUE",
            "DAYS_CREDIT",
            "DAYS_CREDIT_ENDDATE",
        ]
        if c in df.columns
    ]

    print(f"Numeric columns used: {numeric_cols}")

    
    agg_dict = {c: ["mean", "max", "sum"] for c in numeric_cols}
    agg_dict[KEYS["SK_ID_CURR"]] = ["count"]

    bureau_agg = (
        df
        .groupby(KEYS["SK_ID_CURR"])
        .agg(agg_dict)
    )

    
    bureau_agg.columns = [
        f"bureau_{col}_{stat}" if col != KEYS["SK_ID_CURR"] else "bureau_credit_count"
        for col, stat in bureau_agg.columns
    ]

    bureau_agg.reset_index(inplace=True)

    print(f"Aggregated bureau shape: {bureau_agg.shape}")

    
    if (
        "bureau_AMT_CREDIT_SUM_DEBT_sum" in bureau_agg.columns
        and "bureau_AMT_CREDIT_SUM_sum" in bureau_agg.columns
    ):
        bureau_agg["bureau_debt_to_credit_ratio"] = (
            bureau_agg["bureau_AMT_CREDIT_SUM_DEBT_sum"]
            / bureau_agg["bureau_AMT_CREDIT_SUM_sum"]
        )

    
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "feat_bureau.parquet"
    bureau_agg.to_parquet(out_path)

    print(f"[OK] Bureau features saved to: {out_path}")


if __name__ == "__main__":
    main()

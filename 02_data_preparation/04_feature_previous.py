import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, require_columns
from src.config import KEYS


def main():
    
    res = load_parquet("previous_application")
    df = res.df.copy()

    print(f"Loaded previous_application: shape={df.shape}")

    require_columns(df, [KEYS["SK_ID_CURR"], KEYS["SK_ID_PREV"]], df_name="previous_application")

    
    numeric_cols = [
        c for c in [
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "DAYS_DECISION",
        ]
        if c in df.columns
    ]

    print(f"Numeric columns used: {numeric_cols}")

    
    agg_dict = {c: ["mean", "max", "sum"] for c in numeric_cols}
    agg_dict[KEYS["SK_ID_PREV"]] = ["count"]

    prev_agg = (
        df
        .groupby(KEYS["SK_ID_CURR"])
        .agg(agg_dict)
    )

    
    prev_agg.columns = [
        f"prev_{col}_{stat}" if col != KEYS["SK_ID_PREV"] else "prev_app_count"
        for col, stat in prev_agg.columns
    ]

    prev_agg.reset_index(inplace=True)

    print(f"Aggregated previous_application shape: {prev_agg.shape}")

    
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "feat_previous_application.parquet"
    prev_agg.to_parquet(out_path)

    print(f"[OK] Previous application features saved to: {out_path}")


if __name__ == "__main__":
    main()

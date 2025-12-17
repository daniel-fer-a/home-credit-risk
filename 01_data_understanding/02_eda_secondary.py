import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, report_basic, require_columns
from src.config import KEYS


def key_profile(df: pd.DataFrame, key: str, df_name: str) -> dict:
    n = len(df)
    nunique = df[key].nunique(dropna=True) if key in df.columns else None
    missing = int(df[key].isna().sum()) if key in df.columns else None
    top_counts = (
        df[key].value_counts(dropna=True).head(5).to_dict()
        if key in df.columns else None
    )
    return {
        "df": df_name,
        "rows": n,
        "key": key,
        "n_unique": nunique,
        "missing": missing,
        "avg_rows_per_key": (n / nunique) if (nunique and nunique > 0) else None,
        "top_5_key_counts": top_counts,
    }


def main():
    tables = [
        ("bureau", [KEYS["SK_ID_CURR"], KEYS["SK_ID_BUREAU"]]),
        ("bureau_balance", [KEYS["SK_ID_BUREAU"]]),
        ("previous_application", [KEYS["SK_ID_CURR"], KEYS["SK_ID_PREV"]]),
        ("pos_cash_balance", [KEYS["SK_ID_PREV"]]),
        ("installments_payments", [KEYS["SK_ID_PREV"]]),
        ("credit_card_balance", [KEYS["SK_ID_PREV"]]),
    ]

    profiles = []

    print("=" * 80)
    print("SECONDARY TABLES — KEY / GRANULARITY CHECKS")
    print("=" * 80)

    for name, keys in tables:
        res = load_parquet(name)
        df = res.df

        print("\n" + "-" * 80)
        print(f"TABLE: {name} ({res.path.name})")
        print(report_basic(df, df_name=name))
        print("-" * 80)

        
        present_keys = [k for k in keys if k in df.columns]
        print(f"Keys present: {present_keys}")

        
        for k in present_keys:
            prof = key_profile(df, k, name)
            profiles.append(prof)
            print(f"Key profile [{name}] {k}: "
                  f"unique={prof['n_unique']}, missing={prof['missing']}, "
                  f"avg_rows_per_key={prof['avg_rows_per_key']:.2f}" if prof["avg_rows_per_key"] else
                  f"Key profile [{name}] {k}: unique={prof['n_unique']}, missing={prof['missing']}")

        
        if name == "bureau" and KEYS["SK_ID_BUREAU"] in df.columns:
            dup = int(df.duplicated(subset=[KEYS["SK_ID_BUREAU"]]).sum())
            print(f"Duplicated SK_ID_BUREAU rows in bureau: {dup}")

        if name == "previous_application" and KEYS["SK_ID_PREV"] in df.columns:
            dup = int(df.duplicated(subset=[KEYS["SK_ID_PREV"]]).sum())
            print(f"Duplicated SK_ID_PREV rows in previous_application: {dup}")

    
    out_dir = PROJECT_ROOT / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eda_secondary_key_profiles.json"
    pd.DataFrame(profiles).to_json(out_path, orient="records", indent=2)

    print("\n" + "=" * 80)
    print(f"[OK] Secondary key profiles saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

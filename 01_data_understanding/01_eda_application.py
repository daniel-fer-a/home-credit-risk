import sys
from pathlib import Path

# Asegurar import de src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.io import load_parquet, report_basic
from src.config import TARGET_COL


def main():
    # Cargar application
    res = load_parquet("application")
    df = res.df

    print("=" * 80)
    print("APPLICATION DATASET — OVERVIEW")
    print(report_basic(df, df_name="application"))
    print("=" * 80)

    # Tipos de datos
    print("\nDATA TYPES:")
    print(df.dtypes.value_counts())

    # Nulos
    null_ratio = df.isna().mean().sort_values(ascending=False)
    print("\nTOP 20 COLUMNS BY NULL RATIO:")
    print(null_ratio.head(20))

    # Target (si existe)
    if TARGET_COL in df.columns:
        print("\nTARGET DISTRIBUTION:")
        print(df[TARGET_COL].value_counts(normalize=True))

        print("\nTARGET COUNTS:")
        print(df[TARGET_COL].value_counts())
    else:
        print("\n[INFO] No TARGET column found (likely test set).")

    # Columnas ID
    id_cols = [c for c in df.columns if c.startswith("SK_ID")]
    print("\nID COLUMNS:")
    print(id_cols)

    # Columnas constantes
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    print("\nCONSTANT COLUMNS (nunique <= 1):")
    print(constant_cols)

    # Guardar resumen
    summary = {
        "shape": df.shape,
        "n_columns": df.shape[1],
        "id_columns": id_cols,
        "n_constant_columns": len(constant_cols),
        "top_null_columns": null_ratio.head(10).to_dict(),
    }

    out_dir = PROJECT_ROOT / "data" / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eda_application_summary.json"

    pd.Series(summary).to_json(out_path, indent=2)
    print(f"\n[OK] EDA summary saved to: {out_path}")


if __name__ == "__main__":
    main()

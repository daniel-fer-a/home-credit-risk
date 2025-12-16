import sys
from pathlib import Path

# Agregar la raíz del proyecto al PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.io import load_parquet, report_basic

NAMES = [
    "application",
    "bureau",
    "bureau_balance",
    "previous_application",
    "pos_cash_balance",
    "installments_payments",
    "credit_card_balance",
    "columns_description",
]

def main():
    for name in NAMES:
        res = load_parquet(name)
        print(f"OK load: {name} -> {res.path.name}")
        print(report_basic(res.df, df_name=name))
        print("-" * 60)

if __name__ == "__main__":
    main()

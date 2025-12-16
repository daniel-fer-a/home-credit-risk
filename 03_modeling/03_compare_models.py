import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main():
    with open(ARTIFACTS_DIR / "metrics_baseline.json") as f:
        baseline = json.load(f)

    with open(ARTIFACTS_DIR / "metrics_champion.json") as f:
        champion = json.load(f)

    comparison = pd.DataFrame([
        {
            "model": "Baseline (Logistic)",
            "roc_auc_valid": baseline["roc_auc_valid"],
        },
        {
            "model": "Champion (HGB)",
            "roc_auc_valid": champion["roc_auc_valid"],
        },
    ])

    print("\nMODEL COMPARISON (VALID SET)")
    print(comparison)

    out_path = ARTIFACTS_DIR / "model_comparison.csv"
    comparison.to_csv(out_path, index=False)

    print(f"\n[OK] Comparison saved to: {out_path}")


if __name__ == "__main__":
    main()

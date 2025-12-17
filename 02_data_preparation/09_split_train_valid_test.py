import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import TARGET_COL


RANDOM_STATE = 42


def main():
    processed_dir = PROJECT_ROOT / "data" / "processed"

    
    X = pd.read_parquet(processed_dir / "model_X.parquet")
    y = pd.read_parquet(processed_dir / "model_y.parquet")[TARGET_COL]

    print(f"Loaded model_X: {X.shape}")
    print(f"Loaded model_y: {y.shape}")

    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    
    def report_split(name, y_part):
        dist = y_part.value_counts(normalize=True)
        print(f"{name} size={len(y_part)} distribution:")
        print(dist.to_dict())

    report_split("TRAIN", y_train)
    report_split("VALID", y_valid)
    report_split("TEST", y_test)

    
    out_dir = processed_dir
    X_train.to_parquet(out_dir / "X_train.parquet")
    y_train.to_frame(TARGET_COL).to_parquet(out_dir / "y_train.parquet")

    X_valid.to_parquet(out_dir / "X_valid.parquet")
    y_valid.to_frame(TARGET_COL).to_parquet(out_dir / "y_valid.parquet")

    X_test.to_parquet(out_dir / "X_test.parquet")
    y_test.to_frame(TARGET_COL).to_parquet(out_dir / "y_test.parquet")

    
    meta = {
        "train_size": int(len(y_train)),
        "valid_size": int(len(y_valid)),
        "test_size": int(len(y_test)),
        "train_target_dist": y_train.value_counts(normalize=True).to_dict(),
        "valid_target_dist": y_valid.value_counts(normalize=True).to_dict(),
        "test_target_dist": y_test.value_counts(normalize=True).to_dict(),
    }

    meta_path = out_dir / "split_metadata.json"
    pd.Series(meta).to_json(meta_path, indent=2)
    print(f"[OK] Split metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()

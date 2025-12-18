"""
STEP 4: Split cleaned dataset into train / val / test

Input:
    data/processed/cleaned_multilabel_dataset.csv

Output:
    data/processed/train.csv
    data/processed/val.csv
    data/processed/test.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(".")
INPUT_FILE = BASE_DIR / "data/processed/cleaned_multilabel_dataset.csv"

TRAIN_OUT = BASE_DIR / "data/processed/train.csv"
VAL_OUT   = BASE_DIR / "data/processed/val.csv"
TEST_OUT  = BASE_DIR / "data/processed/test.csv"

def main():
    print("[INFO] Loading cleaned dataset...")
    df = pd.read_csv(INPUT_FILE)
    print("[INFO] Shape:", df.shape)

    # optional: shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
    )

    print("[INFO] Train size:", train_df.shape)
    print("[INFO] Val size  :", val_df.shape)
    print("[INFO] Test size :", test_df.shape)

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print("[OK] Saved splits:")
    print("  ->", TRAIN_OUT)
    print("  ->", VAL_OUT)
    print("  ->", TEST_OUT)

if __name__ == "__main__":
    main()

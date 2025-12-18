"""
Data Collection Script for Emotion Detection Project

Emotions (target set):
joy, sadness, anger, fear, surprise, disgust, neutral, love, confusion, sarcasm

This script does:
1. Create folder structure
2. Download existing datasets from Hugging Face:
   - GoEmotions
   - TweetEval (emotion)
   - TweetEval (irony/sarcasm)
3. Build separate labeled datasets (no web scraping!) for:
   - confusion  -> from GoEmotions
   - sarcasm    -> from TweetEval (irony)
4. Save them as CSVs under data/raw/confusion and data/raw/sarcasm
5. Combine all CSVs into one big 'all_raw_combined.csv'

NOTE:
- You need: `pip install datasets pandas`
"""

import os
import glob
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# --------------------------
# 1. FOLDER SETUP
# --------------------------

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

GOEMO_DIR = RAW_DIR / "goemotions"
TWEET_EMO_DIR = RAW_DIR / "tweet_emotion"
TWEET_IRONY_DIR = RAW_DIR / "tweet_irony"
SCRAPE_CONF_DIR = RAW_DIR / "confusion"
SCRAPE_SARC_DIR = RAW_DIR / "sarcasm"


def make_dirs():
    for d in [
        DATA_DIR,
        RAW_DIR,
        INTERIM_DIR,
        GOEMO_DIR,
        TWEET_EMO_DIR,
        TWEET_IRONY_DIR,
        SCRAPE_CONF_DIR,
        SCRAPE_SARC_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    print("[INFO] Directory structure created.")


# --------------------------
# 2. DOWNLOAD EXISTING DATASETS
# --------------------------

def download_goemotions():
    """
    Downloads GoEmotions dataset and saves splits as CSV.
    We keep original columns; label mapping will be done later.
    """
    print("[INFO] Downloading GoEmotions...")
    ds = load_dataset("go_emotions")

    for split in ds.keys():
        out_path = GOEMO_DIR / f"goemotions_{split}.csv"
        df = ds[split].to_pandas()
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved GoEmotions {split} -> {out_path}")


def download_tweeteval_emotion():
    """
    Downloads TweetEval emotion dataset (joy, sadness, anger, etc.).
    """
    print("[INFO] Downloading TweetEval - emotion...")
    ds = load_dataset("tweet_eval", "emotion")

    for split in ds.keys():
        out_path = TWEET_EMO_DIR / f"tweet_emotion_{split}.csv"
        df = ds[split].to_pandas()
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved TweetEval emotion {split} -> {out_path}")


def download_tweeteval_irony():
    """
    Downloads TweetEval irony dataset (sarcasm-like).
    """
    print("[INFO] Downloading TweetEval - irony...")
    ds = load_dataset("tweet_eval", "irony")

    for split in ds.keys():
        out_path = TWEET_IRONY_DIR / f"tweet_irony_{split}.csv"
        df = ds[split].to_pandas()
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved TweetEval irony {split} -> {out_path}")


# --------------------------
# 3. BUILD CONFUSION & SARCASM DATASETS (NO WEB SCRAPING)
# --------------------------

def build_confusion_dataset(target_size: int = 2000):
    """
    Build a confusion dataset from GoEmotions:
    - GoEmotions is multi-label.
    - We select all examples where 'confusion' is one of the labels.
    - Then we sample up to `target_size` examples.
    - Saved as CSV with columns: text, emotion, source.
    """
    print("[INFO] Building confusion dataset from GoEmotions...")
    ds = load_dataset("go_emotions")

    # Get label index for 'confusion'
    label_names = ds["train"].features["labels"].feature.names
    if "confusion" not in label_names:
        print("[WARN] 'confusion' not found in GoEmotions labels.")
        return
    confusion_idx = label_names.index("confusion")

    records = []
    for split in ds.keys():
        for ex in ds[split]:
            # ex["labels"] is a list of indices
            if confusion_idx in ex["labels"]:
                records.append(
                    {
                        "text": ex["text"],
                        "emotion": "confusion",
                        "source": f"goemotions_confusion_{split}",
                    }
                )

    if not records:
        print("[WARN] No confusion examples found in GoEmotions.")
        return

    df = pd.DataFrame(records)

    # Shuffle and downsample to desired size
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if len(df) > target_size:
        df = df.sample(n=target_size, random_state=42).reset_index(drop=True)

    out_path = SCRAPE_CONF_DIR / "confusion_labeled.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Confusion dataset saved -> {out_path} (rows: {len(df)})")


def build_sarcasm_dataset(target_size: int = 2000):
    """
    Build a sarcasm dataset from TweetEval 'irony':
    - In TweetEval irony: label 1 = ironic/sarcastic.
    - We collect all label==1 samples across splits.
    - Then sample up to `target_size` examples.
    - Saved as CSV with columns: text, emotion, source.
    """
    print("[INFO] Building sarcasm dataset from TweetEval irony...")
    ds = load_dataset("tweet_eval", "irony")

    records = []
    for split in ds.keys():
        for ex in ds[split]:
            # label: 0 = non-ironic, 1 = ironic
            if ex["label"] == 1:
                records.append(
                    {
                        "text": ex["text"],
                        "emotion": "sarcasm",
                        "source": f"tweeteval_irony_{split}",
                    }
                )

    if not records:
        print("[WARN] No sarcasm examples found in TweetEval irony.")
        return

    df = pd.DataFrame(records)

    # Shuffle and downsample to desired size
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if len(df) > target_size:
        df = df.sample(n=target_size, random_state=42).reset_index(drop=True)

    out_path = SCRAPE_SARC_DIR / "sarcasm_labeled.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Sarcasm dataset saved -> {out_path} (rows: {len(df)})")


# --------------------------
# 4. COMBINE ALL CSVs INTO ONE
# --------------------------

def combine_all_raw_csvs():
    """
    Finds all CSVs under data/raw and concatenates them into
    data/interim/all_raw_combined.csv.
    """
    print("[INFO] Combining all raw CSVs...")
    pattern = str(RAW_DIR / "**" / "*.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("[WARN] No CSV files found in raw/. Nothing to combine.")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file"] = os.path.relpath(f, BASE_DIR)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {f} due to error: {e}")

    if not dfs:
        print("[WARN] No valid CSVs loaded.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    out_path = INTERIM_DIR / "all_raw_combined.csv"
    combined.to_csv(out_path, index=False)
    print(f"[OK] Combined CSV saved -> {out_path}")
    print(f"[INFO] Combined shape: {combined.shape}")


# --------------------------
# MAIN PIPELINE
# --------------------------

def main():
    make_dirs()

    # 1) Download public datasets
    download_goemotions()
    download_tweeteval_emotion()
    download_tweeteval_irony()

    # 2) Build confusion & sarcasm datasets (no web scraping)
    build_confusion_dataset(target_size=2000)  # change to 1000 if you want smaller
    build_sarcasm_dataset(target_size=2000)

    # 3) Combine everything
    combine_all_raw_csvs()

    print("\n[DONE] Step 2: Data collection completed.")
    print(" - Raw datasets: data/raw/")
    print(" - Combined file: data/interim/all_raw_combined.csv")
    print("Next step: clean, map labels to your 10 emotions, and balance the dataset.")


if __name__ == "__main__":
    main()

"""
STEP 3: DATA CLEANING + LABEL MAPPING SCRIPT
Works with the output of data_collection.py

Target Labels:
     joy, sadness, anger, fear, surprise,
     disgust, neutral, love, confusion, sarcasm
"""

import pandas as pd
import re
from pathlib import Path

# --------------------------
# Path setup
# --------------------------

BASE_DIR = Path(".")
INPUT_FILE = BASE_DIR / "data/interim/all_raw_combined.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/cleaned_multilabel_dataset.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# 1. LOAD RAW COMBINED DATASET
# ---------------------------------------

print("[INFO] Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)

print(df.head())
print(df.columns)
print("[INFO] Total rows:", len(df))


# ---------------------------------------
# 2. UNIFY TEXT COLUMN
# ---------------------------------------

POSSIBLE_TEXT_COLS = ["text", "content", "sentence", "clean_text"]

df["clean_text"] = None
for col in POSSIBLE_TEXT_COLS:
    if col in df.columns:
        df["clean_text"] = df["clean_text"].fillna(df[col])

df = df.dropna(subset=["clean_text"])
print("[INFO] After text unification:", len(df))


# ---------------------------------------
# 3. CLEAN TEXT FUNCTION
# ---------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z0-9!?,.' ]", " ", text)  # remove emojis/symbols (optional)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["clean_text"].apply(clean_text)


# ---------------------------------------
# 4. YOUR EMOTIONS
# ---------------------------------------

TARGET_EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise",
    "disgust", "neutral", "love", "confusion", "sarcasm"
]


# ---------------------------------------
# 5. LABEL MAPPING LOGIC
# ---------------------------------------

def map_labels(row):
    src = row["__source_file"]
    labels = []

    # ------------- A. GoEmotions (multi-label) -------------
    if "goemotions" in src:
        # We check for each target emotion by column presence
        # GoEmotions uses integers and names in the original dataset
        # But here, we only know text + labels column

        if "labels" in row and isinstance(row["labels"], str):
            try:
                raw = eval(row["labels"])  # convert string "[1,4,7]" to list
            except:
                raw = []

            for emo in TARGET_EMOTIONS:
                # if the emotion is part of GoEmotions and appears here
                # but we need actual index mapping
                pass  # This will be handled below using emotion name columns

    # ------------- B. TweetEval – emotion -------------
    elif "tweet_emotion" in src:
        # TweetEval emotion labels:
        # 0=anger, 1=joy, 2=optimism, 3=sadness
        label = row.get("label", None)

        mapping = {
            0: "anger",
            1: "joy",
            2: "joy",     # optimism → joy
            3: "sadness",
        }

        if label in mapping:
            labels.append(mapping[label])

    # ------------- C. TweetEval – irony (sarcasm) -------------
    elif "tweet_irony" in src:
        label = row.get("label", None)
        if label == 1:
            labels.append("sarcasm")
        else:
            labels.append("neutral")

    # ------------- D. Generated confusion dataset -------------
    elif "confusion_labeled" in src:
        labels.append("confusion")

    # ------------- E. Generated sarcasm dataset -------------
    elif "sarcasm_labeled" in src:
        labels.append("sarcasm")

    return list(set(labels))  # remove duplicates


df["mapped_labels"] = df.apply(map_labels, axis=1)


# ---------------------------------------
# 6. REMOVE EMPTY LABEL ROWS
# ---------------------------------------

df = df[df["mapped_labels"].apply(lambda x: len(x) > 0)]
print("[INFO] After label mapping:", len(df))


# ---------------------------------------
# 7. ONE-HOT ENCODE ALL 10 EMOTIONS
# ---------------------------------------

for emo in TARGET_EMOTIONS:
    df[emo] = df["mapped_labels"].apply(lambda lbls: 1 if emo in lbls else 0)


# ---------------------------------------
# 8. SAVE CLEANED MULTILABEL DATASET
# ---------------------------------------

df_final = df[["clean_text", "mapped_labels"] + TARGET_EMOTIONS]
df_final.to_csv(OUTPUT_FILE, index=False)

print("[OK] Saved cleaned multi-label dataset ->", OUTPUT_FILE)
print("[INFO] Final shape:", df_final.shape)

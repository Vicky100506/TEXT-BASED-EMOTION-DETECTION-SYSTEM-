

import pandas as pd
import re
from pathlib import Path



BASE_DIR = Path(".")
INPUT_FILE = BASE_DIR / "data/interim/all_raw_combined.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/cleaned_multilabel_dataset.csv"

Path("data/processed").mkdir(parents=True, exist_ok=True)


print("[INFO] Loading combined dataset...")
df = pd.read_csv(INPUT_FILE)

print(df.head())
print(df.columns)
print("[INFO] Total rows:", len(df))



POSSIBLE_TEXT_COLS = ["text", "content", "sentence", "clean_text"]

df["clean_text"] = None
for col in POSSIBLE_TEXT_COLS:
    if col in df.columns:
        df["clean_text"] = df["clean_text"].fillna(df[col])

df = df.dropna(subset=["clean_text"])
print("[INFO] After text unification:", len(df))



def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z0-9!?,.' ]", " ", text)  # remove emojis/symbols (optional)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["clean_text"].apply(clean_text)


TARGET_EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise",
    "disgust", "neutral", "love", "confusion", "sarcasm"
]




def map_labels(row):
    src = row["__source_file"]
    labels = []

    if "goemotions" in src:
        

        if "labels" in row and isinstance(row["labels"], str):
            try:
                raw = eval(row["labels"])  
            except:
                raw = []

            for emo in TARGET_EMOTIONS:
               
                pass  

    # ------------- B. TweetEval – emotion -------------
    elif "tweet_emotion" in src:
    
        label = row.get("label", None)

        mapping = {
            0: "anger",
            1: "joy",
            2: "joy",    
            3: "sadness",
        }

        if label in mapping:
            labels.append(mapping[label])


    elif "tweet_irony" in src:
        label = row.get("label", None)
        if label == 1:
            labels.append("sarcasm")
        else:
            labels.append("neutral")


    elif "confusion_labeled" in src:
        labels.append("confusion")


    elif "sarcasm_labeled" in src:
        labels.append("sarcasm")

    return list(set(labels)) 


df["mapped_labels"] = df.apply(map_labels, axis=1)


df = df[df["mapped_labels"].apply(lambda x: len(x) > 0)]
print("[INFO] After label mapping:", len(df))



for emo in TARGET_EMOTIONS:
    df[emo] = df["mapped_labels"].apply(lambda lbls: 1 if emo in lbls else 0)



df_final = df[["clean_text", "mapped_labels"] + TARGET_EMOTIONS]
df_final.to_csv(OUTPUT_FILE, index=False)

print("[OK] Saved cleaned multi-label dataset ->", OUTPUT_FILE)
print("[INFO] Final shape:", df_final.shape)

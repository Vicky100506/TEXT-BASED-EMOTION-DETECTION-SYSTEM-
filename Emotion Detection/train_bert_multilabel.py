# =========================================================
# 📦 IMPORTS
# =========================================================
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import precision_recall_fscore_support


# =========================================================
# 🔹 EMOTION MAPPING
# =========================================================
EMOTION2ID = {
    "joy": 0, "sadness": 1, "anger": 2, "fear": 3, "love": 4,
    "surprise": 5, "sarcasm": 6, "disgust": 7, "confusion": 8, "neutral": 9
}

ID2EMOTION = {v: k for k, v in EMOTION2ID.items()}
NUM_LABELS = len(EMOTION2ID)


# =========================================================
# ⚙️ CONFIG
# =========================================================
MAX_LENGTH = 128          # 🔥 faster on CPU
BATCH_SIZE = 16           # 🔥 accelerate handles batching
EPOCHS = 3
LR = 2e-5
THRESHOLD = 0.5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================================================
# 📁 PROJECT ROOT
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"


# =========================================================
# 📄 LOAD DATA
# =========================================================
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

TEXT_COL = next(c for c in ["clean_text", "text"] if c in train_df.columns)


# =========================================================
# 🔁 LABEL ENCODING
# =========================================================
def to_multi_hot(raw):
    vec = [0.0] * NUM_LABELS

    if isinstance(raw, str):
        raw = ast.literal_eval(raw) if raw.startswith("[") else [raw]

    for item in raw:
        idx = int(item) if str(item).isdigit() else EMOTION2ID[item.lower()]
        vec[idx] = 1.0

    return vec


def build_dataset(df):
    return Dataset.from_dict({
        "text": df[TEXT_COL].astype(str).tolist(),
        "labels": df["mapped_labels"].apply(to_multi_hot).tolist()
    })


train_ds = build_dataset(train_df)
val_ds = build_dataset(val_df)


# =========================================================
# 🔤 TOKENIZATION
# =========================================================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# =========================================================
# 🤖 MODEL
# =========================================================
config = AutoConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
    id2label=ID2EMOTION,
    label2id=EMOTION2ID
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    config=config
)


# =========================================================
# 📈 METRICS
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs >= THRESHOLD).int()

    f1_micro = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )[2]

    return {"f1_micro": f1_micro}


# =========================================================
# 🚀 TRAINING (ACCELERATE BACKEND)
# =========================================================
training_args = TrainingArguments(
    output_dir="models/bert_multilabel_accelerate",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

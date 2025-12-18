# test_local_model.py
# Requirements:
# pip install transformers datasets scikit-learn pandas numpy tqdm matplotlib

import os
import ast
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    multilabel_confusion_matrix
)

# --------------------- USER CONFIG ---------------------
# Set the correct local path to your final model folder (from screenshot)
MODEL_PATH = r"models/bert_multilabel_emotions/final"

# Path to the CSV test file (from your 'data/processed' folder)
TEST_CSV = r"data/processed/test.csv"   # change to val.csv if needed

OUTPUT_CSV = "predictions.csv"
SUMMARY_JSON = "eval_summary.json"
F1_PLOT = "per_class_f1.png"

BATCH_SIZE = 32
MAX_LENGTH = 256
THRESHOLD = 0.5   # for multi-label sigmoid -> predicted=prob>=THRESHOLD

# Optional human-readable label names (list length must equal model.num_labels)
LABEL_MAP = None   # e.g. ["joy","anger","sadness", ...] or None to use numeric indices
# -------------------------------------------------------

# sanity checks
assert os.path.isdir(MODEL_PATH), f"Model folder not found: {MODEL_PATH}"
assert os.path.exists(TEST_CSV), f"Test CSV not found: {TEST_CSV}"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load tokenizer + model (local files only to avoid HF Hub)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, config=config, local_files_only=True)
model.to(device)
model.eval()

num_labels = getattr(config, "num_labels", model.config.num_labels)
print("Model num_labels:", num_labels)

# Load CSV with pandas
df = pd.read_csv(TEST_CSV)
print("Test CSV columns:", list(df.columns))
print("Number of examples:", len(df))

# Detect sentence/text column
TEXT_CANDIDATES = ["text", "sentence", "content", "utterance", "review"]
text_col = None
for c in TEXT_CANDIDATES:
    if c in df.columns:
        text_col = c
        break
if text_col is None:
    # fallback: pick first string column
    for c in df.columns:
        if df[c].dtype == object:
            text_col = c
            break
if text_col is None:
    raise ValueError("Could not detect a text column in the CSV. Rename your text column to 'text' or 'sentence'.")

print("Using text column:", text_col)

# Detect label format:
# 1) single integer column named 'label' or 'labels'
# 2) a column with stringified list like "[0,2]"
# 3) multiple binary one-hot columns (detect if many columns of 0/1)
label_col = None
for c in ["label", "labels", "target"]:
    if c in df.columns:
        label_col = c
        break

# helper to test for one-hot columns: numeric and only 0/1 and at least 2 columns look like that
one_hot_cols = []
for c in df.columns:
    if c == text_col:
        continue
    # skip non-numeric columns for one-hot detection
    if pd.api.types.is_numeric_dtype(df[c]):
        vals = df[c].dropna().unique()
        if set(vals).issubset({0,1}):
            one_hot_cols.append(c)

# If found many one-hot columns and no explicit label_col, assume multi-hot per-column format
if (len(one_hot_cols) >= 2) and (label_col is None):
    print("Detected one-hot label columns:", one_hot_cols[:10], "...")
    label_format = "one_hot_cols"
else:
    # if label_col present, inspect a few values to detect stringified lists
    if label_col is not None:
        sample = df[label_col].dropna().astype(str).iloc[0]
        # check if looks like a list
        if sample.strip().startswith("[") and sample.strip().endswith("]"):
            label_format = "string_list"
        else:
            # could be integer or comma-separated
            try:
                int(sample)
                label_format = "single_int"
            except Exception:
                # maybe comma separated "0,2"
                if "," in sample:
                    label_format = "string_list_comma"
                else:
                    label_format = "unknown"
    else:
        label_format = "unknown"

print("Detected label format:", label_format)

# Utility: convert a raw label value into a multi-hot vector (length num_labels)
def to_multi_hot_from_raw(raw):
    # raw may be:
    # - list/tuple/np.ndarray of indices
    # - string like "[0,2]" or "0,2"
    # - int
    # - already one-hot list or comma-separated "0,1,0,..."
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.array(raw)
        # if numeric one-hot
        if arr.dtype in (np.int64, np.int32, np.float64) and arr.size == num_labels:
            return (arr != 0).astype(int)
        # else treat as index list
        vec = np.zeros(num_labels, dtype=int)
        for r in arr:
            vec[int(r)] = 1
        return vec

    # strings
    if isinstance(raw, str):
        s = raw.strip()
        # JSON-like list
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return to_multi_hot_from_raw(parsed)
            except Exception:
                pass
        # comma-separated indices
        if "," in s:
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            # if there are num_labels parts and they look like 0/1 -> treat as one-hot
            if len(parts) == num_labels and set(parts).issubset(set(["0","1","0.0","1.0"])):
                arr = np.array([int(float(x)) for x in parts], dtype=int)
                return arr
            # else treat as indices
            vec = np.zeros(num_labels, dtype=int)
            for p in parts:
                try:
                    vec[int(p)] = 1
                except:
                    pass
            return vec
        # single integer string
        try:
            idx = int(float(s))
            vec = np.zeros(num_labels, dtype=int)
            vec[idx] = 1
            return vec
        except Exception:
            pass

    # numeric
    if isinstance(raw, (int, np.integer)):
        vec = np.zeros(num_labels, dtype=int)
        vec[int(raw)] = 1
        return vec

    # fallback (empty)
    return np.zeros(num_labels, dtype=int)

# Build lists of sentences and true multi-hot labels
sentences = []
true_multi = []

if label_format == "one_hot_cols":
    # use one_hot_cols as label columns
    label_columns = one_hot_cols
    for _, row in df.iterrows():
        sentences.append(str(row[text_col]))
        vec = np.array([int(row[c]) for c in label_columns], dtype=int)
        # if length differs from model num_labels, try to adjust (pad/truncate)
        if len(vec) != num_labels:
            # try to align by assuming label columns correspond to first len(vec)
            tmp = np.zeros(num_labels, dtype=int)
            tmp[:len(vec)] = vec[:min(len(vec), num_labels)]
            vec = tmp
        true_multi.append(vec)
    # if label columns used, no label_map provided: create names
    if LABEL_MAP is None:
        LABEL_MAP = label_columns

elif label_format in ("string_list", "string_list_comma"):
    for _, row in df.iterrows():
        sentences.append(str(row[text_col]))
        raw = row[label_col]
        vec = to_multi_hot_from_raw(raw)
        true_multi.append(vec)

elif label_format == "single_int":
    # single-label dataset: store as single-int and we will treat later
    sentences = df[text_col].astype(str).tolist()
    single_labels = df[label_col].astype(int).tolist()
    # convert to multi-hot for unified processing but mark single_label flag
    true_multi = []
    for lbl in single_labels:
        vec = np.zeros(num_labels, dtype=int)
        vec[int(lbl)] = 1
        true_multi.append(vec)

else:
    # unknown: attempt best-effort: look for 'labels' column and try to parse
    if label_col:
        for _, row in df.iterrows():
            sentences.append(str(row[text_col]))
            raw = row[label_col]
            vec = to_multi_hot_from_raw(raw)
            true_multi.append(vec)
    else:
        raise ValueError("Could not detect labels in your CSV. Please ensure you have label column(s).")

true_multi = np.vstack(true_multi)   # shape (N, num_labels)
N = len(sentences)
print("Prepared", N, "examples. True label matrix shape:", true_multi.shape)

# Determine if this is single-label classification (one-hot vectors with exactly one 1 per row)
ones_per_row = true_multi.sum(axis=1)
is_single_label = np.all((ones_per_row == 1))
print("Is single-label dataset detected?", is_single_label)

# Batched inference
all_probs = np.zeros((N, num_labels), dtype=float)
all_preds_bin = np.zeros((N, num_labels), dtype=int)

for start in tqdm(range(0, N, BATCH_SIZE), desc="Inferring"):
    end = min(N, start + BATCH_SIZE)
    batch_texts = sentences[start:end]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()  # (batch, num_labels) or (batch,1)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)

        # If single-label dataset and model was trained as single-label (softmax), using softmax+argmax
        # However, we can't reliably know model's problem_type, so:
        # - compute sigmoid probabilities for multi-label interpretation
        probs = 1.0 / (1.0 + np.exp(-logits))
        # if the model is actually softmax-trained, softmax on logits might be more appropriate.
        # To be safe, also compute softmax and decide later when is_single_label:
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        softmax_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    all_probs[start:end] = probs
    if is_single_label:
        # use softmax -> single predicted index
        pred_idx = np.argmax(softmax_probs, axis=1)
        tmp = np.zeros((len(pred_idx), num_labels), dtype=int)
        tmp[np.arange(len(pred_idx)), pred_idx] = 1
        all_preds_bin[start:end] = tmp
    else:
        # multi-label: threshold sigmoid probs
        all_preds_bin[start:end] = (probs >= THRESHOLD).astype(int)

# Evaluate
if is_single_label:
    # Convert multi-hot to single integer for metrics
    y_true = np.argmax(true_multi, axis=1)
    y_pred = np.argmax(all_probs, axis=1)  # using softmax-like argmax performed above
    acc = accuracy_score(y_true, y_pred)
    print("\n=== Single-label evaluation ===")
    print("Accuracy:", acc)
    print("\nClassification report:\n")
    if LABEL_MAP:
        print(classification_report(y_true, y_pred, target_names=LABEL_MAP, zero_division=0))
    else:
        print(classification_report(y_true, y_pred, zero_division=0))
    # Save summary
    summary = {"task": "single-label", "accuracy": float(acc)}

else:
    # Multi-label metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(true_multi, all_preds_bin, average="micro", zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_multi, all_preds_bin, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_multi, all_preds_bin, average="weighted", zero_division=0)
    per_prec, per_rec, per_f1, _ = precision_recall_fscore_support(true_multi, all_preds_bin, average=None, zero_division=0)

    print("\n=== Multi-label evaluation ===")
    print(f"Micro  - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"Macro  - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
    print(f"Weighted - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")

    # per class
    print("\nPer-class F1:")
    for i in range(num_labels):
        label_name = LABEL_MAP[i] if (LABEL_MAP and i < len(LABEL_MAP)) else str(i)
        print(f"  {i} ({label_name}): P={per_prec[i]:.4f} R={per_rec[i]:.4f} F1={per_f1[i]:.4f}")

    mcm = multilabel_confusion_matrix(true_multi, all_preds_bin)  # per-class confusion matrices
    summary = {
        "task": "multi-label",
        "micro": {"precision": precision_micro, "recall": recall_micro, "f1": f1_micro},
        "macro": {"precision": precision_macro, "recall": recall_macro, "f1": f1_macro},
        "weighted": {"precision": precision_weighted, "recall": recall_weighted, "f1": f1_weighted},
        "per_class": []
    }
    for i in range(num_labels):
        label_name = LABEL_MAP[i] if (LABEL_MAP and i < len(LABEL_MAP)) else str(i)
        tn, fp, fn, tp = int(mcm[i,0,0]), int(mcm[i,0,1]), int(mcm[i,1,0]), int(mcm[i,1,1])
        summary["per_class"].append({
            "label_index": i,
            "label_name": label_name,
            "precision": float(per_prec[i]),
            "recall": float(per_rec[i]),
            "f1": float(per_f1[i]),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })

# Save predictions to CSV
rows = []
for sent, true_vec, pred_vec, prob_vec in zip(sentences, true_multi, all_preds_bin, all_probs):
    true_indices = [int(i) for i, v in enumerate(true_vec) if v == 1]
    pred_indices = [int(i) for i, v in enumerate(pred_vec) if v == 1]
    if LABEL_MAP:
        true_names = [LABEL_MAP[i] for i in true_indices]
        pred_names = [LABEL_MAP[i] for i in pred_indices]
    else:
        true_names = true_indices
        pred_names = pred_indices
    prob_str = ",".join([f"{p:.4f}" for p in prob_vec])
    rows.append({
        "sentence": sent,
        "true_labels": json.dumps(true_indices, ensure_ascii=False),
        "pred_labels": json.dumps(pred_indices, ensure_ascii=False),
        "true_label_names": json.dumps(true_names, ensure_ascii=False),
        "pred_label_names": json.dumps(pred_names, ensure_ascii=False),
        "pred_probs": prob_str
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved predictions to {OUTPUT_CSV}")

# Save summary JSON
with open(SUMMARY_JSON, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2, ensure_ascii=False)
print("Saved evaluation summary to", SUMMARY_JSON)

# Plot per-class F1 (works for multi-label; for single-label we can plot class-wise f1 from classification_report)
if not is_single_label:
    per_f1 = [c["f1"] for c in summary["per_class"]]
    labels = [c["label_name"] for c in summary["per_class"]]
    plt.figure(figsize=(8, max(4, 0.3 * len(labels))))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, per_f1, align="center")
    plt.yticks(y_pos, labels)
    plt.xlabel("F1 score")
    plt.xlim(0, 1)
    plt.title("Per-class F1")
    for i, v in enumerate(per_f1):
        plt.text(v + 0.01, i, f"{v:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(F1_PLOT, dpi=200, bbox_inches="tight")
    print("Saved per-class F1 plot to", F1_PLOT)
else:
    print("Single-label task — per-class plot omitted. Use classification_report output above for class metrics.")

print("\nDone.")

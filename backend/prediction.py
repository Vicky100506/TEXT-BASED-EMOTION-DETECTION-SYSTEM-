import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models" / "bert_multilabel_emotions" / "final_10labels"

print("Loading model from:", MODEL_DIR)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model.eval()

LABELS = list(model.config.id2label.values())


def postprocess(scores: dict, text: str) -> dict:
    scores = scores.copy()
    text_l = text.lower()
    length = len(text.split())


    if any(w in text_l for w in [
        "happy", "sad", "angry", "scared", "love",
        "disgust", "surprise", "confused"
    ]):
        scores["neutral"] *= 0.3


    sarcasm_cues = {"yeah", "sure", "as if", "great", "just perfect"}
    if scores.get("sarcasm", 0) > 0.35 and not any(c in text_l for c in sarcasm_cues):
        scores["sarcasm"] *= 0.3

    if length <= 3:
        scores["sarcasm"] *= 0.4


    if any(w in text_l for w in ["ew", "gross", "disgusting", "yuck"]):
        scores["disgust"] += 0.25
        scores["anger"] *= 0.5


    if any(w in text_l for w in ["love you", "i love", "lovely"]):
        scores["love"] += 0.30


    if any(w in text_l for w in ["scared", "afraid", "terrified", "panic", "worried"]):
        scores["fear"] += 0.30


    if any(w in text_l for w in ["surprised", "shocked", "unexpected"]):
        scores["surprise"] += 0.30


    if any(w in text_l for w in ["happy", "funny", "lol", "haha", "awesome"]):
        scores["joy"] += 0.25


    if "?" in text_l or any(w in text_l for w in ["what", "why", "how", "huh"]):
        scores["confusion"] += 0.15


    for e in scores:
        scores[e] = max(0.0, min(scores[e], 1.0))

    return scores


def predict(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0].tolist()

    raw_scores = dict(zip(LABELS, probs))
    scores = postprocess(raw_scores, text)


    min_conf = 0.18 if len(text.split()) < 6 else 0.25


    sorted_emotions = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    predictions = {
        emotion: round(score, 3)
        for emotion, score in sorted_emotions[:2]
        if score >= min_conf
    }

    return predictions

if __name__ == "__main__":
    tests = [
        "I am happy",
        "EW DISGUSTING",
        "love you",
        "You are funny bro",
        "I am surprised",
        "I love you but I'm scared",
        "I feel sick just thinking about it",
        "English or Spanish?",
        "Great, another meeting"
    ]

    for t in tests:
        print(f"\nText: {t}")
        print(predict(t))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 🔹 Path to your trained model
MODEL_DIR = "models/bert_multilabel_emotions/final_10labels"

# 🔹 Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict_debug(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits)[0]

    return {
        model.config.id2label[i]: round(float(probs[i]), 3)
        for i in range(len(probs))
    }


# 🔍 TEST
print(predict_debug(
    "I am full of Happiness but my uncle is sad because he fell down the stairs"
))

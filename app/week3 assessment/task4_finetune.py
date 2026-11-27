import inspect
import re
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, pipeline)

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "models/distilbert-feedback"
SEED = 42
torch.manual_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sentences = ["The delivery was fast and the packaging was great!",
             "Where is my refund?"]
enc = tokenizer(sentences, padding=True) 

print("=== 1. DistilBERT tokenizer ===")
for i, s in enumerate(sentences):
    print(f"\nSentence : {s}")
    print("Tokens   :", tokenizer.convert_ids_to_tokens(enc["input_ids"][i]))
    print("Input IDs:", enc["input_ids"][i])
    print("Attn mask:", enc["attention_mask"][i])

print("\n=== 2. Pipelines ===")
sent_pipe = pipeline("sentiment-analysis")         
for s in ["I love this phone, the battery is amazing.",
          "The support team was rude and unhelpful.",
          "The parcel arrived on Monday."]:
    print(s, "->", sent_pipe(s)[0])

ner_pipe = pipeline("ner", aggregation_strategy="simple") 
for s in ["Sundar Pichai announced new Google products in Mumbai.",
          "I bought a Samsung phone from Amazon in Chennai.",
          "Tim Cook visited Apple stores in London last week."]:
    print(s, "->", [(e["word"], e["entity_group"], round(float(e["score"]), 2)) for e in ner_pipe(s)])

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
labels = sorted(train_df["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
print("\nLabels:", label2id)


class TextDataset(Dataset):
    def __init__(self, df):
        self.enc = tokenizer(df["text"].astype(str).tolist(), truncation=True, max_length=64)
        self.labels = df["label"].map(label2id).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


train_ds, test_ds = TextDataset(train_df), TextDataset(test_df)


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels), id2label=id2label, label2id=label2id)

eval_key = "eval_strategy" if "eval_strategy" in inspect.signature(
    TrainingArguments.__init__).parameters else "evaluation_strategy"

args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="no",       
    report_to="none",
    seed=SEED,
    **{eval_key: "epoch"},
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                  data_collator=DataCollatorWithPadding(tokenizer),
                  compute_metrics=compute_metrics)

print("\n=== 3. Fine-tuning ===")
t0 = time.time()
trainer.train()
bert_train_time = time.time() - t0

t0 = time.time()
out = trainer.predict(test_ds)
bert_infer_time = time.time() - t0
bert_metrics = compute_metrics((out.predictions, out.label_ids))
print("DistilBERT:", bert_metrics)

trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved model -> {SAVE_DIR}")

def clean_text(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(t).lower())).strip()

tfidf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                  ("clf", LogisticRegression(max_iter=1000))])
t0 = time.time()
tfidf.fit(train_df["text"].apply(clean_text), train_df["label"])
tf_train_time = time.time() - t0

t0 = time.time()
tf_pred = tfidf.predict(test_df["text"].apply(clean_text))
tf_infer_time = time.time() - t0
tf_acc = accuracy_score(test_df["label"], tf_pred)
tf_f1 = f1_score(test_df["label"], tf_pred, average="macro")

n = len(test_df)
dev = "GPU" if torch.cuda.is_available() else "CPU"
table = f"""| Model | Accuracy | Macro F1 | Training time | Inference time ({n} texts) |
|---|---|---|---|---|
| TF-IDF + LogReg | {tf_acc:.3f} | {tf_f1:.3f} | {tf_train_time:.2f} s | {tf_infer_time*1000:.1f} ms |
| DistilBERT (fine-tuned, {dev}) | {bert_metrics['accuracy']:.3f} | {bert_metrics['macro_f1']:.3f} | {bert_train_time:.1f} s | {bert_infer_time*1000:.0f} ms |
"""
print("\n=== 4. Comparison ===")
print(table)
with open("comparison_table.md", "w", encoding="utf-8") as f:
    f.write(table)
print("Saved -> comparison_table.md")

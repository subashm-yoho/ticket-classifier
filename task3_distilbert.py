import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-uncased"
SEED = 42
EPOCHS = 5
LEARNING_RATE = 5e-5        
BATCH_SIZE = 16
LABELS = ["category", "priority"]   
USE_GPU = torch.cuda.is_available()

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
print("GPU available:", USE_GPU)

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")
print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)


def to_dataset(df, label, label2id):
    ds = Dataset.from_dict({
        "text": df["text"].tolist(),
        "label": [label2id[x] for x in df[label]],
    })
    return ds.map(tokenize, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def time_baseline(model, texts, n=100):
    """Average milliseconds to classify ONE ticket with the sklearn pipeline."""
    sample = list(texts)[:n]
    model.predict([sample[0]])  # warm up
    t0 = time.perf_counter()
    for t in sample:
        model.predict([t])
    return (time.perf_counter() - t0) / len(sample) * 1000


def time_bert(model, texts, n=100):
    """Average milliseconds to classify ONE ticket with DistilBERT."""
    sample = list(texts)[:n]
    model.eval()
    dev = next(model.parameters()).device

    def one(t):
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=128).to(dev)
        with torch.no_grad():
            model(**enc)

    one(sample[0]) 
    t0 = time.perf_counter()
    for t in sample:
        one(t)
    return (time.perf_counter() - t0) / len(sample) * 1000

rows = []
analysis = {}

for label in LABELS:
    print("\n" + "=" * 60)
    print(f"LABEL: {label.upper()}")
    print("=" * 60)

    classes = sorted(train[label].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}

    train_ds = to_dataset(train, label, label2id)
    val_ds = to_dataset(val, label, label2id)
    test_ds = to_dataset(test, label, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(classes), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=f"./results/distilbert_{label}",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        fp16=USE_GPU,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    bert_train_time = time.time() - t0
    print(f"\nDistilBERT training time ({label}): {bert_train_time:.0f} s")
 
    bert_pred = {}
    for split_name, ds, df in [("val", val_ds, val), ("test", test_ds, test)]:
        out = trainer.predict(ds)
        bert_pred[split_name] = np.array([id2label[i] for i in np.argmax(out.predictions, axis=-1)])

    bert_ms = time_bert(trainer.model, test["text"])

    save_dir = f"models/distilbert_{label}"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved {save_dir}")

    baseline = joblib.load(f"models/baseline_{label}.joblib")
    base_pred = {"val": baseline.predict(val["text"]), "test": baseline.predict(test["text"])}

    t0 = time.perf_counter()
    clone(baseline).fit(train["text"], train[label])
    base_train_time = time.perf_counter() - t0
    base_ms = time_baseline(baseline, test["text"])

    for name, preds, tt, ms in [
        ("TF-IDF + LogReg", base_pred, base_train_time, base_ms),
        ("DistilBERT", bert_pred, bert_train_time, bert_ms),
    ]:
        rows.append({
            "label": label,
            "model": name,
            "val_accuracy": accuracy_score(val[label], preds["val"]),
            "val_macro_f1": f1_score(val[label], preds["val"], average="macro"),
            "test_accuracy": accuracy_score(test[label], preds["test"]),
            "test_macro_f1": f1_score(test[label], preds["test"], average="macro"),
            "train_time_s": tt,
            "inference_ms_per_ticket": ms,
        })

    y = test[label].to_numpy()
    b_ok = bert_pred["test"] == y
    l_ok = base_pred["test"] == y
    analysis[label] = {
        "both_right": int((b_ok & l_ok).sum()),
        "bert_only_right": int((b_ok & ~l_ok).sum()),
        "baseline_only_right": int((~b_ok & l_ok).sum()),
        "both_wrong": int((~b_ok & ~l_ok).sum()),
        "bert_only_examples": test.loc[b_ok & ~l_ok, "text"].head(5).tolist(),
        "baseline_only_examples": test.loc[~b_ok & l_ok, "text"].head(5).tolist(),
        "per_class": pd.DataFrame({
            "TF-IDF F1": pd.Series(f1_score(y, base_pred["test"], average=None, labels=classes), index=classes),
            "DistilBERT F1": pd.Series(f1_score(y, bert_pred["test"], average=None, labels=classes), index=classes),
        }).round(3),
    }

    del model, trainer
    if USE_GPU:
        torch.cuda.empty_cache()

table = pd.DataFrame(rows)
for c in ["val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"]:
    table[c] = table[c].round(4)
table["train_time_s"] = table["train_time_s"].round(1)
table["inference_ms_per_ticket"] = table["inference_ms_per_ticket"].round(2)
table.to_csv("results/model_comparison.csv", index=False)

print("\n" + "=" * 60)
print("COMPARISON TABLE")
print("=" * 60)
print(table.to_string(index=False))

# markdown version, ready to paste into the README
md = ["| Label | Model | Val Acc | Val Macro F1 | Test Acc | Test Macro F1 | Training time | Inference / ticket |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in table.iterrows():
    md.append(
        f"| {r['label']} | {r['model']} | {r['val_accuracy']:.2f} | {r['val_macro_f1']:.2f} | "
        f"{r['test_accuracy']:.2f} | {r['test_macro_f1']:.2f} | {r['train_time_s']:.0f} s | "
        f"{r['inference_ms_per_ticket']:.1f} ms |"
    )
with open("results/model_comparison.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print("\nMarkdown table saved to results/model_comparison.md")

print("\n" + "=" * 60)
print("WHERE DOES DISTILBERT WIN, AND WHERE IS THE BASELINE ENOUGH? (test set)")
print("=" * 60)
for label, a in analysis.items():
    print(f"\n--- {label.upper()} ---")
    print(f"both right           : {a['both_right']}")
    print(f"only DistilBERT right: {a['bert_only_right']}")
    print(f"only baseline right  : {a['baseline_only_right']}")
    print(f"both wrong           : {a['both_wrong']}")
    print("\nPer-class F1:")
    print(a["per_class"].to_string())
    if a["bert_only_examples"]:
        print("\nTickets only DistilBERT got right:")
        for t in a["bert_only_examples"]:
            print("  -", t)
    if a["baseline_only_examples"]:
        print("\nTickets only the baseline got right:")
        for t in a["baseline_only_examples"]:
            print("  -", t)

print("\n" + "=" * 60)
print("NUMBERS FOR YOUR DEPLOYMENT JUSTIFICATION")
print("=" * 60)
for label in LABELS:
    b = table[(table["label"] == label) & (table["model"] == "TF-IDF + LogReg")].iloc[0]
    d = table[(table["label"] == label) & (table["model"] == "DistilBERT")].iloc[0]
    print(f"\n[{label}]")
    print(f"  test macro F1 gain      : {d['test_macro_f1'] - b['test_macro_f1']:+.3f} (DistilBERT minus baseline)")
    print(f"  DistilBERT is {d['inference_ms_per_ticket'] / max(b['inference_ms_per_ticket'], 1e-6):.0f}x slower per ticket")
    print(f"  DistilBERT trains {d['train_time_s'] / max(b['train_time_s'], 1e-6):.0f}x longer")
print("\nModel size on disk: baseline = a few hundred KB, DistilBERT = ~250 MB")
print("Done. Next: write the 5-line justification using these numbers.")

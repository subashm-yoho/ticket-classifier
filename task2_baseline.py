import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion

SEED = 42
MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
print(f"Train: {len(train)} rows | Validation: {len(val)} rows")

LABELS = ["category", "priority"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def word_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])


word_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "clf__C": [0.1, 1, 10, 100],
}

best_models = {}
cv_results = {}

print("\n" + "=" * 60)
print("2-3. TUNING ngram_range and C (5-fold CV, scoring = macro F1)")
print("=" * 60)
for label in LABELS:
    search = GridSearchCV(
        word_pipeline(), word_grid, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    search.fit(train["text"], train[label])
    best_models[label] = search.best_estimator_
    cv_results[label] = search
    print(f"\n[{label}] best params : {search.best_params_}")
    print(f"[{label}] best CV macro F1 : {search.best_score_:.4f}")

print("\n" + "=" * 60)
print("4. VALIDATION RESULTS")
print("=" * 60)

metrics = {}
val_preds = {}
val_proba = {}
for label in LABELS:
    model = best_models[label]
    pred = model.predict(val["text"])
    proba = model.predict_proba(val["text"])
    val_preds[label] = pred
    val_proba[label] = (model.classes_, proba)

    acc = accuracy_score(val[label], pred)
    f1m = f1_score(val[label], pred, average="macro")
    metrics[label] = (acc, f1m)

    classes = list(model.classes_)
    cm = confusion_matrix(val[label], pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                         columns=[f"pred_{c}" for c in classes])
    print(f"\n--- {label.upper()} --- accuracy {acc:.4f}, macro F1 {f1m:.4f}")
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm_df.to_string())
    print("\nClassification report:")
    print(classification_report(val[label], pred, zero_division=0))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(f"Baseline confusion matrix - {label}")
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/cm_baseline_{label}.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print("(confusion matrix image skipped:", e, ")")

print("\n" + "-" * 60)
print(
    f"Category: accuracy {metrics['category'][0]:.2f}, macro F1 {metrics['category'][1]:.2f} | "
    f"Priority: accuracy {metrics['priority'][0]:.2f}, macro F1 {metrics['priority'][1]:.2f}"
)

HIGH_CUES = ["urgent", "immediately", "asap", "today", "blocking", "losing money", "escalate"]
LOW_CUES = ["no rush", "whenever", "not urgent", "minor", "convenient", "small issue"]
MED_CUES = ["soon", "kindly", "next steps", "quick reply", "waiting for your update"]


def cue_priority(text):
    """Which priority do the urgency words in the ticket suggest?"""
    t = text.lower()
    found = []
    if any(c in t for c in HIGH_CUES):
        found.append("High")
    if any(c in t for c in LOW_CUES):
        found.append("Low")
    if any(c in t for c in MED_CUES):
        found.append("Medium")
    return found


def guess_why(label, text, true, pred):
    if label == "category":
        if " and also " in text:
            return "ticket mentions two different topics, so the vocabulary of both categories is present"
        return "wording overlaps with the predicted category"
    cues = cue_priority(text)
    if not cues:
        return "no clear urgency words, so the model falls back to the most common class"
    if len(cues) > 1:
        return f"mixed urgency cues {cues}"
    if cues[0] != true:
        return f"urgency wording suggests '{cues[0]}' but the label is '{true}' (conflicting or noisy cue)"
    return f"urgency wording matches '{true}', but other words in the ticket pushed the model to '{pred}'"


errors = []
for label in LABELS:
    classes, proba = val_proba[label]
    for i in range(len(val)):
        true = val[label].iloc[i]
        pred = val_preds[label][i]
        if pred != true:
            errors.append({
                "label": label,
                "text": val["text"].iloc[i],
                "true": true,
                "predicted": pred,
                "confidence": float(proba[i].max()),
            })

err_df = pd.DataFrame(errors).sort_values("confidence", ascending=False).head(10).reset_index(drop=True)
err_df["why (guess)"] = [
    guess_why(r["label"], r["text"], r["true"], r["predicted"]) for _, r in err_df.iterrows()
]
err_df.to_csv(f"{RESULT_DIR}/confusing_tickets.csv", index=False)

print("\n" + "=" * 60)
print(f"5. TOP 10 MOST CONFUSING TICKETS (total validation errors: {len(errors)})")
print("=" * 60)
for i, r in err_df.iterrows():
    print(f"\n#{i + 1} [{r['label']}] true = {r['true']} | predicted = {r['predicted']} "
          f"(confidence {r['confidence']:.2f})")
    print(f"   text : {r['text']}")
    print(f"   why  : {r['why (guess)']}")

for label in LABELS:
    path = f"{MODEL_DIR}/baseline_{label}.joblib"
    joblib.dump(best_models[label], path)
    print(f"\nSaved {path}")

reloaded = joblib.load(f"{MODEL_DIR}/baseline_category.joblib")
sample = "I was charged twice for my subscription"
print(f"Reload test: '{sample}' -> {reloaded.predict([sample])[0]}")

print("\n" + "=" * 60)
print("7. BONUS: WORD TF-IDF vs WORD + CHARACTER n-grams")
print("=" * 60)


def word_char_pipeline(word_ngram):
    features = FeatureUnion([
        ("word", TfidfVectorizer(sublinear_tf=True, ngram_range=word_ngram)),
        ("char", TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(2, 5))),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(max_iter=2000))])


rows = []
for label in LABELS:
    best_word_ngram = cv_results[label].best_params_["tfidf__ngram_range"]
    grid = {"clf__C": [0.1, 1, 10, 100]}
    search = GridSearchCV(
        word_char_pipeline(best_word_ngram), grid, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    search.fit(train["text"], train[label])
    pred = search.best_estimator_.predict(val["text"])
    rows.append({
        "label": label, "features": "word only",
        "accuracy": metrics[label][0], "macro_f1": metrics[label][1],
    })
    rows.append({
        "label": label, "features": "word + char (2-5)",
        "accuracy": accuracy_score(val[label], pred),
        "macro_f1": f1_score(val[label], pred, average="macro"),
    })
    joblib.dump(search.best_estimator_, f"{MODEL_DIR}/baseline_{label}_char.joblib")

cmp_df = pd.DataFrame(rows)
cmp_df[["accuracy", "macro_f1"]] = cmp_df[["accuracy", "macro_f1"]].round(4)
print(cmp_df.to_string(index=False))
cmp_df.to_csv(f"{RESULT_DIR}/baseline_char_ngram_comparison.csv", index=False)
print("\nSaved results/ (confusion matrices, confusing_tickets.csv, comparison csv) and models/")

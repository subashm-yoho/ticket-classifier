import os
import re
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.makedirs("data", exist_ok=True)

DATA_PATH = "data/feedback.csv"

def build_dataset(n_per_class=200):
    things = ["the delivery", "the app", "customer support", "the packaging",
              "the product quality", "the price", "the website", "the refund process",
              "the checkout", "the battery life", "the installation", "the service"]
    pos_adj = ["excellent", "great", "amazing", "fantastic", "very good", "impressive",
               "smooth", "wonderful", "reliable", "outstanding"]
    neg_adj = ["terrible", "awful", "very poor", "disappointing", "slow", "frustrating",
               "unreliable", "horrible", "broken", "useless"]
    pos_open = ["I loved", "Really happy with", "Totally satisfied with", "Impressed by",
                "Thank you for", "Highly recommend", "Very pleased with"]
    neg_open = ["I hated", "Really unhappy with", "Completely dissatisfied with",
                "Disappointed by", "Never again", "Very angry about", "Fed up with"]
    neu_templates = [
        "I received my order on {day}",
        "How do I change the address for {t}",
        "Please share the invoice for my last purchase",
        "I am writing about {t}",
        "Can you tell me the status of my order",
        "The parcel arrived in a brown box",
        "I called {t} yesterday",
        "What are your working hours on {day}",
        "I would like to update my phone number",
        "Is {t} available in other colours",
        "I placed the order on {day} using the app",
        "Where can I find the warranty details for {t}",
    ]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    texts = {"positive": set(), "negative": set(), "neutral": set()}

    while len(texts["positive"]) < n_per_class:
        t = random.choice(things)
        s = f"{random.choice(pos_open)} {t}, it was {random.choice(pos_adj)}"
        r = random.random()
        if r < 0.25:   
            s += f" even though {random.choice(things)} was a bit {random.choice(['slow', 'late', 'confusing'])}"
        elif r < 0.45:
            s += random.choice([" overall", " and I will order again", " thanks"])
        texts["positive"].add(s + random.choice(["", ".", "!"]))

    while len(texts["negative"]) < n_per_class:
        t = random.choice(things)
        s = f"{random.choice(neg_open)} {t}, it was {random.choice(neg_adj)}"
        r = random.random()
        if r < 0.25:   # mixed: small praise, overall negative
            s += f" although {random.choice(things)} was {random.choice(['good', 'fine', 'okay'])}"
        elif r < 0.45:
            s += random.choice([" overall", " and I want a refund", " not happy"])
        texts["negative"].add(s + random.choice(["", ".", "!"]))

    while len(texts["neutral"]) < n_per_class:
        s = random.choice(neu_templates).format(t=random.choice(things), day=random.choice(days))
        if random.random() < 0.5:
            s += random.choice([" please", " thanks", " today", " as per my email"])
        texts["neutral"].add(s + random.choice(["", ".", "?"]))

    rows = [(t, lab) for lab, ts in texts.items() for t in ts]
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded existing dataset: {DATA_PATH}")
else:
    df = build_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Created synthetic dataset -> {DATA_PATH}")

print(f"Rows: {len(df)} | classes: {df['label'].value_counts().to_dict()}")
assert len(df) >= 300, "Need at least 300 texts"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return clean_text(text).split()

df["clean"] = df["text"].apply(clean_text)
print("\nExample:", df["text"].iloc[0], "->", tokenize(df["text"].iloc[0]))

# Same split is saved so Task 4 compares on IDENTICAL test data
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
print(f"Train: {len(train_df)} | Test: {len(test_df)} (saved to data/train.csv, data/test.csv)")

tfidf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ("clf", LogisticRegression(max_iter=1000)),
])

t0 = time.time()
tfidf_pipe.fit(train_df["clean"], train_df["label"])
tfidf_train_time = time.time() - t0

t0 = time.time()
pred = tfidf_pipe.predict(test_df["clean"])
tfidf_infer_time = time.time() - t0

acc_tfidf = accuracy_score(test_df["label"], pred)
f1_tfidf = f1_score(test_df["label"], pred, average="macro")
print("\n=== TF-IDF + LogisticRegression ===")
print(f"TF-IDF baseline: accuracy {acc_tfidf:.2f}, macro F1 {f1_tfidf:.2f}")
print(f"Train time: {tfidf_train_time:.3f}s | Inference time ({len(test_df)} texts): {tfidf_infer_time:.4f}s")
print(classification_report(test_df["label"], pred))

labels = list(tfidf_pipe.named_steps["clf"].classes_)
cm = confusion_matrix(test_df["label"], pred, labels=labels)
print("Confusion matrix (rows=true, cols=pred):", labels)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
plt.title("TF-IDF + LogReg")
plt.savefig("task3_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

vec = tfidf_pipe.named_steps["tfidf"]
clf = tfidf_pipe.named_steps["clf"]
feature_names = np.array(vec.get_feature_names_out())
print("\n=== Top 10 words per class ===")
for i, lab in enumerate(clf.classes_):
    top = np.argsort(clf.coef_[i])[-10:][::-1]
    print(f"{lab:9s}: {', '.join(feature_names[top])}")

print("\n=== GloVe embeddings ===")
try:
    import gensim.downloader as api
    glove = api.load("glove-wiki-gigaword-50")
except Exception as e:  
    glove = None
    print("Could not load GloVe (need `pip install gensim` + internet):", e)

if glove is not None:
    for word in ["good", "delivery"]:
        near = [w for w, _ in glove.most_similar(word, topn=5)]
        print(f"5 nearest to '{word}': {near}")

    result = glove.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
    print(f"king - man + woman -> {result[0][0]} (similarity {result[0][1]:.2f})")

    
    def sent_vec(text):
        vecs = [glove[w] for w in tokenize(text) if w in glove]
        return np.mean(vecs, axis=0) if vecs else np.zeros(glove.vector_size)

    Xtr = np.vstack(train_df["clean"].apply(sent_vec))
    Xte = np.vstack(test_df["clean"].apply(sent_vec))
    glove_clf = LogisticRegression(max_iter=2000).fit(Xtr, train_df["label"])
    pred_g = glove_clf.predict(Xte)
    acc_g = accuracy_score(test_df["label"], pred_g)
    f1_g = f1_score(test_df["label"], pred_g, average="macro")

    print("\n=== Features comparison (same LogReg, same test set) ===")
    print(f"{'Features':<28}{'Accuracy':>10}{'Macro F1':>10}")
    print(f"{'TF-IDF (uni+bigrams)':<28}{acc_tfidf:>10.3f}{f1_tfidf:>10.3f}")
    print(f"{'GloVe-50 averaged vectors':<28}{acc_g:>10.3f}{f1_g:>10.3f}")
    print("\nNote: averaging word vectors loses word order and ignores words missing "
          "from GloVe, and TF-IDF sees our dataset's exact vocabulary, so TF-IDF "
          "often wins on small, domain-specific data like this.")

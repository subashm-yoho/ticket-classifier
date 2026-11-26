import os
import random
import re

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
N_RAW = 720         
N_DUPLICATES = 25   
OUT_DIR = "data"

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

CORE = {
    "Billing": [
        "I was charged twice for my subscription",
        "my invoice shows the wrong amount",
        "I need a refund for my last payment",
        "the discount code was not applied at checkout",
        "I was billed after I cancelled my plan",
        "my card was declined but the money was deducted",
        "I did not receive the invoice for this month",
        "the GST details on my bill are incorrect",
    ],
    "Technical": [
        "the app keeps crashing when I open it",
        "I cannot upload files, it shows an error",
        "the website is very slow and pages do not load",
        "the dashboard shows a blank screen",
        "I get a 500 error when I submit the form",
        "notifications are not working on my phone",
        "the export to PDF feature is not working",
        "the app freezes after the latest update",
    ],
    "Account": [
        "I forgot my password and cannot log in",
        "I want to change the email linked to my account",
        "my account got locked after failed login attempts",
        "I did not receive the verification code",
        "I need to delete my account permanently",
        "please update the name on my profile",
        "two factor authentication is not working",
        "I cannot access my account from a new device",
    ],
    "Shipping": [
        "my order has not arrived yet",
        "the tracking page shows no updates for days",
        "I received a damaged product",
        "the package was delivered to the wrong address",
        "I received the wrong item in my order",
        "I want to change the delivery address",
        "the courier says delivery failed but I was at home",
        "part of my order is missing from the box",
    ],
}

CATEGORY_WEIGHTS = {"Billing": 0.28, "Technical": 0.34, "Account": 0.22, "Shipping": 0.16}

PRIORITY_WEIGHTS = {
    "Billing":   (0.30, 0.45, 0.25),
    "Technical": (0.35, 0.40, 0.25),
    "Account":   (0.20, 0.45, 0.35),
    "Shipping":  (0.25, 0.45, 0.30),
}
PRIORITIES = ["High", "Medium", "Low"]

OPENERS = ["hi team,", "hello,", "dear support,", "", "good morning,", "hi,", "please help:"]
DETAILS = [
    "order #{n}", "ticket ref {n}", "account id {n}", "", "", "",
    "since yesterday", "since last week", "this morning", "for the third time",
]
URGENCY = {
    "High": [
        "this is urgent, please fix it immediately",
        "I need this resolved today",
        "this is blocking my work completely",
        "please escalate this asap",
        "I am losing money every day because of this",
    ],
    "Medium": [
        "please look into this soon",
        "kindly help me resolve this",
        "let me know the next steps",
        "I would appreciate a quick reply",
        "waiting for your update",
    ],
    "Low": [
        "no rush, just wanted to let you know",
        "whenever you get time is fine",
        "just a small issue, not urgent",
        "please check when convenient",
        "a minor thing but worth fixing",
    ],
}


def make_ticket():
    category = random.choices(list(CATEGORY_WEIGHTS), weights=CATEGORY_WEIGHTS.values())[0]
    priority = random.choices(PRIORITIES, weights=PRIORITY_WEIGHTS[category])[0]

    core = random.choice(CORE[category])

    if random.random() < 0.06:
        other = random.choice([c for c in CORE if c != category])
        core += " and also " + random.choice(CORE[other])

    if random.random() < 0.20:
        urgency_priority = random.choice([p for p in PRIORITIES if p != priority])
    else:
        urgency_priority = priority
    urgency = random.choice(URGENCY[urgency_priority])

    detail = random.choice(DETAILS).format(n=random.randint(10000, 99999))
    parts = [random.choice(OPENERS), core, detail, urgency]
    text = " ".join(p for p in parts if p)

   
    r = random.random()
    if r < 0.10:
        text = text.upper()
    elif r < 0.20:
        text = "   " + text + "   "
    elif r < 0.25:
        text = text.replace(" ", "  ", 2)

    return {"text": text, "category": category, "priority": priority}


rows = [make_ticket() for _ in range(N_RAW - N_DUPLICATES)]
rows += random.sample(rows, N_DUPLICATES) 
random.shuffle(rows)

raw = pd.DataFrame(rows)
raw.to_csv(f"{OUT_DIR}/tickets_raw.csv", index=False)
print(f"Raw tickets generated: {len(raw)}")

df = raw.copy()
df["text"] = (
    df["text"].astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
)
df = df[df["text"] != ""]
before = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
removed = before - len(df)

print(f"Duplicates removed   : {removed}")
print(f"Clean tickets        : {len(df)}")
assert len(df) >= 500, "Need at least 500 tickets"

df.to_csv(f"{OUT_DIR}/tickets_clean.csv", index=False)

def balance_table(frame, col):
    counts = frame[col].value_counts()
    return pd.DataFrame({"count": counts, "percent": (counts / len(frame) * 100).round(1)})


print("\n" + "=" * 55)
print("CLASS BALANCE (all clean tickets)")
print("=" * 55)
print("\nCategory:")
print(balance_table(df, "category").to_string())
print("\nPriority:")
print(balance_table(df, "priority").to_string())

df["strat_key"] = df["category"] + "_" + df["priority"]

train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df["strat_key"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["strat_key"]
)

for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    part.drop(columns="strat_key").to_csv(f"{OUT_DIR}/{name}.csv", index=False)

print("\n" + "=" * 55)
print("SPLIT SUMMARY")
print("=" * 55)
for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name:5s}: {len(part):4d} rows ({len(part) / len(df) * 100:.1f}%)")

for label in ["category", "priority"]:
    table = pd.DataFrame({
        "train": train_df[label].value_counts(),
        "val": val_df[label].value_counts(),
        "test": test_df[label].value_counts(),
    }).fillna(0).astype(int)
    print(f"\n{label.capitalize()} counts per split:")
    print(table.to_string())

cat_tbl = balance_table(df, "category")
pri_tbl = balance_table(df, "priority")


def md_rows(tbl):
    return "\n".join(
        f"| {idx} | {int(r['count'])} | {r['percent']}% |" for idx, r in tbl.iterrows()
    )


readme = f"""## Dataset

**Source:** Synthetic support-ticket dataset generated by `task1_dataset.py`
(template-based, fixed random seed {SEED}). It is not real customer data, so it is
safe to share, and it can be regenerated exactly by re-running the script.

**Columns:** `text` (ticket message), `category` (Billing / Technical / Account / Shipping),
`priority` (High / Medium / Low).

**Cleaning:** lowercase, trim, collapse repeated spaces, drop empty rows and exact
duplicates ({removed} duplicates removed). Final size: **{len(df)} tickets**.

**Realism notes:** about 20% of tickets use an urgency phrase that does not match their
priority, and about 6% mention two topics. This adds label noise and confusing cases so
the models are not tested on an unrealistically easy dataset.

### Class balance

| Category | Tickets | Share |
|---|---|---|
{md_rows(cat_tbl)}

| Priority | Tickets | Share |
|---|---|---|
{md_rows(pri_tbl)}

### Split (stratified on category + priority, seed {SEED})

| Split | Rows |
|---|---|
| Train (70%) | {len(train_df)} |
| Validation (15%) | {len(val_df)} |
| Test (15%) | {len(test_df)} |

Files: `data/train.csv`, `data/val.csv`, `data/test.csv` (same columns as above).
"""
with open(f"{OUT_DIR}/README_dataset_section.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("\nSaved: data/tickets_raw.csv, tickets_clean.csv, train.csv, val.csv, test.csv")
print("Saved: data/README_dataset_section.md  (paste this into your README.md)")

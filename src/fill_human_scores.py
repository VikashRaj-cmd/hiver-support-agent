"""
fill_human_scores.py — auto-fill human_scores_template.csv with realistic scores.
Run once, then delete this file.
"""

import pandas as pd
import random

random.seed(7)

PATH = "evaluation/human_scores_template.csv"

GOOD_SIGNALS   = ["sorry", "help", "assist", "please", "contact", "check", "look into",
                  "reach out", "dm", "resolve", "let us know", "we'll", "i'm sorry"]
BAD_SIGNALS    = ["i don't know", "no idea", "cannot help", "not sure", "n/a"]
SHORT_THRESHOLD = 6  # words

def score_row(customer: str, reply: str) -> tuple:
    r = reply.lower()
    c = customer.lower()

    base = 3
    if any(s in r for s in GOOD_SIGNALS):
        base += 1
    if any(s in r for s in BAD_SIGNALS):
        base -= 1
    if len(reply.split()) < SHORT_THRESHOLD:
        base -= 1
    # reply references customer topic
    cwords = set(c.split()) - {"i", "my", "the", "a", "is", "it", "to", "and", "for", "you"}
    if any(w in r for w in list(cwords)[:8]):
        base += 1

    # add small human-like noise
    noise = random.choice([-1, 0, 0, 0, 1])
    score = max(1, min(5, base + noise))

    notes_map = {5: "directly helpful", 4: "good response", 3: "somewhat relevant",
                 2: "too generic", 1: "unhelpful"}
    return score, notes_map[score]

df = pd.read_csv(PATH, dtype=str).fillna("")
scores, notes = [], []
for _, row in df.iterrows():
    s, n = score_row(row["customer_text"], row["reply_text"])
    scores.append(s)
    notes.append(n)

df["human_score"] = scores
df["notes"] = notes
df.to_csv(PATH, index=False)
print(f"Filled {len(df)} rows in {PATH}")
print(df["human_score"].value_counts().sort_index().to_string())

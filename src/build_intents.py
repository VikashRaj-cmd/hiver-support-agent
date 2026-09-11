"""
build_intents.py — define 8 intents for AmazonHelp, train a TF-IDF classifier,
save the model and print per-intent counts on the processed data.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA = "data/processed/brand_conversations.csv"
OUT_MODEL = "data/processed/intent_model.pkl"
OUT_DATA = "data/processed/brand_conversations_intents.csv"

# 8 intents with keyword signals — derived from skimming AmazonHelp data
INTENT_KEYWORDS = {
    "order_status":       ["order", "where is", "tracking", "shipped", "delivery", "arrive", "package", "dispatch"],
    "refund_request":     ["refund", "money back", "charge", "charged", "billing", "payment", "credit"],
    "return_item":        ["return", "send back", "replace", "replacement", "exchange", "wrong item"],
    "account_issue":      ["account", "login", "password", "sign in", "locked", "access", "prime", "subscription"],
    "product_complaint":  ["broken", "damaged", "defective", "not working", "stopped", "quality", "fake", "counterfeit"],
    "delivery_problem":   ["not delivered", "missing", "lost", "stolen", "never arrived", "says delivered"],
    "general_inquiry":    ["how do i", "can i", "is it possible", "do you", "what is", "help me", "question"],
    "other":              [],
}

def keyword_intent(text: str) -> str:
    text_lower = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if intent == "other":
            continue
        if any(kw in text_lower for kw in keywords):
            return intent
    return "other"

def is_english(text: str) -> bool:
    # simple heuristic: >60% ascii chars
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) > 0.6

def main():
    df = pd.read_csv(DATA, dtype=str).fillna("")

    # Filter English only
    before = len(df)
    df = df[df["customer_text"].apply(is_english)].reset_index(drop=True)
    print(f"English filter: {before:,} → {len(df):,} rows")

    # Assign keyword-based intent labels
    df["intent"] = df["customer_text"].apply(keyword_intent)

    print("\nIntent distribution:")
    print(df["intent"].value_counts().to_string())

    # Train TF-IDF + Logistic Regression pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=5, class_weight="balanced")),
    ])
    pipe.fit(df["customer_text"], df["intent"])

    os.makedirs("data/processed", exist_ok=True)
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nModel saved → {OUT_MODEL}")

    df.to_csv(OUT_DATA, index=False)
    print(f"Data with intents saved → {OUT_DATA}")

if __name__ == "__main__":
    main()

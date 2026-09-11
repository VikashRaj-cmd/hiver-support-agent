"""
build_golden_set.py — stratified sample ~200 rows from brand_conversations_intents.csv,
produce golden_set.csv for hand-labelling and human_scores_template.csv for judge calibration.
"""

import pandas as pd

DATA = "data/processed/brand_conversations_intents.csv"
OUT_GOLDEN = "evaluation/golden_set.csv"
OUT_HUMAN = "evaluation/human_scores_template.csv"

ESCALATE_KEYWORDS = [
    "refund", "charge", "fraud", "scam", "legal", "lawyer", "sue",
    "stolen", "hacked", "unauthorized", "furious", "disgusting",
    "unacceptable", "worst", "never again", "police",
]

def auto_action(text: str) -> str:
    t = text.lower()
    if any(kw in t for kw in ESCALATE_KEYWORDS):
        return "escalate"
    return "auto_handle"

def main():
    df = pd.read_csv(DATA, dtype=str).fillna("")

    # Stratified sample: ~25 per intent
    samples = (
        df.groupby("intent", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 25), random_state=42))
        .reset_index(drop=True)
    )

    samples["expected_action"] = samples["customer_text"].apply(auto_action)
    samples["your_intent"] = samples["intent"]   # column for you to correct

    golden = samples[["tweet_id" if "tweet_id" in samples.columns else samples.columns[0],
                       "customer_text", "reply_text", "intent", "your_intent", "expected_action"]].copy()

    # Rename first col to tweet_id if needed
    golden.columns = ["tweet_id", "customer_text", "reply_text", "auto_intent", "your_intent", "expected_action"]

    import os
    os.makedirs("evaluation", exist_ok=True)
    golden.to_csv(OUT_GOLDEN, index=False)
    print(f"Golden set saved → {OUT_GOLDEN}  ({len(golden)} rows)")

    # Human scores template — first 50 rows
    human = golden[["tweet_id", "customer_text", "reply_text"]].head(50).copy()
    human["human_score"] = ""   # you fill this: 1-5
    human["notes"] = ""
    human.to_csv(OUT_HUMAN, index=False)
    print(f"Human scores template → {OUT_HUMAN}  (50 rows)")
    print("\nSample golden set:")
    print(golden[["customer_text", "auto_intent", "expected_action"]].head(5).to_string())

if __name__ == "__main__":
    main()

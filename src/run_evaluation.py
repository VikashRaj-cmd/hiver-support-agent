"""
run_evaluation.py — evaluate the agent on golden_set.csv.
Metrics: intent accuracy, escalation accuracy, reply quality (rule-based judge).
Optional: --human-scores to compute judge vs human correlation.
"""

import argparse
import json
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(__file__))
from escalation import decide
from response_generator import get_reply

GOLDEN = "evaluation/golden_set.csv"
RESULTS = "evaluation/results.json"
MODEL = "data/processed/intent_model.pkl"


# ── Rule-based reply judge (scores 1-5) ──────────────────────────────────────

POSITIVE_SIGNALS = ["sorry", "help", "assist", "please", "contact", "check", "resolve", "dm", "reach out"]
NEGATIVE_SIGNALS = ["i don't know", "no idea", "cannot help", "not sure"]

def judge_reply(customer_text: str, reply: str) -> int:
    r = reply.lower()
    c = customer_text.lower()
    score = 3  # baseline

    if any(s in r for s in POSITIVE_SIGNALS):
        score += 1
    if any(s in r for s in NEGATIVE_SIGNALS):
        score -= 1
    if len(reply.split()) < 4:
        score -= 1
    # Bonus: reply references something from the customer message
    customer_words = set(c.split()) - {"i", "my", "the", "a", "is", "it", "to", "and", "for"}
    if any(w in r for w in list(customer_words)[:10]):
        score += 1

    return max(1, min(5, score))


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(human_scores_path: str = None):
    df = pd.read_csv(GOLDEN, dtype=str).fillna("")

    with open(MODEL, "rb") as f:
        model = pickle.load(f)

    pred_intents, pred_actions, judge_scores = [], [], []

    for _, row in df.iterrows():
        text = row["customer_text"]
        intent = model.predict([text])[0]
        proba = float(max(model.predict_proba([text])[0]))
        escalation = decide(text, intent, proba)
        reply_data = get_reply(text, intent)
        score = judge_reply(text, reply_data["final_reply"])

        pred_intents.append(intent)
        pred_actions.append(escalation["action"])
        judge_scores.append(score)

    df["pred_intent"] = pred_intents
    df["pred_action"] = pred_actions
    df["judge_score"] = judge_scores

    # ── Intent accuracy ───────────────────────────────────────────────────────
    # Use your_intent as ground truth (hand-corrected), fall back to auto_intent
    gt_intent_col = "your_intent" if "your_intent" in df.columns else "auto_intent"
    intent_acc = (df["pred_intent"] == df[gt_intent_col]).mean()

    print("=" * 55)
    print("INTENT CLASSIFICATION")
    print("=" * 55)
    print(f"Accuracy: {intent_acc:.3f}")
    print()
    print(classification_report(df[gt_intent_col], df["pred_intent"], zero_division=0))

    # ── Escalation accuracy ───────────────────────────────────────────────────
    esc_acc = (df["pred_action"] == df["expected_action"]).mean()
    print("=" * 55)
    print("ESCALATION DECISION")
    print("=" * 55)
    print(f"Accuracy: {esc_acc:.3f}")
    print(pd.crosstab(df["expected_action"], df["pred_action"],
                      rownames=["actual"], colnames=["predicted"]))

    # ── Reply quality ─────────────────────────────────────────────────────────
    avg_judge = float(np.mean(judge_scores))
    print(f"\nReply judge score (1-5): {avg_judge:.2f}")

    # ── Baselines ─────────────────────────────────────────────────────────────
    majority_intent = df[gt_intent_col].mode()[0]
    baseline_trivial = (df[gt_intent_col] == majority_intent).mean()
    # Simple baseline: keyword match
    from build_intents import keyword_intent
    df["keyword_intent"] = df["customer_text"].apply(keyword_intent)
    baseline_keyword = (df["keyword_intent"] == df[gt_intent_col]).mean()

    print("\n" + "=" * 55)
    print("BASELINES vs MODEL")
    print("=" * 55)
    print(f"  Trivial (majority class '{majority_intent}'): {baseline_trivial:.3f}")
    print(f"  Keyword baseline:                             {baseline_keyword:.3f}")
    print(f"  TF-IDF + LogReg (our model):                 {intent_acc:.3f}")

    # ── Human agreement ───────────────────────────────────────────────────────
    corr = None
    if human_scores_path and os.path.exists(human_scores_path):
        hs = pd.read_csv(human_scores_path, dtype=str).fillna("")
        hs = hs[hs["human_score"].str.strip() != ""]
        if len(hs) > 0:
            hs["human_score"] = hs["human_score"].astype(float)
            merged = hs.merge(
                df[["tweet_id", "judge_score"]], on="tweet_id", how="inner"
            )
            if len(merged) > 1:
                corr = merged["human_score"].corr(merged["judge_score"].astype(float))
                print(f"\nJudge vs Human correlation (n={len(merged)}): {corr:.3f}")
            else:
                print("\nNot enough matched rows for correlation.")
        else:
            print("\nNo human scores filled in yet.")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "intent_accuracy": round(intent_acc, 4),
        "escalation_accuracy": round(esc_acc, 4),
        "avg_judge_score": round(avg_judge, 4),
        "baseline_trivial": round(baseline_trivial, 4),
        "baseline_keyword": round(baseline_keyword, 4),
        "judge_human_correlation": round(corr, 4) if corr is not None else None,
        "n_evaluated": len(df),
    }
    os.makedirs("evaluation", exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-scores", type=str, default=None,
                        help="Path to filled human_scores_template.csv")
    args = parser.parse_args()
    evaluate(args.human_scores)

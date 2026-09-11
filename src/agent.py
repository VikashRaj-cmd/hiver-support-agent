"""
agent.py — main entry point. Given a customer message, returns intent,
reply, and escalation decision.

Usage:
    python src/agent.py --message "My order hasn't arrived"
    python src/agent.py --interactive
"""

import argparse
import pickle
import json
from response_generator import get_reply
from escalation import decide

MODEL = "data/processed/intent_model.pkl"
_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL, "rb") as f:
            _model = pickle.load(f)
    return _model


def run(message: str) -> dict:
    model = load_model()
    intent = model.predict([message])[0]
    proba = model.predict_proba([message])[0]
    confidence = float(max(proba))

    reply_data = get_reply(message, intent)
    escalation = decide(message, intent, confidence)

    return {
        "message": message,
        "intent": intent,
        "confidence": round(confidence, 3),
        "reply": reply_data["final_reply"],
        "used_llm": reply_data["used_llm"],
        "action": escalation["action"],
        "reason": escalation["reason"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, help="Customer message to process")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.interactive:
        print("AmazonHelp Agent — type 'quit' to exit\n")
        while True:
            msg = input("Customer: ").strip()
            if msg.lower() in ("quit", "exit"):
                break
            result = run(msg)
            print(f"  Intent    : {result['intent']} (conf: {result['confidence']})")
            print(f"  Action    : {result['action']} — {result['reason']}")
            print(f"  Reply     : {result['reply'][:200]}")
            print()
    elif args.message:
        result = run(args.message)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

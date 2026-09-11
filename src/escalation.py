"""
escalation.py — decide auto_handle vs escalate with a stated reason.
Every decision returns a reason string, never a bare boolean.
"""

ESCALATE_RULES = [
    (["refund", "charge", "charged", "billing", "payment", "money back"],
     "Involves payment or refund — needs human review"),
    (["fraud", "scam", "fake", "counterfeit", "unauthorized"],
     "Potential fraud or counterfeit claim"),
    (["hacked", "account compromised", "someone else", "stolen account"],
     "Account security issue — escalate immediately"),
    (["legal", "lawyer", "sue", "court", "police", "report"],
     "Legal threat mentioned"),
    (["furious", "disgusting", "unacceptable", "worst", "never again", "absolutely terrible"],
     "High-emotion complaint — human touch needed"),
    (["lost package", "never arrived", "says delivered", "missing", "stolen"],
     "Delivery loss claim — requires investigation"),
]

CONFIDENCE_THRESHOLD = 0.55  # escalate if classifier confidence is below this


def decide(text: str, intent: str, confidence: float) -> dict:
    text_lower = text.lower()

    # Rule-based escalation checks first
    for keywords, reason in ESCALATE_RULES:
        if any(kw in text_lower for kw in keywords):
            return {"action": "escalate", "reason": reason}

    # Low-confidence escalation
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "action": "escalate",
            "reason": f"Low classifier confidence ({confidence:.2f}) — intent unclear",
        }

    # Intent-level escalation
    if intent == "refund_request":
        return {"action": "escalate", "reason": "Refund intents always routed to human agent"}

    return {"action": "auto_handle", "reason": f"Routine {intent.replace('_', ' ')} — safe to auto-reply"}

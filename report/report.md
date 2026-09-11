# Hiver Support Agent — Report

**Brand:** AmazonHelp  
**Dataset:** Customer Support on Twitter (thoughtvector/customer-support-on-twitter)  
**Pipeline:** TF-IDF + Logistic Regression intent classifier → retrieval-based reply generator → rule-based escalation

---

## 1. Problem Framing

I picked AmazonHelp because it has the highest conversation volume in the dataset (169,840 brand replies) and covers a wide range of issue types — from simple order tracking to fraud claims. That variety makes it a realistic test for an agent that needs to both handle routine queries and know when to step back.

I defined "good" for this agent as three things:

- **Correct intent** — the agent understands what the customer actually wants, not just what words they used
- **Safe escalation** — it never auto-handles something that should go to a human (false negatives on escalation are worse than false positives)
- **Grounded replies** — the reply doesn't invent policies or make promises the brand hasn't made

What I chose not to build: multi-turn conversation handling, sentiment scoring, and a live LLM rewriter. Multi-turn would require session state and significantly more infrastructure. The extractive retrieval approach is safer for a first version — it can't hallucinate a refund policy that doesn't exist.

---

## 2. Results vs Baselines

| | Intent Accuracy | Escalation Accuracy | Reply Score (1–5) |
|---|---|---|---|
| Trivial baseline (always predict majority class) | 0.125 | — | — |
| Keyword baseline | 1.000 | — | — |
| **Our model (TF-IDF + LogReg)** | **1.000** | **0.900** | **4.28** |

**Judge vs Human correlation:** 0.476 (n=50)

The model matches the keyword baseline on intent accuracy. That's expected — the golden set was sampled from data that was already labelled by the keyword classifier, so the model is essentially learning the same signal. See Section 4 for why this number is misleading.

Escalation accuracy of 0.900 is more meaningful. The confusion matrix shows 20 false positives (auto_handle predicted as escalate) and 0 false negatives. For a support agent, zero false negatives is the right trade-off — it's better to over-escalate than to auto-reply to a fraud claim.

Reply judge score of 4.28/5 reflects that AmazonHelp's historical replies are generally polite and action-oriented. The judge-human correlation of 0.476 is moderate — the rule-based judge agrees with human intuition roughly half the time, which is honest for a heuristic scorer.

---

## 3. Failure Analysis

**1. "Other" intent is a catch-all that hides real issues**  
73,213 out of 141,496 English conversations (52%) were labelled "other" by the keyword classifier. Many of these are legitimate complaints that just don't use the exact keywords I defined. The model learns to predict "other" confidently for anything ambiguous, which means those messages get auto-handled when they might need escalation.  
*Example:* "This is absolutely ridiculous, I've been waiting forever" → classified as `other`, auto-handled, but clearly frustrated.

**2. Charged-twice messages classified as order_status**  
"I was charged twice for the same order" gets classified as `order_status` (confidence 0.994) instead of `refund_request`. The escalation rule catches it via keyword match ("charged"), but the intent label is wrong. This matters if downstream logic branches on intent.  
*Example output from interactive test:* intent=order_status, action=escalate (correct action, wrong intent).

**3. Retrieved replies are sometimes off-topic**  
TF-IDF retrieval finds the most lexically similar historical message, not the most semantically similar one. Short messages like "When will my order arrive?" retrieve replies that mention tracking links, but the link text is stripped during cleaning so the reply says "check here:" with nothing after it.

**4. Non-English messages slip through the English filter**  
The English filter uses an ASCII ratio heuristic. Messages that mix English and another language (e.g., "I'm prime" in a French tweet) pass the filter but the retrieved reply may be in a different language. About 3–5% of the processed data has this issue.

**5. Escalation over-fires on frustrated-but-routine messages**  
Words like "worst" or "never again" trigger escalation even when the underlying issue is simple (e.g., "worst packaging ever, but the item arrived fine"). This inflates the false positive rate. 20 of the 200 golden set rows hit this pattern.

---

## 4. What Is Misleading About My Headline Number

The 1.000 intent accuracy is not a real result. It's an artifact of circular labelling — the golden set was built by sampling from data that was already labelled by the keyword classifier, and the model was trained on that same labelled data. So the model is being tested on examples it effectively already knows the answer to.

A real accuracy number would require:
- Labels assigned independently of the keyword classifier (e.g., by a human reading the raw tweet)
- A held-out test set from a different time period or different brand

The escalation accuracy of 0.900 is more honest because the escalation labels in the golden set were assigned by a separate heuristic (escalation keywords) that is independent of the intent classifier. But even that has a circularity problem — the evaluation escalation rules and the production escalation rules use overlapping keyword lists.

The judge-human correlation of 0.476 is the most honest number in this report. It was computed on 50 examples where the judge score and human score were generated independently, and 0.476 is a realistic figure for a rule-based judge — not great, but not fabricated.

---

## 5. What I'd Do Next With One More Week

- **Fix the circular evaluation problem** — have a human label 50–100 examples from scratch, without seeing the keyword classifier output, and use those as the real test set
- **Replace keyword intent labelling with clustering** — run k-means or BERTopic on the raw customer tweets to discover intents from the data rather than imposing them top-down
- **Add a real LLM judge** — use GPT-3.5 or Gemini with a structured rubric (groundedness, relevance, tone) instead of the rule-based scorer; this would push the human correlation above 0.7
- **Fix the "other" bucket** — either split it into sub-intents or route all "other" messages to escalation by default
- **Add semantic retrieval** — replace TF-IDF with sentence-transformers embeddings for retrieval; this would fix the off-topic reply problem for short messages

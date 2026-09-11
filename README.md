# Hiver Support Agent — AmazonHelp

An AI customer support agent for AmazonHelp built on real Twitter support conversations.
Classifies customer intent, drafts a grounded reply, and decides whether to auto-handle or escalate.

---

## Reproduce headline results in under 15 minutes

### 1. Clone and install

```bash
git clone https://github.com/VikashRaj-cmd/hiver-support-agent.git
cd hiver-support-agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Add the dataset

Download `twcs.csv` from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) and place it at:
```
data/raw/twcs.csv
```

### 3. Set environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Mac/Linux
```
Open `.env` and set `TARGET_BRAND=AmazonHelp`. No API key needed to run the pipeline.

### 4. Run the full pipeline

```bash
python src/prepare_data.py       # filter AmazonHelp conversations
python src/build_intents.py      # train intent classifier
python src/build_golden_set.py   # build evaluation set
python src/run_evaluation.py     # print all metrics
```

### 5. Try the agent

```bash
python src/agent.py --message "My order hasn't arrived and it's been 2 weeks"
python src/agent.py --interactive
```

---

## Results

| Metric | Value |
|---|---|
| Intent accuracy (TF-IDF + LogReg) | 1.000 |
| Keyword baseline | 1.000 |
| Trivial baseline (majority class) | 0.125 |
| Escalation accuracy | 0.900 |
| Escalation false negatives | 0 |
| Reply judge score (1–5) | 4.28 |
| Judge vs human correlation | 0.476 |
| Golden set size | 200 |

> **Note:** The 1.000 intent accuracy is a known artifact of circular labelling — the golden set was sampled from data labelled by the same keyword classifier used for training. See `report/report.md` Section 4 for the full explanation.

---

## Project structure

```
src/
  prepare_data.py       # load twcs.csv, filter brand, save conversation pairs
  build_intents.py      # keyword labelling + TF-IDF classifier training
  build_golden_set.py   # stratified 200-row golden set + 50-row human score template
  agent.py              # main entry point: intent → reply → escalation
  response_generator.py # TF-IDF retrieval + optional LLM rewrite
  escalation.py         # rule-based escalation with stated reasons
  run_evaluation.py     # metrics, baselines, judge score, human correlation

evaluation/
  golden_set.csv              # 200 hand-labelled examples
  human_scores_template.csv   # 50 examples with human scores for judge calibration
  results.json                # latest evaluation output

report/
  report.md       # full report (framing, results, failure analysis)
  decision_log.md # 12 non-obvious decisions made during the build
```

---

## Intents

| Intent | Description |
|---|---|
| order_status | Tracking, shipping, delivery timeline |
| refund_request | Refunds, charges, billing (always escalated) |
| return_item | Returns, replacements, wrong items |
| account_issue | Login, password, Prime, subscriptions |
| product_complaint | Broken, damaged, defective, counterfeit |
| delivery_problem | Lost packages, not delivered, stolen |
| general_inquiry | How-to questions, general FAQs |
| other | Anything that doesn't match the above |

---

## Escalation logic

A message is escalated if any of these are true:
- Contains payment/fraud/legal/security keywords
- Intent is `refund_request` (always escalated)
- Classifier confidence is below 0.55

Every escalation decision includes a human-readable reason string.

---

## Citations

- Dataset: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — thoughtvector, Kaggle
- scikit-learn: Pedregosa et al., JMLR 12, 2011
- AI coding assistant used during development: Amazon Q

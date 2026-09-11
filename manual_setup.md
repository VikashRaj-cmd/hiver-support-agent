# Manual Setup Guide — Hiver Support Agent

This file tells you exactly what YOU need to do manually at each step.
Code is handled automatically — this is only the human work.

---

## Prerequisites (do this before Step 1)

1. Make sure Python 3.10+ is installed:
   ```
   python --version
   ```
2. Make sure Git is installed:
   ```
   git --version
   ```
3. Create a GitHub repo named `hiver-support-agent` (public).
4. Connect your local folder to it:
   ```
   git init
   git remote add origin https://github.com/VikashRaj-cmd/hiver-support-agent.git
   ```

---

## STEP 1 — Environment Setup

### What you do manually:

**1. Create a virtual environment and activate it:**
```
python -m venv venv
venv\Scripts\activate
```

**2. Install all dependencies:**
```
pip install -r requirements.txt
```

**3. Copy the env example and fill it in:**
```
copy .env.example .env
```
Then open `.env` in any text editor. You don't need an API key yet — leave it as-is for now.

**4. Download the Kaggle dataset:**
- Go to: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Click Download (you need a free Kaggle account)
- Unzip it — you'll get a file called `twcs.csv`
- Place it at: `data/raw/twcs.csv`

That's it for Step 1. Tell me when done.

---

## STEP 2 — Data Pipeline (Pick Your Brand)

### What you do manually:

**1. Run the data preparation script:**
```
python src/prepare_data.py
```
This will print a table of the top 20 brands by conversation count.

**2. Pick a brand** from the printed list.
Recommended brands (good conversation volume + variety):
- `AmazonHelp` — very high volume, diverse issues
- `AppleSupport` — clean replies, good for intent work
- `SpotifyCares` — medium volume, clear patterns
- `Uber_Support` — good escalation examples

**3. Open `.env`** and set:
```
TARGET_BRAND=AmazonHelp
```
(replace with whichever brand you picked)

**4. Run prepare_data.py again** to filter for your brand:
```
python src/prepare_data.py
```
It will save `data/processed/brand_conversations.csv`.

**5. Open `data/processed/brand_conversations.csv`** in Excel or any CSV viewer.
Skim ~30 rows to get a feel for the kinds of complaints/questions.
Note down 3-5 common themes you see — you'll use these in Step 3.

Tell me when done.

---

## STEP 3 — Intent Classifier + Golden Set

### What you do manually:

**1. Run the intent builder:**
```
python src/build_intents.py
```

**2. Run the golden set builder:**
```
python src/build_golden_set.py
```
This creates `evaluation/golden_set.csv` with ~200 rows.

**3. IMPORTANT — Hand-label the golden set (this is the real work):**
- Open `evaluation/golden_set.csv` in Excel
- There are columns: `tweet_id`, `text`, `auto_intent`, `your_intent`, `expected_action`
- Go through ALL rows and:
  - Fix `your_intent` where `auto_intent` looks wrong
  - Set `expected_action` to either `auto_handle` or `escalate`
  - Use these rules for `expected_action`:
    - escalate: mentions refund/money/legal/angry/urgent/account hacked
    - auto_handle: general questions, tracking, FAQs
- You don't need to fix every single row — fix the ones that are clearly wrong
- Aim to review all ~200 rows (takes ~30-45 minutes)

**4. Save the file** when done (keep it as CSV).

Tell me when done.

---

## STEP 4 — Response Generator + Escalation

### What you do manually:

**1. Get an API key** (pick one — both have free tiers):

Option A — OpenAI:
- Go to: https://platform.openai.com/api-keys
- Create a key, copy it
- Open `.env`, set: `OPENAI_API_KEY=sk-...your key...`

Option B — Google Gemini (free, no credit card):
- Go to: https://aistudio.google.com/app/apikey
- Create a key, copy it
- Open `.env`, set: `GEMINI_API_KEY=...your key...`

Option C — No API key (skip LLM, use extractive only):
- Leave `.env` as-is
- The system will use TF-IDF retrieval only (still works, just less fluent replies)

**2. Run the agent on a test message:**
```
python src/agent.py --message "My order hasn't arrived and it's been 2 weeks"
```
Check that it prints: intent, reply, and escalation decision.

**3. Run it on 5 more messages** of your own to sanity-check the output.
Note anything that looks wrong — you'll use this in the failure analysis.

Tell me when done.

---

## STEP 5 — Evaluation Harness

### What you do manually:

**1. Run the full evaluation:**
```
python src/run_evaluation.py
```
This prints metrics and saves `evaluation/results.json`.

**2. Fill in human scores** (this proves your LLM judge is calibrated):
- Open `evaluation/human_scores_template.csv`
- It has 50 rows with columns: `tweet_id`, `generated_reply`, `human_score`, `notes`
- For each row, read the `generated_reply` and score it 1-5:
  - 5 = perfect, directly addresses the issue
  - 4 = good, minor issues
  - 3 = okay, somewhat relevant
  - 2 = poor, mostly irrelevant
  - 1 = bad, wrong or harmful
- Fill the `human_score` column, add brief `notes` if you want
- Takes ~20-30 minutes

**3. Run the judge agreement check:**
```
python src/run_evaluation.py --human-scores evaluation/human_scores_template.csv
```
This prints the correlation between your scores and the LLM judge.

Tell me when done.

---

## STEP 6 — Report + Submit

### What you do manually:

**1. Run the final pipeline end-to-end once more** to get clean numbers:
```
python src/run_evaluation.py
```
Note down the final numbers for the report.

**2. Review `report/report.md`** — it's pre-written but has placeholders like `[YOUR_BRAND]`.
Fill in:
- Your brand name everywhere it says `[YOUR_BRAND]`
- The actual metric numbers from your evaluation run
- 2-3 real failure examples from your testing in Step 4
- Your honest opinion in the "What is misleading" section

**3. Review `report/decision_log.md`** — add 2-3 decisions you personally made
(e.g., why you picked that brand, any label corrections you made in Step 3).

**4. Push everything to GitHub:**
```
git add .
git commit -m "complete: hiver support agent submission"
git push origin main
```

**5. Submit:**
- Go to: https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f
- Paste your GitHub repo URL
- Attach or paste `report/report.md`
- Submit

---

## Uninstall / Cleanup (after submission)

To remove the virtual environment:
```
rmdir /s /q venv
```

To remove processed data:
```
del data\processed\*.csv
```

Raw data (`data/raw/twcs.csv`) is already gitignored — just delete it locally if you want to free space.

---

## Quick Reference — All Commands in Order

```
# Step 1
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# Step 2
python src/prepare_data.py

# Step 3
python src/build_intents.py
python src/build_golden_set.py

# Step 4
python src/agent.py --message "My order hasn't arrived"

# Step 5
python src/run_evaluation.py
python src/run_evaluation.py --human-scores evaluation/human_scores_template.csv

# Step 6
git add .
git commit -m "complete: hiver support agent submission"
git push origin main
```

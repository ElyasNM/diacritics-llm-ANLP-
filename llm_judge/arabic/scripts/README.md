# LLM-as-Judge Pilot Validation Pipeline

Validates whether GPT-4o-mini can reliably judge Arabic QA answer correctness
before committing to full-scale judging.

---

## Pipeline — Run in This Order

### Step 1 — Build stratified sample

```bash
python 1_sample.py \
    --results_dir /path/to/ANLP-results \
    --output_dir  ./pilot
```

Produces three files:
- `pilot/pilot_sample.json` — full data for judge script
- `pilot/pilot_sample_for_human_labeling.csv` — **BLINDED**: only question, context, gold answers, prediction
- `pilot/pilot_sample_metadata.csv` — model, condition, F1, EM (merge back after labeling)

Sample is balanced: 30 ARCD / 30 TyDi QA / 30 MKQA,
10 per model (Qwen/SILMA/DictaLM2), 5 plain / 5 tashkeel per model,
distributed across high/medium/low F1 levels.

F1 levels are lexical-overlap buckets, NOT correctness judgments:
- high_f1:   F1 >= 0.8
- medium_f1: 0.2 <= F1 < 0.8
- low_f1:    F1 < 0.2

---

### ⚠️ BEFORE running the judge — manual labeling

Open `pilot/pilot_sample_for_human_labeling.csv` and fill in `human_label`
for all 90 rows:
  CORRECT / PARTIALLY_CORRECT / INCORRECT

Rules:
- Label BEFORE looking at GPT output
- The CSV is blinded — model name, condition, F1, EM are hidden deliberately
- If two annotators: each labels independently, then compare between yourselves

Labeling rubric:
  CORRECT:           Answer fully correct. Harmless extra info allowed.
  PARTIALLY_CORRECT: Contains correct info but incomplete, OR correct + material error.
  INCORRECT:         Core answer is wrong, unsupported, or doesn't answer the question.

Key: correct answer + harmless elaboration = CORRECT, not PARTIALLY_CORRECT.
Example: Gold = "2001", Prediction = "عام 2001" → CORRECT

---

### Step 2 — Run GPT judge

```bash
export OPENAI_API_KEY=sk-...

python 2_judge.py \
    --sample  pilot/pilot_sample.json \
    --output  pilot/judge_results.jsonl \
    --model   gpt-4o-mini-2024-07-18
```

- Uses a pinned model snapshot for reproducibility
- Judge is blinded: sees only question, full context, gold answers, prediction
- Automatically adds 15 duplicates for consistency checking
- Saves results incrementally (safe to interrupt and resume)
- Output is JSONL (one record per line)

Estimated cost: < $0.02 for 105 calls with gpt-4o-mini

---

### Step 3 — Analyze agreement

```bash
python 3_agreement.py \
    --judge  pilot/judge_results.jsonl \
    --human  pilot/pilot_sample_for_human_labeling.csv \
    --output pilot/agreement_report.json
```

Computes and prints:
- % agreement and Cohen's kappa (standard unweighted, nominal categories)
- Confusion matrix (human rows x judge columns)
- Per-class precision/recall/F1
- Consistency check (% of duplicates judged identically, errors excluded)
- Disagreements broken down by dataset, condition, F1 level
- Decision recommendation

---

## Decision guide after pilot

| Cohen's kappa | Consistency | Decision |
|---|---|---|
| >= 0.80 | >= 85% | Almost perfect — proceed |
| 0.60–0.79 | >= 80% | Substantial — proceed, review disagreements |
| 0.40–0.59 | any | Moderate — refine rubric, re-pilot |
| < 0.40 | any | Poor — do not scale up |

---

## Requirements

```bash
pip install openai
```

Python 3.8+. No GPU required.

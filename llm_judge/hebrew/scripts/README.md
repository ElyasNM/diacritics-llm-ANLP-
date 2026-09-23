# Hebrew Niqud — LLM Judge Pipeline

Mirrors the validated Arabic pipeline. All hardening fixes from the Arabic run
are already applied (see "Lessons applied" at the bottom).

## Data

| | |
|---|---|
| Datasets | HEQ (600), ParaShoot (500), MKQA (400) = 1,500 questions |
| Models | qwen2.5-7b, dictalm2 |
| Conditions | no_niqud / with_niqud |
| Total judgments | 1,500 x 2 models x 2 conditions = **6,000** |
| Estimated cost | ~$1.14 - $1.37 |
| Estimated runtime | ~80 min with 4 workers |

Note: Hebrew has 2 models vs Arabic's 3, so 6,000 calls instead of 9,000.

---

## Step 0 (recommended) — Pilot validation

Validate the judge on 84 human-labeled examples before the full run.

```bash
python 1_sample_hebrew.py \
    --results_dir /path/to/hebrew/results \
    --output_dir  ./hebrew_pilot
```

Then **manually label** `hebrew_pilot/pilot_sample_for_human_labeling.csv`
(column `human_label`: CORRECT / INCORRECT) **before** running the judge.
The CSV is blinded — model, condition, F1 and EM are in a separate metadata file.

```bash
export OPENAI_API_KEY=sk-...

python 2_judge_hebrew.py \
    --sample  hebrew_pilot/pilot_sample.json \
    --output  hebrew_pilot/judge_results_binary.jsonl \
    --model   gpt-4o-mini-2024-07-18

python 3_agreement_hebrew.py \
    --judge  hebrew_pilot/judge_results_binary.jsonl \
    --human  hebrew_pilot/pilot_sample_for_human_labeling.csv \
    --output hebrew_pilot/agreement_report_binary.json
```

Proceed if Cohen's kappa >= 0.60 and consistency >= 80%.
(Arabic pilot achieved kappa = 0.702, consistency = 100%.)

---

## Step 1 — Build the evaluation set

```bash
python build_eval_set_hebrew.py \
    --results_dir /path/to/hebrew/results \
    --output_dir  ./hebrew_eval
```

Produces `question_set.json`, `judge_input.jsonl`, `to_judge.jsonl`,
`selected_ids.txt`. Verify it prints **Perfectly paired: YES**.

---

## Step 2 — Run the judge

```bash
mkdir -p hebrew_eval

python run_judge_hebrew.py \
    --input       hebrew_eval/to_judge.jsonl \
    --pre_labeled hebrew_eval/judge_input.jsonl \
    --output      hebrew_eval/judge_output.jsonl \
    --workers     4
```

Watch the first progress line at 500 rows. Errors of 0-5 = on track.
20+ = stop and reduce workers.

Safe to interrupt — rerunning resumes and retries any ERROR rows.

---

## Step 3 — Compute deltas

```bash
python compute_judge_delta_hebrew.py \
    --output  hebrew_eval/judge_output.jsonl \
    --report  hebrew_eval/judge_delta_report.json
```

Reports Delta LLM-Judge per model x dataset, per model overall, combined
Hebrew overall, by answer-length bucket, and the four paired transitions
(CC / CI harmed / IC helped / II).

---

## Verify completeness before analysis

```bash
python3 - <<'PY'
import json
VALID = {"CORRECT", "INCORRECT"}
expected = {json.loads(l)["row_id"] for l in open("hebrew_eval/to_judge.jsonl", encoding="utf-8")}
valid = set()
for l in open("hebrew_eval/judge_output.jsonl", encoding="utf-8"):
    try: r = json.loads(l)
    except: continue
    if r.get("judge_label") in VALID: valid.add(r.get("row_id"))
print("Expected:", len(expected), "Valid:", len(valid), "Unresolved:", len(expected - valid))
PY
```

---

## Hebrew-specific judge rules

Beyond the Arabic rules, the Hebrew prompt explicitly instructs the judge to
ignore:

- **Niqud** present or absent
- **Ktiv haser vs. ktiv male** (defective vs. full spelling) — important, since
  the Hebrew report notes this is a real confound between the niqud and plain
  question pairs
- Definite-article prefixes and other harmless morphological differences

---

## Lessons applied from the Arabic run

| Problem hit in Arabic | Fix already in these scripts |
|---|---|
| SDK retries stacked on ours (up to 9 HTTP calls per row) | `OpenAI(..., max_retries=0)` |
| 1s/2s backoff too short for rate-limit windows | 10s / 20s backoff |
| All workers retried at the same instant | `+ random.uniform(0, 3)` jitter |
| ERROR rows skipped on resume | Only valid labels count as completed |
| Aggressive worker counts burned the 10,000 RPD cap | Default 4 workers; start there |
| One unpaired row from an empty prediction | Empty predictions pre-labeled INCORRECT |
| Judge could infer the condition from the question | Judge always sees the plain question |

"""
Step 2: Run GPT-4o-mini as a blinded judge on the pilot sample.

Binary rubric: CORRECT or INCORRECT only.

The judge receives only:
  - Question (in Hebrew)
  - Full context (if available)
  - Gold answer(s)
  - Candidate answer

It does NOT receive: model name, condition (plain/tashkeel), F1, EM.

Includes 15 duplicate examples for consistency checking.
Results are saved incrementally (after each call) to avoid data loss.

Usage:
  export OPENAI_API_KEY=sk-...
  python 2_judge.py \
      --sample  pilot_sample.json \
      --output  pilot/judge_results_binary.jsonl \
      --model   gpt-4o-mini-2024-07-18
"""

import argparse
import json
import os
import random
import random
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Run: pip install openai")

VALID_LABELS = {"CORRECT", "INCORRECT"}

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Hebrew QA evaluation judge. Your task is to assess whether a candidate answer correctly answers a question, given the reference (gold) answer(s).

Judge the candidate based on whether it sufficiently and factually answers the question relative to the provided gold answer(s).

If a context passage is provided, use it as additional evidence.
If no context is provided, judge based on the question and gold answer(s).
Treat the provided gold answer(s) as the reference; do not override them using outside knowledge.

You must respond with valid JSON only. No explanation outside the JSON.

The value of "label" must be exactly one of:
"CORRECT" or "INCORRECT".

Use exactly this JSON structure:
{
  "label": "CORRECT",
  "reason": "one short sentence explaining your judgment"
}

Labeling guidelines:

CORRECT:
- The candidate correctly and sufficiently answers what the question asks.
- It does NOT need to reproduce the entire gold answer if the shorter answer fully answers the question.
- Harmless additional information is allowed.
- Different wording, word order, spelling variants, transliteration, punctuation, capitalization, or formatting are allowed when the meaning is unchanged.
- An equivalent answer in another language or script is allowed when it unambiguously refers to the same entity, value, date, place, or concept.
- Do not penalize Hebrew niqud (vowel diacritics) being present or absent.
- Do not penalize ktiv haser vs. ktiv male spelling differences (defective vs. full spelling).
- Do not penalize harmless grammatical or morphological differences, including prefixes such as the definite article.
- Equivalent date, number, unit, or title formatting is allowed.
- If multiple gold answers are provided, the candidate is CORRECT if it correctly matches any valid gold answer.

Examples:
Gold: "בשנת 2002"
Candidate: "2002"
-> CORRECT

Gold: "ה-24"
Candidate: "יום 24"
-> CORRECT

Gold: "ירושלים"
Candidate: "יְרוּשָׁלַיִם"
-> CORRECT

INCORRECT:
- The core answer is factually wrong.
- The candidate does not answer the question.
- The candidate contains a material factual error or contradiction, even if part of the answer is correct.
- Merely being semantically related to the gold answer is not sufficient.

Important rules:
- Judge correctness, not lexical overlap.
- Judge whether the QUESTION has been sufficiently answered, not whether every word in the gold answer was reproduced.
- A shorter answer is CORRECT when it contains all information necessary to answer the question.
- Extra information is allowed only when it does not introduce a material error.
- Do not use F1, EM, model identity, or experimental condition in your judgment.
"""

USER_TEMPLATE_WITH_CONTEXT = """Question: {question}

Context: {context}

Gold answer(s): {gold_answers}

Candidate answer: {prediction}"""

USER_TEMPLATE_NO_CONTEXT = """Question: {question}

Gold answer(s): {gold_answers}

Candidate answer: {prediction}"""


def build_user_message(ex):
    gold = " | ".join(ex["answers"]) if isinstance(ex["answers"], list) else str(ex["answers"])
    if ex.get("context"):
        return USER_TEMPLATE_WITH_CONTEXT.format(
            question=ex["question"],
            context=ex["context"],
            gold_answers=gold,
            prediction=ex["prediction"],
        )
    return USER_TEMPLATE_NO_CONTEXT.format(
        question=ex["question"],
        gold_answers=gold,
        prediction=ex["prediction"],
    )


def call_judge(client, ex, model):
    user_msg = build_user_message(ex)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            label = parsed.get("label", "").upper().strip()
            if label not in VALID_LABELS:
                raise ValueError(f"Unexpected label: {label!r}")
            return {"label": label, "reason": parsed.get("reason", ""), "raw": raw}
        except Exception as e:
            if attempt == 2:
                return {"label": "ERROR", "reason": str(e), "raw": ""}
            time.sleep(10 * (attempt + 1) + random.uniform(0, 3))


def run_judge(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY environment variable first.")

    client = OpenAI(api_key=api_key, max_retries=0)

    with open(args.sample, encoding="utf-8") as f:
        sample = json.load(f)
    print(f"Loaded {len(sample)} examples from {args.sample}")

    # Add 15 consistency-check duplicates
    random.seed(99)
    consistency_ids = set(
        ex["pilot_id"] for ex in random.sample(sample, min(15, len(sample)))
    )
    duplicates = []
    for ex in sample:
        if ex["pilot_id"] in consistency_ids:
            dup = dict(ex)
            dup["pilot_id_original"] = ex["pilot_id"]
            dup["pilot_id"] = f"{ex['pilot_id']}_dup"
            dup["is_duplicate"] = True
            duplicates.append(dup)

    for ex in sample:
        ex["is_duplicate"] = False
        ex["pilot_id_original"] = ex["pilot_id"]

    full_batch = sample + duplicates
    random.shuffle(full_batch)

    print(f"Running judge on {len(full_batch)} calls "
          f"({len(sample)} originals + {len(duplicates)} consistency duplicates)")

    # Load already-completed results (incremental saving — resume if interrupted)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("judge_label") in VALID_LABELS:
                        completed_ids.add(r["pilot_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(completed_ids)} already done, skipping.")

    errors = 0
    out_file = open(out_path, "a", encoding="utf-8")

    try:
        for i, ex in enumerate(full_batch):
            pid = ex["pilot_id"]
            if pid in completed_ids:
                print(f"  [{i+1:3d}/{len(full_batch)}] pilot_id={pid} — skipped (already done)")
                continue

            print(f"  [{i+1:3d}/{len(full_batch)}] pilot_id={pid} "
                  f"source={ex['source']} condition={ex['condition']}", end=" ... ")

            judgment = call_judge(client, ex, model=args.model)
            print(judgment["label"])

            record = {
                "pilot_id":          pid,
                "pilot_id_original": ex.get("pilot_id_original", pid),
                "is_duplicate":      ex["is_duplicate"],
                "source":            ex["source"],
                "model_key":         ex["model_key"],
                "condition":         ex["condition"],
                "f1_level":          ex["f1_level"],
                "f1":                ex["f1"],
                "em":                ex["em"],
                "question":          ex["question"],
                "gold_answers":      ex["answers"],
                "prediction":        ex["prediction"],
                "judge_label":       judgment["label"],
                "judge_reason":      judgment["reason"],
            }
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()

            if judgment["label"] == "ERROR":
                errors += 1
            time.sleep(0.3)
    finally:
        out_file.close()

    # Load all results for summary
    results = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    print(f"\nSaved {len(results)} judgments -> {out_path}  (JSONL format)")
    if errors:
        print(f"WARNING: {errors} errors in this run")

    from collections import Counter
    originals = [r for r in results if not r["is_duplicate"]]
    label_dist = Counter(r["judge_label"] for r in originals)
    print("\n--- Judge label distribution (originals only) ---")
    for label, count in sorted(label_dist.items()):
        print(f"  {label:<22} {count:3d} ({100*count/len(originals):.1f}%)")

    print("\nReminder: human labeling (binary) should have been completed BEFORE running this script.")
    print("Then run: python 3_agreement.py --judge", args.output,
          "--human pilot_sample_for_human_labeling_binary.csv --output pilot/agreement_report_binary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",  required=True)
    parser.add_argument("--output",  default="pilot/judge_results_binary.jsonl")
    parser.add_argument("--model",   default="gpt-4o-mini-2024-07-18")
    args = parser.parse_args()
    run_judge(args)

"""
Run GPT-4o-mini as a blinded judge with concurrent workers.

Usage:
  export OPENAI_API_KEY=sk-...
  python run_judge.py \\
      --input        full_eval/to_judge.jsonl \\
      --pre_labeled  full_eval/judge_input.jsonl \\
      --output       full_eval/judge_output.jsonl \\
      --model        gpt-4o-mini-2024-07-18 \\
      --workers      4
"""

import argparse
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Run: pip install openai")

VALID_LABELS = {"CORRECT", "INCORRECT"}

SYSTEM_PROMPT = """You are an Arabic QA evaluation judge. Your task is to assess whether a candidate answer correctly answers a question, given the reference (gold) answer(s).

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
- Do not penalize Arabic diacritics (Tashkeel) being present or absent.
- Do not penalize harmless grammatical/case-ending differences.
- Equivalent date, number, unit, or title formatting is allowed.
- If multiple gold answers are provided, the candidate is CORRECT if it correctly matches any valid gold answer.

Examples:
Gold: "في عام 2002"
Candidate: "2002"
-> CORRECT

Gold: "1999م"
Candidate: "1999"
-> CORRECT

Gold: "مسقط"
Candidate: "مَسْقَط"
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

USER_TEMPLATE = """Question: {question}

Gold answer(s): {gold_answers}

Candidate answer: {prediction}"""


def build_message(row):
    gold = " | ".join(row["gold_answers"]) if isinstance(row["gold_answers"], list) else str(row["gold_answers"])
    return USER_TEMPLATE.format(
        question=row["question"],
        gold_answers=gold,
        prediction=row["prediction"],
    )


def call_judge(client, row, model):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_message(row)},
                ],
                temperature=0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            raw    = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            label  = parsed.get("label", "").upper().strip()
            if label not in VALID_LABELS:
                raise ValueError(f"Unexpected label: {label!r}")
            return {"label": label, "reason": parsed.get("reason", "")}
        except Exception as e:
            if attempt == 2:
                return {"label": "ERROR", "reason": str(e)}
            # Backoff with jitter: without the random component, all workers
            # that hit a 429 at the same moment would retry at exactly the
            # same time, producing another synchronized burst.
            time.sleep(10 * (attempt + 1) + random.uniform(0, 3))


def run(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY environment variable first.")

    # A client is created per row (see judge_one below), not per worker.
    # This costs a little connection-reuse efficiency but is simple and safe.
    # max_retries=0 disables the SDK's built-in automatic retries so that
    # only our own retry loop in call_judge() runs — otherwise a single
    # failed call can trigger up to 3 (our attempts) x 3 (SDK retries) = 9
    # HTTP requests, which makes rate-limit backoff much worse and burns
    # through the daily request quota.
    def make_client():
        return OpenAI(api_key=api_key, max_retries=0)

    # Load rows to judge
    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if not r.get("pre_labeled", False):
                    rows.append(r)
    print(f"Loaded {len(rows):,} rows to judge from {args.input}")

    # Load already-completed row_ids (only valid labels, not ERRORs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("judge_label") in VALID_LABELS:
                        completed.add(r["row_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(completed):,} valid results already saved.")

    # Write pre-labeled rows first (only on fresh run)
    if args.pre_labeled and not out_path.exists():
        pre_count = 0
        with open(args.pre_labeled, encoding="utf-8") as f_in, \
             open(out_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("pre_labeled", False):
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    pre_count += 1
        print(f"Wrote {pre_count:,} pre-labeled rows.")

    todo = [r for r in rows if r["row_id"] not in completed]
    print(f"Rows still to judge: {len(todo):,}")

    cost_est = len(todo) * 0.02 / 105
    est_hrs  = len(todo) * 3.24 / args.workers / 3600
    print(f"Estimated cost:    ~${cost_est:.2f}")
    print(f"Estimated runtime: ~{est_hrs:.1f} hrs with {args.workers} workers\n")

    # Thread-safe writer
    write_lock  = threading.Lock()
    counter     = {"done": 0, "errors": 0}
    start_time  = time.time()

    out_file = open(out_path, "a", encoding="utf-8")

    def judge_one(row):
        client   = make_client()
        judgment = call_judge(client, row, args.model)
        result   = {
            "row_id":               row["row_id"],
            "example_id":           row["example_id"],
            "source":               row["source"],
            "model":                row["model"],
            "condition":            row["condition"],
            "answer_length_bucket": row["answer_length_bucket"],
            "f1":                   row["f1"],
            "em":                   row["em"],
            "pre_labeled":          False,
            "judge_label":          judgment["label"],
            "judge_reason":         judgment["reason"],
        }
        with write_lock:
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()
            counter["done"] += 1
            if judgment["label"] == "ERROR":
                counter["errors"] += 1
            done = counter["done"]
            if done % 500 == 0 or done == len(todo):
                elapsed = time.time() - start_time
                rate    = done / elapsed if elapsed > 0 else 0
                eta_hrs = (len(todo) - done) / rate / 3600 if rate > 0 else 0
                cost_so_far = done * 0.02 / 105
                print(f"  [{done:,}/{len(todo):,}] "
                      f"{100*done/len(todo):.1f}%  "
                      f"errors={counter['errors']}  "
                      f"cost≈${cost_so_far:.2f}  "
                      f"ETA≈{eta_hrs:.1f}h")
        return result

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(judge_one, row): row for row in todo}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  Worker error: {e}")
    finally:
        out_file.close()

    # Final summary
    results = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    valid = [r for r in results if r.get("judge_label") in VALID_LABELS]
    print(f"\nTotal saved: {len(results):,}  valid: {len(valid):,}  errors: {counter['errors']}")

    print("\n--- Judge accuracy by model x condition ---")
    from collections import defaultdict
    by_mc = defaultdict(list)
    for r in valid:
        by_mc[(r["model"], r["condition"])].append(r["judge_label"] == "CORRECT")
    for (model, condition), flags in sorted(by_mc.items()):
        acc = 100 * sum(flags) / len(flags)
        print(f"  {model:10} / {condition:15}: {acc:.1f}%  (n={len(flags):,})")

    print(f"\nFull results -> {out_path}")
    print(f"Next: python compute_judge_delta.py --output {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--pre_labeled", default=None)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--model",       default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--workers",     type=int, default=4)
    args = parser.parse_args()
    run(args)

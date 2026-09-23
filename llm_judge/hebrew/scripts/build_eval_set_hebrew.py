"""
Build the Hebrew LLM-judge evaluation set.

Mirrors the Arabic pipeline exactly, adapted for Hebrew:
  - Datasets: HEQ, ParaShoot, MKQA (Hebrew)
  - Models:   qwen2.5-7b, dictalm2
  - Conditions: no_niqud / with_niqud

Targets (symmetric with Arabic 1,500-question design):
  HEQ:        600
  ParaShoot:  500
  MKQA:       400
  Total:    1,500 unique questions
  Calls:    1,500 x 2 models x 2 conditions = 6,000

NOTE: Hebrew has 2 models (not 3 like Arabic), so total calls is 6,000 not 9,000.

Usage:
  python build_eval_set_hebrew.py \
      --results_dir /path/to/hebrew/results \
      --output_dir  ./hebrew_eval
"""

import argparse
import json
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

random.seed(42)

MODELS     = ["qwen2.5-7b", "dictalm2"]
CONDITIONS = ["no_niqud", "with_niqud"]

TARGETS = {
    "heq":       600,
    "parashoot": 500,
    "mkqa":      400,
}

VALID_LABELS = {"CORRECT", "INCORRECT"}


def answer_length_bucket(answers):
    if not answers:
        return "short"
    valid = [a for a in answers if str(a).strip()]
    if not valid:
        return "short"
    min_len = min(len(str(a).split()) for a in valid)
    if min_len <= 1:  return "short"
    if min_len <= 5:  return "medium"
    if min_len <= 15: return "long"
    return "very_long"


HTML_RE    = re.compile(r'<[^>]+>')
PARSOID_RE = re.compile(r'parsoid|data-parsoid', re.IGNORECASE)

def is_corrupted_answer(answers):
    if not answers:
        return True
    for a in answers:
        s = str(a).strip()
        if not s or HTML_RE.search(s) or PARSOID_RE.search(s):
            return True
    return False

def is_empty_prediction(prediction):
    return not prediction or not str(prediction).strip()


def load_all(results_dir, model, condition):
    path = Path(results_dir) / model / f"{condition}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["examples"]


def load_and_filter_canonical(results_dir, model, condition):
    examples = load_all(results_dir, model, condition)
    clean, skip_c, skip_p = [], 0, 0
    for ex in examples:
        if is_corrupted_answer(ex.get("answers", [])):
            skip_c += 1; continue
        if is_empty_prediction(ex.get("prediction", "")):
            skip_p += 1; continue
        clean.append(ex)
    return clean, skip_c, skip_p


def stratified_sample(examples, n, used_questions):
    pool = [ex for ex in examples if ex["question"] not in used_questions]
    if len(pool) <= n:
        return pool

    by_bucket = defaultdict(list)
    for ex in pool:
        by_bucket[answer_length_bucket(ex["answers"])].append(ex)

    total     = len(pool)
    selected  = []
    remaining = n
    buckets   = list(by_bucket.keys())

    for i, bucket in enumerate(buckets):
        if i == len(buckets) - 1:
            target = remaining
        else:
            target = round(n * len(by_bucket[bucket]) / total)
            target = min(target, len(by_bucket[bucket]))
        take = min(target, remaining, len(by_bucket[bucket]))
        selected.extend(random.sample(by_bucket[bucket], take))
        remaining -= take

    used_ids  = {ex["id"] for ex in selected}
    leftover  = [ex for ex in pool if ex["id"] not in used_ids]
    shortfall = n - len(selected)
    if shortfall > 0 and leftover:
        selected.extend(random.sample(leftover, min(shortfall, len(leftover))))

    return selected


def build(args):
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    CANONICAL_MODEL     = "qwen2.5-7b"
    CANONICAL_CONDITION = "no_niqud"

    print("=== Step 1: Build question set ===\n")

    canonical, skip_c, skip_p = load_and_filter_canonical(
        results_dir, CANONICAL_MODEL, CANONICAL_CONDITION
    )
    print(f"Canonical source ({CANONICAL_MODEL}/{CANONICAL_CONDITION}):")
    print(f"  Total: {len(canonical)+skip_c+skip_p}  corrupted: {skip_c}  "
          f"empty: {skip_p}  clean: {len(canonical)}")

    by_source = defaultdict(list)
    for ex in canonical:
        by_source[ex["source"]].append(ex)

    for src, exs in sorted(by_source.items()):
        print(f"  {src}: {len(exs)} clean examples")
    print()

    # Deduplicate at question level
    for src in by_source:
        seen_q = set()
        deduped = []
        for ex in by_source[src]:
            if ex["question"] not in seen_q:
                seen_q.add(ex["question"])
                deduped.append(ex)
        before = len(by_source[src])
        by_source[src] = deduped
        if before != len(deduped):
            print(f"  Deduped {src}: {before} -> {len(deduped)}")

    question_set   = []
    used_questions = set()

    for src in ["heq", "parashoot", "mkqa"]:
        pool     = by_source.get(src, [])
        target   = TARGETS[src]
        selected = stratified_sample(pool, target, used_questions)
        for ex in selected:
            used_questions.add(ex["question"])
            ex["answer_length_bucket"] = answer_length_bucket(ex["answers"])
        question_set.extend(selected)
        print(f"Selected {len(selected):,} from {src} (target {target:,})")
        for b, c in sorted(Counter(answer_length_bucket(ex["answers"]) for ex in selected).items()):
            print(f"    {b}: {c}")

    print(f"\nTotal question set: {len(question_set):,}")

    # Save question set
    qs_path = output_dir / "question_set.json"
    with open(qs_path, "w", encoding="utf-8") as f:
        json.dump([{
            "example_id":           ex["id"],
            "source":               ex["source"],
            "question":             ex["question"],
            "gold_answers":         ex["answers"],
            "answer_length_bucket": ex["answer_length_bucket"],
        } for ex in question_set], f, ensure_ascii=False, indent=2)
    print(f"Saved question set -> {qs_path}")

    ids_path = output_dir / "selected_ids.txt"
    with open(ids_path, "w", encoding="utf-8") as f:
        for ex in question_set:
            f.write(ex["id"] + "\n")
    print(f"Saved selected IDs -> {ids_path}")

    # === Step 2: Expand ===
    print("\n=== Step 2: Build judge input file ===\n")

    id_to_meta = {
        ex["id"]: {
            "example_id":           ex["id"],
            "source":               ex["source"],
            "gold_answers":         ex["answers"],
            "answer_length_bucket": ex["answer_length_bucket"],
            "question_plain":       ex["question"],
        }
        for ex in question_set
    }
    selected_ids = set(id_to_meta.keys())

    all_rows    = []
    to_judge    = []
    row_id      = 1
    pre_labeled = 0

    for model in MODELS:
        for condition in CONDITIONS:
            examples = load_all(results_dir, model, condition)
            matched  = 0
            empty    = 0
            for ex in examples:
                if ex["id"] not in selected_ids:
                    continue
                meta = id_to_meta[ex["id"]]
                base = {
                    "row_id":                row_id,
                    "example_id":            meta["example_id"],
                    "source":                meta["source"],
                    "model":                 model,
                    "condition":             condition,
                    "answer_length_bucket":  meta["answer_length_bucket"],
                    # Always the plain (unvocalized) question — judge stays blind
                    # to which condition produced the prediction
                    "question":              meta["question_plain"],
                    "gold_answers":          meta["gold_answers"],
                    "prediction":            ex["prediction"],
                    "f1":                    ex["f1"],
                    "em":                    ex["em"],
                }
                if is_empty_prediction(ex.get("prediction", "")):
                    base["judge_label"]  = "INCORRECT"
                    base["judge_reason"] = "Empty model prediction."
                    base["pre_labeled"]  = True
                    all_rows.append(base)
                    empty += 1; pre_labeled += 1
                else:
                    base["pre_labeled"] = False
                    all_rows.append(base)
                    to_judge.append(base)
                row_id += 1; matched += 1

            print(f"  {model}/{condition}: {matched:,} rows "
                  f"({empty} pre-labeled INCORRECT)")

    n_total  = len(all_rows)
    n_judge  = len(to_judge)
    expected = len(question_set) * len(MODELS) * len(CONDITIONS)
    counts   = Counter((r["model"], r["condition"]) for r in all_rows)
    paired   = len(set(counts.values())) == 1

    print(f"\nTotal rows:        {n_total:,}  (expected {expected:,})")
    print(f"Pre-labeled:       {pre_labeled:,}")
    print(f"Rows to judge:     {n_judge:,}")
    print(f"Perfectly paired:  {'YES' if paired else 'NO'}")
    if not paired:
        for k, v in sorted(counts.items()):
            print(f"    {k}: {v}")

    cost_low  = n_judge * 0.02 / 105
    cost_high = cost_low * 1.20
    print(f"Estimated cost:    ~${cost_low:.2f} - ${cost_high:.2f}")
    print(f"Estimated runtime: ~{n_judge * 3.24 / 4 / 60:.0f} min with 4 workers")

    judge_path    = output_dir / "judge_input.jsonl"
    to_judge_path = output_dir / "to_judge.jsonl"

    with open(judge_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved judge_input  -> {judge_path}")

    with open(to_judge_path, "w", encoding="utf-8") as f:
        for row in to_judge:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved to_judge     -> {to_judge_path}")

    print(f"\nNext step:")
    print(f"  python run_judge_hebrew.py \\")
    print(f"      --input       {to_judge_path} \\")
    print(f"      --pre_labeled {judge_path} \\")
    print(f"      --output      {output_dir}/judge_output.jsonl \\")
    print(f"      --workers     4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir",  default="hebrew_eval")
    args = parser.parse_args()
    build(args)

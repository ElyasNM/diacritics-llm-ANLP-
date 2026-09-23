"""
Step 1: Build a stratified pilot sample of 90 examples for LLM-as-judge validation.

Sampling design:
  - 30 ARCD  / 30 TyDi QA / 30 MKQA
  - Within each dataset: 10 Qwen / 10 SILMA / 10 DictaLM2
  - Within each model:    5 Plain / 5 Tashkeel
  - Within each cell of 5: balanced across F1 levels
    (high_f1 / medium_f1 / low_f1)

F1 buckets:
  - high_f1:   F1 >= 0.8
  - medium_f1: 0.2 <= F1 < 0.8
  - low_f1:    F1 < 0.2

NOTE: These are lexical-overlap buckets, NOT correctness labels.
A low_f1 prediction may be perfectly correct but phrased differently.

Outputs:
  pilot_sample.json                   -- input for 2_judge.py (full data)
  pilot_sample_for_human_labeling.csv -- BLINDED: no model/condition/F1/EM
  pilot_sample_metadata.csv           -- metadata to merge back after labeling

Usage:
  python 1_sample.py --results_dir /path/to/results --output_dir ./pilot
"""

import argparse
import json
import random
import csv
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

MODELS = {
    "qwen":     "Qwen/Qwen2.5-7B-Instruct",
    "silma":    "silma-ai/SILMA-9B-Instruct-v1.0",
    "dictalm2": "dicta-il/dictalm2.0-instruct",
}

DATASETS   = ["arcd", "tydiqa", "mkqa"]
CONDITIONS = ["no_tashkeel", "with_tashkeel"]
PER_CELL   = 5
F1_LEVEL_TARGETS = {"high_f1": 2, "medium_f1": 1, "low_f1": 2}


def f1_level(f1):
    if f1 >= 0.8:  return "high_f1"
    if f1 >= 0.2:  return "medium_f1"
    return "low_f1"


def load_examples(results_dir, model_key, condition):
    path = results_dir / model_key / f"{condition}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for ex in data["examples"]:
        examples.append({
            "id":        ex["id"],
            "source":    ex["source"],
            "question":  ex["question"],
            "context":   ex.get("context", None),
            "answers":   ex["answers"],
            "prediction":ex["prediction"],
            "f1":        ex["f1"],
            "em":        ex["em"],
            "model_key": model_key,
            "condition": condition,
            "f1_level":  f1_level(ex["f1"]),
        })
    return examples


def stratified_sample_cell(examples, n, targets):
    by_level = defaultdict(list)
    for ex in examples:
        by_level[ex["f1_level"]].append(ex)
    selected = []
    remaining = n
    for level, target_n in targets.items():
        bucket = by_level[level]
        take = min(target_n, len(bucket), remaining)
        selected.extend(random.sample(bucket, take))
        remaining -= take
    used_ids = {ex["id"] for ex in selected}
    leftover = [ex for ex in examples if ex["id"] not in used_ids]
    if remaining > 0 and leftover:
        selected.extend(random.sample(leftover, min(remaining, len(leftover))))
    return selected


def build_sample(results_dir):
    all_by_key = {}
    for model_key in MODELS:
        for condition in CONDITIONS:
            examples = load_examples(results_dir, model_key, condition)
            for ex in examples:
                key = (model_key, condition, ex["source"])
                all_by_key.setdefault(key, []).append(ex)

    sample = []
    pilot_id = 1
    for dataset in DATASETS:
        dataset_sample = []
        for model_key in MODELS:
            for condition in CONDITIONS:
                key = (model_key, condition, dataset)
                pool = all_by_key.get(key, [])
                if not pool:
                    print(f"  WARNING: no examples for {key}")
                    continue
                cell_sample = stratified_sample_cell(pool, PER_CELL, F1_LEVEL_TARGETS)
                dataset_sample.extend(cell_sample)
        random.shuffle(dataset_sample)
        for ex in dataset_sample:
            ex["pilot_id"] = pilot_id
            pilot_id += 1
            sample.append(ex)
    return sample


def save_outputs(sample, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full JSON for judge script
    judge_input = []
    for ex in sample:
        judge_input.append({
            "pilot_id":   ex["pilot_id"],
            "source":     ex["source"],
            "model_key":  ex["model_key"],
            "condition":  ex["condition"],
            "f1_level":   ex["f1_level"],
            "f1":         ex["f1"],
            "em":         ex["em"],
            "question":   ex["question"],
            "context":    ex["context"],
            "answers":    ex["answers"],
            "prediction": ex["prediction"],
        })
    json_path = output_dir / "pilot_sample.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(judge_input, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(judge_input)} examples -> {json_path}")

    # 2. BLINDED CSV for human labeling
    # Exposes ONLY: pilot_id, question, context, gold_answers, prediction
    # Does NOT expose: model_key, condition, f1, em, f1_level
    blind_path = output_dir / "pilot_sample_for_human_labeling.csv"
    with open(blind_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pilot_id", "question", "context", "gold_answers",
            "prediction", "human_label", "human_notes",
        ])
        writer.writeheader()
        for ex in judge_input:
            writer.writerow({
                "pilot_id":    ex["pilot_id"],
                "question":    ex["question"],
                "context":     ex["context"] or "",
                "gold_answers": " | ".join(ex["answers"]),
                "prediction":  ex["prediction"],
                "human_label": "",
                "human_notes": "",
            })
    print(f"Saved blinded human CSV -> {blind_path}")

    # 3. Metadata file (merge back after labeling)
    meta_path = output_dir / "pilot_sample_metadata.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pilot_id", "source", "model_key", "condition", "f1_level", "f1", "em"
        ])
        writer.writeheader()
        for ex in judge_input:
            writer.writerow({
                "pilot_id":  ex["pilot_id"],
                "source":    ex["source"],
                "model_key": ex["model_key"],
                "condition": ex["condition"],
                "f1_level":  ex["f1_level"],
                "f1":        ex["f1"],
                "em":        ex["em"],
            })
    print(f"Saved metadata CSV -> {meta_path}")

    # Breakdown
    print("\n--- Sample breakdown ---")
    print(f"  Dataset:   {dict(Counter(ex['source']    for ex in judge_input))}")
    print(f"  Model:     {dict(Counter(ex['model_key'] for ex in judge_input))}")
    print(f"  Condition: {dict(Counter(ex['condition'] for ex in judge_input))}")
    print(f"  F1 level:  {dict(Counter(ex['f1_level']  for ex in judge_input))}")
    print()
    print("IMPORTANT: Fill in pilot_sample_for_human_labeling.csv BEFORE")
    print("running 2_judge.py. Do not look at GPT output first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir",  default="pilot")
    args = parser.parse_args()
    print("Building stratified pilot sample...")
    sample = build_sample(Path(args.results_dir))
    save_outputs(sample, Path(args.output_dir))
    print(f"Done. Total examples: {len(sample)}")

"""
Step 3: Compare human labels vs GPT judge labels (binary version).

Computes:
  - Overall percentage agreement
  - Cohen's kappa
  - Confusion matrix (human rows, judge columns)
  - Per-class precision/recall/F1
  - Consistency check results (ignores ERROR labels)
  - Disagreement breakdown by dataset, condition, F1 level
  - Decision recommendation

Input:
  judge_results_binary.jsonl           -- from 2_judge.py (binary rubric)
  pilot_sample_for_human_labeling_binary.csv  -- with binary human_label filled in

Usage:
  python 3_agreement.py \
      --judge  pilot/judge_results_binary.jsonl \
      --human  pilot_sample_for_human_labeling_binary.csv \
      --output pilot/agreement_report_binary.json
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["CORRECT", "INCORRECT"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def cohen_kappa(y_human, y_judge, labels):
    """
    Standard unweighted Cohen's kappa (nominal categories).
    Not a macro metric — labels are treated as unordered categories.
    """
    n = len(y_human)
    if n == 0:
        return 0.0
    observed  = sum(h == j for h, j in zip(y_human, y_judge)) / n
    h_counts  = Counter(y_human)
    j_counts  = Counter(y_judge)
    expected  = sum((h_counts.get(l, 0) / n) * (j_counts.get(l, 0) / n) for l in labels)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def confusion_matrix(y_human, y_judge, labels):
    matrix = {h: {j: 0 for j in labels} for h in labels}
    for h, j in zip(y_human, y_judge):
        if h in matrix and j in matrix[h]:
            matrix[h][j] += 1
    return matrix


def per_class_metrics(y_human, y_judge, labels):
    metrics = {}
    for label in labels:
        tp = sum(h == label and j == label for h, j in zip(y_human, y_judge))
        fp = sum(h != label and j == label for h, j in zip(y_human, y_judge))
        fn = sum(h == label and j != label for h, j in zip(y_human, y_judge))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[label] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
    return metrics


# ── Load data ─────────────────────────────────────────────────────────────────

def load_human_labels(csv_path):
    labels = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("human_label", "").strip().upper()
            if label:
                try:
                    labels[int(row["pilot_id"])] = label
                except ValueError:
                    pass
    return labels


def load_judge_results(jsonl_path):
    originals  = []
    duplicates = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("is_duplicate"):
                    duplicates.append(r)
                else:
                    originals.append(r)
            except Exception:
                pass
    return originals, duplicates


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_consistency(duplicates, originals):
    orig_by_id = {r["pilot_id"]: r["judge_label"] for r in originals}
    consistent   = 0
    inconsistent = 0
    skipped      = 0
    cases        = []

    for dup in duplicates:
        orig_id    = dup.get("pilot_id_original")
        orig_label = orig_by_id.get(orig_id)
        dup_label  = dup["judge_label"]

        if orig_label not in LABELS or dup_label not in LABELS:
            skipped += 1
            continue

        match = orig_label == dup_label
        if match:
            consistent += 1
        else:
            inconsistent += 1
        cases.append({
            "pilot_id":     orig_id,
            "first_label":  orig_label,
            "second_label": dup_label,
            "consistent":   match,
        })

    total = consistent + inconsistent
    return {
        "total_valid_checks": total,
        "skipped_errors":     skipped,
        "consistent":         consistent,
        "inconsistent":       inconsistent,
        "consistency_pct":    round(100 * consistent / total, 1) if total else 0,
        "cases":              cases,
    }


def analyze_disagreements(paired):
    disagreements = [p for p in paired if p["human_label"] != p["judge_label"]]
    by_source    = defaultdict(int)
    by_condition = defaultdict(int)
    by_f1_level  = defaultdict(int)
    by_direction = defaultdict(int)

    for d in disagreements:
        by_source[d["source"]]       += 1
        by_condition[d["condition"]] += 1
        by_f1_level[d.get("f1_level", "unknown")] += 1
        direction = f"human={d['human_label']} -> judge={d['judge_label']}"
        by_direction[direction] += 1

    return {
        "total_disagreements": len(disagreements),
        "by_source":           dict(by_source),
        "by_condition":        dict(by_condition),
        "by_f1_level":         dict(by_f1_level),
        "by_direction":        dict(by_direction),
        "examples": [
            {
                "pilot_id":   d["pilot_id"],
                "source":     d["source"],
                "condition":  d["condition"],
                "f1_level":   d.get("f1_level"),
                "f1":         d["f1"],
                "question":   d["question"],
                "gold":       d["gold_answers"],
                "prediction": d["prediction"],
                "human":      d["human_label"],
                "judge":      d["judge_label"],
                "reason":     d.get("judge_reason", ""),
            }
            for d in disagreements
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_agreement(args):
    originals, duplicates = load_judge_results(args.judge)
    human_labels = load_human_labels(args.human)

    paired  = []
    skipped = 0
    for ex in originals:
        pid = ex["pilot_id"]
        try:
            pid_int = int(pid)
        except (ValueError, TypeError):
            skipped += 1
            continue
        h_label = human_labels.get(pid_int)
        if not h_label or h_label not in LABELS:
            skipped += 1
            continue
        if ex["judge_label"] not in LABELS:
            skipped += 1
            continue
        paired.append({**ex, "human_label": h_label})

    print(f"Paired examples: {len(paired)}  (skipped {skipped} missing/invalid labels)")
    if not paired:
        print("ERROR: No paired examples. Fill in human_label column in the binary CSV.")
        return

    y_human = [p["human_label"] for p in paired]
    y_judge = [p["judge_label"] for p in paired]
    n       = len(paired)

    pct_agree = round(100 * sum(h == j for h, j in zip(y_human, y_judge)) / n, 1)
    kappa     = round(cohen_kappa(y_human, y_judge, LABELS), 3)
    cm        = confusion_matrix(y_human, y_judge, LABELS)
    per_class = per_class_metrics(y_human, y_judge, LABELS)
    consistency   = analyze_consistency(duplicates, originals)
    disagreements = analyze_disagreements(paired)

    if kappa >= 0.80:
        kappa_interp = "Almost perfect agreement — proceed with full-scale judging."
    elif kappa >= 0.60:
        kappa_interp = "Substantial agreement — proceed, but review disagreement patterns."
    elif kappa >= 0.40:
        kappa_interp = "Moderate agreement — refine rubric before scaling up."
    else:
        kappa_interp = "Poor agreement — do not use as primary metric without major rubric changes."

    report = {
        "summary": {
            "n_paired":             n,
            "pct_agreement":        pct_agree,
            "cohen_kappa":          kappa,
            "kappa_note":           "Standard unweighted Cohen's kappa (nominal categories).",
            "kappa_interpretation": kappa_interp,
        },
        "confusion_matrix":  cm,
        "per_class_metrics": per_class,
        "consistency":       consistency,
        "disagreements":     disagreements,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Full report saved -> {out_path}")

    print("\n" + "=" * 65)
    print("AGREEMENT REPORT SUMMARY (BINARY)")
    print("=" * 65)
    print(f"  Paired examples:      {n}")
    print(f"  % Agreement:          {pct_agree}%")
    print(f"  Cohen's kappa:        {kappa}  (standard, unweighted)")
    print(f"  Interpretation:       {kappa_interp}")

    print("\n  Confusion matrix (rows = human label, cols = judge label):")
    header = f"  {'':22}" + "".join(f"{l:22}" for l in LABELS)
    print(header)
    for h_label in LABELS:
        row = f"  {h_label:22}" + "".join(
            f"{cm[h_label].get(j_label, 0):<22}" for j_label in LABELS
        )
        print(row)

    print("\n  Per-class metrics (from judge's perspective):")
    for label in LABELS:
        m = per_class[label]
        print(f"    {label:<22} P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}")

    print(f"\n  Consistency check: {consistency['consistency_pct']}% "
          f"({consistency['consistent']}/{consistency['total_valid_checks']} consistent, "
          f"{consistency['skipped_errors']} skipped due to errors)")

    print(f"\n  Total disagreements: {disagreements['total_disagreements']}")
    print("  By dataset:   ", disagreements["by_source"])
    print("  By condition: ", disagreements["by_condition"])
    print("  By F1 level:  ", disagreements["by_f1_level"])
    print("\n  Most common disagreement directions:")
    for direction, count in sorted(
        disagreements["by_direction"].items(), key=lambda x: -x[1]
    )[:5]:
        print(f"    {direction}: {count}")

    print("\n  Decision:")
    if kappa >= 0.60 and consistency["consistency_pct"] >= 80:
        print("  -> PROCEED with full-scale LLM judging.")
    elif kappa >= 0.40:
        print("  -> REFINE rubric, then re-pilot before scaling.")
    else:
        print("  -> DO NOT scale. Consider alternative metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge",  required=True)
    parser.add_argument("--human",  required=True)
    parser.add_argument("--output", default="pilot/agreement_report_binary.json")
    args = parser.parse_args()
    run_agreement(args)

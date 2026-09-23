"""
Step 3: Compare two evaluate.py output files and print the
Delta(tashkeel - no_tashkeel) table.

Usage:
  python compute_delta.py \\
      --baseline  results/silma/no_tashkeel.json \\
      --tashkeel  results/silma/with_tashkeel.json \\
      [--output   results/silma/delta.json]
"""

import argparse
import json
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_delta(args):
    base = load(args.baseline)
    tash = load(args.tashkeel)

    # Align per-example by ID
    base_by_id = {r["id"]: r for r in base["examples"]}
    tash_by_id = {r["id"]: r for r in tash["examples"]}
    common_ids = set(base_by_id) & set(tash_by_id)

    if len(common_ids) < len(base_by_id):
        print(
            f"Warning: {len(base_by_id) - len(common_ids)} IDs in baseline "
            f"not found in tashkeel results."
        )

    per_example = []
    for eid in sorted(common_ids):
        b = base_by_id[eid]
        t = tash_by_id[eid]
        per_example.append({
            "id":           eid,
            "source":       b["source"],
            "delta_em":     t["em"]  - b["em"],
            "delta_f1":     round(t["f1"] - b["f1"], 4),
            "base_em":      b["em"],    "tash_em":  t["em"],
            "base_f1":      b["f1"],    "tash_f1":  t["f1"],
            "base_pred":    b["prediction"],
            "tash_pred":    t["prediction"],
            "base_pred_len": b.get("pred_len_words", len(b["prediction"].split())),
            "tash_pred_len": t.get("pred_len_words", len(t["prediction"].split())),
            "answers":      b["answers"],
        })

    # Aggregate per source + overall
    sources = sorted(set(r["source"] for r in per_example)) + ["overall"]
    agg = {}
    for src in sources:
        subset = (
            per_example
            if src == "overall"
            else [r for r in per_example if r["source"] == src]
        )
        if not subset:
            continue
        n = len(subset)
        agg[src] = {
            "n":             n,
            "base_em":       round(sum(r["base_em"] for r in subset) / n * 100, 2),
            "tash_em":       round(sum(r["tash_em"] for r in subset) / n * 100, 2),
            "delta_em":      round(sum(r["delta_em"] for r in subset) / n * 100, 2),
            "base_f1":       round(sum(r["base_f1"] for r in subset) / n * 100, 2),
            "tash_f1":       round(sum(r["tash_f1"] for r in subset) / n * 100, 2),
            "delta_f1":      round(sum(r["delta_f1"] for r in subset) / n * 100, 2),
            # Flip counts
            "tash_helped_em": round(sum(1 for r in subset if r["delta_em"] > 0) / n * 100, 2),
            "tash_hurt_em":   round(sum(1 for r in subset if r["delta_em"] < 0) / n * 100, 2),
            # Verbosity: mean prediction length in words per condition
            "base_mean_len":  round(sum(r["base_pred_len"] for r in subset) / n, 2),
            "tash_mean_len":  round(sum(r["tash_pred_len"] for r in subset) / n, 2),
        }

    # Print table
    print(f"\n{'':=<74}")
    print(f"  Delta(tashkeel - no_tashkeel)   model: {base['model']}")
    print(f"{'':=<74}")
    header = (
        f"  {'Source':<12}  {'n':>5}  "
        f"{'Base EM':>8}  {'Tash EM':>8}  {'D EM':>6}  "
        f"{'Base F1':>8}  {'Tash F1':>8}  {'D F1':>6}"
    )
    print(header)
    print(f"  {'-'*70}")
    for src in sources:
        if src not in agg:
            continue
        s = agg[src]
        marker = " <--" if src == "overall" else ""
        print(
            f"  {src:<12}  {s['n']:>5}  "
            f"{s['base_em']:>7.1f}%  {s['tash_em']:>7.1f}%  {s['delta_em']:>+6.1f}%  "
            f"{s['base_f1']:>7.1f}%  {s['tash_f1']:>7.1f}%  {s['delta_f1']:>+6.1f}%"
            f"{marker}"
        )
    print(f"{'':=<74}")

    print("\n-- Tashkeel effect on individual examples (EM) --")
    for src in sources:
        if src not in agg:
            continue
        s = agg[src]
        churn = s["tash_helped_em"] + s["tash_hurt_em"]
        print(
            f"  {src:<12}  "
            f"helped: {s['tash_helped_em']:5.1f}%  "
            f"hurt: {s['tash_hurt_em']:5.1f}%  "
            f"churn: {churn:.1f}%"
        )

    print("\n-- Prediction verbosity (mean words) --")
    for src in sources:
        if src not in agg:
            continue
        s = agg[src]
        delta_len = round(s["tash_mean_len"] - s["base_mean_len"], 2)
        print(
            f"  {src:<12}  "
            f"base: {s['base_mean_len']:5.1f}  "
            f"tash: {s['tash_mean_len']:5.1f}  "
            f"delta: {delta_len:+.2f}"
        )

    output = {
        "model":       base["model"],
        "baseline":    args.baseline,
        "tashkeel":    args.tashkeel,
        "aggregate":   agg,
        "per_example": per_example,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nFull delta saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="results/no_tashkeel.json")
    parser.add_argument("--tashkeel", required=True, help="results/with_tashkeel.json")
    parser.add_argument("--output",   default=None,  help="Optional: save delta JSON")
    args = parser.parse_args()
    compute_delta(args)

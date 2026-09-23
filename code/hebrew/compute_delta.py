"""
Step 4: Compare two evaluate.py output files and print the Δ(niqud − no_niqud) table.

Usage:
  python compute_delta.py \\
      --baseline  results/no_niqud.json \\
      --niqud     results/with_niqud.json \\
      [--output   results/delta.json]
"""
import argparse
import json
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_delta(args):
    base = load(args.baseline)
    niq  = load(args.niqud)

    # Per-example delta (matched by id)
    base_by_id = {r["id"]: r for r in base["examples"]}
    niq_by_id  = {r["id"]: r for r in niq["examples"]}
    common_ids = set(base_by_id) & set(niq_by_id)

    if len(common_ids) < len(base_by_id):
        print(f"Warning: {len(base_by_id) - len(common_ids)} IDs in baseline not found in niqud results.")

    per_example = []
    for eid in sorted(common_ids):
        b = base_by_id[eid]
        n = niq_by_id[eid]
        per_example.append({
            "id":         eid,
            "source":     b["source"],
            "delta_em":   n["em"]  - b["em"],
            "delta_f1":   round(n["f1"] - b["f1"], 4),
            "base_em":    b["em"],  "niq_em":  n["em"],
            "base_f1":    b["f1"],  "niq_f1":  n["f1"],
            "base_pred":  b["prediction"],
            "niq_pred":   n["prediction"],
            "answers":    b["answers"],
        })

    # Aggregate delta per source + overall
    sources = sorted(set(r["source"] for r in per_example)) + ["overall"]
    agg = {}
    for src in sources:
        subset = per_example if src == "overall" else [r for r in per_example if r["source"] == src]
        if not subset:
            continue
        n = len(subset)
        agg[src] = {
            "n":        n,
            "base_em":  round(sum(r["base_em"] for r in subset) / n * 100, 2),
            "niq_em":   round(sum(r["niq_em"]  for r in subset) / n * 100, 2),
            "delta_em": round(sum(r["delta_em"] for r in subset) / n * 100, 2),
            "base_f1":  round(sum(r["base_f1"] for r in subset) / n * 100, 2),
            "niq_f1":   round(sum(r["niq_f1"]  for r in subset) / n * 100, 2),
            "delta_f1": round(sum(r["delta_f1"] for r in subset) / n * 100, 2),
            # fraction of examples where niqud changed the EM outcome
            "niq_helped_em": round(sum(1 for r in subset if r["delta_em"] > 0) / n * 100, 2),
            "niq_hurt_em":   round(sum(1 for r in subset if r["delta_em"] < 0) / n * 100, 2),
        }

    # Print table
    print(f"\n{'':=<70}")
    print(f"  Delta(niqud - no_niqud)   model: {base['model']}")
    print(f"{'':=<70}")
    header = f"  {'Source':<12}  {'n':>5}  {'Base EM':>8}  {'Niq EM':>7}  {'D EM':>6}  {'Base F1':>8}  {'Niq F1':>7}  {'D F1':>6}"
    print(header)
    print(f"  {'-'*66}")
    for src in sources:
        if src not in agg:
            continue
        s = agg[src]
        marker = " <--" if src == "overall" else ""
        print(
            f"  {src:<12}  {s['n']:>5}  "
            f"{s['base_em']:>7.1f}%  {s['niq_em']:>6.1f}%  {s['delta_em']:>+6.1f}%  "
            f"{s['base_f1']:>7.1f}%  {s['niq_f1']:>6.1f}%  {s['delta_f1']:>+6.1f}%"
            f"{marker}"
        )
    print(f"{'':=<70}")

    print("\n-- Niqud effect on individual examples (EM) --")
    for src in sources:
        if src not in agg:
            continue
        s = agg[src]
        print(f"  {src:<12}  helped: {s['niq_helped_em']:5.1f}%  hurt: {s['niq_hurt_em']:5.1f}%")

    output = {
        "model":       base["model"],
        "baseline":    args.baseline,
        "niqud":       args.niqud,
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
    parser.add_argument("--baseline", required=True, help="results/no_niqud.json")
    parser.add_argument("--niqud",    required=True, help="results/with_niqud.json")
    parser.add_argument("--output",   default=None,  help="Optional: save delta JSON")
    args = parser.parse_args()
    compute_delta(args)

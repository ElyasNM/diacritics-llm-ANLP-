"""
Compute Delta LLM-Judge = Judge_Accuracy(niqud) - Judge_Accuracy(no_niqud)

Since questions are perfectly paired, also computes per-question transitions:
  Plain CORRECT   -> Niqud CORRECT    (stable correct)
  Plain CORRECT   -> Niqud INCORRECT  (harmed by niqud)
  Plain INCORRECT -> Niqud CORRECT    (helped by niqud)
  Plain INCORRECT -> Niqud INCORRECT  (stable incorrect)

Reports:
  - Per model x dataset
  - Per model overall
  - Combined Hebrew overall (all models + datasets)
  - By answer length bucket
  - Transition table per model x dataset

Usage:
  python compute_judge_delta.py \\
      --output  hebrew_eval/judge_output.jsonl \\
      --report  hebrew_eval/judge_delta_report.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

VALID_LABELS = {"CORRECT", "INCORRECT"}
MODELS       = ["qwen2.5-7b", "dictalm2"]
DATASETS     = ["heq", "parashoot", "mkqa"]
BUCKETS      = ["short", "medium", "long", "very_long"]


def accuracy(labels):
    valid = [l for l in labels if l in VALID_LABELS]
    if not valid:
        return None, 0
    return round(100 * sum(l == "CORRECT" for l in valid) / len(valid), 2), len(valid)


def compute_transitions(plain_labels_by_id, niqud_labels_by_id):
    """
    Compute the four transition counts for paired examples.
    Returns dict with keys: CC, CI, IC, II and totals.
    """
    common_ids = set(plain_labels_by_id.keys()) & set(niqud_labels_by_id.keys())
    CC = CI = IC = II = 0
    for eid in common_ids:
        p = plain_labels_by_id[eid]
        t = niqud_labels_by_id[eid]
        if p not in VALID_LABELS or t not in VALID_LABELS:
            continue
        if   p == "CORRECT"   and t == "CORRECT":   CC += 1
        elif p == "CORRECT"   and t == "INCORRECT": CI += 1
        elif p == "INCORRECT" and t == "CORRECT":   IC += 1
        elif p == "INCORRECT" and t == "INCORRECT": II += 1
    n = CC + CI + IC + II
    return {
        "n_paired": n,
        "CC": CC, "CI": CI, "IC": IC, "II": II,
        "pct_harmed":  round(100 * CI / n, 2) if n else 0,
        "pct_helped":  round(100 * IC / n, 2) if n else 0,
        "net_churn":   CI - IC,
    }


def compute_delta(args):
    # Load all valid results
    rows = []
    with open(args.output, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("judge_label") in VALID_LABELS:
                    rows.append(r)
            except Exception:
                pass

    print(f"Loaded {len(rows):,} valid judgments\n")

    # Index by (model, condition, source) -> {example_id: label}
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["model"], r["condition"], r["source"])][r["example_id"]] = r["judge_label"]

    # Also index by (model, condition) for overall
    idx_mc = defaultdict(dict)
    for r in rows:
        idx_mc[(r["model"], r["condition"])][r["example_id"]] = r["judge_label"]

    # Index by (model, condition, bucket)
    idx_mcb = defaultdict(dict)
    for r in rows:
        idx_mcb[(r["model"], r["condition"], r.get("answer_length_bucket", "unknown"))][r["example_id"]] = r["judge_label"]

    report = {
        "per_model_dataset": {},
        "per_model_overall": {},
        "combined_overall":  {},
        "per_bucket":        {},
    }

    def fmt_row(model, label, n, plain_acc, niqud_acc, delta):
        return (f"{model:<12} {label:<12} {n:>6}  "
                f"{plain_acc:>9.1f}% {niqud_acc:>9.1f}% {delta:>+8.1f}%")

    # ── Per model x dataset ───────────────────────────────────────────────────
    print("=" * 75)
    print("DELTA REPORT — per model x dataset")
    print("=" * 75)
    print(f"{'Model':<12} {'Dataset':<12} {'n':>6}  "
          f"{'Plain':>10} {'Tash':>10} {'Δ Judge':>9}")
    print("-" * 65)

    for model in MODELS:
        for src in DATASETS:
            plain_d = idx.get((model, "no_niqud",  src), {})
            niqud_d  = idx.get((model, "with_niqud", src), {})
            plain_acc, pn = accuracy(list(plain_d.values()))
            niqud_acc,  tn = accuracy(list(niqud_d.values()))
            if plain_acc is None or niqud_acc is None:
                continue
            delta = round(niqud_acc - plain_acc, 2)
            n = min(pn, tn)
            trans = compute_transitions(plain_d, niqud_d)
            print(fmt_row(model, src, n, plain_acc, niqud_acc, delta))
            report["per_model_dataset"][f"{model}_{src}"] = {
                "n": n, "plain_acc": plain_acc, "niqud_acc": niqud_acc,
                "delta": delta, "transitions": trans,
            }
        print()

    # ── Per model overall ─────────────────────────────────────────────────────
    print("=" * 75)
    print("DELTA REPORT — per model overall")
    print("=" * 75)
    print(f"{'Model':<12} {'':12} {'n':>6}  "
          f"{'Plain':>10} {'Tash':>10} {'Δ Judge':>9}")
    print("-" * 65)

    for model in MODELS:
        plain_d = idx_mc.get((model, "no_niqud"),  {})
        niqud_d  = idx_mc.get((model, "with_niqud"), {})
        plain_acc, pn = accuracy(list(plain_d.values()))
        niqud_acc,  tn = accuracy(list(niqud_d.values()))
        if plain_acc is None or niqud_acc is None:
            continue
        delta = round(niqud_acc - plain_acc, 2)
        n = min(pn, tn)
        trans = compute_transitions(plain_d, niqud_d)
        print(fmt_row(model, "OVERALL", n, plain_acc, niqud_acc, delta))
        report["per_model_overall"][model] = {
            "n": n, "plain_acc": plain_acc, "niqud_acc": niqud_acc,
            "delta": delta, "transitions": trans,
        }

    # ── Combined Hebrew overall ───────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("DELTA REPORT — combined Hebrew overall (all models + datasets)")
    print("=" * 75)

    all_plain = [r["judge_label"] for r in rows if r["condition"] == "no_niqud"]
    all_tash  = [r["judge_label"] for r in rows if r["condition"] == "with_niqud"]
    plain_acc, pn = accuracy(all_plain)
    niqud_acc,  tn = accuracy(all_tash)
    delta = round(niqud_acc - plain_acc, 2)
    print(fmt_row("ALL MODELS", "ALL DATA", min(pn, tn), plain_acc, niqud_acc, delta))
    report["combined_overall"] = {
        "n": min(pn, tn), "plain_acc": plain_acc,
        "niqud_acc": niqud_acc, "delta": delta,
    }

    # ── Transition tables ─────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("TRANSITION TABLE — per model (paired question-level)")
    print("=" * 75)
    print(f"{'Model':<12} {'Dataset':<12} "
          f"{'CC':>6} {'CI (harmed)':>12} {'IC (helped)':>12} {'II':>6} "
          f"{'%harmed':>8} {'%helped':>8} {'net churn':>10}")
    print("-" * 85)

    for model in MODELS:
        for src in DATASETS:
            key = f"{model}_{src}"
            if key not in report["per_model_dataset"]:
                continue
            t = report["per_model_dataset"][key]["transitions"]
            print(f"{model:<12} {src:<12} "
                  f"{t['CC']:>6} {t['CI']:>12} {t['IC']:>12} {t['II']:>6} "
                  f"{t['pct_harmed']:>7.1f}% {t['pct_helped']:>7.1f}% "
                  f"{t['net_churn']:>+10}")
        print()

    # ── By answer length bucket ───────────────────────────────────────────────
    print("=" * 75)
    print("DELTA REPORT — by answer length bucket")
    print("=" * 75)
    print(f"{'Model':<12} {'Bucket':<12} {'n':>6}  "
          f"{'Plain':>10} {'Tash':>10} {'Δ Judge':>9}")
    print("-" * 65)

    for model in MODELS:
        for bucket in BUCKETS:
            plain_d = idx_mcb.get((model, "no_niqud",  bucket), {})
            niqud_d  = idx_mcb.get((model, "with_niqud", bucket), {})
            plain_acc, pn = accuracy(list(plain_d.values()))
            niqud_acc,  tn = accuracy(list(niqud_d.values()))
            if plain_acc is None or niqud_acc is None:
                continue
            delta = round(niqud_acc - plain_acc, 2)
            n = min(pn, tn)
            print(fmt_row(model, bucket, n, plain_acc, niqud_acc, delta))
            report["per_bucket"][f"{model}_{bucket}"] = {
                "n": n, "plain_acc": plain_acc, "niqud_acc": niqud_acc, "delta": delta,
            }
        print()

    # Save
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Full report saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True,
                        help="judge_output.jsonl from run_judge.py")
    parser.add_argument("--report", default="hebrew_eval/judge_delta_report.json")
    args = parser.parse_args()
    compute_delta(args)

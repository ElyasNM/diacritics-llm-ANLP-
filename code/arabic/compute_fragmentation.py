"""
Step 4 (Arabic-specific): Compute the tokenization fragmentation ratio for
each model x dataset combination, then correlate with the Delta F1.

Fragmentation ratio = token_count(tashkeel_text) / token_count(plain_text)

A ratio > 1 means tashkeel inflates the token count. The hypothesis is that
higher fragmentation correlates with larger F1 degradation (more negative D F1).

Usage:
  python compute_fragmentation.py \\
      --no_tashkeel_dir  data/no_tashkeel \\
      --with_tashkeel_dir data/with_tashkeel \\
      --delta_files results/silma/delta.json results/qwen/delta.json \\
                    results/llama/delta.json results/dictalm2/delta.json \\
      [--output fragmentation_report.json]

Output:
  - Per model x dataset: fragmentation ratio, delta EM, delta F1
  - Pearson r between fragmentation ratio and delta F1 across all combinations
  - Printed table + optional JSON
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


# ── tokenization helpers ──────────────────────────────────────────────────────

def count_tokens(texts: list, tokenizer) -> int:
    """Count total tokens across a list of texts."""
    total = 0
    for text in texts:
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        total += len(ids)
    return total


def load_texts_from_jsonl(path: Path) -> dict:
    """
    Load context + question texts from a JSONL file.
    Returns dict: source -> list of (context, question) pairs.
    """
    by_source = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            src = ex["source"]
            ctx = ex.get("context") or ""
            q   = ex.get("question") or ""
            by_source.setdefault(src, []).append(ctx + " " + q)
    return by_source


def compute_fragmentation_for_model(
    no_tash_dir: Path,
    with_tash_dir: Path,
    model_name: str,
    delta_path: Path,
) -> dict:
    """
    For one model, compute fragmentation ratio per dataset and load delta scores.
    Returns dict: source -> {frag_ratio, delta_em, delta_f1, n}
    """
    print(f"\n  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load delta file for this model
    with open(delta_path, encoding="utf-8") as f:
        delta = json.load(f)

    results = {}

    for jsonl_file in sorted(no_tash_dir.glob("*.jsonl")):
        with_jsonl = with_tash_dir / jsonl_file.name
        if not with_jsonl.exists():
            print(f"  Warning: {with_jsonl} not found, skipping.")
            continue

        no_texts   = load_texts_from_jsonl(jsonl_file)
        with_texts = load_texts_from_jsonl(with_jsonl)

        for src in no_texts:
            if src not in with_texts:
                continue
            no_tok_count   = count_tokens(no_texts[src],   tokenizer)
            with_tok_count = count_tokens(with_texts[src], tokenizer)

            frag_ratio = (
                round(with_tok_count / no_tok_count, 4)
                if no_tok_count > 0
                else None
            )

            # Get delta scores for this source from the delta file
            src_agg = delta["aggregate"].get(src, {})
            results[src] = {
                "n":            src_agg.get("n", 0),
                "frag_ratio":   frag_ratio,
                "no_tok_count": no_tok_count,
                "with_tok_count": with_tok_count,
                "delta_em":     src_agg.get("delta_em"),
                "delta_f1":     src_agg.get("delta_f1"),
            }

    return results


# ── correlation ───────────────────────────────────────────────────────────────

def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
    if std_x == 0 or std_y == 0:
        return None
    return round(cov / (n * std_x * std_y), 4)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    no_tash_dir   = Path(args.no_tashkeel_dir)
    with_tash_dir = Path(args.with_tashkeel_dir)

    # Map model name -> delta file path
    # Infer model names from delta JSON files
    model_results = {}

    for delta_path in args.delta_files:
        delta_path = Path(delta_path)
        with open(delta_path, encoding="utf-8") as f:
            delta_meta = json.load(f)
        model_name = delta_meta["model"]

        print(f"\nProcessing model: {model_name}")
        per_source = compute_fragmentation_for_model(
            no_tash_dir, with_tash_dir, model_name, delta_path
        )
        model_results[model_name] = per_source

    # Print table
    print(f"\n\n{'':=<90}")
    print("  Fragmentation Ratio x Delta F1 per Model x Dataset")
    print(f"{'':=<90}")
    header = (
        f"  {'Model':<35}  {'Dataset':<10}  {'n':>5}  "
        f"{'No-Tash Tok':>12}  {'Tash Tok':>10}  {'Frag Ratio':>11}  "
        f"{'D EM':>6}  {'D F1':>6}"
    )
    print(header)
    print(f"  {'-'*86}")

    all_frag_ratios = []
    all_delta_f1s   = []

    for model_name, per_source in model_results.items():
        short_name = model_name.split("/")[-1]
        for src, vals in sorted(per_source.items()):
            if vals["frag_ratio"] is None:
                continue
            print(
                f"  {short_name:<35}  {src:<10}  {vals['n']:>5}  "
                f"{vals['no_tok_count']:>12,}  {vals['with_tok_count']:>10,}  "
                f"{vals['frag_ratio']:>11.4f}  "
                f"{vals['delta_em']:>+6.1f}%  {vals['delta_f1']:>+6.1f}%"
            )
            if vals["delta_f1"] is not None:
                all_frag_ratios.append(vals["frag_ratio"])
                all_delta_f1s.append(vals["delta_f1"])

    print(f"{'':=<90}")

    # Correlation across all model x dataset points
    r = pearson_r(all_frag_ratios, all_delta_f1s)
    n_points = len(all_frag_ratios)
    print(f"\n  Pearson r (frag_ratio vs delta_F1) across all {n_points} model x dataset points: {r}")
    if r is not None:
        if r < -0.5:
            print("  -> Strong negative correlation: higher fragmentation predicts lower F1.")
        elif r < -0.2:
            print("  -> Moderate negative correlation.")
        elif abs(r) < 0.2:
            print("  -> Weak / no correlation.")
        else:
            print("  -> Positive correlation (higher fragmentation does not hurt F1).")

    # Save output
    output = {
        "model_results":             model_results,
        "pearson_r_frag_vs_delta_f1": r,
        "n_model_dataset_points":    n_points,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nFragmentation report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_tashkeel_dir",   required=True)
    parser.add_argument("--with_tashkeel_dir", required=True)
    parser.add_argument(
        "--delta_files",
        required=True,
        nargs="+",
        help="One delta.json per model (from compute_delta.py)"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args)

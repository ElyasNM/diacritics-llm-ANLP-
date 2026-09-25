"""
Hebrew tokenizer fragmentation ratio (niqud vs. plain), the Hebrew counterpart
of the Arabic compute_fragmentation.py.

For each dataset, a fixed random sample (seed 0) of --n examples present in both
conditions is tokenized (context + question, no special tokens) with each model's
tokenizer, and the ratio  total_tokens(niqud) / total_tokens(plain)  is reported.
Tokenizer calls only; no model inference. Sampling keeps this cheap; the ratios
are stable across datasets (within ~0.3x), so a few hundred examples suffice.

Usage (from the repo root):
  python code/compute_fragmentation_hebrew.py
  # GitHub-folder layout (data/hebrew/...):
  python compute_fragmentation_hebrew.py --data_root data/hebrew
"""

import argparse
import json
import os
import random

from transformers import AutoTokenizer

MODELS = ["Qwen/Qwen2.5-7B-Instruct", "dicta-il/dictalm2.0-instruct"]
DATASETS = [("heq", "json"), ("parashoot", "jsonl"), ("mkqa", "json")]


def load(path):
    with open(path, encoding="utf8") as f:
        rows = (json.loads(line) for line in f)
        return {r["id"]: r for r in rows if r.get("id")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data",
                    help="folder containing no_niqud/ and with_niqud/")
    ap.add_argument("--n", type=int, default=200, help="examples sampled per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline", action="store_true",
                    help="use only locally cached tokenizers")
    args = ap.parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    toks = {m: AutoTokenizer.from_pretrained(m) for m in MODELS}
    for ds, ext in DATASETS:
        plain = load(os.path.join(args.data_root, "no_niqud", f"{ds}_eval.jsonl"))
        niqud = load(os.path.join(args.data_root, "with_niqud", f"{ds}_eval.{ext}"))
        ids = sorted(set(plain) & set(niqud))
        random.Random(args.seed).shuffle(ids)
        ids = ids[: args.n]
        for m, t in toks.items():
            tp = tn = 0
            for i in ids:
                for field in ("context", "question"):
                    tp += len(t(plain[i].get(field) or "", add_special_tokens=False)["input_ids"])
                    tn += len(t(niqud[i].get(field) or "", add_special_tokens=False)["input_ids"])
            print(f"{m.split('/')[0]:9s} {ds:9s} n={len(ids)} plain={tp} niqud={tn} ratio={tn / tp:.2f}x")


if __name__ == "__main__":
    main()

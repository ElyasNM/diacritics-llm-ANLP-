"""
Step 3: Run a causal LM on one condition (no_niqud or with_niqud) and save
per-example predictions + scores.

Usage:
  python evaluate.py \\
      --model  dicta-il/dictalm2.0-instruct \\
      --data   data/no_niqud \\
      --output results/no_niqud.json \\
      [--max_samples 1500] \\
      [--batch_size 8] \\
      [--max_new_tokens 80] \\
      [--max_length 3072] \\
      [--device cuda]

Run twice (no_niqud + with_niqud), then run compute_delta.py.
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── prompts ──────────────────────────────────────────────────────────────────

EXTRACTIVE_PROMPT = (
    "קרא את הקטע הבא וענה על השאלה. כתוב את התשובה בדיוק כפי שהיא מופיעה בקטע, "
    "ללא הסבר נוסף.\n\n"
    "קטע: {context}\n\n"
    "שאלה: {question}\n\n"
    "תשובה:"
)

OPEN_DOMAIN_PROMPT = (
    "ענה על השאלה הבאה בקצרה. כתוב רק את התשובה.\n\n"
    "שאלה: {question}\n\n"
    "תשובה:"
)

# ── normalisation ─────────────────────────────────────────────────────────────

NIQUD_RE  = re.compile(r"[ְ-ׇ֑-֯]")   # vowel points U+05B0-05C7 + cantillation U+0591-05AF
PUNCT_RE  = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    text = NIQUD_RE.sub("", text)          # strip any residual niqud
    text = text.lower()                    # for Latin chars
    text = PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


# ── metrics ───────────────────────────────────────────────────────────────────

def exact_match(pred: str, golds: list[str]) -> int:
    p = normalize(pred)
    return int(any(normalize(g) == p for g in golds))


def token_f1(pred: str, golds: list[str]) -> float:
    pred_toks = normalize(pred).split()
    best = 0.0
    for gold in golds:
        gold_toks = normalize(gold).split()
        # SQuAD-standard: multiset intersection
        common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
        if not common:
            continue
        p = common / len(pred_toks) if pred_toks else 0.0
        r = common / len(gold_toks) if gold_toks else 0.0
        if p + r:
            best = max(best, 2 * p * r / (p + r))
    return best


# ── inference ─────────────────────────────────────────────────────────────────

def build_prompt(ex: dict, tokenizer) -> str:
    if ex["context"]:
        user_msg = EXTRACTIVE_PROMPT.format(context=ex["context"], question=ex["question"])
    else:
        user_msg = OPEN_DOMAIN_PROMPT.format(question=ex["question"])

    # Apply chat template if the tokenizer defines one (e.g. DictaLM2, Qwen-Instruct)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_msg


def run_batch(prompts: list[str], tokenizer, model, device: str, max_new_tokens: int, max_length: int) -> list[str]:
    actual_device = next(model.parameters()).device
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(actual_device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    preds = []
    for ids in out:
        gen_ids = ids[enc["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        text = text.split("\n")[0].strip()   # first line only
        preds.append(text)
    return preds


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    data_dir = Path(args.data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all JSONL files in the data dir
    examples = []
    for jsonl in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))
    if not examples:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    if args.max_samples and args.max_samples < len(examples):
        # Stratified subsample to keep source proportions
        from collections import defaultdict
        import random
        random.seed(42)
        by_source = defaultdict(list)
        for ex in examples:
            by_source[ex["source"]].append(ex)
        n_per_source = args.max_samples // len(by_source)
        examples = []
        for src, exs in by_source.items():
            examples.extend(random.sample(exs, min(n_per_source, len(exs))))

    print(f"Loaded {len(examples)} examples from {data_dir}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    results = []
    t0 = time.time()

    for i in tqdm(range(0, len(examples), args.batch_size), desc="Evaluating"):
        batch = examples[i : i + args.batch_size]
        prompts = [build_prompt(ex, tokenizer) for ex in batch]
        preds = run_batch(prompts, tokenizer, model, args.device, args.max_new_tokens, args.max_length)

        for ex, pred in zip(batch, preds):
            em = exact_match(pred, ex["answers"])
            f1 = token_f1(pred, ex["answers"])
            results.append({
                "id":         ex["id"],
                "source":     ex["source"],
                "question":   ex["question"],
                "answers":    ex["answers"],
                "prediction": pred,
                "em":         em,
                "f1":         round(f1, 4),
            })

    elapsed = time.time() - t0

    # Aggregate per source and overall
    agg = {}
    for src in set(r["source"] for r in results):
        src_res = [r for r in results if r["source"] == src]
        agg[src] = {
            "n":      len(src_res),
            "em":     round(sum(r["em"] for r in src_res) / len(src_res) * 100, 2),
            "f1":     round(sum(r["f1"] for r in src_res) / len(src_res) * 100, 2),
        }
    agg["overall"] = {
        "n":  len(results),
        "em": round(sum(r["em"] for r in results) / len(results) * 100, 2),
        "f1": round(sum(r["f1"] for r in results) / len(results) * 100, 2),
    }

    output = {
        "model":     args.model,
        "data_dir":  str(data_dir),
        "elapsed_s": round(elapsed, 1),
        "aggregate": agg,
        "examples":  results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {out_path}  ({elapsed:.0f}s)")
    print("\n-- Aggregate scores ------------------")
    for src, scores in agg.items():
        print(f"  {src:12s}  EM={scores['em']:5.1f}%  F1={scores['f1']:5.1f}%  (n={scores['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          required=True,  help="HF model name or local path")
    parser.add_argument("--data",           required=True,  help="Directory with .jsonl files (no_niqud or with_niqud)")
    parser.add_argument("--output",         required=True,  help="Path to write results JSON")
    parser.add_argument("--max_samples",    type=int,       default=None, help="Cap total examples (stratified)")
    parser.add_argument("--batch_size",     type=int,       default=8)
    parser.add_argument("--max_new_tokens", type=int,       default=80)
    parser.add_argument("--max_length",     type=int,       default=3072)
    parser.add_argument("--device",         default="auto", help="device_map value: 'auto', 'cuda:0', 'cpu'")
    args = parser.parse_args()
    evaluate(args)

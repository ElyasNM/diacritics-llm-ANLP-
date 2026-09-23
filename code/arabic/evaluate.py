"""
Step 2: Run a causal LM on one condition (no_tashkeel or with_tashkeel)
and save per-example predictions + scores.

Usage:
  python evaluate.py \\
      --model  silma-ai/SILMA-9B-Instruct-v1.0 \\
      --data   data/no_tashkeel \\
      --output results/silma/no_tashkeel.json \\
      [--max_samples 2000] \\
      [--batch_size 8] \\
      [--max_new_tokens 80] \\
      [--max_length 3072] \\
      [--device auto]

Run twice per model (no_tashkeel + with_tashkeel), then run compute_delta.py.

Supported models:
  silma-ai/SILMA-9B-Instruct-v1.0        (Arabic-specific)
  Qwen/Qwen2.5-7B-Instruct               (multilingual)
  meta-llama/Llama-3.1-8B-Instruct       (multilingual)
  dicta-il/dictalm2.0-instruct           (Hebrew-specific, cross-language control)
"""

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── prompts ──────────────────────────────────────────────────────────────────

EXTRACTIVE_PROMPT = (
    "اقرأ النص التالي وأجب على السؤال. "
    "اكتب الإجابة كما وردت في النص بالضبط، دون أي شرح إضافي.\n\n"
    "النص: {context}\n\n"
    "السؤال: {question}\n\n"
    "الإجابة:"
)

OPEN_DOMAIN_PROMPT = (
    "أجب على السؤال التالي باختصار. اكتب الإجابة فقط.\n\n"
    "السؤال: {question}\n\n"
    "الإجابة:"
)


# ── normalisation ─────────────────────────────────────────────────────────────

# Strip Arabic diacritics before scoring so scores are comparable across
# the two conditions (model may echo tashkeel in its output).
TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670]")
PUNCT_RE    = re.compile(r"[^\w\s]")

# Common Arabic stopwords to ignore during scoring (mirrors Hebrew stop-word
# removal in the Hebrew evaluate.py normalizer).
ARABIC_STOPWORDS = {
    "في", "من", "إلى", "على", "عن", "مع", "هو", "هي", "هم", "هن",
    "أن", "إن", "كان", "كانت", "التي", "الذي", "الذين", "وهو", "وهي",
    "هذا", "هذه", "ذلك", "تلك", "قد", "لا", "ما", "لم", "لن",
    "أو", "ثم", "حتى", "إذا", "بعد", "قبل", "كل", "بين",
}


def normalize(text: str) -> str:
    text = TASHKEEL_RE.sub("", text)       # strip tashkeel
    text = PUNCT_RE.sub(" ", text)          # remove punctuation
    text = text.lower()
    tokens = [t for t in text.split() if t not in ARABIC_STOPWORDS]
    return " ".join(tokens)


# ── metrics ───────────────────────────────────────────────────────────────────

def exact_match(pred: str, golds: list) -> int:
    p = normalize(pred)
    return int(any(normalize(g) == p for g in golds))


def token_f1(pred: str, golds: list) -> float:
    pred_toks = normalize(pred).split()
    best = 0.0
    for gold in golds:
        gold_toks = normalize(gold).split()
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
        user_msg = EXTRACTIVE_PROMPT.format(
            context=ex["context"], question=ex["question"]
        )
    else:
        user_msg = OPEN_DOMAIN_PROMPT.format(question=ex["question"])

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_msg


def run_batch(
    prompts: list,
    tokenizer,
    model,
    max_new_tokens: int,
    max_length: int,
) -> list:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

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
    ckpt_path = out_path.with_suffix(".ckpt.jsonl")

    # Load all JSONL files in the data dir
    examples = []
    for jsonl in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))
    if not examples:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    # Resume from checkpoint: reload cached predictions, skip their IDs
    cached_results = []
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cached_results.append(json.loads(line))
        done_ids = {r["id"] for r in cached_results}
        examples = [ex for ex in examples if ex["id"] not in done_ids]
        print(f"Resuming from checkpoint: {len(cached_results)} cached, "
              f"{len(examples)} remaining")

    # Optional stratified subsample
    if args.max_samples and args.max_samples < len(examples):
        import random
        random.seed(42)
        by_source = defaultdict(list)
        for ex in examples:
            by_source[ex["source"]].append(ex)
        n_per_source = args.max_samples // len(by_source)
        examples = []
        for src, exs in by_source.items():
            examples.extend(random.sample(exs, min(n_per_source, len(exs))))

    print(f"Loaded {len(examples)} examples from {data_dir} "
          f"({len(cached_results)} already cached)")

    t0 = time.time()
    new_results = []

    if examples:
        print(f"Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )
        model.eval()

        # Append mode: keep any cached lines already on disk from a prior run.
        ckpt_file = open(ckpt_path, "a", encoding="utf-8")

        for i in tqdm(range(0, len(examples), args.batch_size), desc="Evaluating"):
            batch = examples[i: i + args.batch_size]
            prompts = [build_prompt(ex, tokenizer) for ex in batch]
            preds = run_batch(
                prompts, tokenizer, model, args.max_new_tokens, args.max_length
            )

            for ex, pred in zip(batch, preds):
                em = exact_match(pred, ex["answers"])
                f1 = token_f1(pred, ex["answers"])
                rec = {
                    "id":              ex["id"],
                    "source":          ex["source"],
                    "question":        ex["question"],
                    "answers":         ex["answers"],
                    "prediction":      pred,
                    "pred_len_words":  len(pred.split()),   # for verbosity analysis
                    "em":              em,
                    "f1":              round(f1, 4),
                }
                new_results.append(rec)
                ckpt_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

            ckpt_file.flush()  # persist after every batch, not just at exit

        ckpt_file.close()

    results = cached_results + new_results
    elapsed = time.time() - t0

    # Aggregate per source and overall
    agg = {}
    for src in set(r["source"] for r in results):
        src_res = [r for r in results if r["source"] == src]
        agg[src] = {
            "n":            len(src_res),
            "em":           round(sum(r["em"] for r in src_res) / len(src_res) * 100, 2),
            "f1":           round(sum(r["f1"] for r in src_res) / len(src_res) * 100, 2),
            "mean_pred_len": round(sum(r["pred_len_words"] for r in src_res) / len(src_res), 2),
        }
    agg["overall"] = {
        "n":            len(results),
        "em":           round(sum(r["em"] for r in results) / len(results) * 100, 2),
        "f1":           round(sum(r["f1"] for r in results) / len(results) * 100, 2),
        "mean_pred_len": round(sum(r["pred_len_words"] for r in results) / len(results), 2),
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
        print(
            f"  {src:<12}  EM={scores['em']:5.1f}%  "
            f"F1={scores['f1']:5.1f}%  "
            f"mean_len={scores['mean_pred_len']:.1f}  "
            f"(n={scores['n']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          required=True,  help="HF model name or local path")
    parser.add_argument("--data",           required=True,  help="Directory with .jsonl files")
    parser.add_argument("--output",         required=True,  help="Path to write results JSON")
    parser.add_argument("--max_samples",    type=int,       default=None)
    parser.add_argument("--batch_size",     type=int,       default=8)
    parser.add_argument("--max_new_tokens", type=int,       default=80)
    parser.add_argument("--max_length",     type=int,       default=3072)
    parser.add_argument("--device",         default="auto", help="device_map value")
    args = parser.parse_args()
    evaluate(args)

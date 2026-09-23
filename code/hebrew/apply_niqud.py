"""
Step 3: Diacritize context + question fields using
        dicta-il/dictabert-large-char-menaked (DICTA SOTA, ~0.3B).

Reads:   data/no_niqud/*.jsonl
Writes:  data/with_niqud/*.jsonl  (same schema, answers untouched)

Features:
  - Chunking: splits long contexts on sentence boundaries so no text
    exceeds the model's safe input length
  - Batching: collects chunks from multiple examples into one predict() call
  - Resumable: skips already-written IDs if you re-run after a crash

Usage:
  python apply_niqud.py [--batch_size 32] [--max_chars 400] [--device cuda]
"""
import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "dicta-il/dictabert-large-char-menaked"
NO_NIQ   = Path("data/no_niqud")
WITH_NIQ = Path("data/with_niqud")


# ── text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks <= max_chars, breaking on sentence boundaries."""
    if not text or len(text) <= max_chars:
        return [text] if text else []

    # Try splitting on Hebrew/Latin sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, current_len = [], [], 0

    for sent in sentences:
        if current and current_len + len(sent) + 1 > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [sent], len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    # Safety: if any single sentence is still too long, hard-split it
    final = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            final.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        if chunk:
            final.append(chunk)
    return final


# ── core diacritization ───────────────────────────────────────────────────────

def diacritize_texts(texts: list[str], tokenizer, model, max_chars: int) -> list[str]:
    """
    Diacritize a list of texts (may include long contexts).
    Chunks each text, runs one batched predict() call, then reassembles.
    """
    # Build flat chunk list, tracking which text each chunk belongs to
    all_chunks:  list[str] = []
    text_slices: list[tuple[int, int]] = []   # (start, end) into all_chunks per text

    for text in texts:
        start = len(all_chunks)
        all_chunks.extend(chunk_text(text, max_chars))
        text_slices.append((start, len(all_chunks)))

    if not all_chunks:
        return texts

    with torch.no_grad():
        diac = model.predict(all_chunks, tokenizer)

    results = []
    for text, (start, end) in zip(texts, text_slices):
        if start == end:            # empty text
            results.append(text)
        else:
            results.append(" ".join(diac[start:end]))
    return results


# ── file processing ───────────────────────────────────────────────────────────

def load_done_ids(path: Path) -> set:
    """Return IDs already written to an output file (for resumability)."""
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return done


def process_file(src: Path, dst: Path, tokenizer, model, batch_size: int, max_chars: int):
    done_ids = load_done_ids(dst)

    examples = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    pending = [ex for ex in examples if ex["id"] not in done_ids]
    if not pending:
        print(f"  {src.name}: already complete ({len(examples)} examples), skipping.")
        return

    print(f"  {src.name}: {len(examples)} total, {len(pending)} to process "
          f"({len(done_ids)} already done)")

    with open(dst, "a", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(pending), batch_size), desc=f"  {src.stem}"):
            batch = pending[i : i + batch_size]

            # Separate contexts (may be None for MKQA) and questions
            contexts  = [ex["context"]  or "" for ex in batch]
            questions = [ex["question"]        for ex in batch]

            # Single batched call per field type
            diac_ctx = diacritize_texts(contexts,  tokenizer, model, max_chars)
            diac_q   = diacritize_texts(questions, tokenizer, model, max_chars)

            for ex, ctx, q in zip(batch, diac_ctx, diac_q):
                out = dict(ex)
                out["context"]  = ctx if ex["context"] is not None else None
                out["question"] = q
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")

            out_f.flush()           # survive cluster pre-emption
            torch.cuda.empty_cache()  # release fragmented memory between batches

    # Verify output count
    written = sum(1 for _ in open(dst, encoding="utf-8"))
    print(f"  -> {dst.name}: {written} examples written")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    WITH_NIQ.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,   # halves weight memory vs float32
    )
    model.eval()
    if args.device != "cpu":
        model.to(args.device)
    print(f"Model loaded on {args.device}\n")

    for src in sorted(NO_NIQ.glob("*.jsonl")):
        dst = WITH_NIQ / src.name
        print(f"==> {src.name}")
        process_file(src, dst, tokenizer, model, args.batch_size, args.max_chars)

    print("\nDone. data/with_niqud/ is ready for evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Examples per predict() call (reduce if OOM)")
    parser.add_argument("--max_chars",  type=int, default=400,
                        help="Max chars per chunk before splitting (model safe limit)")
    parser.add_argument("--device",     default="cuda",
                        help="'cuda', 'cuda:0', 'cpu'")
    main(parser.parse_args())

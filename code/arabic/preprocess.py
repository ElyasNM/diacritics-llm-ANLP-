"""
Step 1: Load the three Arabic QA datasets from Drive (produced by the
vocalization notebook) and convert them to a unified JSONL schema.

Produces:
  data/no_tashkeel/{arcd,tydiqa,mkqa}_eval.jsonl   <- baseline (no diacritics)
  data/with_tashkeel/{arcd,tydiqa,mkqa}_eval.jsonl  <- fully vocalized

Unified schema per line:
  {
    "id":           str,
    "source":       "arcd" | "tydiqa" | "mkqa",
    "context":      str | null,   # null for open-domain MKQA
    "question":     str,
    "answers":      [str, ...],   # one or more valid answer strings
    "answer_start": [int, ...]    # char offsets into context (null for MKQA)
  }

Only context + question fields are tashkeel-stripped for the no_tashkeel
condition. Answers are kept as-is; evaluate.py normalizes them at scoring time.

Usage:
  python preprocess.py --drive_dir /content/drive/MyDrive/Arabic_Tashkeel
"""

import argparse
import json
import re
from pathlib import Path

from datasets import load_from_disk

NO_TASH  = Path("data/no_tashkeel")
WITH_TASH = Path("data/with_tashkeel")

# Arabic diacritics (Tashkeel) Unicode range U+064B-U+0652 + U+0670 superscript alef
TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670]")


def strip_tashkeel(text: str) -> str:
    return TASHKEEL_RE.sub("", text) if text else text


# ── ARCD ─────────────────────────────────────────────────────────────────────

def process_arcd(drive_dir: Path):
    ds = load_from_disk(str(drive_dir / "arcd" / "arcd_tashkeel"))

    no_records, with_records = [], []

    for split in ds:
        for row in ds[split]:
            # Answers: SQuAD format {'text': [...], 'answer_start': [...]}
            ans_obj = row["answers"]
            texts  = ans_obj.get("text", [])
            starts = [int(s) for s in ans_obj.get("answer_start", [])]
            if not texts:
                continue

            uid = f"arcd_{split}_{row['id']}"

            no_records.append({
                "id":           uid,
                "source":       "arcd",
                "context":      strip_tashkeel(row["context"]),
                "question":     strip_tashkeel(row["question"]),
                "answers":      texts,
                "answer_start": starts,
            })
            with_records.append({
                "id":           uid,
                "source":       "arcd",
                "context":      row["context_tashkeel"],
                "question":     row["question_tashkeel"],
                "answers":      texts,
                "answer_start": starts,
            })

    _write(no_records,   NO_TASH  / "arcd_eval.jsonl",  "ARCD (no tashkeel)")
    _write(with_records, WITH_TASH / "arcd_eval.jsonl", "ARCD (with tashkeel)")


# ── TyDi QA (Arabic) ─────────────────────────────────────────────────────────

def process_tydiqa(drive_dir: Path):
    ds = load_from_disk(str(drive_dir / "tydiqa_arabic" / "tydiqa_arabic_tashkeel"))

    no_records, with_records = [], []

    for split in ds:
        for row in ds[split]:
            # Answers: {'text': [...], 'start_byte': [...], 'limit_byte': [...]}
            # We use text only; answer_start is set to None (byte offsets
            # are not directly usable as char offsets for this dataset).
            ans_obj = row["answers"]
            texts = ans_obj.get("text", [])
            if not texts:
                continue

            uid = f"tydiqa_{split}_{row['id']}"

            # Baseline: use passage_text_clean (already stripped of selective
            # diacritics during vocalization notebook — this is the true
            # unvocalized baseline, not the original which had partial tashkeel).
            no_records.append({
                "id":           uid,
                "source":       "tydiqa",
                "context":      row["passage_text_clean"],
                "question":     row["question_text_clean"],
                "answers":      texts,
                "answer_start": None,
            })
            with_records.append({
                "id":           uid,
                "source":       "tydiqa",
                "context":      row["passage_text_tashkeel"],
                "question":     row["question_text_tashkeel"],
                "answers":      texts,
                "answer_start": None,
            })

    _write(no_records,   NO_TASH  / "tydiqa_eval.jsonl",  "TyDi QA Arabic (no tashkeel)")
    _write(with_records, WITH_TASH / "tydiqa_eval.jsonl", "TyDi QA Arabic (with tashkeel)")


# ── MKQA (Arabic) ────────────────────────────────────────────────────────────

def process_mkqa(drive_dir: Path):
    ds = load_from_disk(str(drive_dir / "mkqa_arabic" / "mkqa_arabic_tashkeel"))

    no_records, with_records = [], []

    for split in ds:
        for row in ds[split]:
            question = row.get("query_ar", "").strip()
            if not question:
                continue

            # answers_ar: list of dicts with keys: text, aliases, type, entity
            ar_answers = row.get("answers_ar", []) or []
            texts = []
            for a in ar_answers:
                if a.get("type") in ("entity", "short_phrase", "value", "number_with_unit"):
                    t = a.get("text", "")
                    if t:
                        texts.append(t)
                    texts.extend(a.get("aliases", []))
            if not texts:
                continue

            uid = f"mkqa_{row['example_id']}"

            no_records.append({
                "id":           uid,
                "source":       "mkqa",
                "context":      None,
                "question":     strip_tashkeel(question),
                "answers":      texts,
                "answer_start": None,
            })
            with_records.append({
                "id":           uid,
                "source":       "mkqa",
                "context":      None,
                "question":     row["query_ar_tashkeel"],
                "answers":      texts,
                "answer_start": None,
            })

    _write(no_records,   NO_TASH  / "mkqa_eval.jsonl",  "MKQA Arabic (no tashkeel)")
    _write(with_records, WITH_TASH / "mkqa_eval.jsonl", "MKQA Arabic (with tashkeel)")


# ── helpers ──────────────────────────────────────────────────────────────────

def _write(records, path: Path, name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   {name}: {len(records)} examples -> {path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drive_dir",
        required=True,
        help="Path to the Arabic_Tashkeel Drive folder (e.g. /content/drive/MyDrive/Arabic_Tashkeel)",
    )
    args = parser.parse_args()
    drive_dir = Path(args.drive_dir)

    print("==> Processing ARCD...")
    process_arcd(drive_dir)

    print("==> Processing TyDi QA Arabic...")
    process_tydiqa(drive_dir)

    print("==> Processing MKQA Arabic...")
    process_mkqa(drive_dir)

    print("\nPreprocessing done.")
    print(f"  Baseline  -> data/no_tashkeel/")
    print(f"  Vocalized -> data/with_tashkeel/")

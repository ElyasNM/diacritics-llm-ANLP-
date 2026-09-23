"""
Step 2: Convert raw data to unified JSONL and strip niqud.

Produces:
  data/no_niqud/{heq,parashoot,mkqa}_eval.jsonl   ← baseline (no diacritics)
  data/with_niqud/                                 ← placeholder; fill with Nakdan output

Unified schema per line:
  {
    "id":           str,
    "source":       "heq" | "parashoot" | "mkqa",
    "context":      str | null,   # null for open-domain MKQA
    "question":     str,
    "answers":      [str, ...],   # one or more valid answer strings
    "answer_start": [int, ...]    # char offsets into context (null for MKQA)
  }

Only context + question are niqud-stripped (those are the fields passed to the
model and to Nakdan). Answers are kept as-is; evaluate.py normalizes them at
scoring time so they stay comparable across both conditions.
"""
import ast
import json
import re
from pathlib import Path

RAW      = Path("data/raw")
NO_NIQ   = Path("data/no_niqud")
WITH_NIQ = Path("data/with_niqud")

# Unicode range for Hebrew niqqud (vowel points + cantillation marks)
NIQUD_RE = re.compile(r"[ְ-ׇ֑-֯]")


def strip_niqud(text: str) -> str:
    return NIQUD_RE.sub("", text) if text else text


# ── HeQ ──────────────────────────────────────────────────────────────────────

def process_heq():
    out = NO_NIQ / "heq_eval.jsonl"
    records = []
    for split in ("train", "validation", "test"):
        src = RAW / "heq" / f"{split}.jsonl"
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("Is_Impossible"):
                    continue
                answers_obj = row["Answers"]
                if isinstance(answers_obj, str):
                    answers_obj = ast.literal_eval(answers_obj)
                texts  = answers_obj.get("text", [])
                starts = [int(s) for s in answers_obj.get("answer_start", [])]
                if not texts:
                    continue
                records.append({
                    "id":           f"heq_{row['ID']}",
                    "source":       "heq",
                    "context":      strip_niqud(row["Context"]),
                    "question":     strip_niqud(row["Question"]),
                    "answers":      texts,
                    "answer_start": starts,
                })
    _write(records, out, "HeQ (all splits)")


# ── ParaShoot ────────────────────────────────────────────────────────────────

def process_parashoot():
    out = NO_NIQ / "parashoot_eval.jsonl"
    records = []
    for split in ("train", "dev", "test"):
        src = RAW / "parashoot" / f"{split}.json"
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        for qa in data["data"]:
            if qa.get("is_impossible"):
                continue
            answers = qa.get("answers", {})
            if isinstance(answers, str):
                answers = ast.literal_eval(answers)
            texts  = answers.get("text", [])
            starts = [int(s) for s in answers.get("answer_start", [])]
            if not texts:
                continue
            records.append({
                "id":           f"parashoot_{qa['id']}",
                "source":       "parashoot",
                "context":      strip_niqud(qa["context"]),
                "question":     strip_niqud(qa["question"]),
                "answers":      texts,
                "answer_start": starts,
            })
    _write(records, out, "ParaShoot (all splits)")


# ── MKQA (Hebrew) ────────────────────────────────────────────────────────────

def process_mkqa():
    src = RAW / "mkqa" / "mkqa.jsonl"
    out = NO_NIQ / "mkqa_eval.jsonl"
    records = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            question = (row.get("queries") or {}).get("he", "").strip()
            if not question:
                continue
            he_answers = (row.get("answers") or {}).get("he", [])
            texts = []
            for a in he_answers:
                if a.get("type") in ("entity", "short_phrase", "value"):
                    t = a.get("text", "")
                    if t:
                        texts.append(t)
                    texts.extend(a.get("aliases", []))
            if not texts:
                continue
            records.append({
                "id":           f"mkqa_{row['example_id']}",
                "source":       "mkqa",
                "context":      None,
                "question":     strip_niqud(question),
                "answers":      texts,
                "answer_start": None,
            })
    _write(records, out, "MKQA-Hebrew")


# ── helpers ──────────────────────────────────────────────────────────────────

def _write(records, path, name):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"   {name}: {len(records)} examples -> {path}")


def seed_with_niqud_dir():
    """Copy no_niqud files into with_niqud as templates (user replaces context/question with Nakdan output)."""
    readme = WITH_NIQ / "README.txt"
    readme.write_text(
        "Run Nakdan (or another diacritization tool) on the 'context' and 'question'\n"
        "fields of each JSONL in ../no_niqud/, then place the results here with the\n"
        "same filenames. The 'answers' and 'id' fields must remain identical.\n",
        encoding="utf-8",
    )
    print(f"   Created placeholder README in {WITH_NIQ}")


if __name__ == "__main__":
    print("==> Processing HeQ (all splits)...")
    process_heq()

    print("==> Processing ParaShoot (all splits)...")
    process_parashoot()

    print("==> Processing MKQA (Hebrew)...")
    process_mkqa()

    print("==> Seeding with_niqud/ placeholder...")
    seed_with_niqud_dir()

    print("\nPreprocessing done. data/no_niqud/ is ready.")
    print("Next: run Nakdan on context+question fields, save outputs to data/with_niqud/.")

"""
Step 1b (variant): Rewrite fully-vocalized Tashkeel JSONL into a
"postfix" vocalization scheme, to test whether moving diacritics out of
their interleaved position changes tokenization fragmentation / LLM QA
performance.

Interleaved (standard) Unicode form:
    letter, mark, letter, mark, letter, mark, mark, letter
    (diacritics sit right after each consonant they modify)
Postfix form (diacritics of each Arabic word moved to the end of that
word, letters kept in original relative order):
    letter, letter, letter, letter, mark, mark, mark, mark, mark
    (all consonants of the word, followed by its diacritics as one block)

Only `context` and `question` are rewritten. `id`, `source`, `answers`,
`answer_start` are copied through unchanged, matching how with_tashkeel
data is already produced by preprocess.py (answers are held fixed across
conditions; evaluate.py strips diacritics before scoring anyway).

Reads from data already sitting in the standard eval_data layout:
    Arabic_Tashkeel/eval_data/with_tashkeel/{arcd,tydiqa,mkqa}_eval.jsonl
and writes to a sibling folder:
    Arabic_Tashkeel/eval_data/postfix_tashkeel/{arcd,tydiqa,mkqa}_eval.jsonl

Modular by design: which source files get processed is controlled by
--include_sources / --exclude_sources, so re-adding mkqa later (or
restricting to just one dataset) doesn't require touching this file.

Usage:
  python prepare_postfix_tashkeel.py \\
      --input_dir  eval_data/with_tashkeel \\
      --output_dir eval_data/postfix_tashkeel \\
      --exclude_sources mkqa
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Character ranges are built from codepoints (range()/chr()), not literal
# glyphs or \u escapes, so this file's meaning can't drift with editor or
# terminal encoding.
#   0x064B-0x065F : Tashkeel diacritics (Fathatan..Sukun, plus the extra
#                   Quranic/diacritic marks in the same block).
#                   preprocess.py / evaluate.py use 0x064B-0x0652 + 0x0670;
#                   we cover the slightly wider contiguous block for safety.
#   0x0670        : superscript alef (diacritic).
#   0x0621-0x063A : main Arabic letter block (hamza .. ghain).
#   0x0640        : tatweel / kashida (elongation char, travels with the word).
#   0x0641-0x064A : main Arabic letter block (feh .. yeh).
_DIACRITIC_CODEPOINTS = list(range(0x064B, 0x0660)) + [0x0670]
_LETTER_CODEPOINTS = (
    list(range(0x0621, 0x063B)) + [0x0640] + list(range(0x0641, 0x064B))
)

TASHKEEL_CHARS = "".join(chr(c) for c in _DIACRITIC_CODEPOINTS)
ARABIC_RUN_CHARS = "".join(chr(c) for c in _LETTER_CODEPOINTS) + TASHKEEL_CHARS

# Matches a single diacritic mark (used to classify chars within a word run).
TASHKEEL_RE = re.compile("[" + re.escape(TASHKEEL_CHARS) + "]")

# A "word" for reordering purposes: any maximal run of Arabic base letters,
# tatweel/kashida, and/or diacritics. Non-Arabic runs (spaces, digits,
# Latin text, punctuation) are left untouched and act as word boundaries.
ARABIC_RUN_RE = re.compile("[" + re.escape(ARABIC_RUN_CHARS) + "]+")


def _postfix_word(run: str) -> str:
    """Given one Arabic run, move its diacritics to the end, keep letters in order."""
    letters, marks = [], []
    for ch in run:
        (marks if TASHKEEL_RE.match(ch) else letters).append(ch)
    return "".join(letters) + "".join(marks)


def postfix_vocalize(text: str) -> str:
    if not text:
        return text
    return ARABIC_RUN_RE.sub(lambda m: _postfix_word(m.group(0)), text)


def _assert_reorder_only(original: str, transformed: str):
    """Sanity check: postfixing must not add/drop/change any characters."""
    if Counter(original) != Counter(transformed):
        raise ValueError(
            "Postfix transform changed the character multiset "
            f"(expected pure reordering).\n  original:    {original!r}\n"
            f"  transformed: {transformed!r}"
        )


def process_file(path: Path, out_path: Path, verify: bool) -> int:
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            ctx = ex.get("context")
            q = ex.get("question")
            new_ctx = postfix_vocalize(ctx) if ctx else ctx
            new_q = postfix_vocalize(q) if q else q

            if verify:
                if ctx:
                    _assert_reorder_only(ctx, new_ctx)
                if q:
                    _assert_reorder_only(q, new_q)

            ex["context"] = new_ctx
            ex["question"] = new_q
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    jsonl_files = sorted(in_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {in_dir}")

    include = set(args.include_sources) if args.include_sources else None
    exclude = set(args.exclude_sources or [])

    print(f"Input:  {in_dir}")
    print(f"Output: {out_dir}")
    if include:
        print(f"Processing only sources: {sorted(include)}")
    if exclude:
        print(f"Skipping sources: {sorted(exclude)}")

    for jf in jsonl_files:
        # Source name is inferred from filename (arcd_eval.jsonl -> arcd),
        # matching the one-file-per-source convention used throughout eval_data/.
        source = jf.stem.replace("_eval", "")

        if include is not None and source not in include:
            print(f"  skip {jf.name} (source '{source}' not in --include_sources)")
            continue
        if source in exclude:
            print(f"  skip {jf.name} (source '{source}' in --exclude_sources)")
            continue

        out_path = out_dir / jf.name
        n = process_file(jf, out_path, verify=not args.no_verify)
        print(f"  {jf.name}: {n} examples -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default="eval_data/with_tashkeel",
        help="Dir with fully-vocalized (interleaved) JSONL files",
    )
    parser.add_argument(
        "--output_dir", default="eval_data/postfix_tashkeel",
        help="Dir to write postfix-vocalized JSONL files",
    )
    parser.add_argument(
        "--include_sources", nargs="+", default=None,
        help="Only process these sources (by filename stem, e.g. arcd tydiqa). "
             "Overrides --exclude_sources.",
    )
    parser.add_argument(
        "--exclude_sources", nargs="+", default=["mkqa"],
        help="Skip these sources (default: mkqa, per current experiment scope)",
    )
    parser.add_argument(
        "--no_verify", action="store_true",
        help="Skip the per-example character-multiset sanity check (faster)",
    )
    args = parser.parse_args()
    main(args)

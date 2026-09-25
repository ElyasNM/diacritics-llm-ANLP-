# Vowel Diacritization and LLM QA (Hebrew Niqud / Arabic Tashkeel)

Code, data, and results for a study measuring whether adding vowel
diacritics (Niqud in Hebrew, Tashkeel in Arabic) to QA inputs helps or
hurts LLM performance, across three models per language and six QA
benchmarks, with an LLM-as-judge re-scoring pass and a postfix
diacritic-placement ablation.

Core metric: **ΔF1 / ΔEM = score(diacritized) − score(plain)**, per
dataset/model, computed by `evaluate.py` + `compute_delta.py` in each
language's `code/` subdirectory.

## Datasets

| Dataset     | Language | n       | Type                     |
|-------------|----------|---------|--------------------------|
| ARCD        | Arabic   | 1,395   | Extractive RC            |
| TyDiQA      | Arabic   | 15,726  | Extractive/info-seeking  |
| MKQA-AR     | Arabic   | 4,966   | Open-domain QA           |
| HeQ         | Hebrew   | 8,363   | Extractive RC            |
| ParaShoot   | Hebrew   | 3,038   | Extractive RC            |
| MKQA-HE     | Hebrew   | 4,573   | Open-domain QA           |

All six are publicly available, human-authored benchmarks, originally
distributed without diacritics. Full citations and per-dataset notes are
in `DatasetDetails.txt`. Hebrew Niqud was applied with
`dicta-il/dictabert-large-char-menaked` (Nakdan/DICTA); Arabic Tashkeel
with Farasa.

## Models

- **Arabic:** SILMA-9B-Instruct, Qwen2.5-7B-Instruct, dictalm2.0-instruct
  (Hebrew-tuned, used as a cross-lingual control)
- **Hebrew:** Qwen2.5-7B-Instruct, dictalm2.0-instruct

## Repository layout

```
.
├── code/
│   ├── hebrew/           # download -> preprocess -> apply_niqud -> evaluate -> compute_delta (+ recalc_heq_filtered, compute_fragmentation_hebrew)
│   └── arabic/           # preprocess -> evaluate -> compute_delta / compute_fragmentation / prepare_postfix_tashkeel
├── data/
│   ├── hebrew/
│   │   ├── raw/           # untouched HF dumps (HeQ, MKQA, ParaShoot)
│   │   ├── no_niqud/      # preprocessed baseline JSONL
│   │   └── with_niqud/    # diacritized JSONL (paired with no_niqud by id)
│   └── arabic/
│       ├── no_tashkeel/   # plain baseline JSONL (ARCD, TyDiQA, MKQA-AR)
│       ├── with_tashkeel/ # Tashkeel (Farasa, standard Unicode prefix placement)
│       └── postfix_tashkeel/  # postfix ablation: same diacritics, moved to the end of each word (ARCD, TyDiQA only)
├── results/
│   ├── hebrew/{dictalm2,qwen2.5-7b}/       # no_niqud.json, with_niqud.json, delta.json, heq_filtered_* (quality-filtered HeQ subset)
│   └── arabic/
│       ├── {qwen,silma,dictalm2}/          # no_tashkeel.json, with_tashkeel.json, delta.json
│       ├── postfix_silma/                  # SILMA postfix-ablation predictions + delta
│       └── fragmentation_report.json       # tokenizer fragmentation ratios (plain -> Tashkeel)
├── llm_judge/
│   ├── hebrew/           # judge input/output, question set, delta report, judge scripts, human-agreement pilot
│   └── arabic/           # same, for the Arabic judge run
├── failure_case_corpus/
│   ├── niqud_hurt_examples_ALL.txt   # ~900 judge-confirmed diacritic-induced correct->incorrect cases (Hebrew and Arabic)
│   └── mkqa_harm_dump.json           # MKQA-specific harm cases (structured)
├── report/
│   └── examples_for_report.txt       # hand-annotated qualitative failure/success examples
└── DatasetDetails.txt
```

## Pipeline

### Hebrew (`code/hebrew/`)

```bash
python download.py                 # -> data/hebrew/raw/
python preprocess.py                # -> data/hebrew/no_niqud/
# run an external diacritizer (e.g. Nakdan/DICTA) on context+question fields
# of each file in data/hebrew/no_niqud/, save to data/hebrew/with_niqud/
# with identical filenames/ids/answers

python evaluate.py --model <hf-model> --data data/hebrew/no_niqud   --output results/hebrew/<model>/no_niqud.json
python evaluate.py --model <hf-model> --data data/hebrew/with_niqud --output results/hebrew/<model>/with_niqud.json
python compute_delta.py --baseline results/hebrew/<model>/no_niqud.json --niqud results/hebrew/<model>/with_niqud.json --output results/hebrew/<model>/delta.json

# optional: quality-filtered HeQ recomputation
python recalc_heq_filtered.py
```

### Arabic (`code/arabic/`)

> `preprocess.py` and `prepare_postfix_tashkeel.py` use fixed relative paths
> (`data/no_tashkeel`, `data/with_tashkeel`, `eval_data/with_tashkeel`,
> `eval_data/postfix_tashkeel`) rather than this repo's `data/arabic/...`
> nesting — either run them with `data/arabic/` as your working directory,
> or adjust the path constants/`--input_dir`/`--output_dir` defaults at the
> top of each script to point at `data/arabic/`.

```bash
python preprocess.py                          # build plain/Tashkeel JSONL -> data/arabic/{no_tashkeel,with_tashkeel}/
python prepare_postfix_tashkeel.py             # postfix ablation -> data/arabic/postfix_tashkeel/

python evaluate.py --model <hf-model> --data data/arabic/no_tashkeel   --output results/arabic/<model>/no_tashkeel.json
python evaluate.py --model <hf-model> --data data/arabic/with_tashkeel --output results/arabic/<model>/with_tashkeel.json
python compute_delta.py --baseline results/arabic/<model>/no_tashkeel.json --niqud results/arabic/<model>/with_tashkeel.json --output results/arabic/<model>/delta.json

python compute_fragmentation.py                # tokenizer fragmentation ratio report
```

### LLM-as-judge (`llm_judge/{hebrew,arabic}/scripts/`)

Each language directory has the same 6-step pipeline (see each `scripts/README.md`
for details): sample a stratified evaluation set → build the judge input →
run the judge (`gpt-4o-mini-2024-07-18`, binary CORRECT/INCORRECT, blinded
to model identity and condition) → compute the judge-based delta →
check agreement against human labels (Arabic binary pilot: 94.4% agreement,
κ=0.874; Arabic post-run Qwen sanity check: 90.8%, κ=0.813; Hebrew pilot:
94.0%, κ=0.879). The judge always sees the plain (unvocalized) question,
regardless of condition.

## Setup

```bash
pip install -r requirements.txt
```

See `code/arabic/requirements.txt` for Arabic-pipeline-specific dependencies
(Farasa, tokenizer libraries).

# Arabic Tashkeel Evaluation Pipeline

Measures the effect of automatic diacritization (Tashkeel) on LLM performance
on Arabic QA tasks. Core metric: **Δ F1 / Δ EM = score(with tashkeel) − score(no tashkeel)**.

## Datasets

| Dataset    | Type              | Size (eval) | Role              |
|------------|-------------------|-------------|-------------------|
| ARCD       | Extractive RC     | ~1,395      | Primary           |
| TyDi QA    | Extractive RC     | ~15,726     | Primary           |
| MKQA-Arabic| Open-domain QA    | ~10,000     | Negative control  |

Datasets were diacritized using Farasa (see vocalization notebook).
TyDi QA baseline uses `passage_text_clean` / `question_text_clean`
(selective diacritics stripped) — NOT the original columns.

## Models

| Model                              | Type                  |
|------------------------------------|-----------------------|
| silma-ai/SILMA-9B-Instruct-v1.0    | Arabic-specific       |
| Qwen/Qwen2.5-7B-Instruct           | Multilingual          |
| meta-llama/Llama-3.1-8B-Instruct   | Multilingual          |
| dicta-il/dictalm2.0-instruct       | Hebrew-specific (control) |

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### Step 1 — Preprocess (build no/with tashkeel JSONL files)
```bash
python preprocess.py --drive_dir /content/drive/MyDrive/Arabic_Tashkeel
```
Outputs unified JSONL to `data/no_tashkeel/` and `data/with_tashkeel/`.

### Step 2 — Evaluate both conditions per model
```bash
# Example for SILMA:
python evaluate.py \
    --model  silma-ai/SILMA-9B-Instruct-v1.0 \
    --data   data/no_tashkeel \
    --output results/silma/no_tashkeel.json

python evaluate.py \
    --model  silma-ai/SILMA-9B-Instruct-v1.0 \
    --data   data/with_tashkeel \
    --output results/silma/with_tashkeel.json
```

Repeat for all four models (qwen, llama, dictalm2).

Key arguments:

| Flag               | Default | Description                                    |
|--------------------|---------|------------------------------------------------|
| `--model`          | —       | HuggingFace model name or local path           |
| `--data`           | —       | Directory with `.jsonl` files                  |
| `--output`         | —       | Path to write results JSON                     |
| `--max_samples`    | None    | Cap total examples (stratified across sources) |
| `--batch_size`     | 8       | Inference batch size                           |
| `--max_new_tokens` | 80      | Max tokens generated per answer                |
| `--max_length`     | 3072    | Max input tokens (truncates long contexts)     |
| `--device`         | auto    | device_map value passed to from_pretrained     |

### Step 3 — Compute delta
```bash
python compute_delta.py \
    --baseline  results/silma/no_tashkeel.json \
    --tashkeel  results/silma/with_tashkeel.json \
    --output    results/silma/delta.json
```
Prints delta table per dataset + overall, flip counts, and verbosity analysis.

### Step 4 — Tokenization fragmentation analysis
```bash
python compute_fragmentation.py \
    --no_tashkeel_dir   data/no_tashkeel \
    --with_tashkeel_dir data/with_tashkeel \
    --delta_files results/silma/delta.json \
                  results/qwen/delta.json \
                  results/llama/delta.json \
                  results/dictalm2/delta.json \
    --output fragmentation_report.json
```
Computes per-model x per-dataset fragmentation ratio and correlates with Δ F1.

## File structure

```
arabic_tashkeel_eval/
├── data/
│   ├── no_tashkeel/         # baseline JSONL (step 1 output)
│   └── with_tashkeel/       # vocalized JSONL (step 1 output)
├── results/
│   ├── silma/
│   │   ├── no_tashkeel.json
│   │   ├── with_tashkeel.json
│   │   └── delta.json
│   ├── qwen/
│   ├── llama/
│   └── dictalm2/
├── preprocess.py
├── evaluate.py
├── compute_delta.py
├── compute_fragmentation.py
├── requirements.txt
└── README.md
```

## Schema of unified JSONL

```json
{
  "id":           "arcd_train_abc123",
  "source":       "arcd",
  "context":      "...",
  "question":     "...",
  "answers":      ["answer text", ...],
  "answer_start": [42, ...]
}
```
- `context` is `null` for MKQA (open-domain, no passage)
- `answer_start` is `null` for MKQA and TyDi QA (byte offsets not usable as char offsets)
- `answers` are always held fixed across both conditions

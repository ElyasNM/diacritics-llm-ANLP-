"""
Re-compute aggregate metrics and delta for HEQ (quality-filtered),
and regenerate overall numbers. Does NOT modify any existing files.
Writes results to: results/{model}/heq_filtered_*.json
                   results/{model}/heq_filtered_delta.json
                   heq_filtered_summary.txt
"""
import json, os, sys, collections
sys.stdout.reconfigure(encoding='utf-8')

base       = r'C:\Users\Mossa\Desktop\NLP-Multilingual Analysis\hebrew_niqud_eval\results'
raw_base   = r'C:\Users\Mossa\Desktop\NLP-Multilingual Analysis\hebrew_niqud_eval\data\raw\heq'
KEEP_QUALS = {'good', 'verified', 'gold', 'checked'}

# ── 1. Build ID → quality map ────────────────────────────────────────────────
id_to_quality = {}
for split in ['train', 'validation', 'test']:
    with open(os.path.join(raw_base, f'{split}.jsonl'), encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            id_to_quality[row['ID']] = row['Question_Quality']

# ── 2. F1 helper (identical to eval logic) ──────────────────────────────────
import string, re

def normalize(s):
    s = s.lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def token_f1(pred, gold):
    p_tok = normalize(pred).split()
    g_tok = normalize(gold).split()
    if not p_tok or not g_tok:
        return 0.0
    common = collections.Counter(p_tok) & collections.Counter(g_tok)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(p_tok)
    rec  = n_common / len(g_tok)
    return 2 * prec * rec / (prec + rec)

def best_f1(pred, golds):
    return max(token_f1(pred, g) for g in golds)

def best_em(pred, golds):
    np = normalize(pred)
    return int(any(np == normalize(g) for g in golds))

# ── 3. Process each model ────────────────────────────────────────────────────
MODELS = ['dictalm2', 'qwen2.5-7b']
summary_lines = []

for model in MODELS:
    for cond in ['with_niqud', 'no_niqud']:
        path = os.path.join(base, model, f'{cond}.json')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)

        filtered = []
        skipped  = 0
        for ex in data['examples']:
            if not ex['id'].startswith('heq_'):
                continue
            raw_id = ex['id'][len('heq_'):]
            if id_to_quality.get(raw_id, '') in KEEP_QUALS:
                filtered.append(ex)
            else:
                skipped += 1

        n = len(filtered)
        em  = 100 * sum(e['em']  for e in filtered) / n
        f1  = 100 * sum(e['f1']  for e in filtered) / n

        out = {
            'model':     data['model'],
            'condition': cond,
            'filter':    'good+verified+gold+checked',
            'aggregate': {
                'heq_filtered': {'n': n, 'em': round(em, 2), 'f1': round(f1, 2)}
            },
            'examples': filtered
        }
        out_path = os.path.join(base, model, f'heq_filtered_{cond}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'[{model}] {cond}  n={n}  skipped={skipped}  EM={em:.2f}  F1={f1:.2f}')

    # ── 4. Delta for filtered HEQ ────────────────────────────────────────────
    with open(os.path.join(base, model, 'heq_filtered_with_niqud.json'), encoding='utf-8') as f:
        wn = json.load(f)
    with open(os.path.join(base, model, 'heq_filtered_no_niqud.json'), encoding='utf-8') as f:
        nn = json.load(f)

    wn_idx = {e['id']: e for e in wn['examples']}
    nn_idx = {e['id']: e for e in nn['examples']}
    shared = list(set(wn_idx) & set(nn_idx))

    helped = sum(1 for eid in shared if nn_idx[eid]['em']==0 and wn_idx[eid]['em']==1)
    hurt   = sum(1 for eid in shared if nn_idx[eid]['em']==1 and wn_idx[eid]['em']==0)
    base_em = wn['aggregate']['heq_filtered']['em']  # wrong — recalc from nn
    base_em  = 100 * sum(nn_idx[e]['em'] for e in shared) / len(shared)
    niq_em   = 100 * sum(wn_idx[e]['em'] for e in shared) / len(shared)
    base_f1  = 100 * sum(nn_idx[e]['f1'] for e in shared) / len(shared)
    niq_f1   = 100 * sum(wn_idx[e]['f1'] for e in shared) / len(shared)

    delta = {
        'model': wn['model'],
        'heq_filtered': {
            'n': len(shared),
            'base_em': round(base_em, 2), 'niq_em': round(niq_em, 2),
            'delta_em': round(niq_em - base_em, 2),
            'base_f1': round(base_f1, 2), 'niq_f1': round(niq_f1, 2),
            'delta_f1': round(niq_f1 - base_f1, 2),
            'niq_helped_em': round(100*helped/len(shared), 2),
            'niq_hurt_em':   round(100*hurt/len(shared), 2),
        }
    }
    delta_path = os.path.join(base, model, 'heq_filtered_delta.json')
    with open(delta_path, 'w', encoding='utf-8') as f:
        json.dump(delta, f, ensure_ascii=False, indent=2)

    d = delta['heq_filtered']
    print(f'[{model}] HEQ-filtered delta  n={d["n"]}  '
          f'Δ_EM={d["delta_em"]:+.2f}  Δ_F1={d["delta_f1"]:+.2f}  '
          f'helped={d["niq_helped_em"]}%  hurt={d["niq_hurt_em"]}%')
    summary_lines.append((model, d))

print()
print('='*60)
print('COMPARISON: original HEQ vs quality-filtered HEQ')
print('='*60)

# Load original deltas for comparison
original = {}
for model in MODELS:
    with open(os.path.join(base, model, 'delta.json'), encoding='utf-8') as f:
        orig = json.load(f)
    original[model] = orig['heq']

for model, d in summary_lines:
    o = original[model]
    print(f'\n{model}:')
    print(f'  Original HEQ  (n={o["n"]:5d}): Δ_EM={o["delta_em"]:+.2f}  Δ_F1={o["delta_f1"]:+.2f}  '
          f'helped={o["niq_helped_em"]}%  hurt={o["niq_hurt_em"]}%')
    print(f'  Filtered HEQ  (n={d["n"]:5d}): Δ_EM={d["delta_em"]:+.2f}  Δ_F1={d["delta_f1"]:+.2f}  '
          f'helped={d["niq_helped_em"]}%  hurt={d["niq_hurt_em"]}%')

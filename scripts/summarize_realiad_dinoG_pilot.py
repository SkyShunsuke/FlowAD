"""Noise-level pilot summary: DiT-S (50 ep, DINOv2-G) mAD per RealIAD category x inv_stop_time t.

  python scripts/summarize_realiad_dinoG_pilot.py [log_root] [out.md]
Prints the markdown table and the t with the best mean mAD (last line: BEST_T=<t>).
"""
import os
import sys

import yaml

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'logs/realiad_ditS_dinoG_pilot'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'noise_sweep.md')
TS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro', 'mad']
HDR = ['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO', 'mAD']

cats = sorted(c for c in os.listdir(ROOT) if not c.startswith('_') and os.path.isdir(os.path.join(ROOT, c)))
res = {}  # (cat, t) -> metrics dict
for c in cats:
    for t in TS:
        p = os.path.join(ROOT, c, 'eval_noise_sweep', f't{t}', 'eval_results.yaml')
        if os.path.exists(p):
            with open(p) as f:
                r = yaml.unsafe_load(f)  # numpy scalars
            r = {k: float(v) for k, v in r.get(c, r.get("average")).items()}
            r['mad'] = sum(r[m] for m in METRICS[:-1]) / 7
            res[c, t] = r

fmt = lambda v: f'{100 * v:.1f}'
complete = [c for c in cats if all((c, t) in res for t in TS)]
mean = {t: {m: sum(res[c, t][m] for c in complete) / len(complete) for m in METRICS} for t in TS} if complete else {}
best_t = max(TS, key=lambda t: mean[t]['mad']) if complete else None

lines = [
    '# RealIAD noise-level pilot: DiT-S + DINOv2-G, FM + 1-step recon (50 epochs, per-category models)',
    '',
    '- Model: DiT-S (hidden 384, depth 12, 6 heads, patch 1), DINOv2-giant last-block features after the final '
    'LayerNorm (256px → 224px → 1536×16×16), plain FM (uniform t)',
    '- Training: 50 epochs, batch 64 (1 GPU), AdamW, lr warmup 10 ep → 5e-5 → cosine 5e-6; checkpoint at epoch 50',
    '- Scoring: x_t = t·x + (1−t)·ε, one Euler step to t=1, `diff+sum` image aggregation; '
    't = inv_stop_time (smaller t = more noise)',
    f'- Categories with all 9 noise levels evaluated: {len(complete)}/{len(cats)}. mAD = mean of 7 metrics, %.',
    '',
]
if complete:
    lines += ['## Mean over categories', '', '| t | ' + ' | '.join(HDR) + ' |', '|---:|' + '---:|' * len(HDR)]
    for t in TS:
        cells = [fmt(mean[t][m]) for m in METRICS]
        if t == best_t:
            cells = [f'**{v}**' for v in cells]
        lines.append(f'| {t} | ' + ' | '.join(cells) + ' |')
    lines += ['', f'**Best noise level: t = {best_t}** (mean mAD {fmt(mean[best_t]["mad"])})', '']

lines += ['## mAD per category', '', '| Category | ' + ' | '.join(f't={t}' for t in TS) + ' | best t |',
          '|---|' + '---:|' * (len(TS) + 1)]
for c in cats:
    row = [res[c, t]['mad'] if (c, t) in res else None for t in TS]
    have = [(v, t) for v, t in zip(row, TS) if v is not None]
    bt = max(have)[1] if have else '-'
    lines.append(f'| {c} | ' + ' | '.join(fmt(v) if v is not None else '-' for v in row) + f' | {bt} |')
if complete:
    hist = {}
    for c in complete:
        bt = max(TS, key=lambda t: res[c, t]['mad'])
        hist[bt] = hist.get(bt, 0) + 1
    lines += ['', '| Per-category best t | ' + ' | '.join(str(t) for t in sorted(hist)) + ' |',
              '|---|' + '---:|' * len(hist),
              '| #categories | ' + ' | '.join(str(hist[t]) for t in sorted(hist)) + ' |']

with open(OUT, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(f'BEST_T={best_t}')

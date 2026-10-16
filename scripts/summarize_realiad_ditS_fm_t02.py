"""Summarize per-category RealIAD DiT-S FM (t=0.2, 1-step recon) runs into markdown."""
import csv
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else 'logs/realiad_ditS_enet_fm_t02'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'summary.md')
REF = 'logs/realiad_ditB_enet'  # DiT-B (logistic t, t=0.1) per-category reference
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro', 'mad']
HDR = ['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO', 'mAD']


def last_epoch(run_dir):
    path = os.path.join(run_dir, 'log.csv')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        eps = [int(r['epoch']) for r in csv.DictReader(f) if r.get('epoch')]
    return max(eps) if eps else None


def load_best(run_dir):
    path = os.path.join(run_dir, 'best_results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d['epoch'], d['results']['average']


cats = sorted(c for c in os.listdir(ROOT) if not c.startswith('_') and os.path.isdir(os.path.join(ROOT, c)))
rows, ref_mads = [], []
for c in cats:
    best = load_best(os.path.join(ROOT, c))
    if best is None:
        continue
    ep, res = best
    stop = last_epoch(os.path.join(ROOT, c))
    ref = load_best(os.path.join(REF, c))
    rows.append((c, ep, stop, os.path.exists(os.path.join(ROOT, c, 'DONE')), res, ref))

fmt = lambda v: f'{100 * v:.1f}'
lines = [
    '# RealIAD: DiT-S, FM + 1-step recon from t=0.2 (per-category models)',
    '',
    '- Model: DiT-S (hidden 384, depth 12, 6 heads, patch 2), EfficientNet-B4 features, plain FM (uniform t)',
    '- Scoring: inversion to t=0.2 (1 Euler step) + 1-step reconstruction, `diff+sum` image aggregation',
    '- Training: 300 max epochs, global batch 64 (1 GPU x 64), AdamW, lr warmup 40 ep → 5e-5',
    '- Early stopping: evaluate every 25 epochs, stop when mAD does not improve for 50 epochs; best-mAD checkpoint saved',
    '- mAD = mean of the 7 metrics. All numbers in %.',
    '- Ref: DiT-B (logistic-t FM, t=0.1) per-category runs from `logs/realiad_ditB_enet` (fixed 300 epochs, best eval epoch).',
    '',
    '| Category | Best epoch | Stopped at | ' + ' | '.join(HDR) + ' | Ref DiT-B mAD (ep) |',
    '|---|---:|---:|' + '---:|' * len(HDR) + '---:|',
]
for c, ep, stop, done, res, ref in rows:
    stop_s = f'{stop}' + ('' if done else ' (running)')
    ref_s = f'{fmt(ref[1]["mad"])} ({ref[0]})' if ref else '-'
    if ref:
        ref_mads.append(ref[1]['mad'])
    lines.append(f'| {c} | {ep} | {stop_s} | ' + ' | '.join(fmt(res[m]) for m in METRICS) + f' | {ref_s} |')

if rows:
    n = len(rows)
    mean = {m: sum(r[4][m] for r in rows) / n for m in METRICS}
    mean_ep = sum(r[1] for r in rows) / n
    mean_stop = sum(r[2] for r in rows if r[2]) / n
    ref_s = f'{fmt(sum(ref_mads) / len(ref_mads))}' if len(ref_mads) == n else '-'
    lines.append(f'| **Mean ({n})** | {mean_ep:.1f} | {mean_stop:.1f} | '
                 + ' | '.join(f'**{fmt(mean[m])}**' for m in METRICS) + f' | {ref_s} |')
    lines += ['', '## Best-epoch distribution', '']
    hist = {}
    for r in rows:
        hist[r[1]] = hist.get(r[1], 0) + 1
    lines.append('| Best epoch | ' + ' | '.join(str(k) for k in sorted(hist)) + ' |')
    lines.append('|---|' + '---:|' * len(hist))
    lines.append('| #categories | ' + ' | '.join(str(hist[k]) for k in sorted(hist)) + ' |')
    early = sum(1 for r in rows if r[3] and r[2] and r[2] < 300)
    lines += ['', f'- Early-stopped before epoch 300: {early}/{sum(r[3] for r in rows)} finished categories.']

with open(OUT, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))

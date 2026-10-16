"""Aggregate RealIAD DiT-G + EfficientNet-B4 (logistic-t FM, 1-step recon from t=0.1) over 5 seeds into markdown.
Seed 0: logs/realiads/realiad_mc_ditG_enet_0903_<cat> (best eval epoch from log.csv; last run in the file if restarted).
Seeds 1-4: logs/realiad_ditG_enet_seeds/seed<s>/<cat>/best_results.json (early stopping, patience 50)."""
import ast
import csv
import json
import os
import statistics
import sys

ROOT = 'logs/realiad_ditG_enet_seeds'
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'summary.md')
SEED0 = 'logs/realiads/realiad_mc_ditG_enet_0903_{}'
SEEDS = [0, 1, 2, 3, 4]
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro', 'mad']
HDR = ['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO', 'mAD']
CATS = sorted(d.split('0903_')[1] for d in os.listdir('logs/realiads') if d.startswith('realiad_mc_ditG_enet_0903_'))


def load_seed0(cat):
    """(best epoch, metrics, last epoch) over the eval rows of the final run in log.csv."""
    with open(os.path.join(SEED0.format(cat), 'log.csv')) as f:
        rows = list(csv.DictReader(f))
    evals, epoch = [], 0
    for r in rows:
        if r.get('epoch'):
            if int(r['epoch']) < epoch:  # restarted run appended to the same csv
                evals = []
            epoch = int(r['epoch'])
        if r.get('average'):
            evals.append((epoch, ast.literal_eval(r['average'])))
    ep, res = max(evals, key=lambda e: e[1]['mad'])  # first max on ties, as in train.py
    return ep, res, epoch


def load_seed(seed, cat):
    run = os.path.join(ROOT, f'seed{seed}', cat)
    if not os.path.exists(os.path.join(run, 'DONE')):
        return None
    with open(os.path.join(run, 'best_results.json')) as f:
        d = json.load(f)
    with open(os.path.join(run, 'log.csv')) as f:
        last = max(int(r['epoch']) for r in csv.DictReader(f) if r.get('epoch'))
    return d['epoch'], d['results']['average'], last


res = {(0, c): load_seed0(c) for c in CATS}
res.update({(s, c): load_seed(s, c) for s in SEEDS[1:] for c in CATS})
done = {s: [c for c in CATS if res[(s, c)]] for s in SEEDS}
full = [s for s in SEEDS if len(done[s]) == len(CATS)]

fmt = lambda v: f'{100 * v:.1f}'
pm = lambda xs: f'{100 * statistics.mean(xs):.1f} ± {100 * statistics.stdev(xs):.1f}' if len(xs) > 1 else fmt(xs[0])
lines = [
    '# RealIAD: DiT-G + EfficientNet-B4, FM + 1-step recon (per-category models), 5 seeds',
    '',
    '- Model: DiT-G (hidden 2048, depth 16, 16 heads, patch 2), EfficientNet-B4 features, logistic-t FM (t_mu=logit(0.1), t_sigma=0.35)',
    '- Scoring: inversion to t=0.1 (1 Euler step) + 1-step reconstruction, `diff+sum` image aggregation',
    '- Training: 300 max epochs, global batch 64, AdamW, lr warmup 40 ep → 5e-5; eval every 25 epochs, best-mAD epoch reported',
    '- Seed 0 (previous): `logs/realiads/realiad_mc_ditG_enet_0903_*` (8 GPU x 8; runs stopped at various epochs, best eval epoch taken)',
    '- Seeds 1-4 (new): `logs/realiad_ditG_enet_seeds/seed*` (1 GPU x 64; early stopping when mAD does not improve for 50 epochs)',
    '- mAD = mean of the 7 metrics. All numbers in %. Mean ± std = sample std over seeds.',
    f'- Completed seeds: {", ".join(f"{s} ({len(done[s])}/{len(CATS)})" for s in SEEDS)}',
    '',
    '## Category-mean per seed',
    '',
    '| Seed | #cats | Mean best epoch | ' + ' | '.join(HDR) + ' |',
    '|---|---:|---:|' + '---:|' * len(HDR),
]
seed_means = {}
for s in SEEDS:
    cs = done[s]
    if not cs:
        continue
    seed_means[s] = {m: statistics.mean(res[(s, c)][1][m] for c in cs) for m in METRICS}
    ep = statistics.mean(res[(s, c)][0] for c in cs)
    lines.append(f'| {s} | {len(cs)} | {ep:.1f} | ' + ' | '.join(fmt(seed_means[s][m]) for m in METRICS) + ' |')
if len(full) > 1:
    lines.append(f'| **Mean ± std ({len(full)} seeds)** | {len(CATS)} | | '
                 + ' | '.join(f'**{pm([seed_means[s][m] for s in full])}**' for m in METRICS) + ' |')

lines += ['', '## Per-category (mean ± std over completed seeds)', '',
          '| Category | #seeds | Best epochs (seed 0..4) | ' + ' | '.join(HDR) + ' |',
          '|---|---:|---|' + '---:|' * len(HDR)]
for c in CATS:
    ss = [s for s in SEEDS if res[(s, c)]]
    eps = '/'.join(str(res[(s, c)][0]) if res[(s, c)] else '-' for s in SEEDS)
    lines.append(f'| {c} | {len(ss)} | {eps} | ' + ' | '.join(pm([res[(s, c)][1][m] for s in ss]) for m in METRICS) + ' |')

lines += ['', '## Per-category mAD by seed (best epoch / stopped epoch)', '',
          '| Category | ' + ' | '.join(f'Seed {s}' for s in SEEDS) + ' |', '|---|' + '---:|' * len(SEEDS)]
for c in CATS:
    cells = [f'{fmt(r[1]["mad"])} ({r[0]}/{r[2]})' if (r := res[(s, c)]) else '-' for s in SEEDS]
    lines.append(f'| {c} | ' + ' | '.join(cells) + ' |')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))

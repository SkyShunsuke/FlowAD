"""Summarize DiT-G + FM + 1-step + EfficientNet-b4 (0829 setting) over 5 seeds on MVTec AD / VisA / MPDD.

Seed 0 = the original runs (logs/<ds>_ditG_enet_0829), seeds 10/20/30/40 = logs/ditG_seeds/<ds>_ditG_enet_0829_seed<s>.
For each run the best epoch is the eval epoch with the highest average mAD (same criterion as save_best).
Writes logs/ditG_seeds/summary.md (mean +- sample std over the available seeds).
    python scripts/summarize_ditG_seeds.py
"""
import ast
import csv
import os

import numpy as np

DATASETS = {'ad': 'MVTec AD', 'visa': 'VisA', 'mpdd': 'MPDD'}
SEEDS = [0, 10, 20, 30, 40]
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro', 'mad']
HEADER = ['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'AUPRO', 'mAD']
OUT = 'logs/ditG_seeds/summary.md'


def run_dir(ds, seed):
    return f'logs/{ds}_ditG_enet_0829' if seed == 0 else f'logs/ditG_seeds/{ds}_ditG_enet_0829_seed{seed}'


def best_epoch(log_csv):
    """Return (best_epoch, average metrics at that epoch, last evaluated epoch) or None."""
    if not os.path.exists(log_csv):
        return None
    best, last = None, 0
    with open(log_csv) as f:
        for row in csv.DictReader(f):
            if not row.get('average'):
                continue
            avg = ast.literal_eval(row['average'])
            ep = int(row['epoch'])
            last = max(last, ep)
            if best is None or avg['mad'] > best[1]['mad']:
                best = (ep, avg)
    return None if best is None else (best[0], best[1], last)


def fmt(x):
    return f'{100 * x:.1f}'


def main():
    per_run, agg = [], []
    for ds, name in DATASETS.items():
        vals = []
        for s in SEEDS:
            d = run_dir(ds, s)
            r = best_epoch(os.path.join(d, 'log.csv'))
            if r is None:
                per_run.append(f'| {name} | {s} | – | – | ' + ' | '.join(['–'] * len(METRICS)) + ' |')
                continue
            ep, avg, last = r
            done = os.path.exists(os.path.join(d, 'DONE')) or s == 0
            status = f'{last}/300' + ('' if done or last == 300 else ' (running)')
            per_run.append(f'| {name} | {s} | {ep} | {status} | ' + ' | '.join(fmt(avg[m]) for m in METRICS) + ' |')
            if last == 300:
                vals.append([avg[m] for m in METRICS])
        if vals:
            v = 100 * np.array(vals)
            std = v.std(0, ddof=1) if len(v) > 1 else np.zeros(v.shape[1])
            agg.append(f'| {name} | {len(v)} | ' + ' | '.join(f'{m:.1f} ± {sd:.1f}' for m, sd in zip(v.mean(0), std)) + ' |')

    lines = [
        '# DiT-G + FM + 1-step + EfficientNet-b4: 5-seed results',
        '',
        'Best epoch per run = eval epoch (every 25) with the highest average mAD. '
        'Seed 0 = original 0829 runs; seeds 10/20/30/40 = new runs (identical config, `opt.global_seed` changed, '
        'train sampler seeded with it). Mean ± sample std (ddof=1) over completed runs only.',
        '',
        '## Aggregate (mean ± std)',
        '',
        '| Dataset | #seeds | ' + ' | '.join(HEADER) + ' |',
        '|---|---|' + '---|' * len(HEADER),
        *agg,
        '',
        '## Per run (best epoch)',
        '',
        '| Dataset | Seed | Best ep. | Evaluated | ' + ' | '.join(HEADER) + ' |',
        '|---|---|---|---|' + '---|' * len(HEADER),
        *per_run,
        '',
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

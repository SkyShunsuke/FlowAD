"""Summarize multi-seed DiT-G + EfficientNet-B4 FM (ad_ditG_enet.yaml setting, 1 GPU x 64, no DDP) on MVTec AD / VisA / MPDD.
Best epoch per run = eval epoch (every 25) with the highest average mAD, read from log.csv (so unfinished runs show progress).
Writes logs/ditG_enet_fm_fr<FR>_tf32_seeds/summary.md (mean +- sample std over finished runs).
    python scripts/summarize_ditG_enet_fm_seeds.py [feature_res=16]
"""
import ast
import csv
import os
import sys

import numpy as np

FR = int(sys.argv[1]) if len(sys.argv) > 1 else 16
ROOT = f'logs/ditG_enet_fm_fr{FR}_tf32_seeds'
DATASETS = {'ad': 'MVTec AD', 'visa': 'VisA', 'mpdd': 'MPDD'}
SEEDS = [0, 1, 2, 3, 4]
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro', 'mad']
HEADER = ['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'AUPRO', 'mAD']
REF = {'ad': ('ad_mc_ditG_enet_0903 (8 GPU x 8, seed 0)', 'logs/ad_mc_ditG_enet_0903/log.csv')}


def best_epoch(log_csv):
    """(best epoch, average metrics at that epoch, last evaluated epoch) or None."""
    if not os.path.exists(log_csv):
        return None
    evals = []
    with open(log_csv) as f:
        for row in csv.DictReader(f):
            if row.get('average'):
                evals.append((int(row['epoch']), ast.literal_eval(row['average'])))
    if not evals:
        return None
    ep, res = max(evals, key=lambda e: e[1]['mad'])  # first max on ties, as in train.py
    return ep, res, max(e for e, _ in evals)


def fmt(res):
    return ' | '.join(f'{100 * res[m]:.1f}' for m in METRICS)


lines = [f'# DiT-G + EfficientNet-B4 FM, 1-step recon (feature_res {FR}): multi-seed results', '',
         'Config: `configs/train/ad_ditG_enet.yaml` (logistic-t FM, t_mu=logit(0.1), t_sigma=0.35, patch 2, 300 ep, '
         'lr 5e-5), dataset swapped; 1 GPU x 64 (no DDP), fp32 + TF32 matmuls (single-GPU bf16 diverged), `opt.global_seed` = seed. Best epoch = eval epoch (every 25) '
         'with the highest mAD. Mean ± sample std (ddof=1) over finished runs.', '',
         '## Aggregate (mean ± std)', '',
         '| Dataset | #seeds | ' + ' | '.join(HEADER) + ' |', '|---|---' + '|---' * len(METRICS) + '|']
per_run = []
for ds, name in DATASETS.items():
    done = []
    for s in SEEDS:
        run = f'{ROOT}/{ds}_s{s}'
        r = best_epoch(f'{run}/log.csv')
        finished = os.path.exists(f'{run}/DONE')
        if r is None:
            per_run.append(f'| {name} | {s} | – | – |' + ' – |' * len(METRICS))
            continue
        ep, res, last = r
        per_run.append(f'| {name} | {s} | {ep} | {last}/300{"" if finished else " (running)"} | {fmt(res)} |')
        if finished:
            done.append([100 * res[m] for m in METRICS])
    if done:
        a = np.array(done)
        sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros(len(METRICS))
        lines.append(f'| {name} | {len(a)} | ' + ' | '.join(f'{m:.1f} ± {v:.1f}' for m, v in zip(a.mean(0), sd)) + ' |')
    else:
        lines.append(f'| {name} | 0 |' + ' – |' * len(METRICS))

lines += ['', '## Per run (best epoch)', '',
          '| Dataset | Seed | Best ep. | Evaluated | ' + ' | '.join(HEADER) + ' |', '|---|---|---|---' + '|---' * len(METRICS) + '|']
lines += per_run
lines += ['', '## Reference', '', '| Run | Best ep. | ' + ' | '.join(HEADER) + ' |', '|---|---' + '|---' * len(METRICS) + '|']
for ds, (label, path) in REF.items():
    r = best_epoch(path)
    if r:
        lines.append(f'| {label} | {r[0]} | {fmt(r[1])} |')

os.makedirs(ROOT, exist_ok=True)
with open(f'{ROOT}/summary.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n'.join(lines))

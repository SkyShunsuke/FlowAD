"""Summarize single-class DiT-G + EfficientNet-B4 FM on MVTec AD (0829 setting, one model per category).
Per category: eval epoch (every 10) with the highest mAD, read from log.csv (so unfinished runs show progress).
Writes logs/ad_sc_ditG_enet_0829/summary.md with the multi-class 0829 run (final epoch) as reference.
    python scripts/summarize_ad_sc_ditG_enet_0829.py
"""
import ast
import csv
import os

import numpy as np

ROOT = 'logs/ad_sc_ditG_enet_0829'
REF = 'logs/ad_ditG_enet_0829/log.csv'
CLASSES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
           'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro', 'mad']
HEADER = ['I-AUROC', 'I-AP', 'I-F1max', 'P-AUROC', 'P-AP', 'P-F1max', 'AUPRO', 'mAD']


def evals(log_csv, key='average'):
    out = []
    if os.path.exists(log_csv):
        with open(log_csv) as f:
            for row in csv.DictReader(f):
                if row.get(key):
                    out.append((int(row['epoch']), ast.literal_eval(row[key])))
    return out


def fmt(vals):
    return ' | '.join(f'{v:.1f}' for v in vals)


rows, best = [], []
for c in CLASSES:
    ev = evals(f'{ROOT}/{c}/log.csv')
    if not ev:
        rows.append(f'| {c} | – | – |' + ' – |' * len(METRICS))
        continue
    ep, res = max(ev, key=lambda e: e[1]['mad'])  # first max on ties, as in train.py
    last = max(e for e, _ in ev)
    status = 'done' if os.path.exists(f'{ROOT}/{c}/DONE') else 'running'
    stop = f'{last}' + (' (early stop)' if status == 'done' and last < 300 else '') + ('' if status == 'done' else ' (running)')
    vals = [100 * res[m] for m in METRICS]
    best.append(vals)
    rows.append(f'| {c} | {ep} | {stop} | {fmt(vals)} |')

lines = ['# MVTec AD single-class DiT-G + EfficientNet-B4 FM (0829 setting)', '',
         'One DiT-G per category. Config = `logs/ad_ditG_enet_0829/config.yaml` (linear-t FM, patch 1, feature_res 16, '
         '1-step recon from t=0.1, lr 5e-5, 40 warmup ep, max 300 ep) with: `dataset_name: mvtec_ad` + category, '
         '`use_class_labels: false`, 1 GPU x 8 (0829: 8 GPUs x 8), fp32 + TF32. Eval every 10 epochs; early stop when mAD '
         'has not improved for 20 epochs. Reported = best-mAD eval epoch per category (selected on the test set).', '',
         '| Category | Best ep | Last ep | ' + ' | '.join(HEADER) + ' |', '|---|---|---' + '|---' * len(METRICS) + '|']
lines += rows
if best:
    lines.append(f'| **Mean ({len(best)}/{len(CLASSES)})** | | | ' + ' | '.join(f'**{v:.1f}**' for v in np.mean(best, 0)) + ' |')

ref = evals(REF)
if ref:
    ep, res = ref[-1]
    lines += ['', f'Reference: multi-class `ad_ditG_enet_0829` (8 GPUs x 8, bf16), final epoch {ep}:', '',
              '| Model | ' + ' | '.join(HEADER) + ' |', '|---' + '|---' * len(METRICS) + '|',
              f'| multi-class 0829 (ep {ep}) | {fmt([100 * res[m] for m in METRICS])} |']
    if best:
        lines.append(f'| single-class (best ep) | {fmt(np.mean(best, 0))} |')

open(f'{ROOT}/summary.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

"""Collect the DiT-G 2D recon sweeps into CSVs (see scripts/gen_ditG_2d_sweep_configs.py).
Writes <run>/eval_2d_sweep/results.csv (long: one row per (noise, steps), all 7 metrics + mAD, in %)
and <run>/eval_2d_sweep/mad_grid.csv (mAD, rows = noise strength, cols = recon_steps). Missing runs are blank.
    python scripts/summarize_ditG_2d_sweep.py mvtec visa mpdd
    python scripts/summarize_ditG_2d_sweep.py realiad_ditB  # per-class models: class mean, only (s, n) done for all classes;
                                                             # also writes per_class.csv (one row per class, s, n)
"""
import csv
import os
import sys

import yaml

RUNS = {'mvtec': 'logs/ad_mc_ditG_enet_0903', 'visa': 'logs/visa_ditG_enet_0829', 'mpdd': 'logs/mpdd_ditG_enet_0829',
        'mvtec0829': 'logs/ad_ditG_enet_0829', 'mvtec_ditB': 'logs/mvtec_mc_ditB_enet',
        'visa_ditB': 'logs/visa_ditB_enet', 'mpdd_ditB': 'logs/mpdd_mc_ditB_enet'}
PER_CLASS = {'realiad_ditB': 'logs/realiad_ditB_enet'}  # run root with one run dir per class
NOISE = [round(0.1 * i, 1) for i in range(1, 10)]
STEPS = [1, 2, 4, 8, 20, 40, 80, 100, 200]
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro', 'mad']


def load(f):
    avg = yaml.unsafe_load(open(f))['average']
    return {m: 100 * float(avg[m]) for m in METRICS}


def collect_per_class(root, ROOT):
    classes = sorted(c for c in os.listdir(root) if os.path.isdir(f'{root}/{c}/checkpoints'))
    res, rows = {}, []
    for s in NOISE:
        for n in STEPS:
            per = {c: load(f) for c in classes if os.path.exists(f := f'{root}/{c}/eval_2d_sweep/s{s}_n{n}/eval_results.yaml')}
            rows += [[c, s, n] + [f'{r[m]:.2f}' for m in METRICS] for c, r in per.items()]
            if len(per) == len(classes):
                res[s, n] = {m: sum(r[m] for r in per.values()) / len(per) for m in METRICS}
    os.makedirs(ROOT, exist_ok=True)
    with open(f'{ROOT}/per_class.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class', 'noise_strength', 'recon_steps'] + METRICS)
        w.writerows(rows)
    return res, len(rows), len(classes)


for ds in sys.argv[1:] or RUNS:
    if ds in PER_CLASS:
        ROOT = f'{PER_CLASS[ds]}/eval_2d_sweep'
        res, n_runs, n_cls = collect_per_class(PER_CLASS[ds], ROOT)
        print(f'{ds}: {n_runs}/{n_cls * len(NOISE) * len(STEPS)} class runs done')
    else:
        ROOT = f'{RUNS[ds]}/eval_2d_sweep'
        res = {(s, n): load(f) for s in NOISE for n in STEPS
               if os.path.exists(f := f'{ROOT}/s{s}_n{n}/eval_results.yaml')}

    with open(f'{ROOT}/results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['noise_strength', 'inv_stop_time', 'recon_steps'] + METRICS)
        for (s, n), r in sorted(res.items()):
            w.writerow([s, round(1 - s, 1), n] + [f'{r[m]:.2f}' for m in METRICS])
    with open(f'{ROOT}/mad_grid.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['noise_strength\\recon_steps'] + STEPS)
        for s in NOISE:
            w.writerow([s] + [f'{res[s, n]["mad"]:.2f}' if (s, n) in res else '' for n in STEPS])

    print(f'{ds}: {len(res)}/{len(NOISE) * len(STEPS)} done -> {ROOT}/results.csv, {ROOT}/mad_grid.csv')
    print(open(f'{ROOT}/mad_grid.csv').read())

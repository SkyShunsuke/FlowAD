"""Collect DiffusionAD results: final-epoch averages, upstream `best` selection,
and the eval curve (every 50 epochs). Usage: python summarize.py [--per_class]"""
import argparse
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_aupro', 'px_ap', 'px_f1max', 'mad']
RUNS = [('MVTec-AD', 'mvtec_300ep'), ('VisA', 'visa_300ep')]


def read(path):
    with open(path) as f:
        return {r['category']: {c: float(r[c]) * 100 for c in COLS} for r in csv.DictReader(f)}


def row(name, r):
    return f"| {name} | " + ' | '.join(f'{r[c]:.1f}' for c in COLS) + ' |'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_class', action='store_true')
    a = ap.parse_args()
    hdr = '| | I-AUROC | I-AP | I-F1max | P-AUROC | P-AUPRO | P-AP | P-F1max | mAD |\n|---|' + '---|' * 8
    finals, bests, curves = [], [], {}
    for ds, run in RUNS:
        out = os.path.join(HERE, 'out', run)
        evals = {int(re.search(r'ep(\d+)', p).group(1)): read(p) for p in glob.glob(f'{out}/eval_ep*.csv')}
        final = os.path.join(out, 'final_last.csv')
        if os.path.exists(final):
            ep = int(run.split('_')[1].replace('ep', ''))
            evals[ep] = read(final)
            finals.append((f'{ds} (ep {ep})', read(final)))
        if not evals:
            continue
        curves[ds] = sorted(evals.items())
        # upstream train.py: keep the eval with img+px AUROC >= best sum and img >= best img
        # (both rounded as round(x, 3) * 100)
        best, be = None, None
        for ep, r in sorted(evals.items()):
            img, px = round(r['average']['img_auroc'] / 100, 3), round(r['average']['px_auroc'] / 100, 3)
            if best is None or (img + px >= best[0] + best[1] and img >= best[0]):
                best, be = (img, px), ep
        bests.append((f'{ds} (ep {be})', evals[be]['average']))
    print('## Final epoch (reported)\n\n' + hdr)
    for n, r in finals:
        print(row(n, r['average']))
    print('\n## Upstream selection (`best` on test set, every 50 epochs) — for reference only\n\n' + hdr)
    for n, r in bests:
        print(row(n, r))
    print('\n## Eval curve (average over classes)\n')
    for ds, cur in curves.items():
        print(f'**{ds}**\n\n| epoch | I-AUROC | P-AUROC | P-AUPRO | mAD |\n|---|---|---|---|---|')
        for ep, r in cur:
            if ep in (50, 100, 300, 500, 1000, 1500, 2000, 2500, 3000) or ep == cur[-1][0]:
                x = r['average']
                print(f"| {ep} | {x['img_auroc']:.1f} | {x['px_auroc']:.1f} | {x['px_aupro']:.1f} | {x['mad']:.1f} |")
        print()
    if a.per_class:
        for n, r in finals:
            print(f'\n### {n} per class\n\n' + hdr.replace('| |', '| class |', 1))
            for c in r:
                print(row(c, r[c]))


if __name__ == '__main__':
    main()

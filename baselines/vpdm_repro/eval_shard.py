"""Multi-process final evaluation for large test sets (e.g. Real-IAD).

train_vpdm.evaluate() splits batches over GPUs with threads in one process; the
1000-step chain is launch-bound, so the GIL caps each GPU at ~1/3 utilisation.
Here each process handles batches b = shard, shard + n, ... of the same fixed
shuffled order, with the same per-batch chain seed (seed * 1_000_003 + b), so
the scores are identical to evaluate()'s. Each shard extracts features only for
its own test images and writes its scores to <log_dir>/shards/; `--merge`
computes the per-class metrics and writes final_*.csv / final_summary.yaml.

  # one process per GPU, then merge (run from the repo root)
  for i in 0 1 2 3 4; do CUDA_VISIBLE_DEVICES=$((i+3)) python baselines/vpdm_repro/eval_shard.py \
      --log_dir <run dir> --shard $i --num_shards 5 & done; wait
  python baselines/vpdm_repro/eval_shard.py --log_dir <run dir> --num_shards 5 --merge
"""
import os
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.getcwd())

from baselines.vpdm_repro.train_vpdm import (build_vpdm, extract_all, write_csv, logger,
                                             BackboneWrapper, build_efficient, build_dataset,
                                             calculate_img_metrics, calculate_px_metrics, divide_by_class)

VARIANTS = {'full_batch': ('full', 'batch'), 'yhat_batch': ('yhat', 'batch')}
# FlowAD one-step plug-in (NOT in paper): oN_batch = one x0 prediction from q(x_N | x0, y_hat)
VARIANTS.update({f'o{t}_batch': (f'o{t}', 'batch') for t in (50, 100, 200, 300, 400, 527, 658, 762, 900)})


def shard_eval(p, args):
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ep = p['logging']['eval']
    seed = p['opt']['global_seed']
    variants = args.variants

    train_ds = build_dataset(train=True, **p['data'])
    test_ds = build_dataset(train=False, **p['data'])
    N, bs = len(test_ds), ep['test_batch_size']
    order = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).numpy() \
        if ep.get('shuffle_test', True) else np.arange(N)
    batches = [order[s:s + bs] for s in range(0, N, bs)]
    mine = list(range(args.shard, len(batches), args.num_shards))
    if args.max_batches:
        mine = mine[:args.max_batches]
    idx = np.concatenate([batches[b] for b in mine])
    logger.info(f"shard {args.shard}/{args.num_shards}: {len(mine)} batches, {len(idx)} images")

    bb, fs = p['model']['backbone'], p['model']['feature_size']
    fe = BackboneWrapper(build_efficient(model_name=bb['model_name'], pretrained=True,
                                         outblocks=bb['outblocks'], outstrides=bb['outstrides']),
                         target_size=(fs, fs)).to(device).eval()
    train = extract_all(fe, train_ds, device)
    test = extract_all(fe, torch.utils.data.Subset(test_ds, idx.tolist()), device, test=True)
    del fe
    torch.cuda.empty_cache()
    fn = p['data'].get('feat_norm', 'none')
    if fn != 'none':
        dims = (0, 2, 3) if fn == 'channel' else (0, 1, 2, 3)
        mu = train['feats'].mean(dims, keepdim=True).reshape(1, -1, 1, 1)
        sd = train['feats'].std(dims, keepdim=True).reshape(1, -1, 1, 1)
        train['feats'] = (train['feats'] - mu) / sd
        test['feats'] = (test['feats'] - mu) / sd

    model = build_vpdm(p).to(device)
    ck = torch.load(os.path.join(p['logging']['log_dir'], 'checkpoints', args.ckpt), map_location='cpu', weights_only=True)
    model.load_state_dict(ck['model'])   # raw (non-EMA) weights, as in the paper
    if p['diffusion'].get('clip_x0') == 'pixel':
        model.x0_range = (train['feats'].amin(0, keepdim=True), train['feats'].amax(0, keepdim=True))
    del train
    model.eval()

    img_sz, k = p['data']['img_size'], ep.get('img_pool_kernel', 16)
    out = {v: {'img': np.zeros(len(idx), np.float32), 'px': np.zeros((len(idx), img_sz, img_sz), np.float16)}
           for v in variants}
    pos = 0
    with torch.no_grad():
        for j, b in enumerate(mine):
            n = len(batches[b])
            x = test['feats'][pos:pos + n]
            for v in variants:
                kind, mode = VARIANTS[v]
                if kind == 'yhat':
                    y_hat, _, _ = model.vpot(x, mode=mode)
                    S = (x - y_hat).pow(2).sum(1).sqrt()
                else:
                    g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + b)
                    one = kind.startswith('o')
                    S = model.score(x, ot_mode=mode, generator=g, t_start=int(kind[1:]) if one else None, one_step=one)
                S = F.interpolate(S[:, None], size=(img_sz, img_sz), mode='bilinear', align_corners=False)
                out[v]['img'][pos:pos + n] = F.avg_pool2d(S, k, stride=1).amax(dim=(1, 2, 3)).cpu().numpy()
                out[v]['px'][pos:pos + n] = S[:, 0].cpu().numpy().astype(np.float16)
            pos += n
            if j % 10 == 0 or j == len(mine) - 1:
                logger.info(f"shard {args.shard}: batch {j + 1}/{len(mine)}")

    d = os.path.join(p['logging']['log_dir'], 'shards')
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f'shard{args.shard}of{args.num_shards}.npz'), idx=idx, cls=test['cls'],
             label=test['label'], mask=np.packbits(test['mask'], axis=-1),
             **{f'{v}_{a}': out[v][a] for v in variants for a in ('img', 'px')})
    logger.info(f"shard {args.shard} done")


def merge(p, args):
    log_dir = p['logging']['log_dir']
    ep = p['logging']['eval']
    files = sorted(glob.glob(os.path.join(log_dir, 'shards', f'shard*of{args.num_shards}.npz')))
    assert len(files) == args.num_shards, f"found {len(files)} of {args.num_shards} shards"
    sh = [np.load(f) for f in files]
    img_sz = p['data']['img_size']
    label = np.concatenate([s['label'] for s in sh])
    mask = np.concatenate([np.unpackbits(s['mask'], axis=-1, count=img_sz) for s in sh])
    # class from the global test index: per-class datasets (e.g. Real-IAD) may all
    # report clslabel 0, so the saved 'cls' is not reliable across datasets
    test_ds = build_dataset(train=False, **p['data'])
    idx = np.concatenate([s['idx'] for s in sh])
    cls = np.searchsorted(np.asarray(test_ds.cumulative_sizes), idx, side='right')
    clsname_map = {k: d.category for k, d in enumerate(test_ds.datasets)}
    metrics = ep['metrics']
    img_m = [m for m in metrics if m.startswith('img_')]
    px_m = [m for m in metrics if m.startswith('px_')]
    sp = os.path.join(log_dir, 'final_summary.yaml')
    summary = yaml.safe_load(open(sp)) if os.path.isfile(sp) else {}
    summary = summary or {}
    for v in args.variants:
        img_all = np.concatenate([s[f'{v}_img'] for s in sh])
        px_all = np.concatenate([s[f'{v}_px'] for s in sh]).astype(np.float32)
        by = lambda a: divide_by_class(a, cls)
        img_s, img_g, px_s, px_g = by(img_all), by(label), by(px_all), by(mask)
        res = {}
        for c in sorted(img_s.keys()):
            r = {k2: float(x) for k2, x in calculate_img_metrics(gt_labels=img_g[c], pred_scores=img_s[c], metrics=img_m).items()}
            r.update({k2: float(x) for k2, x in calculate_px_metrics(gt_masks=px_g[c], pred_scores=px_s[c], metrics=px_m).items()})
            r['mad'] = float(np.mean(list(r.values())))
            res[clsname_map[c]] = r
        res['average'] = {m: float(np.mean([res[n][m] for n in res])) for m in next(iter(res.values()))}
        write_csv(res, os.path.join(log_dir, f'final_raw_{v}.csv'))
        summary[f'raw_{v}'] = res['average']
        with open(sp, 'w') as f:
            yaml.dump(summary, f)
        logger.info(f"[FINAL raw {v}] " + ' '.join(f"{k}={x:.4f}" for k, x in res['average'].items()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', required=True, help='training run dir (holds config.yaml and checkpoints/)')
    ap.add_argument('--ckpt', default='ckpt_last.pth')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--variants', nargs='+', default=['full_batch', 'yhat_batch'], choices=list(VARIANTS))
    ap.add_argument('--max_batches', type=int, default=0, help='debug: stop after this many batches')
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    with open(os.path.join(args.log_dir, 'config.yaml')) as f:
        params = yaml.safe_load(f)
    params['logging']['log_dir'] = args.log_dir
    merge(params, args) if args.merge else shard_eval(params, args)

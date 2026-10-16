"""Step-1 diagnostic: is the Alg. 2 reconstruction error dominated by
sampling variance (the chain forgets the input) or by a sampler bug?

For a trained checkpoint, on subsets of normal-train / normal-test /
anomalous-test features, reports (mean over pixels of per-pixel L2 over
channels, in standardised-feature units):
  err_x0      ||x0 - rec||                 (averaged over chain seeds)
  spread      ||rec_a - rec_b|| / sqrt(2)  (seed-to-seed std of the chain)
  err_yhat    ||x0 - y_hat||
  err_mean    ||x0 - mean_seeds(rec)||
for several start steps t_start (999 = full Alg. 2). Sampler sanity: err_x0
must go -> ~0 as t_start -> 0 and grow monotonically with t_start.

  CUDA_VISIBLE_DEVICES=4 python baselines/vpdm_repro/diag_sampling.py \
      --run_dir baselines/vpdm_repro/out/vpdm300_mvtec_s0
"""

import os
import sys
import argparse

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

from src.backbones import BackboneWrapper
from src.backbones.efficientnet import build_efficient
from src.datasets import build_dataset
from baselines.vpdm_repro.train_vpdm import build_vpdm, extract_all


def img_score(S, k=16, sz=224):
    S = F.interpolate(S[:, None], size=(sz, sz), mode='bilinear', align_corners=False)
    return F.avg_pool2d(S, k, stride=1).amax(dim=(1, 2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--n', type=int, default=128, help='images per group')
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--t_starts', default='0,10,100,250,500,999')
    args = ap.parse_args()
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    p = yaml.safe_load(open(os.path.join(args.run_dir, 'config.yaml')))
    train_ds = build_dataset(train=True, **p['data'])
    test_ds = build_dataset(train=False, **p['data'])
    bb, fs = p['model']['backbone'], p['model']['feature_size']
    fe = BackboneWrapper(build_efficient(model_name=bb['model_name'], pretrained=True,
                                         outblocks=bb['outblocks'], outstrides=bb['outstrides']),
                         target_size=(fs, fs)).to(device).eval()
    train = extract_all(fe, train_ds, device)
    test = extract_all(fe, test_ds, device, test=True)
    del fe
    # same normalisation as train_vpdm.py
    fn = p['data'].get('feat_norm', 'none')
    if fn != 'none':
        dims = (0, 2, 3) if fn == 'channel' else (0, 1, 2, 3)
        mu = train['feats'].mean(dims, keepdim=True).reshape(1, -1, 1, 1)
        sd = train['feats'].std(dims, keepdim=True).reshape(1, -1, 1, 1)
        train['feats'] = (train['feats'] - mu) / sd
        test['feats'] = (test['feats'] - mu) / sd

    model = build_vpdm(p).to(device).eval()
    ck = torch.load(os.path.join(args.run_dir, 'checkpoints', 'ckpt_last.pth'), map_location='cpu', weights_only=True)
    model.load_state_dict(ck['model'])
    if p['diffusion'].get('clip_x0') == 'pixel':
        model.x0_range = (train['feats'].amin(0, keepdim=True), train['feats'].amax(0, keepdim=True))

    rng = np.random.default_rng(0)
    lab = test['label']
    groups = {
        'train_normal': train['feats'][torch.as_tensor(rng.choice(len(train['feats']), args.n, replace=False))],
        'test_normal': test['feats'][torch.as_tensor(rng.choice(np.where(lab == 0)[0], args.n, replace=False))],
        'test_anom': test['feats'][torch.as_tensor(rng.choice(np.where(lab == 1)[0], args.n, replace=False))],
    }
    # mixed-class batches: shuffle all groups together, as in evaluation
    X = torch.cat(list(groups.values()))
    gid = np.repeat(np.arange(len(groups)), args.n)
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    X, gid = X[order], gid[order.numpy()]
    bs = p['logging']['eval']['test_batch_size']

    out = {}
    with torch.no_grad():
        yhat = torch.cat([model.vpot(X[s:s + bs], mode='batch')[0] for s in range(0, len(X), bs)])
        e_y = (X - yhat).pow(2).sum(1).sqrt()
        for ts in [int(t) for t in args.t_starts.split(',')]:
            recs = []
            for k in range(args.seeds):
                g = torch.Generator(device=device).manual_seed(1000 + k)
                recs.append(torch.cat([model.reconstruct(X[s:s + bs], ot_mode='batch', generator=g, t_start=ts)
                                       for s in range(0, len(X), bs)]))
            R = torch.stack(recs)                                     # (K, N, C, H, W)
            e_x0 = (X[None] - R).pow(2).sum(2).sqrt().mean(0)         # (N, H, W)
            spread = (R[0] - R[1]).pow(2).sum(1).sqrt() / np.sqrt(2)
            e_mean = (X - R.mean(0)).pow(2).sum(1).sqrt()
            out[ts] = dict(err_x0=e_x0, spread=spread, err_mean=e_mean)
            print(f't_start={ts} done', flush=True)

    tn, ta = gid == 1, gid == 2
    auc = lambda S: roc_auc_score(np.r_[np.zeros(tn.sum()), np.ones(ta.sum())],
                                  np.r_[img_score(S[tn]).cpu().numpy(), img_score(S[ta]).cpu().numpy()])
    lines = [f"# {args.run_dir}  n={args.n}/group seeds={args.seeds}",
             f"{'t_start':>7} {'quantity':>9} " + ' '.join(f'{g:>13}' for g in groups) + '  img-AUROC(test n vs a)']
    lines.append(f"{'-':>7} {'err_yhat':>9} " + ' '.join(f'{e_y[gid == i].mean().item():13.3f}' for i in range(3))
                 + f'  {auc(e_y):.3f}')
    for ts, d in out.items():
        for q, S in d.items():
            lines.append(f'{ts:>7} {q:>9} ' + ' '.join(f'{S[gid == i].mean().item():13.3f}' for i in range(3))
                         + (f'  {auc(S):.3f}' if q != 'spread' else ''))
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(args.run_dir, 'diag_sampling.txt'), 'w') as f:
        f.write(txt + '\n')


if __name__ == '__main__':
    main()

"""Plug FlowAD's one-step reconstruction into a pretrained InvAD model.

InvAD (CVPR'26, ~/projects/InversionAD) trains a class-unconditional eps-prediction DiT on
EfficientNet-b4 features (linear DDPM schedule, T=1000) and scores by DDIM *inversion*
(S steps, x_0 -> x_T; score = deviation of x_T from N(0, I)).

One-step recon (FlowAD, src/vfad/eval_recon.py) on the same weights:
    x_t   = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps        (one fixed noise draw, as FlowAD's noise_base)
    x0hat = (x_t - sqrt(1 - ab_t) eps_theta(x_t, t)) / sqrt(ab_t)   (one NFE)
    map   = sum_c (x0hat - x_0)^2, bilinear to image size; image score = 'diff+sum'.
FlowAD perturbs to t_FM = 0.1 (x_t = 0.1 x_1 + 0.9 eps); the DDPM step with the same SNR
((0.1/0.9)^2) is reported as `t_snr` and always included in the sweep.

Both inversion (InvAD's native scoring: per-pixel ||x_T||_2 map; image score = minmax(|max-min|)
+ minmax(NLL) per class) and one-step recon are scored with FlowAD's metrics
(src/utils/adeval/eval_utils.py): 7 metrics per class + mAD, class average.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

INVAD_ROOT = os.path.expanduser('~/projects/InversionAD')
FLOWAD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_ap', 'px_f1max', 'px_aupro']


def import_invad():
    sys.path.insert(0, INVAD_ROOT)
    sys.path.insert(0, os.path.join(INVAD_ROOT, 'src', 'backbones'))  # InvAD imports `dinov2` from here
    os.chdir(INVAD_ROOT)  # InvAD resolves backbone weights / data roots relative to its repo
    from src.datasets import build_dataset
    from src.denoiser import get_denoiser
    from src.backbones import get_backbone, get_backbone_feature_shape
    return build_dataset, get_denoiser, get_backbone, get_backbone_feature_shape


def import_flowad_metrics():
    """InvAD and FlowAD both ship a top-level `src` package; swap it once InvAD is no longer needed."""
    for k in [k for k in sys.modules if k == 'src' or k.startswith('src.')]:
        del sys.modules[k]
    sys.path[:] = [q for q in sys.path if not q.startswith(INVAD_ROOT) and q not in ('./src/backbones', '')]
    sys.path.insert(0, FLOWAD_ROOT)
    os.chdir(FLOWAD_ROOT)
    from src.utils.adeval.eval_utils import aggregate_px_values, calculate_img_metrics, calculate_px_metrics
    return aggregate_px_values, calculate_img_metrics, calculate_px_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', required=True, help='InvAD results dir with config.yaml + checkpoint')
    p.add_argument('--ckpt', default='model_best.pth')
    p.add_argument('--t_list', default='50,100,200,300,400,500,600,700,800,900',
                   help='DDPM timesteps (0..999) for one-step recon; SNR-matched t is added')
    p.add_argument('--inv_steps', type=int, default=-1, help='DDIM inversion steps (-1: config eval_step)')
    p.add_argument('--no_inversion', action='store_true')
    p.add_argument('--recon_bf16', action='store_true', help='bf16 autocast for the one-step NFE (as InvAD inversion)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', required=True)
    return p.parse_args()


def snr_matched_t(alphas_cumprod, t_fm=0.1):
    snr = (t_fm / (1 - t_fm)) ** 2
    ab = snr / (1 + snr)
    return int(np.argmin(np.abs(alphas_cumprod - ab)))


@torch.no_grad()
def run_inference(args, config, device):
    build_dataset, get_denoiser, get_backbone, get_backbone_feature_shape = import_invad()

    in_sh = get_backbone_feature_shape(model_type=config['backbone']['model_type'])
    inv_steps = args.inv_steps if args.inv_steps > 0 else config['evaluation']['eval_step']
    config['diffusion']['num_sampling_steps'] = str(inv_steps)
    model = get_denoiser(**config['diffusion'], input_shape=in_sh)
    sd = torch.load(os.path.join(args.run_dir, args.ckpt), map_location='cpu', weights_only=True)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    fe = get_backbone(**config['backbone']).to(device).eval()

    diff = model.train_diffusion  # unspaced, T = 1000
    ab = np.asarray(diff.alphas_cumprod)
    t_snr = snr_matched_t(ab)
    t_list = sorted(set(int(t) for t in args.t_list.split(',')) | {t_snr})

    data_cfg = dict(config['data'])
    data_cfg['data_root'] = os.path.join(INVAD_ROOT, data_cfg['data_root'])
    data_cfg['train'] = False
    loaders = []
    for normal_only, anom_only in [(True, False), (False, True)]:
        data_cfg['normal_only'], data_cfg['anom_only'] = normal_only, anom_only
        ds = build_dataset(**data_cfg)
        loaders.append(DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8))

    g = torch.Generator(device=device).manual_seed(args.seed)
    noise_base = torch.randn((1, *in_sh), device=device, generator=g)

    out = {'recon': {t: [] for t in t_list}, 'inv_map': [], 'inv_nll': [], 'cls': [], 'label': [], 'mask': [],
           'names': [], 't_snr': t_snr, 't_list': t_list, 'inv_steps': inv_steps}
    times = {'recon': [], 'inv': []}
    start_t = None
    for loader in loaders:
        for batch in tqdm(loader):
            imgs = batch['samples'].to(device)
            x0, _ = fe(imgs)
            x0 = x0.float()
            B = x0.shape[0]
            out['cls'].append(batch['clslabels'].numpy())
            out['label'].append((batch['labels'].numpy() > 0).astype(np.uint8))
            out['mask'].append((batch['masks'].squeeze(1).numpy() > 0).astype(np.uint8))
            out['names'].extend(batch['clsnames'])

            noise = noise_base.expand(B, -1, -1, -1)
            for t in t_list:
                torch.cuda.synchronize(); s = time.perf_counter()
                tt = torch.full((B,), t, device=device, dtype=torch.long)
                x_t = diff.q_sample(x0, tt, noise=noise)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.recon_bf16):
                    x0_hat = diff.p_mean_variance(model.net, x_t, tt, clip_denoised=False,
                                                  model_kwargs=dict(y=None))['pred_xstart'].float()
                m = ((x0_hat - x0) ** 2).sum(dim=1)  # (B, h, w)
                torch.cuda.synchronize(); times['recon'].append((time.perf_counter() - s) / B)
                out['recon'][t].append(m.cpu().numpy().astype(np.float32))

            if not args.no_inversion:
                # identical to InvAD evaluate_inv: bf16 DDIM inversion over all respaced steps
                if start_t is None or start_t.shape[0] != B:
                    start_t = torch.zeros(B, device=device, dtype=torch.long)
                torch.cuda.synchronize(); s = time.perf_counter()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    xT = model.ddim_reverse_sample(x0, start_t, batch['clslabels'].to(device), eta=0.0)
                xT = xT.float()
                torch.cuda.synchronize(); times['inv'].append((time.perf_counter() - s) / B)
                out['inv_map'].append(torch.sum(xT ** 2, dim=1).sqrt().cpu().numpy())
                out['inv_nll'].append((0.5 * (xT ** 2 + np.log(2 * np.pi))).sum(dim=(1, 2, 3)).cpu().numpy())

    cat = lambda xs: np.concatenate(xs, axis=0)
    for k in ['cls', 'label', 'mask', 'inv_map', 'inv_nll']:
        out[k] = cat(out[k]) if out[k] else None
    out['recon'] = {t: cat(v) for t, v in out['recon'].items()}
    out['ms_per_img'] = {k: 1000 * float(np.median(v)) for k, v in times.items() if v}
    return out


def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def score(out, args):
    aggregate_px_values, calculate_img_metrics, calculate_px_metrics = import_flowad_metrics()
    cls, label, mask = out['cls'], out['label'], out['mask']
    img_sz = mask.shape[-2:]
    names = np.array(out['names'])
    cls_names = {c: names[cls == c][0] for c in np.unique(cls)}

    def up(m):  # (N, h, w) -> (N, H, W), in chunks to bound memory
        return np.concatenate([
            F.interpolate(torch.from_numpy(m[i:i + 256]).unsqueeze(1).cuda(), size=img_sz,
                          mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
            for i in range(0, len(m), 256)])

    def evaluate(px, img):
        rows = []
        for c in np.unique(cls):
            idx = cls == c
            r = {'class': cls_names[c]}
            r.update(calculate_img_metrics(label[idx], img[idx], METRICS))
            r.update(calculate_px_metrics(mask[idx], px[idx], METRICS))
            rows.append(r)
        df = pd.DataFrame(rows)[['class'] + METRICS]
        df['mad'] = df[METRICS].mean(axis=1)
        df.loc[len(df)] = ['mean'] + df[METRICS + ['mad']].mean().tolist()
        return df

    results = {}
    if out['inv_map'] is not None:
        px = up(out['inv_map'])
        # InvAD image score: per-class minmax(|max - min| of ||x_T|| map) + minmax(NLL)
        lo = out['inv_map'].reshape(len(cls), -1)
        ats, img = lo.max(1) - lo.min(1), np.zeros(len(cls))
        for c in np.unique(cls):
            idx = cls == c
            img[idx] = minmax(ats[idx]) + minmax(out['inv_nll'][idx])
        results[f'inversion_S{out["inv_steps"]}'] = evaluate(px, img)
    for t, m in out['recon'].items():
        px = up(m)
        img = aggregate_px_values('diff+sum', px)
        tag = f'onestep_t{t}' + ('_snr' if t == out['t_snr'] else '')
        results[tag] = evaluate(px, img)
    return results


def main():
    args = parse_args()
    args.out_dir, args.run_dir = os.path.abspath(args.out_dir), os.path.abspath(args.run_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(os.path.join(args.run_dir, 'config.yaml')) as f:
        config = yaml.safe_load(f)
    out = run_inference(args, config, 'cuda')
    results = score(out, args)

    summary = []
    for tag, df in results.items():
        df.to_csv(os.path.join(args.out_dir, f'{tag}.csv'), index=False)
        row = df[df['class'] == 'mean'].iloc[0].to_dict()
        row['method'] = tag
        summary.append(row)
    summary = pd.DataFrame(summary)[['method'] + METRICS + ['mad']]
    summary.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)
    meta = {'run_dir': args.run_dir, 'ckpt': args.ckpt, 't_snr': out['t_snr'], 'inv_steps': out['inv_steps'],
            'ms_per_img': out['ms_per_img'], 'n_images': int(len(out['cls']))}
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    pd.set_option('display.width', 200)
    print(json.dumps(meta, indent=2))
    print((summary.set_index('method') * 100).round(1))


if __name__ == '__main__':
    main()

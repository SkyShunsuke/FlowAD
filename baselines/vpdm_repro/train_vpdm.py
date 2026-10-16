"""VPDM reproduction driver (multi-class / unified MVTec-AD), single GPU,
batch 32 (paper Appendix D: one RTX 3090, Adam lr 1e-3, batch 32).

Backbone features are deterministic (no augmentation), so train and test
features are extracted once and kept on the GPU.

Evaluation: FlowAD 7-metric protocol (src/utils/adeval) per class, class mean.
Periodic eval is logged for monitoring only; the reported numbers are from the
final-epoch checkpoint (no test-set model selection).

Run from the repo root:
  CUDA_VISIBLE_DEVICES=0 python baselines/vpdm_repro/train_vpdm.py \
      --config_file baselines/vpdm_repro/configs/vpdm_mvtec.yaml
"""

import os
import sys
import copy
import time
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())

from src.backbones import BackboneWrapper
from src.backbones.efficientnet import build_efficient
from src.datasets import build_dataset
from src.utils.adeval.eval_utils import calculate_img_metrics, calculate_px_metrics, divide_by_class

from baselines.amdm_repro.guided_diffusion.unet import UNetModel
from baselines.vpdm_repro.vpdm import VPOT, VPDM

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('vpdm')

CSV_COLUMNS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_aupro', 'px_ap', 'px_f1max', 'mad']


def build_vpdm(p):
    m, v, d = p['model'], p['vpot'], p['diffusion']
    C = m['in_channels']
    unet = UNetModel(
        image_size=m['feature_size'],
        in_channels=(2 if p.get('x0_cond', {}).get('mode', 'none') == 'none' else 3) * C, model_channels=m['model_channels'],
        out_channels=C, num_res_blocks=m['num_res_blocks'],
        attention_resolutions=tuple(m['attention_ds']), dropout=m.get('dropout', 0.0),
        channel_mult=tuple(m['channel_mult']), num_classes=None, use_checkpoint=False,
        use_fp16=False, num_heads=m.get('num_heads', 4), num_head_channels=m.get('num_head_channels', -1),
        use_scale_shift_norm=m.get('use_scale_shift_norm', True),
        resblock_updown=m.get('resblock_updown', False), use_new_attention_order=False,
    )
    vpot = VPOT(channels=C, feat_size=m['feature_size'], down=v['down'],
                num_prototypes=v['num_prototypes'], eps=v['eps'], sinkhorn_iters=v['sinkhorn_iters'])
    return VPDM(unet, vpot, num_timesteps=d['num_timesteps'], beta_start=d['beta_start'],
                beta_end=d['beta_end'], endpoint=d.get('endpoint', 'yhat'), x0_cond=p.get('x0_cond'))


@torch.no_grad()
def extract_all(fe, dataset, device, bs=64, test=False, out_device=None):
    """out_device: where the features are kept (default: the backbone device)."""
    dl = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=8)
    feats, cls, lab, masks = [], [], [], []
    for b in tqdm(dl, desc='extract'):
        z, _ = fe(b['img'].to(device))
        feats.append(z if out_device is None else z.to(out_device))
        if test:
            cls.append(b['clslabel'])
            lab.append((b['label'] > 0).to(torch.uint8))
            masks.append((b['mask'].squeeze(1) > 0.5).to(torch.uint8))
    out = {'feats': torch.cat(feats)}
    if test:
        out.update(cls=torch.cat(cls).numpy(), label=torch.cat(lab).numpy(),
                   mask=torch.cat(masks).numpy())
    return out



def eval_devices(ep, main_dev):
    """logging.eval.devices: 'all' (every visible GPU), an int count, or a list
    of visible-GPU indices. The training device is always first."""
    spec = ep.get('devices', 1)
    n = torch.cuda.device_count()
    ids = list(range(n)) if spec == 'all' else list(range(min(int(spec), n))) if isinstance(spec, int) else list(spec)
    main = main_dev.index if main_dev.index is not None else torch.cuda.current_device()
    return [torch.device('cuda', main)] + [torch.device('cuda', i) for i in ids if i != main]


def replica(model, dev):
    m = copy.deepcopy(model).to(dev).eval()
    if model.x0_range is not None:   # plain attribute, not a buffer: move explicitly
        m.x0_range = tuple(t.to(dev) for t in model.x0_range)
    return m


@torch.no_grad()
def evaluate(model, test, clsname_map, img_sz, ep, ot_mode='batch', seed=0, save_maps=None):
    """Alg. 2 reconstruction for the whole test set; per-class metrics."""
    model.eval()
    feats = test['feats']
    N = feats.shape[0]
    bs = ep['test_batch_size']
    # fixed shuffled order -> mixed-class test batches (matters only for 'batch' OT)
    order = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).numpy() \
        if ep.get('shuffle_test', True) else np.arange(N)
    px_all = np.zeros((N, img_sz, img_sz), dtype=np.float32)
    img_all = np.zeros((N,), dtype=np.float32)
    k = ep.get('img_pool_kernel', 16)
    batches = [order[s:s + bs] for s in range(0, N, bs)]
    # Split evaluation: the fixed test batches (they matter for 'batch' OT) are
    # distributed round-robin over all visible GPUs, one model replica and one
    # thread per GPU. Each batch draws its chain noise from its own generator
    # seeded by (seed, batch index), so scores do not depend on the GPU count.
    devs = eval_devices(ep, feats.device)
    reps = [model] + [replica(model, d) for d in devs[1:]]
    pbar = tqdm(total=len(batches), desc=f'eval[{ot_mode}] x{len(devs)}gpu')

    def work(r):
        m, dev = reps[r], devs[r]
        # no_grad is thread-local: worker threads do not inherit the caller's
        with torch.cuda.device(dev), torch.no_grad():
            for b in range(r, len(batches), len(devs)):
                idx = batches[b]
                x = feats[idx].to(dev, non_blocking=True)
                if ep.get('score_mode') == 'yhat':   # diagnostic: condition-only residual
                    y_hat, _, _ = m.vpot(x, mode=ot_mode)
                    S = (x - y_hat).pow(2).sum(1).sqrt()
                else:
                    g = torch.Generator(device=dev).manual_seed(seed * 1_000_003 + b)
                    S = m.score(x, ot_mode=ot_mode, generator=g, t_start=ep.get('t_start'),
                                one_step=ep.get('one_step', False))  # (b, 32, 32)
                S = F.interpolate(S[:, None], size=(img_sz, img_sz), mode='bilinear', align_corners=False)
                img_all[idx] = F.avg_pool2d(S, k, stride=1).amax(dim=(1, 2, 3)).cpu().numpy()
                px_all[idx] = S[:, 0].cpu().numpy()
                pbar.update(1)

    with torch.no_grad():
        if len(devs) == 1:
            work(0)
        else:
            with ThreadPoolExecutor(len(devs)) as ex:
                list(ex.map(work, range(len(devs))))   # list() re-raises worker exceptions
    pbar.close()
    del reps
    if save_maps:
        np.savez_compressed(save_maps, px=px_all.astype(np.float16), img=img_all)

    metrics = ep['metrics']
    img_m = [m for m in metrics if m.startswith('img_')]
    px_m = [m for m in metrics if m.startswith('px_')]
    res = {}
    by = lambda a: divide_by_class(a, test['cls'])
    img_s, img_g, px_s, px_g = by(img_all), by(test['label']), by(px_all), by(test['mask'])
    for c in sorted(img_s.keys()):
        name = clsname_map[c]
        r = {k2: float(v) for k2, v in calculate_img_metrics(gt_labels=img_g[c], pred_scores=img_s[c], metrics=img_m).items()}
        r.update({k2: float(v) for k2, v in calculate_px_metrics(gt_masks=px_g[c], pred_scores=px_s[c], metrics=px_m).items()})
        r['mad'] = float(np.mean(list(r.values())))
        res[name] = r
    res['average'] = {m: float(np.mean([res[n][m] for n in res])) for m in next(iter(res.values()))}
    model.train()
    return res


def write_csv(res, path):
    with open(path, 'w') as f:
        f.write('category,' + ','.join(CSV_COLUMNS) + '\n')
        for name in [n for n in res if n != 'average'] + ['average']:
            f.write(name + ',' + ','.join(f"{res[name].get(c, float('nan')):.6f}" for c in CSV_COLUMNS) + '\n')


def main(p, args):
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log_dir = p['logging']['log_dir']
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(p, f)
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)

    seed = p['opt']['global_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -- data + frozen backbone
    train_ds = build_dataset(train=True, **p['data'])
    test_ds = build_dataset(train=False, **p['data'])
    clsname_map = test_ds.datasets[0].labels_to_names
    bb = p['model']['backbone']
    fs = p['model']['feature_size']
    fe = BackboneWrapper(build_efficient(model_name=bb['model_name'], pretrained=True,
                                         outblocks=bb['outblocks'], outstrides=bb['outstrides']),
                         target_size=(fs, fs)).to(device).eval()
    train = extract_all(fe, train_ds, device)
    # data.test_feats_device: cpu keeps large test sets (e.g. Real-IAD) in host memory;
    # evaluate() moves each batch to its GPU
    test = extract_all(fe, test_ds, device, test=True, out_device=p['data'].get('test_feats_device'))
    del fe
    torch.cuda.empty_cache()
    # feature standardisation (not in paper). DDPM's schedule assumes ~unit
    # variance data; raw EfficientNet features have std ~5.7 (max ~47), and
    # eps-prediction on them plateaus / collapses (see out/_collapsed_rawfeat_*).
    # 'global' is a single scalar shift/scale, so the anomaly map is the
    # paper's ||x0 - x0_rec||_2 up to a constant factor.
    fn = p['data'].get('feat_norm', 'none')
    if fn != 'none':
        dims = (0, 2, 3) if fn == 'channel' else (0, 1, 2, 3)
        mu = train['feats'].mean(dims, keepdim=True)
        sd = train['feats'].std(dims, keepdim=True)
        mu, sd = mu.reshape(1, -1, 1, 1), sd.reshape(1, -1, 1, 1)
        train['feats'] = (train['feats'] - mu) / sd
        test['feats'].sub_(mu.to(test['feats'].device)).div_(sd.to(test['feats'].device))
        logger.info(f"feat_norm={fn}: mean {mu.mean().item():.3f}, std {sd.mean().item():.3f}")
    logger.info(f"train feats {tuple(train['feats'].shape)}, test feats {tuple(test['feats'].shape)}")

    model = build_vpdm(p).to(device)
    if p['diffusion'].get('clip_x0') == 'pixel':   # per-(c,h,w) train min/max
        model.x0_range = (train['feats'].amin(0, keepdim=True), train['feats'].amax(0, keepdim=True))
    logger.info(f"U-Net params {sum(q.numel() for q in model.model.parameters())/1e6:.1f}M, "
                f"prototypes {tuple(model.vpot.prototypes.shape)}")
    o = p['opt']
    proto_lr = o.get('proto_lr') or o['lr']
    optimizer = torch.optim.Adam([
        {'params': model.model.parameters(), 'lr': o['lr'], 'base_lr': o['lr']},
        {'params': model.vpot.parameters(), 'lr': proto_lr, 'base_lr': proto_lr},
    ])
    ema = copy.deepcopy(model).eval()
    ema.x0_range = model.x0_range
    for q in ema.parameters():
        q.requires_grad_(False)
    ema_rate = o.get('ema_rate', 0.9999)

    ep = p['logging']['eval']
    img_sz = p['data']['img_size']
    hist = os.path.join(log_dir, 'history.csv')
    last_ckpt = os.path.join(ckpt_dir, 'ckpt_last.pth')
    start_epoch = 0
    if os.path.isfile(last_ckpt):
        ck = torch.load(last_ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model']); ema.load_state_dict(ck['ema'])
        optimizer.load_state_dict(ck['opt']); start_epoch = ck['epoch']
        logger.info(f"resumed at epoch {start_epoch}")
    elif not args.eval_only:
        with open(hist, 'w') as f:
            f.write('epoch,weights,ot_mode,' + ','.join(CSV_COLUMNS) + '\n')

    def save(path, epoch):
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'ema': ema.state_dict(),
                    'opt': optimizer.state_dict()}, path)

    X = train['feats']
    N = X.shape[0]
    bs = p['data']['train_batch_size']
    ipe = N // bs                                     # drop_last
    epochs = o['epochs']
    model.train()
    t0 = time.time()
    snapshot = None
    lr_scale = 1.0   # cumulative collapse-guard LR factor (the snapshot's optimizer state carries its own lr)
    init_base_lr = [g['base_lr'] for g in optimizer.param_groups]
    for epoch in range(start_epoch + 1, epochs + 1) if not args.eval_only else []:
        perm = torch.randperm(N, device=device)
        tot = el = ot = 0.0
        for i in range(ipe):
            x0 = X[perm[i * bs:(i + 1) * bs]]
            loss, l_eps, l_ot = model(x0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if o.get('warmup_steps', 0) > 0:  # linear LR warmup (not in paper)
                step = (epoch - 1) * ipe + i + 1
                for gr in optimizer.param_groups:
                    gr['lr'] = gr['base_lr'] * min(1.0, step / o['warmup_steps'])
            if o.get('clip_grad', 0) > 0:
                # denoiser only: the OT-loss gradient on the prototypes is ~600x
                # larger and would otherwise dominate a joint norm
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), o['clip_grad'])
            optimizer.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(ema_rate).add_(pm, alpha=1 - ema_rate)
            tot += loss.item(); el += l_eps.item(); ot += l_ot.item()
        if not np.isfinite(tot):
            raise RuntimeError(f'non-finite loss at epoch {epoch}')
        # collapse guard (not in paper): the denoiser occasionally falls into a
        # dead constant-output state (eps-MSE == 1.0, never recovers). Roll back
        # to the last healthy snapshot and halve the LRs; every event is logged.
        guard = o.get('collapse_guard', {})
        if guard:
            forced = epoch == guard.get('debug_force_epoch', -1)   # test hook
            if forced or (epoch >= guard.get('min_epoch', 3) and el / ipe > guard.get('threshold', 0.95)):
                guard['debug_force_epoch'] = -1
                if snapshot is None:
                    raise RuntimeError(f'collapse at epoch {epoch} with no snapshot')
                model.load_state_dict(snapshot['model']); ema.load_state_dict(snapshot['ema'])
                optimizer.load_state_dict(snapshot['opt'])
                # halve cumulatively: consecutive rollbacks to the same snapshot
                # must not all restore the snapshot's lr and halve it only once
                lr_scale *= 0.5
                step = (epoch - 1) * ipe
                wu = min(1.0, step / o['warmup_steps']) if o.get('warmup_steps', 0) > 0 else 1.0
                for gr, b0 in zip(optimizer.param_groups, init_base_lr):
                    gr['base_lr'] = b0 * lr_scale; gr['lr'] = gr['base_lr'] * wu
                logger.info(f"[COLLAPSE-GUARD] epoch {epoch}: eps {el/ipe:.4f} -> rolled back to epoch "
                            f"{snapshot['epoch']}, lr now {[g['lr'] for g in optimizer.param_groups]}")
                continue
            if epoch % guard.get('snapshot_every', 5) == 0 and el / ipe < guard.get('threshold', 0.95):
                snapshot = {'epoch': epoch,
                            'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            'ema': {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()},
                            'opt': copy.deepcopy(optimizer.state_dict())}
        if epoch % p['logging']['log_epoch_freq'] == 0 or epoch == 1:
            with torch.no_grad():
                _, _, W = model.vpot(X[perm[:bs]], mode='batch')
                used = W.argmax(1).unique().numel()
            logger.info(f"epoch {epoch}/{epochs} loss {tot/ipe:.4f} eps {el/ipe:.4f} ot {ot/ipe:.3f} "
                        f"| argmax-prototypes used in a batch: {used} | {time.time()-t0:.0f}s")
        if epoch % ep['eval_epoch_freq'] == 0 and epoch != epochs:
            save(last_ckpt, epoch)
            for tag, m in (('ema', ema),):
                r = evaluate(m, test, clsname_map, img_sz, ep, ot_mode='batch', seed=seed)['average']
                logger.info(f"[eval ep{epoch} {tag} batch] " + ' '.join(f"{k}={v:.4f}" for k, v in r.items()))
                with open(hist, 'a') as f:
                    f.write(f"{epoch},{tag},batch," + ','.join(f"{r[c]:.6f}" for c in CSV_COLUMNS) + '\n')

    if not args.eval_only:
        save(os.path.join(ckpt_dir, f'ckpt_epoch_{epochs}.pth'), epochs)
        save(last_ckpt, epochs)

    # -- final evaluation of the final-epoch weights (raw and EMA).
    # Variants: 'full' = Alg. 2 (paper); 'tN' = start from q(x_N | x0, y_hat)
    # (NOT in paper); 'oN' = one x0 prediction from q(x_N | x0, y_hat) (FlowAD plug-in); 'yhat' = ||x0 - y_hat|| (condition only, diagnostic).
    summary = {}
    for tag, m in (('raw', model), ('ema', ema)):
        variants = ep.get('final_variants', ['full_batch', 'full_semi'])
        if tag == 'ema':   # EMA is secondary: optionally a subset (each full chain ~15 min)
            variants = ep.get('ema_final_variants', variants)
        for var in variants:
            kind, mode = var.split('_')
            e = dict(ep)
            if kind == 'yhat':
                e['score_mode'] = 'yhat'
            elif kind.startswith('t'):
                e['t_start'] = int(kind[1:])
            elif kind.startswith('o'):   # FlowAD one-step plug-in: single x0 prediction at t = N
                e['t_start'], e['one_step'] = int(kind[1:]), True
            res = evaluate(m, test, clsname_map, img_sz, e, ot_mode=mode, seed=seed,
                           save_maps=os.path.join(log_dir, f'maps_{tag}_{var}.npz') if ep.get('save_maps') else None)
            write_csv(res, os.path.join(log_dir, f'final_{tag}_{var}.csv'))
            summary[f'{tag}_{var}'] = res['average']
            logger.info(f"[FINAL {tag} {var}] " + ' '.join(f"{k}={v:.4f}" for k, v in res['average'].items()))
            with open(os.path.join(log_dir, 'final_summary.yaml'), 'w') as f:
                yaml.dump(summary, f)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_file', required=True)
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--override', nargs='*', default=[],
                    help='dotted key=value overrides, e.g. opt.global_seed=1')
    args = ap.parse_args()
    with open(args.config_file) as f:
        params = yaml.safe_load(f)
    for kv in args.override:
        k, v = kv.split('=', 1)
        d = params
        *path, last = k.split('.')
        for key in path:
            d = d[key]
        d[last] = yaml.safe_load(v)
    main(params, args)

"""DiffusionAD (Zhang et al., arXiv 2303.08730), multi-class, DDP.

Port of upstream train.py / eval.py (github.com/HuiZhang0812/DiffusionAD @ 8a64c03,
vendored in orig/). Kept from upstream: U-Net and segmentation sub-network,
linear T=1000 schedule, two-scale noise (normal t~U[0,300), noisier t~U[300,1000)),
noise loss on normal samples only, norm-guided one-step x0 fed (with gradient) to
the segmentation net, loss = noise + 5*focal + smoothL1, Adam (1e-4 / 1e-5, wd 0),
per-iteration CosineAnnealingLR(T_max=10) on the seg optimiser, batch 16, 256x256
inputs in [0,1], eval with t_normal=200 / t_noisier=400, anomaly map = seg output,
image score = mean of its top-50 pixels.

Changes (see README.md):
  * multi-class: one model per benchmark trained on the union of its classes;
  * DDP: batch 16 per GPU (so BatchNorm in the seg net sees upstream's batch),
    global batch 16 x world size, same learning rates;
  * epoch budget 300 for all benchmarks, set in configs/ (upstream: 3000 per class);
  * reported weights = final epoch (upstream picks `best` on the test set every
    50 epochs; that selection is also tracked and reported separately);
  * FlowAD 7-metric evaluation on FlowAD's splits / GT masks.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from baselines.diffusionad_repro.data import DiffusionADTrain, worker_init_fn  # noqa: E402
from baselines.diffusionad_repro.orig.DDPM import GaussianDiffusionModel, get_beta_schedule  # noqa: E402
from baselines.diffusionad_repro.orig.Recon_subnetwork import UNetModel  # noqa: E402
from baselines.diffusionad_repro.orig.Seg_subnetwork import SegmentationSubNetwork  # noqa: E402
from src.utils.adeval.eval_utils import calculate_img_metrics, calculate_px_metrics  # noqa: E402

CACHE = os.environ.get('DIFFAD_CACHE', os.path.join(ROOT, 'data', 'diffusionad_aux', 'cache'))
METRICS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_aupro', 'px_ap', 'px_f1max']
CSV_COLUMNS = METRICS + ['mad']


class BinaryFocalLoss(nn.Module):   # upstream train.py
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiffusionAD(nn.Module):
    """Both sub-networks in one module so a single DDP wrapper syncs them."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.unet_model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args.get('channel_mults', ''),
                                    dropout=args['dropout'], n_heads=args['num_heads'],
                                    n_head_channels=args['num_head_channels'], in_channels=args['channels'])
        self.seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1)
        betas = get_beta_schedule(args['T'], args['beta_schedule'])
        # upstream passes args['loss_weight'] / args['loss-type'], both unset -> '' (plain eps-MSE)
        self.ddpm_sample = GaussianDiffusionModel(args['img_size'], betas, loss_weight='', loss_type='',
                                                  noise=args['noise_fn'], img_channels=args['channels'])

    def forward(self, aug_image, anomaly_label):
        noise_loss, pred_x0, normal_t, x_normal_t, x_noiser_t = \
            self.ddpm_sample.norm_guided_one_step_denoising(self.unet_model, aug_image, anomaly_label, self.args)
        pred_mask = self.seg_model(torch.cat((aug_image, pred_x0), dim=1))
        return noise_loss, pred_mask

    @torch.no_grad()
    def infer(self, image):
        n = image.shape[0]
        normal_t = torch.tensor([self.args['eval_normal_t']], device=image.device).repeat(n)
        noiser_t = torch.tensor([self.args['eval_noisier_t']], device=image.device).repeat(n)
        out = self.ddpm_sample.norm_guided_one_step_denoising_eval(self.unet_model, image, normal_t, noiser_t, self.args)
        pred_mask = self.seg_model(torch.cat((image, out[1]), dim=1))   # out[1] = pred_x_0_condition
        return pred_mask


def is_main():
    return dist.get_rank() == 0


def log(msg, path=None):
    if is_main():
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        if path:
            with open(path, 'a') as f:
                f.write(line + '\n')


@torch.no_grad()
def evaluate(model, cache_dir, tmp_dir, tag, bs=32, seed=0):
    """Sharded inference over all ranks, then per-class metrics sharded over ranks."""
    model.eval()
    rank, ws = dist.get_rank(), dist.get_world_size()
    dev = torch.device('cuda', torch.cuda.current_device())
    meta = json.load(open(os.path.join(cache_dir, 'meta.json')))
    imgs = np.load(os.path.join(cache_dir, 'test_img.npy'), mmap_mode='r')
    N = len(imgs)
    px_path = os.path.join(tmp_dir, f'px_{tag}.npy')
    sc_path = os.path.join(tmp_dir, f'img_{tag}.npy')
    if rank == 0:
        np.lib.format.open_memmap(px_path, 'w+', np.float32, (N, 256, 256))
        np.lib.format.open_memmap(sc_path, 'w+', np.float32, (N,))
    dist.barrier()
    px = np.load(px_path, mmap_mode='r+')
    sc = np.load(sc_path, mmap_mode='r+')
    # upstream draws the eval noise from the global RNG; fix it per rank for repeatability
    torch.manual_seed(seed * 1000 + rank)
    starts = list(range(rank * bs, N, ws * bs))
    for s in starts:
        x = torch.from_numpy(np.ascontiguousarray(imgs[s:s + bs])).to(dev).permute(0, 3, 1, 2).float() / 255.0
        pred_mask = model.infer(x)
        # upstream eval: image score = mean(top-50 of flatten(out_mask[0]))
        top = torch.topk(torch.flatten(pred_mask, start_dim=1), 50, dim=1, largest=True)[0]
        sc[s:s + len(x)] = top.mean(1).cpu().numpy()
        px[s:s + len(x)] = pred_mask[:, 0].float().cpu().numpy()
    px.flush(); sc.flush()
    dist.barrier()
    # metrics: classes round-robin over ranks
    cls = np.load(os.path.join(cache_dir, 'test_cls.npy'))
    labels = np.load(os.path.join(cache_dir, 'test_label.npy'))
    masks = np.load(os.path.join(cache_dir, 'test_mask.npy'), mmap_mode='r')
    names = meta['classes']
    local = {}
    for c in range(rank, len(names), ws):
        sel = np.where(cls == c)[0]
        r = calculate_img_metrics(gt_labels=labels[sel].astype(np.int64), pred_scores=np.asarray(sc[sel]),
                                  metrics=[m for m in METRICS if m.startswith('img_')])
        r.update(calculate_px_metrics(gt_masks=np.asarray(masks[sel]), pred_scores=np.asarray(px[sel]),
                                      metrics=[m for m in METRICS if m.startswith('px_')]))
        r = {k: float(v) for k, v in r.items()}
        r['mad'] = float(np.mean([r[m] for m in METRICS]))
        local[names[c]] = r
        torch.cuda.empty_cache()
    gathered = [None] * ws
    dist.all_gather_object(gathered, local)
    res = {}
    for g in gathered:
        res.update(g)
    res = {n: res[n] for n in names}
    res['average'] = {m: float(np.mean([res[n][m] for n in names])) for m in CSV_COLUMNS}
    del px, sc
    dist.barrier()
    if rank == 0:
        os.remove(px_path); os.remove(sc_path)
    model.train()
    return res


def write_csv(res, path):
    with open(path, 'w') as f:
        f.write('category,' + ','.join(CSV_COLUMNS) + '\n')
        for name in res:
            f.write(name + ',' + ','.join(f'{res[name][c]:.6f}' for c in CSV_COLUMNS) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True, help='logs / CSVs')
    ap.add_argument('--ckpt_dir', default=None, help='checkpoints (default <out>/checkpoints)')
    ap.add_argument('--eval_only', default=None, help='checkpoint to evaluate')
    ap.add_argument('--max_steps', type=int, default=0, help='debug: stop after N steps')
    cli = ap.parse_args()

    dist.init_process_group('nccl')
    rank, ws = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dev = torch.device('cuda', local_rank)

    args = json.load(open(os.path.join(ROOT, 'baselines/diffusionad_repro/orig/args1.json')))
    args.update(json.load(open(cli.config)))
    if args.get('tf32', True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out = cli.out
    ck_dir = cli.ckpt_dir or os.path.join(out, 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    logf = os.path.join(out, 'train.log')
    tmp_dir = os.environ.get('PBS_LOCALDIR') or out
    if rank == 0:
        json.dump(args, open(os.path.join(out, 'config.json'), 'w'), indent=1)
    cache_dir = os.path.join(CACHE, args['dataset'])

    seed = args.get('seed', 42)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    model = DiffusionAD(args).to(dev)

    if cli.eval_only:
        ck = torch.load(cli.eval_only, map_location='cpu', weights_only=False)
        model.unet_model.load_state_dict(ck['unet_model_state_dict'])
        model.seg_model.load_state_dict(ck['seg_model_state_dict'])
        res = evaluate(model, cache_dir, tmp_dir, 'evalonly', bs=args.get('eval_bs', 32), seed=args.get('eval_seed', 0))
        if rank == 0:
            write_csv(res, os.path.join(out, f"eval_{os.path.basename(cli.eval_only).replace('.pt', '')}.csv"))
            log(f"eval {cli.eval_only}: {json.dumps(res['average'])}", logf)
        dist.destroy_process_group()
        return

    ddp = DDP(model, device_ids=[local_rank])
    optimizer_ddpm = optim.Adam(model.unet_model.parameters(), lr=args['diffusion_lr'], weight_decay=args['weight_decay'])
    optimizer_seg = optim.Adam(model.seg_model.parameters(), lr=args['seg_lr'], weight_decay=args['weight_decay'])
    scheduler_seg = optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=10, eta_min=0, last_epoch=-1)
    loss_focal = BinaryFocalLoss().to(dev)
    loss_smL1 = nn.SmoothL1Loss().to(dev)

    ds = DiffusionADTrain(cache_dir, img_size=args['img_size'])
    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True, seed=seed, drop_last=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args['Batch_Size'], sampler=sampler,
                                         num_workers=args.get('num_workers', 16), pin_memory=True, drop_last=True,
                                         worker_init_fn=worker_init_fn, persistent_workers=True,
                                         generator=torch.Generator().manual_seed(seed + rank))

    start_epoch, best = 0, {'img': 0.0, 'px': 0.0, 'epoch': -1}
    last_ck = os.path.join(ck_dir, 'params-last-running.pt')
    if os.path.isfile(last_ck):
        ck = torch.load(last_ck, map_location='cpu', weights_only=False)
        model.unet_model.load_state_dict(ck['unet_model_state_dict'])
        model.seg_model.load_state_dict(ck['seg_model_state_dict'])
        optimizer_ddpm.load_state_dict(ck['opt_ddpm']); optimizer_seg.load_state_dict(ck['opt_seg'])
        scheduler_seg.load_state_dict(ck['sched_seg'])
        start_epoch, best = ck['n_epoch'], ck['best']
        torch.set_rng_state(ck['rng'][rank]['torch']); torch.cuda.set_rng_state(ck['rng'][rank]['cuda'])
        log(f'resumed from epoch {start_epoch}', logf)

    def save(path, epoch):
        rng = [None] * ws
        dist.all_gather_object(rng, {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()})
        if rank == 0:
            torch.save({'n_epoch': epoch, 'unet_model_state_dict': model.unet_model.state_dict(),
                        'seg_model_state_dict': model.seg_model.state_dict(), 'args': args,
                        'opt_ddpm': optimizer_ddpm.state_dict(), 'opt_seg': optimizer_seg.state_dict(),
                        'sched_seg': scheduler_seg.state_dict(), 'best': best, 'rng': rng}, path + '.tmp')
            os.replace(path + '.tmp', path)
        dist.barrier()

    E = args['EPOCHS']
    steps_per_epoch = len(loader)
    log(f"dataset {args['dataset']}: {len(ds)} train images, world {ws} x batch {args['Batch_Size']}, "
        f"{steps_per_epoch} steps/epoch, epochs {start_epoch}->{E}", logf)
    eval_every = args.get('eval_every', 0)
    ckpt_every = args.get('ckpt_every', 10)
    t0, step = time.time(), 0
    model.train()
    for epoch in range(start_epoch, E):
        sampler.set_epoch(epoch)
        tot = torch.zeros(4, device=dev)
        for sample in loader:
            aug_image = sample['augmented_image'].to(dev, non_blocking=True)
            anomaly_mask = sample['anomaly_mask'].to(dev, non_blocking=True)
            anomaly_label = sample['has_anomaly'].to(dev, non_blocking=True).squeeze()

            noise_loss, pred_mask = ddp(aug_image, anomaly_label)
            focal_loss = loss_focal(pred_mask, anomaly_mask)
            smL1_loss = loss_smL1(pred_mask, anomaly_mask)
            loss = noise_loss + 5*focal_loss + smL1_loss

            optimizer_ddpm.zero_grad()
            optimizer_seg.zero_grad()
            loss.backward()
            optimizer_ddpm.step()
            optimizer_seg.step()
            scheduler_seg.step()
            tot += torch.stack([loss.detach(), noise_loss.detach(), 5*focal_loss.detach(), smL1_loss.detach()])
            step += 1
            if cli.max_steps and step >= cli.max_steps:
                break
        dist.all_reduce(tot)
        tot /= ws * max(steps_per_epoch, 1)
        el = time.time() - t0
        log(f'epoch {epoch+1}/{E} loss {tot[0]:.4f} noise {tot[1]:.4f} focal5 {tot[2]:.4f} smL1 {tot[3]:.5f} '
            f'| {el/60:.1f} min, {step/el:.2f} it/s', logf)
        if cli.max_steps and step >= cli.max_steps:
            break
        # upstream: eval every 50 epochs, keep `best` by (img+px AUROC, img AUROC)
        if eval_every and (epoch + 1) % eval_every == 0 and epoch + 1 < E:
            res = evaluate(model, cache_dir, tmp_dir, f'ep{epoch+1}', bs=args.get('eval_bs', 32))
            a = res['average']
            img, px = round(a['img_auroc'], 3)*100, round(a['px_auroc'], 3)*100   # upstream rounding
            log(f"eval ep{epoch+1}: " + ' '.join(f'{k} {v*100:.2f}' for k, v in a.items()), logf)
            if rank == 0:
                write_csv(res, os.path.join(out, f'eval_ep{epoch+1}.csv'))
            if img + px >= best['img'] + best['px'] and img >= best['img']:
                best = {'img': img, 'px': px, 'epoch': epoch + 1}
                save(os.path.join(ck_dir, 'params-best.pt'), epoch + 1)
                log(f'new best (upstream criterion) at epoch {epoch+1}', logf)
        if (epoch + 1) % ckpt_every == 0 or epoch + 1 == E:
            save(last_ck, epoch + 1)

    if not cli.max_steps:
        save(os.path.join(ck_dir, 'params-last.pt'), E)
    res = evaluate(model, cache_dir, tmp_dir, 'final', bs=args.get('eval_bs', 32), seed=args.get('eval_seed', 0))
    if rank == 0:
        write_csv(res, os.path.join(out, 'final_last.csv'))
    log('final (last epoch): ' + ' '.join(f'{k} {v*100:.2f}' for k, v in res['average'].items()), logf)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

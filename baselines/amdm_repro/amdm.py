"""AMDM (Adaptive Masked Diffusion Model) — the multi-step teacher of OmiAD
(Feng et al., "OmiAD: One-Step Adaptive Masked Diffusion Model for Multi-class
Anomaly Detection via Adversarial Distillation", ICML 2025, PMLR v267) —
reproduced WITHOUT the adversarial-distillation (ASD) stage, per user request.

The official repo (github.com/luolundashu/OmiAD) contains no code (README only),
so this is a faithful reimplementation from the paper:

Training (Sec. 3.2, Appendix F):
  - x0 = EfficientNet-b4 stages 1-4 feature maps, each resized to 32x32 and
    concatenated -> (272, 32, 32); input images 224x224 (ImageNet norm).
  - DDPM, T=1000, linear betas 1e-4 -> 0.02 (Ho et al. 2020).
  - Adaptive masking: p(t) = pmin + (pmax - pmin) * (t/T)^k with
    pmin=0.1, pmax=0.4, k=2; binary mask M(t) zeroes feature entries with
    probability p(t) (Eq. 6-8); we mask whole spatial locations (the mask is
    shared across channels), matching the paper's "broader context /
    reduce pixel-level shortcut reliance" motivation.
  - x_t^m = sqrt(a-bar_t) * (x0 * M) + sqrt(1 - a-bar_t) * eps  (Eq. 9)
  - The network predicts x0-hat directly (guided-diffusion's predict_xstart
    parameterization; equivalent to Eq. 10's eps-form via the same identity)
    and is trained with plain MSE against the UNMASKED x0 (Eq. 11, gamma(t)=1),
    i.e. the model denoises AND inpaints. [An eps-parameterized variant of the
    same loss was tried first and does not train: the 1/sqrt(a-bar) factor at
    high t makes those timesteps dominate all gradients, and both image- and
    pixel-level AUROC stay at chance for 100+ epochs. Direct x0 prediction
    keeps the Eq. 11 target bounded at every t and trains stably.]
  - Adam, lr 1e-3, global batch 32 (Appendix F). Network: guided-diffusion
    U-Net (paper's default architecture, Table 14), unconditional (UniAD-style
    unified multi-class model, no class labels).

Inference (Appendix I.1 + user instruction):
  - Perturb the test feature to t_init = 960 (paper-recommended), then a
    40-step reconstruction (user-prescribed, OmiAD protocol) back to x0-hat
    with deterministic DDIM over 40 evenly spaced timesteps in [0, t_init).
    No masking at inference.
  - Anomaly map = per-location channel-summed squared error, bilinearly
    upsampled to image size (identical to src/vfad/eval_recon.py);
    image score = max over the map (UniAD-lineage original scoring,
    via FlowAD's aggregate_px_values('max')).

Evaluation protocol: FlowAD's 7 metrics + mAD, mirroring src/vfad/eval_recon.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.distributed import concat_all_gather
from src.utils.adeval.eval_utils import (
    calculate_img_metrics,
    calculate_px_metrics,
    divide_by_class,
    extract_features,
    aggregate_px_values,
)

import logging
logger = logging.getLogger(__name__)


class AMDM(nn.Module):
    """DDPM with adaptive masking (AMDM) on frozen backbone features."""

    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 p_min=0.1, p_max=0.4, k=2.0):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.p_min = p_min
        self.p_max = p_max
        self.k = k

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer('sqrt_ab', alphas_cumprod.sqrt().float())
        self.register_buffer('sqrt_1mab', (1.0 - alphas_cumprod).sqrt().float())

    def mask_prob(self, t):
        # p(t) = pmin + (pmax - pmin) * (t/T)^k   (Eq. 7; t is the 1-based step)
        frac = (t.float() + 1.0) / self.num_timesteps
        return self.p_min + (self.p_max - self.p_min) * frac.pow(self.k)

    def forward(self, x0, y=None):
        """AMDM training loss (Eq. 6-11); y is ignored (unconditional model)."""
        B = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device)
        p = self.mask_prob(t).view(B, 1, 1, 1)

        # spatial binary mask, shared across channels: zeroed with prob p(t)
        keep = (torch.rand(B, 1, *x0.shape[2:], device=x0.device) >= p).float()
        xm = x0 * keep

        eps = torch.randn_like(x0)
        sab = self.sqrt_ab[t].view(B, 1, 1, 1)
        s1mab = self.sqrt_1mab[t].view(B, 1, 1, 1)
        xt = sab * xm + s1mab * eps                                   # Eq. 9

        x0_hat = self.model(xt, t.float())      # predict_xstart parameterization
        return F.mse_loss(x0_hat, x0)                                  # Eq. 11


    @torch.no_grad()
    def reconstruct(self, x0, t_init=960, steps=40, noise=None):
        """Perturb clean features to t_init, then DDIM-reconstruct in `steps` steps."""
        B = x0.shape[0]
        dev = x0.device
        ts = np.linspace(0, t_init - 1, steps).round().astype(np.int64)[::-1].copy()
        if steps == 1:  # one-step recon (FlowAD plug-in): single x0 prediction at t_init - 1, not at t=0
            ts = np.array([t_init - 1], dtype=np.int64)

        if noise is None:
            noise = torch.randn_like(x0)
        t0 = int(ts[0])
        x = self.sqrt_ab[t0] * x0 + self.sqrt_1mab[t0] * noise

        for i, t in enumerate(ts):
            t_b = torch.full((B,), float(t), device=dev)
            x0_hat = self.model(x, t_b)
            if i + 1 < len(ts):
                eps_hat = (x - self.sqrt_ab[t] * x0_hat) / self.sqrt_1mab[t]
                t_prev = int(ts[i + 1])
                x = self.sqrt_ab[t_prev] * x0_hat + self.sqrt_1mab[t_prev] * eps_hat
            else:
                x = x0_hat
        return x

    @torch.no_grad()
    def score(self, x0, t_init=960, steps=40, noise=None):
        recon = self.reconstruct(x0, t_init=t_init, steps=steps, noise=noise)
        px_map = (recon - x0).pow(2).sum(dim=1)      # (B, h, w), as in eval_recon.py
        return px_map


@torch.no_grad()
def evaluate_amdm(
    vf,
    fe,
    norm_fn,
    dataloader,
    device,
    img_sz,
    verbose=True,
    distributed=False,
    eval_params=None,
    **kwargs,
):
    """FlowAD 7-metric protocol (mirrors src/vfad/eval_recon.py) with AMDM scoring."""
    was_train = vf.training
    vf.eval()

    eval_metrics = eval_params.get('metrics', ['img_auroc', 'px_auroc']) if eval_params else ['img_auroc', 'px_auroc']
    img_score_agg = eval_params.get('img_score_agg', 'max') if eval_params else 'max'
    t_init = eval_params.get('t_init', 960) if eval_params else 960
    recon_steps = eval_params.get('recon_steps', 40) if eval_params else 40

    # RealIAD-scale options (the (N, H, W) accumulators cost ~29GB for 114k
    # images at 224^2):
    #  - accumulate_rank0_only: ranks != 0 run their inference shard + gathers
    #    but skip accumulation and metric computation (arrays once, on rank 0)
    #  - distributed_metrics: every rank accumulates ONLY the rows of its class
    #    shard (classes[rank::world]) and computes those classes' metrics on
    #    its own GPU; per-class results are gathered and merged on rank 0.
    #    Takes precedence over accumulate_rank0_only when world > 1.
    acc_rank0_only = bool(eval_params.get('accumulate_rank0_only', False)) if eval_params else False
    dist_metrics = bool(eval_params.get('distributed_metrics', False)) if eval_params else False
    dist_on = distributed and torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if dist_on else 1
    rank = torch.distributed.get_rank() if dist_on else 0
    if dist_metrics and world > 1:
        mode = 'shard'
    elif acc_rank0_only:
        mode = 'rank0'
    else:
        mode = 'all'
    store = (mode != 'rank0') or rank == 0

    clsname_map = dataloader.dataset.datasets[0].labels_to_names
    all_class_labels = sorted(clsname_map.keys())
    my_class_labels = all_class_labels[rank::world] if mode == 'shard' else all_class_labels

    N = len(dataloader.dataset)
    if store:
        # np.zeros pages lazily: in 'shard' mode only the ~N/world written
        # rows consume physical RAM
        masks_all = np.zeros((N, *img_sz), dtype=np.uint8)
        clslabels_all = np.zeros((N,), dtype=np.uint8)
        anom_labels_all = np.zeros((N,), dtype=np.uint8)
        px_all = np.zeros((N, *img_sz), dtype=np.float32)
        written = np.zeros((N,), dtype=bool)

    # vf may be a DDP wrapper or a plain AMDM (e.g. the EMA copy); `distributed`
    # only controls cross-rank result gathering
    module = vf.module if hasattr(vf, 'module') else vf

    # one shared perturbation noise for the whole eval (eval_recon.py protocol),
    # from a fixed-seed generator so every eval / rank uses the same noise
    g = torch.Generator(device='cpu').manual_seed(0)
    feat_shape = None

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose):
        imgs_local = batch["img"].to(device, non_blocking=True)
        clslabels_local = batch["clslabel"].to(device, non_blocking=True)
        anom_labels_local = batch['label'].to(device, non_blocking=True)
        anom_masks_local = batch['mask'].to(device, non_blocking=True)
        idx_local = batch["index"].to(device, non_blocking=True)

        z1, _ = extract_features(fe, imgs_local, device)
        z1 = norm_fn(z1)

        if feat_shape is None:
            feat_shape = z1.shape[1:]
            noise_base = torch.randn(1, *feat_shape, generator=g).to(device)

        px_local = module.score(
            z1, t_init=t_init, steps=recon_steps,
            noise=noise_base.repeat(len(z1), 1, 1, 1),
        )

        if distributed:
            px = concat_all_gather(px_local.contiguous())
            anom_labels = concat_all_gather(anom_labels_local)
            cls_labels = concat_all_gather(clslabels_local)
            anom_masks = concat_all_gather(anom_masks_local.contiguous())
            idx = concat_all_gather(idx_local)
        else:
            px = px_local
            anom_labels, cls_labels = anom_labels_local, clslabels_local
            anom_masks, idx = anom_masks_local, idx_local

        if not store:
            continue

        if mode == 'shard':
            sel = torch.isin(cls_labels, torch.as_tensor(my_class_labels, device=cls_labels.device))
            if not bool(sel.any()):
                continue
            px, anom_labels, cls_labels = px[sel], anom_labels[sel], cls_labels[sel]
            anom_masks, idx = anom_masks[sel], idx[sel]

        px = F.interpolate(px.unsqueeze(1), size=img_sz, mode='bilinear', align_corners=False).squeeze(1)

        anom_masks = anom_masks.squeeze(1)
        anom_masks = F.interpolate(
            anom_masks.unsqueeze(1).float(), size=img_sz, mode='nearest'
        ).squeeze(1).long()

        idx_np = idx.cpu().numpy()
        px_all[idx_np] = px.cpu().numpy()
        clslabels_all[idx_np] = cls_labels.cpu().numpy().astype(np.uint8)
        masks_all[idx_np] = anom_masks.cpu().numpy().astype(np.uint8)
        anom_labels_all[idx_np] = (anom_labels.cpu().numpy() > 0).astype(np.uint8)
        written[idx_np] = True

    if distributed:
        torch.distributed.barrier()

    if not store:
        if was_train:
            vf.train()
        return {'average': {'mad': float('nan')}}

    if written.all():
        px_use, cls_use = px_all, clslabels_all
        masks_use, anom_use = masks_all, anom_labels_all
    else:  # 'shard' mode: restrict to this rank's class-shard rows
        px_use, cls_use = px_all[written], clslabels_all[written]
        masks_use, anom_use = masks_all[written], anom_labels_all[written]

    img_scores_all = aggregate_px_values(agg_method=img_score_agg, px_values=px_use)

    img_scores_by_class = divide_by_class(img_scores_all, cls_use)
    img_gts_by_class = divide_by_class(anom_use, cls_use)
    px_scores_by_class = divide_by_class(px_use, cls_use)
    px_gts_by_class = divide_by_class(masks_use, cls_use)

    eval_results = {v: {} for v in clsname_map.values()}
    img_metrics = [m for m in eval_metrics if m.startswith('img_')]
    px_metrics = [m for m in eval_metrics if m.startswith('px_')]

    for cls_label in img_scores_by_class.keys():
        cls_name = clsname_map[cls_label]
        cls_img_metrics = calculate_img_metrics(
            gt_labels=img_gts_by_class[cls_label],
            pred_scores=img_scores_by_class[cls_label],
            metrics=img_metrics,
        )
        for k, v in cls_img_metrics.items():
            eval_results[cls_name][k] = float(v)

    for cls_label in px_scores_by_class.keys():
        cls_name = clsname_map[cls_label]
        cls_px_metrics = calculate_px_metrics(
            gt_masks=px_gts_by_class[cls_label],
            pred_scores=px_scores_by_class[cls_label],
            metrics=px_metrics,
        )
        for k, v in cls_px_metrics.items():
            eval_results[cls_name][k] = float(v)

    if mode == 'shard':
        # merge the per-class results computed on each rank
        gathered = [None] * world
        torch.distributed.all_gather_object(
            gathered, {k: v for k, v in eval_results.items() if len(v) > 0})
        if rank != 0:
            if was_train:
                vf.train()
            return {'average': {'mad': float('nan')}}
        eval_results = {v: {} for v in clsname_map.values()}
        for part in gathered:
            eval_results.update(part)

    # -- class average + per-class mad (same as src/vfad/eval_recon.py)
    eval_results['average'] = {}
    for cls_name in eval_results.keys():
        if cls_name == 'average':
            continue
        for met_name, met_value in eval_results[cls_name].items():
            eval_results['average'][met_name] = eval_results['average'].get(met_name, 0.0) + met_value
    num_classes = len(dataloader.dataset.datasets)
    for met_name in eval_results['average'].keys():
        eval_results['average'][met_name] /= num_classes

    del_classes = []
    for cls_name in eval_results.keys():
        if len(eval_results[cls_name]) > 0:
            eval_results[cls_name]['mad'] = sum(eval_results[cls_name].values()) / len(eval_results[cls_name])
        else:
            del_classes.append(cls_name)
    for cls_name in del_classes:
        del eval_results[cls_name]

    logger.info(f"AMDM evaluation completed. Results: {eval_results}")

    if was_train:
        vf.train()
    return eval_results


CSV_COLUMNS = ['img_auroc', 'img_ap', 'img_f1max', 'px_auroc', 'px_aupro', 'px_ap', 'px_f1max', 'mad']


def write_results_csv(eval_results, csv_path):
    import csv
    class_names = [c for c in eval_results.keys() if c != 'average'] + ['average']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category'] + CSV_COLUMNS)
        for cls_name in class_names:
            row = [cls_name] + [eval_results[cls_name].get(c, '') for c in CSV_COLUMNS]
            writer.writerow(row)

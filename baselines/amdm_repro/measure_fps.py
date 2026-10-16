"""Inference-FPS measurement for the AMDM reproduction (OmiAD's multi-step
Adaptive Masked Diffusion Model teacher, Feng et al. ICML 2025; no official
code upstream — this lab reimplementation at baselines/amdm_repro/ is the
reportable baseline).

Protocol (shared with the other modality-FPS rounds, user-specified):
  - Batch size 1, single GPU (run with CUDA_VISIBLE_DEVICES=0).
  - Times the full test-time path for ONE image exactly as
    baselines/amdm_repro/amdm.py::evaluate_amdm runs it, config
    configs/amdm_mvtec_ft.yaml:
      image tensor already on GPU (random 1x3x224x224)
        -> frozen EfficientNet-b4 stages 1-4 (outblocks [1,5,9,21]),
           each resized to 32x32 and concatenated -> (272, 32, 32)
        -> norm_fn = Identity (config `normalization: null`, raw features)
        -> AMDM.score: perturb to t_init=960 with the FIXED shared noise
           (fixed-seed CPU generator, seed 0 — the eval_recon.py protocol),
           then deterministic DDIM (eta=0) reconstruction over 40 evenly
           spaced timesteps (np.linspace(0, 959, 40), descending), no mask
           at inference; per-location channel-summed squared error
        -> bilinear upsample of the (32,32) map to 224x224
        -> image score = map max (img_score_agg: max).
    Dataset/disk I/O and metric computation are excluded.
  - PRIMARY: the 40-step teacher inference (what the baseline reports).
    CONTEXT: the same pipeline with steps=1 (a single UNet forward through
    the identical reconstruct() code path) — the cost profile of OmiAD's
    one-step distilled student, which shares this architecture.
  - torch.no_grad + .eval(), torch.backends.cudnn.benchmark = True,
    fp32 (evaluate_amdm uses no autocast).
  - Weights: EMA state from
    out/amdm_mc_unet_enet_mvtec_ft_0918/checkpoints/ckpt_best.pth
    (EMA is what eval_results_final.* / the reported numbers used);
    falls back to random init if unavailable (FPS is weight-independent).
  - Timing: >=10 full warmup iterations, then 5 independent repeats of
    N full iterations each, torch.cuda.synchronize() + time.perf_counter()
    around every iteration. Per-stage breakdown (backbone / DDIM loop /
    anomaly map) is sync-bracketed inside each iteration.
  - Reported: per-repeat mean ms/img, final mean +/- std ACROSS the 5
    repeats of ms/img and FPS (= 1000 / ms_per_img).

Run from the FlowAD repo root (baselines code imports src.* absolutely):
  cd /newssd/projects/sakai/FlowAD && CUDA_VISIBLE_DEVICES=0 \
    /home/haselab/anaconda3/envs/flowad/bin/python \
    baselines/amdm_repro/measure_fps.py | tee baselines/amdm_repro/fps.txt
"""

import os
import sys
import time
import statistics

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.getcwd())

from src.backbones import BackboneWrapper, get_normalization_func
from src.backbones.efficientnet import build_efficient
from baselines.amdm_repro.amdm import AMDM
from baselines.amdm_repro.guided_diffusion.unet import UNetModel

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'amdm_mvtec_ft.yaml')
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'out', 'amdm_mc_unet_enet_mvtec_ft_0918', 'checkpoints', 'ckpt_best.pth')

IMG_SIZE = 224
WARMUP = 10
REPEATS = 5
N_ITERS_40 = 20      # per repeat, 40-step primary
N_ITERS_1 = 200      # per repeat, 1-step context (much faster)


def build(params, device):
    m = params['model']
    net = UNetModel(
        image_size=m['feature_size'],
        in_channels=m['in_channels'],
        model_channels=m['model_channels'],
        out_channels=m['in_channels'],
        num_res_blocks=m['num_res_blocks'],
        attention_resolutions=tuple(m['attention_ds']),
        dropout=m.get('dropout', 0.0),
        channel_mult=tuple(m['channel_mult']),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=m.get('num_heads', 4),
        num_head_channels=m.get('num_head_channels', -1),
        use_scale_shift_norm=m.get('use_scale_shift_norm', True),
        resblock_updown=m.get('resblock_updown', False),
        use_new_attention_order=False,
    )
    d = params['diffusion']
    amdm = AMDM(net, num_timesteps=d['num_timesteps'], beta_start=d['beta_start'],
                beta_end=d['beta_end'], p_min=d['p_min'], p_max=d['p_max'], k=d['k'])

    weights = 'random init'
    if os.path.isfile(CKPT):
        ck = torch.load(CKPT, map_location='cpu', weights_only=True)
        key = 'ema' if 'ema' in ck else 'model'
        sd = {k.replace('module.', ''): v for k, v in ck[key].items()}
        amdm.load_state_dict(sd)
        weights = f"{CKPT} [{key} weights, epoch {ck.get('epoch', '?')}]"
    amdm.to(device).eval()
    for p in amdm.parameters():
        p.requires_grad = False

    bb = m['backbone']
    bnet = build_efficient(model_name=bb['model_name'], pretrained=bb['pretrained'],
                           outblocks=bb['outblocks'], outstrides=bb['outstrides'])
    fs = m['feature_size']
    fe = BackboneWrapper(bnet, target_size=(fs, fs)).to(device).eval()
    for p in fe.parameters():
        p.requires_grad = False
    norm_fn = get_normalization_func(bb.get('normalization', None))
    return amdm, fe, norm_fn, weights


@torch.no_grad()
def one_iter(amdm, fe, norm_fn, img, noise, t_init, steps, img_sz):
    """One full inference iteration, evaluate_amdm's path, with sync-bracketed
    stage timings. Returns (total_ms, backbone_ms, ddim_ms, map_ms)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    z1, _ = fe(img)                      # backbone -> (1, 272, 32, 32)
    z1 = norm_fn(z1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    recon = amdm.reconstruct(z1, t_init=t_init, steps=steps, noise=noise)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    px = (recon - z1).pow(2).sum(dim=1)  # AMDM.score's map
    px = F.interpolate(px.unsqueeze(1), size=img_sz, mode='bilinear',
                       align_corners=False).squeeze(1)
    score = px.amax()
    _ = score.item()                     # image score = max (forces the value)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    return ((t3 - t0) * 1e3, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)


def run_benchmark(label, amdm, fe, norm_fn, img, noise, t_init, steps, img_sz, n_iters):
    print(f"\n=== {label} (t_init={t_init}, DDIM steps={steps}, eta=0) ===")
    for _ in range(WARMUP):
        one_iter(amdm, fe, norm_fn, img, noise, t_init, steps, img_sz)

    repeat_means, stage_means = [], []
    for r in range(REPEATS):
        totals, stages = [], []
        for _ in range(n_iters):
            tot, b, d, m = one_iter(amdm, fe, norm_fn, img, noise, t_init, steps, img_sz)
            totals.append(tot)
            stages.append((b, d, m))
        mean_ms = statistics.mean(totals)
        repeat_means.append(mean_ms)
        sb, sd, sm = (statistics.mean([s[i] for s in stages]) for i in range(3))
        stage_means.append((sb, sd, sm))
        print(f"  repeat {r + 1}/{REPEATS}: {mean_ms:8.2f} ms/img over {n_iters} iters "
              f"(backbone {sb:6.2f} | ddim {sd:8.2f} | map {sm:5.2f} ms)")

    mean = statistics.mean(repeat_means)
    std = statistics.stdev(repeat_means)
    fps_mean = statistics.mean([1000.0 / m for m in repeat_means])
    fps_std = statistics.stdev([1000.0 / m for m in repeat_means])
    ab, ad, am = (statistics.mean([s[i] for s in stage_means]) for i in range(3))
    print(f"  stage breakdown (mean of repeat means): "
          f"backbone {ab:.2f} ms | DDIM loop {ad:.2f} ms | anomaly map {am:.2f} ms")
    print(f"  FINAL {label}: {mean:.2f} +/- {std:.2f} ms/img  ->  "
          f"{fps_mean:.3f} +/- {fps_std:.3f} FPS  (mean +/- std across {REPEATS} repeats)")
    return mean, std, fps_mean, fps_std


def main():
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    with open(CONFIG) as f:
        params = yaml.safe_load(f)

    print("AMDM (OmiAD teacher, lab reimplementation) inference FPS")
    print(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})")
    print(f"config: {CONFIG}")
    print("precision: fp32 (no autocast, as in evaluate_amdm); "
          "torch.no_grad, .eval(), cudnn.benchmark=True")
    print(f"batch size: 1 | input: 3x{IMG_SIZE}x{IMG_SIZE} (random, already on GPU) "
          f"| latent: (272, 32, 32) raw EfficientNet-b4 stages 1-4")

    amdm, fe, norm_fn, weights = build(params, device)
    print(f"weights: {weights}")
    n_unet = sum(p.numel() for p in amdm.model.parameters())
    n_bb = sum(p.numel() for p in fe.parameters())
    print(f"UNet params: {n_unet / 1e6:.2f}M | backbone (EfficientNet-b4) params: {n_bb / 1e6:.2f}M")

    ev = params['logging']['eval']
    t_init, steps = ev['t_init'], ev['recon_steps']
    img_sz = (IMG_SIZE, IMG_SIZE)

    img = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    # fixed shared perturbation noise, exactly as evaluate_amdm builds it
    g = torch.Generator(device='cpu').manual_seed(0)
    with torch.no_grad():
        feat_shape = norm_fn(fe(img)[0]).shape[1:]
    noise = torch.randn(1, *feat_shape, generator=g).to(device)
    print(f"fixed shared noise: shape {tuple(noise.shape)}, CPU generator seed 0 "
          f"(evaluate_amdm protocol)")
    print(f"warmup: {WARMUP} full iterations; {REPEATS} repeats")

    r40 = run_benchmark("PRIMARY 40-step teacher", amdm, fe, norm_fn, img, noise,
                        t_init, steps, img_sz, N_ITERS_40)
    r1 = run_benchmark("CONTEXT 1-step (single UNet forward, student cost profile)",
                       amdm, fe, norm_fn, img, noise, t_init, 1, img_sz, N_ITERS_1)

    print("\n=== SUMMARY ===")
    print(f"PRIMARY  AMDM 40-step teacher : {r40[0]:.2f} +/- {r40[1]:.2f} ms/img "
          f"= {r40[2]:.3f} +/- {r40[3]:.3f} FPS")
    print(f"CONTEXT  1-step (student cost): {r1[0]:.2f} +/- {r1[1]:.2f} ms/img "
          f"= {r1[2]:.3f} +/- {r1[3]:.3f} FPS")


if __name__ == "__main__":
    main()

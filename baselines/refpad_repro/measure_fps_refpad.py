"""FPS measurement for ReFP-AD inference, mirroring FlowAD's
src/utils/measure_fps.py protocol.

Times the exact computation of ReFP-AD's ``pipeline._score_image`` (the path
``pipe.inference`` runs per test image), starting from a pixel tensor already
resident on the GPU (host image loading / PIL decoding excluded, as in
measure_fps.py where the batch is pre-staged on the GPU):

    pixel tensor -> backbone patch tokens
                 -> TokenStats z-score
                 -> RectifiedFlow.encode_rk4 (cfg.flow_steps RK4 steps, EMA)
                 -> EBM energy gradient norm per token
                 -> reshape (gh, gw) -> bilinear upsample to input res
                 -> scipy gaussian_filter(sigma=cfg.score_sigma)
                 -> top-1% mean image score

CUDA effects are controlled identically: cudnn.benchmark, warmup iterations,
CUDA events + torch.cuda.synchronize() bracket every timed iteration; FPS is
reported as mean +/- std over --repeats independent measurements.

Usage (from baselines/refpad_repro/):

    CUDA_VISIBLE_DEVICES=0 python measure_fps_refpad.py \
        --variant enet --ckpt out/mvtec_unified_enet/unified_checkpoint.pt \
        --dataset MVTec --category bottle --batch_size 1 --repeats 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_repro  # noqa: F401,E402  (sys.path setup + MPDD registration)
from config import CFG, Mode  # noqa: E402
import pipeline  # noqa: E402

from enet_backbone import EnetBackbone, IMG_SIZE  # noqa: E402


def tokens_from_tensor(dino, x):
    """Backbone patch tokens for a GPU-resident pixel batch (B, 3, H, W).

    Numerically identical to dino.extract() minus the PIL loading/transform.
    """
    if isinstance(dino, EnetBackbone):
        feats, _ = dino.model(x)                       # (B, 272, 16, 16)
        return feats.flatten(2).permute(0, 2, 1)       # (B, 256, 272)
    outputs = dino.model(pixel_values=x, output_hidden_states=True, return_dict=True)
    return dino._tokens_from_hidden(outputs.hidden_states, dino.layers)


def score_from_tensor(pipe, x, category):
    """ReFP-AD _score_image computation from a pre-staged pixel tensor.

    Follows pipeline._score_image line by line (MEAN aggregation: no
    projector), including the CPU-side upsample + gaussian smoothing + top-1%
    aggregation that produce its outputs.
    """
    cfg = pipe.cfg
    dino = pipe.dino
    stats = pipe.stats[category]

    with torch.no_grad():
        tok = tokens_from_tensor(dino, x)[0].to(cfg.device, dtype=torch.float32)
        z = stats.normalize(tok)
        u = (
            pipe.flow.encode_rk4(z, steps=cfg.flow_steps, use_ema=cfg.use_ema)
            if pipe.flow is not None
            else z
        )

    u_init = u.detach().clone()
    with torch.enable_grad():
        u_init.requires_grad_(True)
        e_ebm = pipe.model(u_init).sum()
        g_init = torch.autograd.grad(e_ebm, u_init)[0]
        score_grad = torch.sqrt(g_init.pow(2).sum(1))
    score_patch = score_grad.view(dino.gh, dino.gw).detach().cpu().numpy()

    h, w = cfg.center_crop
    score_up = (
        F.interpolate(
            torch.from_numpy(score_patch)[None, None].float(),
            size=(h, w), mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0).numpy()
    )
    score_up = gaussian_filter(score_up, sigma=cfg.score_sigma)

    flat = score_up.reshape(-1)
    n = flat.size
    k = max(1, int(n * cfg.img_top_ratio))
    topk = np.partition(flat, n - k)[n - k:]
    return float(topk.mean()), score_up


def main():
    ap = argparse.ArgumentParser(description="FPS measurement for ReFP-AD inference")
    ap.add_argument("--variant", choices=["enet", "dino"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="MVTec", choices=["MVTec", "VisA", "MPDD"])
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert args.batch_size == 1, "ReFP-AD inference is per-image (batch 1)."
    assert torch.cuda.is_available(), "CUDA device required for FPS measurement."
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.variant == "enet":
        pipeline.DinoBackbone = EnetBackbone
        res = (IMG_SIZE, IMG_SIZE)
    else:
        res = (700, 700)

    cfg = CFG(
        dataset=args.dataset,
        data_path="unused",
        out_dir=str(HERE / "out" / "_fps_tmp"),
        cache_dir=HERE / "out" / "_fps_tmp" / "cache",
        mode=Mode.UNIFIED,
        seed=42,
        device="cuda",
        use_flow=True,
        buffer_init="flow_prior",
        resize=res,
        center_crop=(IMG_SIZE, IMG_SIZE) if args.variant == "enet" else (672, 672),
    )
    if args.variant == "enet":
        cfg.dino_model = "efficientnet-b4"

    pipe = pipeline.build_pipeline(cfg)
    pipe.load_checkpoint(Path(args.ckpt))
    if args.category not in pipe.stats:
        raise ValueError(f"category {args.category} not in checkpoint stats "
                         f"({sorted(pipe.stats)})")

    in_hw = cfg.center_crop[0]
    print(f"Device        : {torch.cuda.get_device_name(0)}")
    print(f"Variant       : {args.variant} | checkpoint: {args.ckpt}")
    print(f"Backbone      : {cfg.dino_model} -> {pipe.dino.out_dim}d tokens, "
          f"grid {pipe.dino.gh}x{pipe.dino.gw}, input {in_hw}")
    print(f"Flow          : hidden={cfg.flow_hidden} depth={cfg.flow_depth} "
          f"RK4 steps={cfg.flow_steps} (EMA={cfg.use_ema}) | EBM hidden={cfg.hidden_dim}")
    print(f"Batch size    : {args.batch_size} | score_sigma={cfg.score_sigma} "
          f"| img_top_ratio={cfg.img_top_ratio}")
    print(f"Warmup/iters  : {args.warmup}/{args.iters} | repeats={args.repeats}\n")

    # -- dummy input, pre-staged on GPU (timing excludes host I/O)
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.batch_size, 3, in_hw, in_hw, generator=g).to(cfg.device)
    if args.variant == "dino":
        x = x.to(next(pipe.dino.model.parameters()).dtype)

    # -- warmup (also lets cudnn.benchmark pick algorithms)
    for _ in range(args.warmup):
        score_from_tensor(pipe, x, args.category)
    torch.cuda.synchronize()

    # -- repeated timed measurements with CUDA events
    rep_mean_ms, rep_fps = [], []
    for rep in range(args.repeats):
        times_ms = []
        for _ in range(args.iters):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            score_from_tensor(pipe, x, args.category)
            end_ev.record()
            torch.cuda.synchronize()
            times_ms.append(start_ev.elapsed_time(end_ev))
        mean_ms = float(np.mean(times_ms))
        rep_mean_ms.append(mean_ms)
        rep_fps.append(args.batch_size / (mean_ms / 1e3))

    rep_mean_ms, rep_fps = np.asarray(rep_mean_ms), np.asarray(rep_fps)
    mean_ms, ms_std = rep_mean_ms.mean(), rep_mean_ms.std()
    fps, fps_std = rep_fps.mean(), rep_fps.std()
    rep_str = ", ".join(f"{v:.1f}" for v in rep_fps)
    print(f"{mean_ms:.2f} ± {ms_std:.2f} ms/img | FPS = {fps:.2f} ± {fps_std:.2f} "
          f"(repeats: [{rep_str}])")
    print("\nvariant,ms_per_img,ms_std,fps,fps_std")
    print(f"{args.variant},{mean_ms:.3f},{ms_std:.3f},{fps:.2f},{fps_std:.2f}")


if __name__ == "__main__":
    main()

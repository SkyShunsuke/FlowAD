"""ReFP-AD reproduction with the backbone swapped to FlowAD's EfficientNet-B4.

Controlled-setting variant of run_repro.py, per user request: the ONLY change
vs. original ReFP-AD is the token backbone — DINOv2-G patch tokens are replaced
by FlowAD's efficientnet-b4 multi-scale features (272-dim, 16x16 grid, input
256x256, see enet_backbone.py). Training and inference code paths are the
unmodified ReFP-AD implementation (UNIFIED mode, default hyperparameters,
seed 42, buffer_init="flow_prior"); the swap is done by monkey-patching the
`DinoBackbone` symbol in their `pipeline` module — no repo files are touched.

Because the input resolution is FlowAD's 256x256 (resize only, no crop —
center_crop is set equal to resize, so ReFP-AD's CenterCrop is a no-op), the
score maps come out natively at 256 and the FlowAD 7-metric protocol is
computed without any resizing. GT masks: NEAREST resize to 256.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# run_repro sets up sys.path for both the ReFP-AD clone and the FlowAD root,
# registers MPDD, and provides the shared eval helpers.
from run_repro import (  # noqa: E402
    compute_flowad_metrics,
    load_gt_masks,
    refpad_native_protocol,
    write_csv,
)
from config import CFG, Mode  # noqa: E402  (ReFP-AD module)
from utils import seed_all  # noqa: E402
import pipeline  # noqa: E402

from enet_backbone import EnetBackbone, IMG_SIZE  # noqa: E402

# ---- the one and only change vs. original ReFP-AD: the token backbone ----
pipeline.DinoBackbone = EnetBackbone


def use_one_step_encode():
    """FlowAD one-step plug-in (NOT in ReFP-AD): every flow encode (EBM training buffer and
    test-time scoring both call RectifiedFlow.encode_rk4 with cfg.flow_steps=10 -> 40 NFE)
    becomes a single Euler step u = z + v(0, z) (1 NFE), with the same EMA weights."""
    from flow import RectifiedFlow

    @torch.no_grad()
    def encode_one_step(self, z, steps=10, use_ema=True):
        def step():
            return z + self.v(torch.zeros(z.shape[0], device=z.device), z)
        if self.ema is not None and use_ema:
            with self.ema.apply(self.v):
                return step()
        return step()

    RectifiedFlow.encode_rk4 = encode_one_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MVTec", "VisA", "MPDD"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--encode", default="native", choices=["native", "euler1"],
                    help="native: RK4 x flow_steps (ReFP-AD); euler1: FlowAD one-step plug-in")
    args = ap.parse_args()
    if args.encode == "euler1":
        use_one_step_encode()

    cfg = CFG(
        dataset=args.dataset,
        data_path=args.data_path,
        out_dir=args.out_dir,
        cache_dir=Path(args.cache_dir),
        mode=Mode.UNIFIED,
        seed=args.seed,
        device="cuda",
        use_flow=True,
        buffer_init="flow_prior",  # main.py CLI default
        resize=(IMG_SIZE, IMG_SIZE),        # FlowAD protocol input
        center_crop=(IMG_SIZE, IMG_SIZE),   # no-op crop
    )
    cfg.dino_model = "efficientnet-b4"  # recorded in config.json + cache filename
    cfg.encode = args.encode

    if os.environ.get("REFPAD_SMOKE") == "1":
        cfg.flow_epochs, cfg.flow_min_epochs, cfg.flow_patience = 2, 0, 1
        cfg.epochs = 1
        cfg.categories = cfg.categories[:2]
        print("!!! SMOKE TEST MODE: 2 categories, tiny epochs !!!")

    # identical to ReFP-AD main.run_single()
    seed_all(cfg.seed)
    cfg.save()
    print(f"### ReFP-AD repro (EfficientNet-B4 backbone) | mode=UNIFIED "
          f"dataset={cfg.dataset} seed={cfg.seed}")

    pipe = pipeline.build_pipeline(cfg)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.out_dir) / "unified_checkpoint.pt"
    if args.eval_only:
        pipe.load_checkpoint(ckpt_path)
    else:
        t0 = time.time()
        pipe.train()
        print(f"[-] Training took {(time.time() - t0) / 3600:.2f} h")
        pipe.save_checkpoint()

    # ---------------- evaluation (FlowAD metrics protocol) ----------------
    crop_hw = cfg.center_crop[0]   # 256
    resize_hw = cfg.resize[0]      # 256 -> aligned mask == direct NEAREST resize
    root = Path(cfg.data_path)
    maps_dir = Path(cfg.out_dir) / "score_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    cat_data = {}
    t0 = time.time()
    for cat in cfg.categories:
        test_root = root / cat / "test"
        img_scores, img_labels, maps, masks_a, masks_d, paths = [], [], [], [], [], []
        for d in sorted(p for p in test_root.iterdir() if p.is_dir()):
            imgs = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.JPG")))
            for img_p in imgs:
                s_img, s_map, _ = pipe.inference(img_p, category=cat)  # UNMODIFIED
                ma, md = load_gt_masks(cfg.dataset, root, cat, img_p, d.name,
                                       crop_hw, resize_hw)
                img_scores.append(float(s_img))
                img_labels.append(0 if d.name == "good" else 1)
                maps.append(s_map.astype(np.float16))
                masks_a.append(ma)
                masks_d.append(md)
                paths.append(str(img_p))
        cat_data[cat] = {
            "img_scores": img_scores,
            "img_labels": img_labels,
            "maps": np.stack(maps),
            "masks_aligned": np.stack(masks_a),
            "masks_direct": np.stack(masks_d),
        }
        np.savez_compressed(
            maps_dir / f"{cat}.npz",
            maps=cat_data[cat]["maps"],
            masks_aligned=cat_data[cat]["masks_aligned"],
            img_scores=np.asarray(img_scores, dtype=np.float32),
            img_labels=np.asarray(img_labels, dtype=np.uint8),
            paths=np.asarray(paths),
        )
        print(f"[-] Inference done: {cat} ({len(img_scores)} images, "
              f"{time.time() - t0:.0f}s elapsed)")

    print("\n=== ReFP-AD native protocol (I/P-AUROC, reference) ===")
    native = refpad_native_protocol(cat_data, cfg.categories)

    print("\n=== FlowAD protocol @256 (primary) ===")
    res_256 = compute_flowad_metrics(cat_data, cfg.categories, 256, crop_hw)

    out = Path(cfg.out_dir)
    write_csv(res_256, cfg.categories, out / "eval_results_flowad_256.csv")
    with open(out / "eval_results_full.json", "w") as f:
        json.dump({"flowad_256": res_256, "refpad_native": native}, f, indent=2)
    print(f"\n[-] DONE. mAD@256 = {res_256['average']['mad']:.4f} | "
          f"avg img_auroc={res_256['average']['img_auroc']:.4f} "
          f"px_auroc={res_256['average']['px_auroc']:.4f}")


if __name__ == "__main__":
    main()

"""ReFP-AD reproduction (EfficientNet-B4 backbone) on RealIAD, multi-class.

Deltas vs run_repro_enet.py (MVTec/VisA/MPDD):

- 30 categories from FlowAD's src/datasets/realiad.py (sorted), registered in
  ReFP-AD's config.DATASET_CATEGORIES. Data comes from the MVTec-style symlink
  layout built by prepare_realiad_1cls.py (masks linked as <stem>_mask.png, so
  the shared run_repro.load_gt_masks MVTec-style branch applies).
- NO early stopping: cfg.flow_min_epochs = cfg.flow_epochs, so the rectified
  flow trains the full flow_epochs (default 250) — per protocol request:
  train all epochs, no intermediate results, one evaluation at the end. The
  EBM stage always runs its fixed cfg.epochs (15) anyway. Note ReFP-AD's
  train_flow still restores the best-internal-fitness epoch weights at the
  end; that diagnostic uses train-side samples only (no test data) and is
  unmodified ReFP-AD behaviour.
- The test set is 114,585 images, so metrics are computed per category as
  soon as its inference finishes (streaming — maps of all 30 classes are
  never held in RAM together), and score maps are persisted per category to
  <out_dir>/score_maps/<cat>.npz. The final eval can optionally be sharded
  across GPUs with --eval_only --categories a,b,c and combined afterwards
  with --merge_only (recomputes all metrics from the npz files).

Training/inference remain the unmodified ReFP-AD code (UNIFIED mode, default
hyperparameters, seed 42, buffer_init="flow_prior"); the backbone swap is the
same pipeline.DinoBackbone monkey-patch as run_repro_enet.py. With
resize == center_crop == 256, score maps are native 256 and the aligned /
direct GT masks coincide, so one mask array serves both protocols.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_repro  # noqa: E402  (sys.path setup, MPDD registration, eval helpers)
from run_repro import (  # noqa: E402
    compute_flowad_metrics,
    load_gt_masks,
    refpad_native_protocol,
    write_csv,
)
import config as refpad_config  # noqa: E402  (ReFP-AD module)
from config import CFG, Mode  # noqa: E402
from utils import seed_all  # noqa: E402
import pipeline  # noqa: E402

from enet_backbone import EnetBackbone, IMG_SIZE  # noqa: E402
from src.datasets.realiad import REALIAD_CLASSES  # noqa: E402

REALIAD_CATEGORIES = sorted(REALIAD_CLASSES)
refpad_config.DATASET_CATEGORIES.setdefault("RealIAD", REALIAD_CATEGORIES)
run_repro.MVTEC_STYLE_GT.add("RealIAD")  # masks linked as <stem>_mask.png

# ---- the one and only change vs. original ReFP-AD: the token backbone ----
pipeline.DinoBackbone = EnetBackbone

METRIC_ORDER = run_repro.METRIC_ORDER


def infer_category(pipe, root: Path, cat: str, hw: int):
    """Full test-set inference for one category (unmodified pipe.inference)."""
    test_root = root / cat / "test"
    max_imgs = int(os.environ.get("REFPAD_SMOKE_MAX_IMGS", "0"))  # 0 = no cap
    img_scores, img_labels, maps, masks, paths = [], [], [], [], []
    for d in sorted(p for p in test_root.iterdir() if p.is_dir()):
        imgs = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.JPG")))
        if max_imgs:
            imgs = imgs[:max_imgs]
        for img_p in imgs:
            s_img, s_map, _ = pipe.inference(img_p, category=cat)  # UNMODIFIED
            # resize == center_crop == 256 -> aligned and direct masks coincide
            ma, _ = load_gt_masks("RealIAD", root, cat, img_p, d.name, hw, hw)
            img_scores.append(float(s_img))
            img_labels.append(0 if d.name == "good" else 1)
            maps.append(s_map.astype(np.float16))
            masks.append(ma)
            paths.append(str(img_p))
    return {
        "img_scores": img_scores,
        "img_labels": img_labels,
        "maps": np.stack(maps),
        "masks": np.stack(masks),
        "paths": paths,
    }


def save_npz(maps_dir: Path, cat: str, d: dict):
    np.savez_compressed(
        maps_dir / f"{cat}.npz",
        maps=d["maps"],
        masks=d["masks"],
        img_scores=np.asarray(d["img_scores"], dtype=np.float32),
        img_labels=np.asarray(d["img_labels"], dtype=np.uint8),
        paths=np.asarray(d["paths"]),
    )


def load_npz(maps_dir: Path, cat: str):
    z = np.load(maps_dir / f"{cat}.npz")
    return {
        "img_scores": z["img_scores"].tolist(),
        "img_labels": z["img_labels"].tolist(),
        "maps": z["maps"],
        "masks": z["masks"],
    }


def category_metrics(cat: str, d: dict, hw: int):
    dd = {**d, "masks_aligned": d["masks"], "masks_direct": d["masks"]}
    flowad = compute_flowad_metrics({cat: dd}, [cat], hw, hw)[cat]
    native = refpad_native_protocol({cat: dd}, [cat])[cat]
    return flowad, native


def finalize(results: dict, native: dict, categories: list, out: Path):
    avg = {m: float(np.mean([results[c][m] for c in categories])) for m in METRIC_ORDER}
    avg["mad"] = float(np.mean([avg[m] for m in METRIC_ORDER]))
    results = {**results, "average": avg}
    native = {**native, "average": {
        "i_auroc": float(np.nanmean([native[c]["i_auroc"] for c in categories])),
        "p_auroc": float(np.nanmean([native[c]["p_auroc"] for c in categories])),
    }}
    write_csv(results, categories, out / "eval_results_flowad_256.csv")
    with open(out / "eval_results_full.json", "w") as f:
        json.dump({"flowad_256": results, "refpad_native": native}, f, indent=2)
    print(f"\n[-] DONE. mAD@256 = {results['average']['mad']:.4f} | "
          f"avg img_auroc={results['average']['img_auroc']:.4f} "
          f"px_auroc={results['average']['px_auroc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=str(run_repro.FLOWAD_ROOT / "data" / "realiad_1cls"))
    ap.add_argument("--out_dir", default=str(HERE / "out" / "realiad_unified_enet"))
    ap.add_argument("--cache_dir", default=str(HERE / "cache_tokens_enet"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--categories", type=str, default=None,
                    help="comma-separated eval shard (use with --eval_only)")
    ap.add_argument("--merge_only", action="store_true",
                    help="rebuild CSV/JSON from saved score_maps/*.npz (after sharded eval)")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    cfg = CFG(
        dataset="RealIAD",
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
    # protocol request: train ALL epochs — disable flow early stopping entirely
    cfg.flow_min_epochs = cfg.flow_epochs

    hw = cfg.center_crop[0]  # 256
    root = Path(cfg.data_path)
    out = Path(cfg.out_dir)
    maps_dir = out / "score_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        res, nat = {}, {}
        for cat in cfg.categories:
            res[cat], nat[cat] = category_metrics(cat, load_npz(maps_dir, cat), hw)
        finalize(res, nat, cfg.categories, out)
        return

    if os.environ.get("REFPAD_SMOKE") == "1":
        cfg.flow_epochs, cfg.flow_min_epochs, cfg.flow_patience = 2, 2, 1
        cfg.epochs = 1
        cfg.categories = cfg.categories[:2]
        print("!!! SMOKE TEST MODE: 2 categories, tiny epochs !!!")

    # identical to ReFP-AD main.run_single()
    seed_all(cfg.seed)
    cfg.save()
    print(f"### ReFP-AD repro (EfficientNet-B4 backbone) | mode=UNIFIED "
          f"dataset=RealIAD seed={cfg.seed} "
          f"flow_epochs={cfg.flow_epochs} (early stopping disabled)")

    pipe = pipeline.build_pipeline(cfg)

    ckpt_path = Path(args.ckpt) if args.ckpt else out / "unified_checkpoint.pt"
    if args.eval_only:
        pipe.load_checkpoint(ckpt_path)
    else:
        t0 = time.time()
        pipe.train()  # all epochs, no intermediate evaluation
        print(f"[-] Training took {(time.time() - t0) / 3600:.2f} h")
        pipe.save_checkpoint()

    # ------------- evaluation (FlowAD metrics protocol, streaming) -------------
    cats = args.categories.split(",") if args.categories else list(cfg.categories)
    res, nat = {}, {}
    t0 = time.time()
    for cat in cats:
        d = infer_category(pipe, root, cat, hw)
        save_npz(maps_dir, cat, d)
        res[cat], nat[cat] = category_metrics(cat, d, hw)
        print(f"[-] Inference done: {cat} ({len(d['img_scores'])} images, "
              f"{time.time() - t0:.0f}s elapsed)", flush=True)
        del d

    if set(cats) == set(cfg.categories):
        finalize(res, nat, cfg.categories, out)
    else:
        print(f"[-] Shard done ({len(cats)}/{len(cfg.categories)} categories). "
              f"Run with --merge_only once all shards have finished.")


if __name__ == "__main__":
    main()

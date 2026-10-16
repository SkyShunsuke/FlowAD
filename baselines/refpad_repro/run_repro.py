"""Reproduce ReFP-AD (https://github.com/CLendering/ReFP-AD) under the FlowAD
evaluation-metrics protocol.

- Training and inference are the ORIGINAL ReFP-AD code, untouched: this script
  imports the cloned repo and drives it exactly like their main.py run_single()
  (UNIFIED mode, default hyperparameters, seed 42, buffer_init="flow_prior"
  which is their CLI default).
- Evaluation replaces only the METRIC computation: per-image scores and pixel
  anomaly maps come from the unmodified `pipe.inference`; metrics are the
  FlowAD 7 (img_auroc, img_ap, img_f1max, px_auroc, px_ap, px_f1max, px_aupro)
  + mAD, computed per class with src/utils/adeval/eval_utils, at the FlowAD
  protocol resolution 256x256 (primary) and at ReFP-AD's native 672x672
  (reference). GT masks are geometrically aligned with ReFP-AD's input
  transform (Resize(700,700) NEAREST + CenterCrop(672)).
- Also reports ReFP-AD's own-protocol I-AUROC / P-AUROC as a sanity check
  against the paper (their mask handling: direct NEAREST resize to 672,
  pixels pooled per category).

MPDD support is injected by registering its category list in
config.DATASET_CATEGORIES (MVTec-style layout; no repo files are modified).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
REFPAD_ROOT = HERE.parent / "ReFP-AD"
FLOWAD_ROOT = HERE.parent.parent
sys.path.insert(0, str(FLOWAD_ROOT))
sys.path.insert(0, str(REFPAD_ROOT))

import config as refpad_config  # noqa: E402  (ReFP-AD module)
from config import CFG, Mode  # noqa: E402
from utils import seed_all  # noqa: E402
from pipeline import build_pipeline  # noqa: E402

from src.utils.adeval.eval_utils import (  # noqa: E402
    calculate_img_metrics,
    calculate_px_metrics,
)

MPDD_CATEGORIES = [
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
]
refpad_config.DATASET_CATEGORIES.setdefault("MPDD", MPDD_CATEGORIES)

METRIC_ORDER = [
    "img_auroc", "img_ap", "img_f1max",
    "px_auroc", "px_aupro", "px_ap", "px_f1max",
]
IMG_METRICS = ["img_auroc", "img_ap", "img_f1max"]
PX_METRICS = ["px_auroc", "px_ap", "px_f1max", "px_aupro"]

# datasets with MVTec-style ground_truth/<defect>/<stem>_mask.png
MVTEC_STYLE_GT = {"MVTec", "MPDD"}


def load_gt_masks(dataset: str, root: Path, category: str, img_path: Path,
                  defect: str, crop_hw: int, resize_hw: int):
    """Return (aligned_mask, direct_mask) uint8 arrays at crop_hw x crop_hw.

    aligned: Resize(resize,NEAREST) + CenterCrop(crop) -- matches the image
             transform of ReFP-AD's _score_image, pixel-aligned to the map.
    direct:  NEAREST resize straight to crop x crop -- ReFP-AD's own eval.
    """
    if defect == "good":
        z = np.zeros((crop_hw, crop_hw), dtype=np.uint8)
        return z, z
    if dataset in MVTEC_STYLE_GT:
        mpath = root / category / "ground_truth" / defect / (img_path.stem + "_mask.png")
    else:  # VisA 1cls
        mpath = root / category / "ground_truth" / "bad" / (img_path.stem + ".png")
    if not mpath.exists():
        # mirror ReFP-AD behaviour: anomalous image without a mask file -> all-zero
        z = np.zeros((crop_hw, crop_hw), dtype=np.uint8)
        return z, z
    m = Image.open(mpath).convert("L")
    direct = np.array(m.resize((crop_hw, crop_hw), Image.NEAREST))
    aligned_pil = m.resize((resize_hw, resize_hw), Image.NEAREST)
    off = (resize_hw - crop_hw) // 2
    aligned = np.array(aligned_pil)[off:off + crop_hw, off:off + crop_hw]
    return (aligned > 0).astype(np.uint8), (direct > 0).astype(np.uint8)


@torch.no_grad()
def resize_maps(maps: np.ndarray, size: int, mode: str) -> np.ndarray:
    t = torch.from_numpy(maps).float().unsqueeze(1).cuda()
    if mode == "bilinear":
        r = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    else:
        r = F.interpolate(t, size=(size, size), mode="nearest")
    return r.squeeze(1).cpu().numpy()


def compute_flowad_metrics(cat_data: dict, categories: list, at_size: int, crop_hw: int):
    """Per-class 7 metrics + mad, plus 'average' row (FlowAD protocol)."""
    results = {}
    for cat in categories:
        d = cat_data[cat]
        img_scores = np.asarray(d["img_scores"], dtype=np.float32)
        img_gts = np.asarray(d["img_labels"], dtype=np.uint8)
        px_scores = d["maps"].astype(np.float32)  # (N, 672, 672)
        px_gts = d["masks_aligned"]  # (N, 672, 672) uint8

        if at_size != crop_hw:
            px_scores = resize_maps(px_scores, at_size, "bilinear")
            px_gts = resize_maps(px_gts.astype(np.float32), at_size, "nearest").astype(np.uint8)

        res = {}
        res.update(calculate_img_metrics(gt_labels=img_gts, pred_scores=img_scores,
                                         metrics=IMG_METRICS))
        res.update({k: float(v) for k, v in calculate_px_metrics(
            gt_masks=px_gts, pred_scores=px_scores, metrics=PX_METRICS).items()})
        res["mad"] = float(np.mean([res[m] for m in METRIC_ORDER]))
        results[cat] = res
        print(f"  [{at_size}px] {cat}: " +
              " ".join(f"{m}={res[m]:.4f}" for m in METRIC_ORDER + ["mad"]))

    avg = {m: float(np.mean([results[c][m] for c in categories]))
           for m in METRIC_ORDER}
    avg["mad"] = float(np.mean([avg[m] for m in METRIC_ORDER]))
    results["average"] = avg
    return results


def refpad_native_protocol(cat_data: dict, categories: list):
    """ReFP-AD's own evaluate.py protocol: sklearn AUROC, direct-resized masks."""
    from sklearn.metrics import roc_auc_score
    out = {}
    for cat in categories:
        d = cat_data[cat]
        y_img = np.asarray(d["img_labels"])
        p_img = np.asarray(d["img_scores"])
        i_auc = float(roc_auc_score(y_img, p_img)) if len(set(y_img.tolist())) > 1 else float("nan")
        y_pix = d["masks_direct"].reshape(-1)
        p_pix = d["maps"].astype(np.float32).reshape(-1)
        p_auc = float(roc_auc_score(y_pix, p_pix)) if y_pix.max() > 0 else float("nan")
        out[cat] = {"i_auroc": i_auc, "p_auroc": p_auc}
        print(f"  [native-protocol] {cat}: I-AUROC={i_auc:.4f} P-AUROC={p_auc:.4f}")
    out["average"] = {
        "i_auroc": float(np.nanmean([out[c]["i_auroc"] for c in categories])),
        "p_auroc": float(np.nanmean([out[c]["p_auroc"] for c in categories])),
    }
    return out


def write_csv(results: dict, categories: list, path: Path):
    cols = ["img_auroc", "img_ap", "img_f1max", "px_auroc", "px_aupro", "px_ap", "px_f1max", "mad"]
    with open(path, "w") as f:
        f.write("category," + ",".join(cols) + "\n")
        for cat in categories + ["average"]:
            f.write(cat + "," + ",".join(str(results[cat][c]) for c in cols) + "\n")
    print(f"[-] Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MVTec", "VisA", "MPDD"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

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
        resize=(700, 700),
        center_crop=(672, 672),
    )

    # smoke-test gate: tiny epochs / 2 categories, ONLY when REFPAD_SMOKE=1
    if os.environ.get("REFPAD_SMOKE") == "1":
        cfg.flow_epochs, cfg.flow_min_epochs, cfg.flow_patience = 2, 0, 1
        cfg.epochs = 1
        cfg.categories = cfg.categories[:2]
        print("!!! SMOKE TEST MODE: 2 categories, tiny epochs !!!")

    # identical to ReFP-AD main.run_single()
    seed_all(cfg.seed)
    cfg.save()
    print(f"### ReFP-AD repro | mode=UNIFIED dataset={cfg.dataset} seed={cfg.seed}")

    pipe = build_pipeline(cfg)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.out_dir) / "unified_checkpoint.pt"
    if args.eval_only:
        pipe.load_checkpoint(ckpt_path)
    else:
        t0 = time.time()
        pipe.train()
        print(f"[-] Training took {(time.time() - t0) / 3600:.2f} h")
        pipe.save_checkpoint()

    # ---------------- evaluation (FlowAD metrics protocol) ----------------
    crop_hw = cfg.center_crop[0]   # 672
    resize_hw = cfg.resize[0]      # 700
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

    print("\n=== ReFP-AD native protocol (sanity vs paper) ===")
    native = refpad_native_protocol(cat_data, cfg.categories)

    print("\n=== FlowAD protocol @256 (primary) ===")
    res_256 = compute_flowad_metrics(cat_data, cfg.categories, 256, crop_hw)

    print("\n=== FlowAD metrics @672 native (reference) ===")
    res_672 = compute_flowad_metrics(cat_data, cfg.categories, crop_hw, crop_hw)

    out = Path(cfg.out_dir)
    write_csv(res_256, cfg.categories, out / "eval_results_flowad_256.csv")
    write_csv(res_672, cfg.categories, out / "eval_results_flowad_672.csv")
    with open(out / "eval_results_full.json", "w") as f:
        json.dump({"flowad_256": res_256, "flowad_672": res_672,
                   "refpad_native": native}, f, indent=2)
    print(f"\n[-] DONE. mAD@256 = {res_256['average']['mad']:.4f} | "
          f"avg img_auroc={res_256['average']['img_auroc']:.4f} "
          f"px_auroc={res_256['average']['px_auroc']:.4f}")


if __name__ == "__main__":
    main()

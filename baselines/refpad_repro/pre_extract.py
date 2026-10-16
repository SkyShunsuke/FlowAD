"""Pre-build ReFP-AD train-token caches for a subset of categories.

Uses the unmodified DinoBackbone + TokenCache from the cloned repo, so the
cache files are byte-identical to what pipe.train() would create; training
runs simply find them already present. Purely a parallelization of the
(deterministic) extraction phase across GPUs.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ReFP-AD"))

import config as refpad_config  # noqa: E402
from config import CFG, Mode  # noqa: E402
from backbone import DinoBackbone  # noqa: E402
from token_stats import TokenCache  # noqa: E402

refpad_config.DATASET_CATEGORIES.setdefault("MPDD", [
    "bracket_black", "bracket_brown", "bracket_white",
    "connector", "metal_plate", "tubes",
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["MVTec", "VisA", "MPDD"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--categories", required=True, help="comma-separated")
    args = ap.parse_args()

    cfg = CFG(dataset=args.dataset, data_path=args.data_path,
              out_dir=str(HERE / "out" / "_pre_extract_tmp"),
              cache_dir=Path(args.cache_dir), mode=Mode.UNIFIED, device="cuda")

    dino = DinoBackbone(cfg.dino_model, cfg.device, cfg.resize, cfg.center_crop,
                        cfg.patch, cfg.aggregation, cfg.layers)
    cache = TokenCache(cfg, dino)
    for cat in args.categories.split(","):
        t = cache.load_category(cat)
        print(f"[pre-extract] {cat}: {tuple(t.shape)}", flush=True)
    print("[pre-extract] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

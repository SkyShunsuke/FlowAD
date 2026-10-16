"""Pre-build EfficientNet-B4 train-token caches for a subset of categories.

Enet-backbone counterpart of pre_extract.py: uses the unmodified TokenCache
with EnetBackbone and the same CFG-derived cache filename as
run_repro_enet*.py (dino_model="efficientnet-b4", resize=center_crop=256,
default layers/aggregation fields), so training runs find the cache files
already present, byte-identical to what pipe.train() would create. Purely a
parallelization of the deterministic extraction phase across GPUs.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_repro  # noqa: E402,F401  (sys.path setup for ReFP-AD + FlowAD, MPDD registration)
import config as refpad_config  # noqa: E402  (ReFP-AD module)
from config import CFG, Mode  # noqa: E402
from token_stats import TokenCache  # noqa: E402

from enet_backbone import EnetBackbone, IMG_SIZE  # noqa: E402
from src.datasets.realiad import REALIAD_CLASSES  # noqa: E402

refpad_config.DATASET_CATEGORIES.setdefault("RealIAD", sorted(REALIAD_CLASSES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["MVTec", "VisA", "MPDD", "RealIAD"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--categories", required=True, help="comma-separated")
    args = ap.parse_args()

    cfg = CFG(dataset=args.dataset, data_path=args.data_path,
              out_dir=str(HERE / "out" / "_pre_extract_tmp"),
              cache_dir=Path(args.cache_dir), mode=Mode.UNIFIED, device="cuda",
              resize=(IMG_SIZE, IMG_SIZE), center_crop=(IMG_SIZE, IMG_SIZE))
    cfg.dino_model = "efficientnet-b4"

    backbone = EnetBackbone(cfg.dino_model, cfg.device, cfg.resize,
                            cfg.center_crop, cfg.patch, cfg.aggregation,
                            cfg.layers)
    cache = TokenCache(cfg, backbone)
    for cat in args.categories.split(","):
        t = cache.load_category(cat)
        print(f"[pre-extract] {cat}: {tuple(t.shape)}", flush=True)
    print("[pre-extract] ALL DONE", flush=True)


if __name__ == "__main__":
    main()

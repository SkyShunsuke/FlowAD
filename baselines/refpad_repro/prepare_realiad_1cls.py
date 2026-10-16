"""Build an MVTec-style 1cls layout for RealIAD via symlinks (expected by
ReFP-AD / refpad_repro), from the official realiad_jsons split files — the
same split FlowAD's src/datasets/realiad.py uses.

Layout produced:
  <out>/<category>/train/good/<name>.jpg                      (OK only)
  <out>/<category>/test/good/<name>.jpg                       (OK)
  <out>/<category>/test/<anomaly_class>/<name>.jpg            (NG: AK/BX/HS/...)
  <out>/<category>/ground_truth/<anomaly_class>/<stem>_mask.png
    (mask symlink renamed to the MVTec convention so run_repro.load_gt_masks'
     MVTec-style branch resolves it)
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOWAD_ROOT = HERE.parent.parent
sys.path.insert(0, str(FLOWAD_ROOT))

from src.datasets.realiad import REALIAD_CLASSES  # noqa: E402

REALIAD_ROOT = Path("/home/acg17237wy/projects/FlowAD/data/realiad")
META_DIR = Path("/home/acg17237wy/projects/FlowAD/data/realiad_meta/realiad_jsons")
OUT_ROOT = FLOWAD_ROOT / "data" / "realiad_1cls"


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(src, dst)


def main():
    n = {"train": 0, "test_good": 0, "test_ng": 0, "mask": 0, "mask_missing": 0}
    for cat in sorted(REALIAD_CLASSES):
        meta = json.loads((META_DIR / f"{cat}.json").read_text())
        img_root = REALIAD_ROOT / cat / cat  # images live under <root>/<cat>/<cat>/
        for e in meta["train"]:
            assert e["anomaly_class"] == "OK", f"NG image in train split: {e}"
            src = img_root / e["image_path"]
            assert src.exists(), src
            link(src, OUT_ROOT / cat / "train" / "good" / src.name)
            n["train"] += 1
        for e in meta["test"]:
            src = img_root / e["image_path"]
            assert src.exists(), src
            if e["anomaly_class"] == "OK":
                link(src, OUT_ROOT / cat / "test" / "good" / src.name)
                n["test_good"] += 1
            else:
                defect = e["anomaly_class"]
                link(src, OUT_ROOT / cat / "test" / defect / src.name)
                n["test_ng"] += 1
                mask = img_root / e["mask_path"] if e["mask_path"] else None
                if mask is not None and mask.exists():
                    link(mask, OUT_ROOT / cat / "ground_truth" / defect / (src.stem + "_mask.png"))
                    n["mask"] += 1
                else:
                    n["mask_missing"] += 1  # harness scores it with an all-zero mask
        print(f"[prep] {cat} done", flush=True)
    print(n)


if __name__ == "__main__":
    main()

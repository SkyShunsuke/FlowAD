"""Build the VisA_pytorch/1cls-style layout (expected by ReFP-AD) from the
original VisA layout via symlinks, using the official split_csv/1cls.csv.

Layout produced:
  <out>/<category>/train/good/<stem>.JPG
  <out>/<category>/test/good/<stem>.JPG
  <out>/<category>/test/bad/<stem>.JPG
  <out>/<category>/ground_truth/bad/<stem>.png
"""

import csv
import os
from pathlib import Path

VISA_ROOT = Path("/newssd/projects/sakai/FlowAD/data/visa")
OUT_ROOT = Path("/newssd/projects/sakai/FlowAD/data/visa_pytorch/1cls")


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(src, dst)


def main():
    csv_path = VISA_ROOT / "split_csv" / "1cls.csv"
    n = {"train": 0, "test_good": 0, "test_bad": 0, "mask": 0}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cat, split, label = row["object"], row["split"], row["label"]
            img = VISA_ROOT / row["image"]
            assert img.exists(), img
            stem = img.name
            if split == "train":
                assert label == "normal"
                link(img, OUT_ROOT / cat / "train" / "good" / stem)
                n["train"] += 1
            else:
                if label == "normal":
                    link(img, OUT_ROOT / cat / "test" / "good" / stem)
                    n["test_good"] += 1
                else:
                    link(img, OUT_ROOT / cat / "test" / "bad" / stem)
                    n["test_bad"] += 1
                    mask = VISA_ROOT / row["mask"]
                    assert mask.exists(), mask
                    link(mask, OUT_ROOT / cat / "ground_truth" / "bad" / mask.name)
                    n["mask"] += 1
    print(n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
eval_results_{s}.csv (s = 1..1000) を探索し、
category ごとにフォルダを作成して category/eval_results_{s}.csv に分割するスクリプト。

各出力ファイルには元のヘッダ行と、該当 category の1行のみが書き込まれます。
"""

import csv
from pathlib import Path

# 入力ファイルが置かれているディレクトリ(必要に応じて変更してください)
INPUT_DIR = Path("/home/haselab/projects/sakai_ssd/FlowAD/artifacts/dataset_ablations/rawdata/bmad")
# 出力先のルートディレクトリ(category フォルダはこの下に作られます)
OUTPUT_DIR = Path("/home/haselab/projects/sakai_ssd/FlowAD/artifacts/dataset_ablations/rawdata")


def split_file(csv_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # category,img_auroc,img_ap,img_f1max,mad
        for row in reader:
            if not row:
                continue
            category = row[0]
            out_dir = OUTPUT_DIR / category
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / csv_path.name
            with out_path.open("w", newline="", encoding="utf-8") as out_f:
                writer = csv.writer(out_f)
                writer.writerow(header)
                row[0] = "average"
                writer.writerow(row)


def main() -> None:
    found = 0
    for s in range(1, 1001):
        csv_path = INPUT_DIR / f"eval_results_{s}.csv"
        if csv_path.exists():
            split_file(csv_path)
            found += 1
    print(f"処理完了: {found} 個のファイルを分割しました。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from pathlib import Path
import argparse
from PIL import Image


def parse_grid(grid_str: str) -> tuple[int, int]:
    try:
        h, w = grid_str.lower().split("x")
        h, w = int(h), int(w)
        if h <= 0 or w <= 0:
            raise ValueError
        return h, w
    except Exception:
        raise argparse.ArgumentTypeError(
            "グリッド形式は 'HxW' で指定してください（例: 3x4）"
        )


def load_png_files(folder: Path) -> list[Path]:
    # 大文字小文字の拡張子両対応
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
    return files


def make_grid_image(
    image_paths: list[Path],
    rows: int,
    cols: int,
    output_path: Path,
    fill_empty: bool = True,
) -> None:
    if not image_paths:
        raise ValueError("PNGファイルが見つかりませんでした。")

    # 1枚目のサイズに合わせて全画像をリサイズ
    first = Image.open(image_paths[0]).convert("RGBA")
    cell_w, cell_h = first.size

    total_cells = rows * cols
    used_paths = image_paths[:total_cells]

    canvas = Image.new("RGBA", (cols * cell_w, rows * cell_h), (255, 255, 255, 0))

    for idx, img_path in enumerate(used_paths):
        img = Image.open(img_path).convert("RGBA")
        if img.size != (cell_w, cell_h):
            img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)

        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = r * cell_h
        canvas.paste(img, (x, y), img)

    # 空きセルを白で埋めたい場合
    if fill_empty and len(used_paths) < total_cells:
        for idx in range(len(used_paths), total_cells):
            r = idx // cols
            c = idx % cols
            x = c * cell_w
            y = r * cell_h
            blank = Image.new("RGBA", (cell_w, cell_h), (255, 255, 255, 255))
            canvas.paste(blank, (x, y))

    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="フォルダ内のPNG画像を数え、HxWグリッド画像に結合します。"
    )
    parser.add_argument("folder", type=str, help="PNGファイルが入ったフォルダ")
    parser.add_argument("grid", type=parse_grid, help="グリッドサイズ（例: 3x4）")
    parser.add_argument(
        "-o", "--output", type=str, default="grid.png", help="出力画像ファイル名"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    rows, cols = args.grid
    output_path = Path(args.output)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"フォルダが存在しません: {folder}")

    png_files = load_png_files(folder)
    print(f"PNGファイル数: {len(png_files)}")

    total_cells = rows * cols
    if len(png_files) < total_cells:
        print(
            f"注意: グリッドは {total_cells} 枠ですが、PNGは {len(png_files)} 枚です。"
            " 空きセルは白で埋めます。"
        )
    elif len(png_files) > total_cells:
        print(
            f"注意: PNGは {len(png_files)} 枚ありますが、グリッドは {total_cells} 枠です。"
            f" 先頭から {total_cells} 枚だけ使用します。"
        )

    make_grid_image(
        image_paths=png_files,
        rows=rows,
        cols=cols,
        output_path=output_path,
        fill_empty=True,
    )
    print(f"保存しました: {output_path}")


if __name__ == "__main__":
    main()
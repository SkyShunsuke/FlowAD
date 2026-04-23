import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# 例:
# eval_results_0.1.csv
# eval_results_1.csv
# eval_results_10.5.csv
FILENAME_PATTERN = re.compile(r"eval_results_([0-9]+(?:\.[0-9]+)?)\.csv$")

VALID_METRICS = [
    "img_auroc",
    "img_ap",
    "img_f1max",
    "px_auroc",
    "px_aupro",
    "px_ap",
    "px_f1max",
    "mad",
]


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text))


def read_eval_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "category" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "category"})

    return df


def collect_metric_values(folder: Path, metric: str) -> pd.DataFrame:
    """
    指定フォルダ内の eval_results_{time}.csv をすべて読み込み、
    全カテゴリ・全timeについて metric を集約した DataFrame を返す。

    return columns:
        category, time, <metric>
    """
    records = []

    for csv_path in sorted(folder.glob("eval_results_*.csv")):
        match = FILENAME_PATTERN.match(csv_path.name)
        if match is None:
            continue

        time_value = float(match.group(1))

        try:
            df = read_eval_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")
            continue

        if metric not in df.columns:
            print(f"[WARN] '{metric}' not found in {csv_path.name}. Skipped.")
            continue

        for _, row in df.iterrows():
            category = str(row["category"])
            value = row[metric]
            records.append({
                "category": category,
                "time": time_value,
                metric: value,
            })

    if not records:
        raise ValueError(f"No valid data found for metric='{metric}' in folder='{folder}'")

    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(["category", "time"]).reset_index(drop=True)
    return result_df


def choose_xticks(existing_times):
    """
    データに存在する time をそのまま使う。
    多すぎる場合だけ間引く。
    """
    xs = sorted(set(float(x) for x in existing_times))

    if len(xs) <= 12:
        return xs

    # 多すぎるときは等間隔に間引く
    n_ticks = 10
    idxs = [round(i * (len(xs) - 1) / (n_ticks - 1)) for i in range(n_ticks)]
    ticks = [xs[i] for i in sorted(set(idxs))]
    return ticks


def format_tick_label(x: float) -> str:
    # 0.100000 のような表示を避ける
    return f"{x:g}"


def plot_metric_for_category(
    data_df: pd.DataFrame,
    metric: str,
    category: str,
    output_path: Path,
    xscale: str = "log",
    show: bool = False,
):
    category_df = data_df[data_df["category"].astype(str) == str(category)].copy()
    if category_df.empty:
        print(f"[WARN] category='{category}' has no data. Skipped.")
        return

    category_df = category_df.sort_values("time")
    x = category_df["time"].to_numpy()
    y = category_df[metric].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, y, marker="o", linewidth=1.8, markersize=5)

    ax.set_xlabel("time")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs time (category={category})")
    ax.grid(True, which="major", alpha=0.3)

    if xscale == "log":
        if (x <= 0).any():
            raise ValueError("xscale='log' requires all time values to be positive.")
        ax.set_xscale("log")
        ax.grid(True, which="minor", alpha=0.12)

    xticks = choose_xticks(x)
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_tick_label(t) for t in xticks])

    if len(x) >= 2:
        xmin, xmax = min(x), max(x)
        if xscale == "log":
            ax.set_xlim(xmin * 0.9, xmax * 1.1)
        else:
            margin = (xmax - xmin) * 0.05 if xmax > xmin else 0.1
            ax.set_xlim(xmin - margin, xmax + margin)
    else:
        if xscale == "log":
            ax.set_xlim(x[0] * 0.8, x[0] * 1.2)
        else:
            ax.set_xlim(x[0] - 0.1, x[0] + 0.1)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="eval_results_{time}.csv が入っているフォルダ",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=VALID_METRICS,
        help="プロットしたい指標名",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="average",
        help="対象 category 名。--all-categories 指定時は無視される",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="全カテゴリを個別にプロットして保存する",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="単一カテゴリ描画時の保存先ファイル名",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="全カテゴリ描画時の保存先ディレクトリ",
    )
    parser.add_argument(
        "--xscale",
        type=str,
        default="log",
        choices=["linear", "log"],
        help="x軸スケール",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存に加えて表示も行う",
    )

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    data_df = collect_metric_values(folder, args.metric)

    if args.all_categories:
        categories = sorted(data_df["category"].astype(str).unique())

        output_dir = Path(args.output_dir) if args.output_dir else folder / f"plots_{args.metric}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Found categories: {categories}")
        print(f"[INFO] Output directory: {output_dir}")

        for category in categories:
            filename = f"{args.metric}_{sanitize_filename(category)}.png"
            output_path = output_dir / filename
            plot_metric_for_category(
                data_df=data_df,
                metric=args.metric,
                category=category,
                output_path=output_path,
                xscale=args.xscale,
                show=args.show,
            )
    else:
        category = args.category
        available = set(data_df["category"].astype(str))
        if category not in available:
            raise ValueError(
                f"category='{category}' not found. Available categories: {sorted(available)}"
            )

        output_path = Path(args.output) if args.output else folder / f"{args.metric}_{sanitize_filename(category)}.png"

        plot_metric_for_category(
            data_df=data_df,
            metric=args.metric,
            category=category,
            output_path=output_path,
            xscale=args.xscale,
            show=args.show,
        )

if __name__ == "__main__":
    main()
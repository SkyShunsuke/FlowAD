#!/usr/bin/env python3
"""Plot evaluation metrics across s, optionally as differences from s=1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-08-18-explicit-dataset-colors-v8"
Mode = Literal["only-average", "no-average"]


@dataclass(frozen=True)
class PlotOptions:
    """Visual and aggregation options for :func:`plot_curves`."""

    median_color: str = "#0072B2"
    individual_colormap: str = "turbo"
    # Explicit per-dataset colors. Keys must match the dataset names in the
    # (dataset, category) MultiIndex; values are any Matplotlib color spec
    # ("#0072B2", "tab:orange", (r, g, b), ...). When provided, this takes
    # precedence over ``individual_colormap``. Missing datasets raise an
    # error so a silent color mix-up cannot occur.
    dataset_colors: Mapping[str, str] | None = None
    individual_linewidth: float = 1.0
    individual_alpha: float = 0.72
    individual_markersize: float = 3.0
    median_linewidth: float = 2.4
    median_markersize: float = 4.2
    iqr_alpha: float = 0.16
    reference_line_color: str = "0.30"
    reference_linewidth: float = 0.9
    reference_line_alpha: float = 0.90
    reference_linestyle: str = "-"
    figsize: tuple[float, float] = (3.35, 2.35)
    xscale: str = "linear"
    yscale: str = "linear"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    show_all_s_ticks: bool = True
    tick_fontsize: float = 8.0
    rasterize_background: bool = False

    # Compact dataset legend for combined paper figures. When the input uses
    # a (dataset, category) MultiIndex, all trajectories from the same dataset
    # share one color and the dataset appears only once in the legend.
    show_dataset_legend: bool = True
    dataset_display_names: Mapping[str, str] | None = None
    legend_loc: str = "lower left"
    legend_bbox_to_anchor: tuple[float, float] = (0.015, 0.020)
    legend_ncol: int = 2
    legend_fontsize: float = 6.4

    # Difference mode:
    #   False -> plot the original metric values.
    #   True  -> plot metric(s) - metric(diff_reference_s) per trajectory.
    diff: bool = False
    diff_reference_s: float = 1.0
    # Do not draw individual trajectories that are identically zero after
    # reference subtraction. They still participate in median/IQR statistics.
    hide_zero_diff_curves: bool = True
    # A trajectory that exceeds zero at least once is drawn less transparently
    # (with a larger Matplotlib alpha) to emphasize positive differences.
    positive_diff_alpha: float = 1.0


def _sample_curve_colors(
    colormap: str,
    count: int,
) -> np.ndarray:
    """Return deterministic RGBA colors for trajectories or datasets."""
    if count < 1:
        return np.empty((0, 4), dtype=float)

    try:
        cmap = mpl.colormaps[colormap]
    except KeyError as exc:
        available = ", ".join(sorted(mpl.colormaps))
        raise ValueError(
            f"Unknown individual_colormap={colormap!r}. "
            f"Available Matplotlib colormaps include: {available}"
        ) from exc

    # Qualitative (listed) colormaps such as "tab10" or "Set2" define a small
    # number of discrete entries. Use them verbatim in order, cycling if more
    # curves than entries are requested, instead of interpolating along the
    # continuous [0, 1] range. Interpolation would blend neighboring entries
    # and defeat the purpose of a categorical palette.
    listed = getattr(cmap, "colors", None)
    if listed is not None:
        if count > len(listed):
            warnings.warn(
                f"individual_colormap={colormap!r} provides only "
                f"{len(listed)} discrete colors but {count} are needed; "
                "colors will repeat. Consider dataset_colors for explicit "
                "assignment.",
                stacklevel=2,
            )
        return np.asarray(
            [mcolors.to_rgba(listed[i % len(listed)]) for i in range(count)],
            dtype=float,
        )

    # Continuous colormaps: avoid the extreme ends, which are often too dark
    # or too light for thin paper-figure trajectories.
    positions = (
        np.array([0.50])
        if count == 1
        else np.linspace(0.04, 0.96, count)
    )
    return np.asarray(cmap(positions), dtype=float)


def _resolve_curve_colors(
    unique_color_keys: Sequence[object],
    options: PlotOptions,
) -> dict[object, np.ndarray]:
    """Map each color key to an RGBA color.

    Explicit ``options.dataset_colors`` wins when provided; otherwise colors
    are sampled from ``options.individual_colormap``.
    """
    if options.dataset_colors is not None:
        missing = [
            key
            for key in unique_color_keys
            if str(key) not in options.dataset_colors
        ]
        if missing:
            configured = sorted(options.dataset_colors)
            raise ValueError(
                "dataset_colors is missing entries for: "
                f"{sorted(str(key) for key in missing)}. "
                f"Configured keys: {configured}"
            )
        try:
            return {
                key: np.asarray(
                    mcolors.to_rgba(options.dataset_colors[str(key)]),
                    dtype=float,
                )
                for key in unique_color_keys
            }
        except ValueError as exc:
            raise ValueError(
                f"dataset_colors contains an invalid color spec: {exc}"
            ) from exc

    sampled_colors = _sample_curve_colors(
        options.individual_colormap,
        len(unique_color_keys),
    )
    return dict(zip(unique_color_keys, sampled_colors, strict=True))


def _dataset_keys_from_index(index: pd.Index) -> np.ndarray | None:
    """Return one dataset key per row when a dataset index level is present."""
    if not isinstance(index, pd.MultiIndex):
        return None

    if "dataset" in index.names:
        dataset_values = index.get_level_values("dataset")
    elif index.nlevels >= 2:
        # ``load_curves`` creates (dataset, category), so retain compatibility
        # even if the caller removed the level names.
        dataset_values = index.get_level_values(0)
    else:
        return None

    return np.asarray([str(value) for value in dataset_values], dtype=object)


def load_curves(
    result_dir: str | Path,
    dataset_modes: Mapping[str, Mode],
    s_values: Sequence[int | float | str],
    metric: str,
    *,
    average_label: str = "average",
) -> pd.DataFrame:
    """Read result CSVs and return one row per selected trajectory.

    Files are read from::

        {result_dir}/{dataset}/eval_results_{s}.csv

    The returned DataFrame has a ``(dataset, category)`` MultiIndex and one
    column for each numeric s value.
    """
    root = Path(result_dir).expanduser()

    # The string token is retained for the filename. This permits entries such
    # as "0.0100", while float(token) is used for the numeric x coordinate.
    try:
        s_points = sorted((float(s), str(s)) for s in s_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Every s value must be convertible to float.") from exc

    if not s_points:
        raise ValueError("s_values is empty.")
    if len({x for x, _ in s_points}) != len(s_points):
        raise ValueError("s_values contains duplicate numeric values.")

    records: list[dict[str, object]] = []

    for dataset, mode in dataset_modes.items():
        if mode not in {"only-average", "no-average"}:
            raise ValueError(f"{dataset}: invalid mode {mode!r}")

        for x, token in s_points:
            csv_path = root / dataset / f"eval_results_{token}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)

            df = pd.read_csv(csv_path)
            required = {"category", metric}
            missing_columns = required.difference(df.columns)
            if missing_columns:
                raise ValueError(
                    f"{csv_path}: missing column(s): "
                    f"{sorted(missing_columns)}"
                )

            df = df[["category", metric]].copy()
            df["category"] = df["category"].astype(str).str.strip()

            if df["category"].duplicated().any():
                duplicates = sorted(
                    df.loc[
                        df["category"].duplicated(keep=False), "category"
                    ].unique()
                )
                raise ValueError(
                    f"{csv_path}: duplicate categories: {duplicates}"
                )

            is_average = df["category"].eq(average_label)
            if int(is_average.sum()) != 1:
                raise ValueError(
                    f"{csv_path}: expected exactly one {average_label!r} "
                    f"row, found {int(is_average.sum())}."
                )

            selected = (
                df.loc[is_average]
                if mode == "only-average"
                else df.loc[~is_average]
            )
            values = pd.to_numeric(selected[metric], errors="coerce")
            if values.isna().any():
                bad_categories = selected.loc[
                    values.isna(), "category"
                ].tolist()
                raise ValueError(
                    f"{csv_path}: invalid {metric!r} value for "
                    f"{bad_categories}"
                )

            for category, value in zip(selected["category"], values):
                records.append(
                    {
                        "dataset": dataset,
                        "category": category,
                        "s": x,
                        "value": float(value),
                    }
                )

    if not records:
        raise ValueError("No result rows were loaded.")

    long_df = pd.DataFrame.from_records(records)
    curves = long_df.pivot(
        index=["dataset", "category"],
        columns="s",
        values="value",
    )
    x_order = [x for x, _ in s_points]
    return curves.reindex(columns=x_order).sort_index()


def difference_from_reference(
    curves: pd.DataFrame,
    reference_s: int | float = 1,
) -> pd.DataFrame:
    """Compute per-trajectory absolute performance differences.

    For every selected ``(dataset, category)`` trajectory, this computes

    ``diff(s) = metric(s) - metric(reference_s)``.

    Subtraction is performed *before* the pointwise median and IQR are
    calculated. Thus, this is not ``median(s) - median(reference_s)``.
    """
    if curves.empty:
        raise ValueError("curves is empty.")

    try:
        x = curves.columns.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "All curve columns must be numeric s values to compute "
            "differences."
        ) from exc

    reference = float(reference_s)
    matches = np.flatnonzero(
        np.isclose(x, reference, rtol=1e-9, atol=1e-12)
    )
    if len(matches) != 1:
        available = ", ".join(f"{value:g}" for value in x)
        raise ValueError(
            f"Diff mode requires exactly one s={reference:g} column; "
            f"available s values: [{available}]"
        )

    reference_column = curves.columns[matches[0]]
    reference_values = curves[reference_column]
    if reference_values.isna().any():
        missing_rows = reference_values.index[reference_values.isna()]
        preview = ", ".join(map(str, missing_rows[:5]))
        suffix = " ..." if len(missing_rows) > 5 else ""
        raise ValueError(
            f"Cannot compute differences because s={reference:g} is "
            f"missing for {len(missing_rows)} trajectory/trajectories: "
            f"{preview}{suffix}"
        )

    differences = curves.sub(reference_values, axis=0)

    # Avoid signed zero or tiny floating-point residuals at the reference.
    differences.loc[:, reference_column] = 0.0
    return differences


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Return pointwise Q1, median, Q3, and the number of valid curves."""
    return pd.DataFrame(
        {
            "q1": curves.quantile(0.25, axis=0),
            "median": curves.quantile(0.50, axis=0),
            "q3": curves.quantile(0.75, axis=0),
            "n": curves.notna().sum(axis=0).astype(int),
        }
    ).rename_axis("s")


def _light_tick_font(size: float, strict: bool) -> FontProperties:
    """Find a requested Light-weight font in priority order."""
    candidates = [
        ("Helvetica Neue Light", "light"),
        ("Helvetica Light", "light"),
        ("Arial Light", "light"),
        ("Helvetica Neue", "light"),
        ("Helvetica", "light"),
        ("Arial", "light"),
    ]
    for family, weight in candidates:
        prop = FontProperties(family=family, weight=weight, size=size)
        try:
            font_manager.findfont(prop, fallback_to_default=False)
            return prop
        except ValueError:
            continue

    message = (
        "Helvetica Neue Light / Helvetica Light / Arial Light was not found."
    )
    if strict:
        raise RuntimeError(message)

    warnings.warn(
        message + " Falling back to DejaVu Sans ExtraLight.",
        stacklevel=2,
    )
    return FontProperties(
        family="DejaVu Sans",
        weight="ultralight",
        size=size,
    )


def plot_curves(
    curves: pd.DataFrame,
    output_path: str | Path,
    *,
    options: PlotOptions = PlotOptions(),
    strict_font: bool = False,
    scale: float = 1.0,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot individual curves, the pointwise median, and the IQR.

    With ``options.diff=True``, each trajectory is first converted to
    ``metric(s) - metric(options.diff_reference_s)``. The returned statistics
    and saved summary therefore also contain differences rather than original
    metric values.
    """
    if curves.empty:
        raise ValueError("curves is empty.")

    plotted_curves = (
        difference_from_reference(curves, options.diff_reference_s)
        if options.diff
        else curves.copy()
    )
    x = plotted_curves.columns.to_numpy(dtype=float)
    unscaled_y = plotted_curves.to_numpy(dtype=float)
    y = unscaled_y * scale

    if options.xscale == "log" and np.any(x <= 0):
        raise ValueError("A logarithmic x-axis requires s > 0.")

    if options.diff and options.yscale == "log":
        raise ValueError(
            "diff=True is incompatible with yscale='log' because the "
            "reference difference is zero and differences may be negative. "
            "Use yscale='linear' or 'symlog'."
        )

    finite_y = y[np.isfinite(y)]
    if options.yscale == "log" and np.any(finite_y <= 0):
        raise ValueError("A logarithmic y-axis requires metric > 0.")

    if not 0.0 <= options.individual_alpha <= 1.0:
        raise ValueError("individual_alpha must be between 0 and 1.")
    if not 0.0 <= options.positive_diff_alpha <= 1.0:
        raise ValueError("positive_diff_alpha must be between 0 and 1.")
    if not 0.0 <= options.iqr_alpha <= 1.0:
        raise ValueError("iqr_alpha must be between 0 and 1.")
    if options.individual_linewidth <= 0:
        raise ValueError("individual_linewidth must be positive.")
    if options.individual_markersize < 0:
        raise ValueError("individual_markersize cannot be negative.")
    if options.median_linewidth <= 0:
        raise ValueError("median_linewidth must be positive.")
    if options.median_markersize < 0:
        raise ValueError("median_markersize cannot be negative.")
    if options.reference_linewidth <= 0:
        raise ValueError("reference_linewidth must be positive.")
    if not 0.0 <= options.reference_line_alpha <= 1.0:
        raise ValueError("reference_line_alpha must be between 0 and 1.")
    if options.legend_ncol < 1:
        raise ValueError("legend_ncol must be at least 1.")
    if options.legend_fontsize <= 0:
        raise ValueError("legend_fontsize must be positive.")
    if len(options.legend_bbox_to_anchor) != 2:
        raise ValueError("legend_bbox_to_anchor must contain exactly two values.")

    summary = summarize_curves(plotted_curves)

    # In diff mode, a trajectory can be exactly zero for every sampled s.
    # Such a curve simply lies on top of the y=0 reference line and adds a
    # colored horizontal line/markers without conveying additional structure.
    # Hide it visually while retaining it in the median/IQR calculation.
    visible_mask = np.ones(y.shape[0], dtype=bool)
    if options.diff and options.hide_zero_diff_curves:
        for i, row in enumerate(unscaled_y):
            finite = row[np.isfinite(row)]
            if finite.size > 0 and np.all(
                np.isclose(finite, 0.0, rtol=0.0, atol=1e-12)
            ):
                visible_mask[i] = False

    visible_y = y[visible_mask]
    visible_unscaled_y = unscaled_y[visible_mask]

    # For the combined plot, color is a dataset-level visual encoding. This
    # keeps the legend unambiguous when one dataset contributes multiple
    # category trajectories. Inputs without a dataset MultiIndex retain the
    # original one-color-per-trajectory behavior.
    dataset_keys = _dataset_keys_from_index(plotted_curves.index)
    if dataset_keys is None:
        color_keys: np.ndarray = np.arange(len(plotted_curves), dtype=int)
    else:
        color_keys = dataset_keys

    unique_color_keys = list(dict.fromkeys(color_keys.tolist()))
    color_by_key = _resolve_curve_colors(unique_color_keys, options)
    all_curve_colors = np.asarray(
        [color_by_key[key] for key in color_keys],
        dtype=float,
    )
    curve_colors = all_curve_colors[visible_mask]
    visible_dataset_keys = (
        dataset_keys[visible_mask] if dataset_keys is not None else None
    )

    # In difference mode, emphasize an entire trajectory when at least one
    # sampled point has a strictly positive difference. The decision is made
    # from unscaled differences so it is unaffected by the display scale.
    curve_alphas = np.full(
        visible_y.shape[0],
        options.individual_alpha,
        dtype=float,
    )
    if options.diff and visible_unscaled_y.shape[0] > 0:
        has_positive_difference = np.any(
            np.isfinite(visible_unscaled_y) & (visible_unscaled_y > 0.0),
            axis=1,
        )
        curve_alphas[has_positive_difference] = options.positive_diff_alpha

    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, ax = plt.subplots(
            figsize=options.figsize,
            constrained_layout=True,
        )

        ax.grid(
            axis="y",
            color="0.90",
            linewidth=0.55,
            alpha=0.8,
            zorder=0,
        )

        q1 = summary["q1"].to_numpy(dtype=float)
        median = summary["median"].to_numpy(dtype=float)
        q3 = summary["q3"].to_numpy(dtype=float)
        valid_iqr = np.isfinite(q1) & np.isfinite(q3)

        ax.fill_between(
            x,
            q1,
            q3,
            where=valid_iqr,
            interpolate=True,
            color=options.median_color,
            alpha=options.iqr_alpha,
            linewidth=0,
            zorder=1,
        )

        for row, color, alpha in zip(
            visible_y, curve_colors, curve_alphas, strict=True
        ):
            # Draw the line below the y=0 reference so crossings do not hide
            # the reference line. Markers are drawn separately above it.
            ax.plot(
                x,
                row,
                color=color,
                alpha=float(alpha),
                linewidth=options.individual_linewidth,
                solid_capstyle="round",
                solid_joinstyle="round",
                rasterized=options.rasterize_background,
                zorder=2,
            )

        ax.set_xscale(options.xscale)
        ax.set_yscale(options.yscale)

        if options.xlim is not None:
            ax.set_xlim(*options.xlim)
        else:
            ax.margins(x=0.025)

        if options.ylim is not None:
            ax.set_ylim(*options.ylim)

        # Zero means "same performance as s=reference_s" in diff mode.
        # This is intentionally added after xscale/xlim are finalized.
        # axhline uses axes coordinates in x, hence xmin=0 and xmax=1 cover
        # the entire plotting width, not merely min(s) through max(s).
        if options.diff:
            reference_line = ax.axhline(
                0.0,
                xmin=0.0,
                xmax=1.0,
                color=options.reference_line_color,
                alpha=options.reference_line_alpha,
                linewidth=options.reference_linewidth,
                linestyle=options.reference_linestyle,
                solid_capstyle="butt",
                clip_on=False,
                zorder=3,
            )
            reference_line.set_gid("diff-reference-y0")

        # Same-color circles show every sampled s point. They are separated
        # from the trajectory lines so they can remain visible above the
        # reference line without allowing the lines to obscure that line.
        if options.individual_markersize > 0:
            for row, color, alpha in zip(
                visible_y, curve_colors, curve_alphas, strict=True
            ):
                ax.plot(
                    x,
                    row,
                    linestyle="None",
                    marker="o",
                    markersize=options.individual_markersize,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.0,
                    alpha=float(alpha),
                    rasterized=options.rasterize_background,
                    zorder=3.5,
                )

        # ax.plot(
        #     x,
        #     median,
        #     color=options.median_color,
        #     linewidth=options.median_linewidth,
        #     marker="o" if options.median_markersize > 0 else None,
        #     markersize=options.median_markersize,
        #     markerfacecolor=options.median_color,
        #     markeredgecolor=options.median_color,
        #     markeredgewidth=0.0,
        #     solid_capstyle="round",
        #     solid_joinstyle="round",
        #     zorder=4,
        # )

        if options.show_all_s_ticks:
            ax.set_xticks(x)
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:g}")
            )
            if options.xscale == "log":
                ax.xaxis.set_minor_formatter(NullFormatter())

        # Axis labels and title are intentionally omitted for compact paper
        # figures; the dataset legend is added below after font resolution.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("0.22")
            ax.spines[side].set_linewidth(0.8)

        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            length=3.2,
            width=0.7,
            colors="0.18",
            pad=2.5,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            length=2.0,
            width=0.7,
            colors="0.18",
        )

        tick_font = _light_tick_font(
            options.tick_fontsize,
            strict=strict_font,
        )
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontproperties(tick_font)

        if (
            options.show_dataset_legend
            and visible_dataset_keys is not None
            and visible_dataset_keys.size > 0
        ):
            visible_datasets = list(
                dict.fromkeys(visible_dataset_keys.tolist())
            )
            display_names = options.dataset_display_names or {}
            legend_handles: list[Line2D] = []

            for dataset in visible_datasets:
                color = color_by_key[dataset]
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=color,
                        linewidth=max(1.15, options.individual_linewidth),
                        marker=(
                            "o" if options.individual_markersize > 0 else None
                        ),
                        markersize=max(3.2, 0.72 * options.individual_markersize),
                        markerfacecolor=color,
                        markeredgecolor=color,
                        markeredgewidth=0.0,
                        label=display_names.get(dataset, dataset),
                    )
                )

            legend_font = tick_font.copy()
            legend_font.set_size(options.legend_fontsize)
            legend_font.set_weight("normal")
            legend = ax.legend(
                handles=legend_handles,
                loc=options.legend_loc,
                bbox_to_anchor=options.legend_bbox_to_anchor,
                bbox_transform=ax.transAxes,
                ncol=min(options.legend_ncol, len(legend_handles)),
                prop=legend_font,
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.94,
                facecolor="white",
                edgecolor="0.78",
                borderpad=0.28,
                labelspacing=0.26,
                handlelength=1.35,
                handletextpad=0.42,
                columnspacing=0.80,
                borderaxespad=0.0,
            )
            legend.get_frame().set_linewidth(0.55)
            legend.set_zorder(10)
            for text_artist in legend.get_texts():
                text_artist.set_color("0.16")

        # Draw once after the legend is present so constrained_layout accounts
        # for every paper-figure artist before export.
        fig.canvas.draw()

        output_path = Path(output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.02,
        )

    return fig, ax, summary


def _parse_cli_args() -> argparse.Namespace:
    """Parse optional overrides for difference plotting."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot configured evaluation curves. --diff subtracts the "
            "reference-s value separately from every trajectory."
        )
    )
    parser.add_argument(
        "--diff",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable difference mode. Equivalent to changing "
            "PlotOptions.diff in the settings block."
        ),
    )
    parser.add_argument(
        "--diff-reference-s",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Reference s for difference mode. Supplying this option also "
            "enables difference mode unless --no-diff is supplied."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    return parser.parse_args()


def _apply_cli_overrides(
    options: PlotOptions,
    args: argparse.Namespace,
) -> PlotOptions:
    """Apply only the CLI values explicitly supplied by the user."""
    if args.diff is False and args.diff_reference_s is not None:
        raise ValueError(
            "--no-diff and --diff-reference-s cannot be used together."
        )

    diff_enabled = options.diff if args.diff is None else bool(args.diff)
    reference_s = (
        options.diff_reference_s
        if args.diff_reference_s is None
        else float(args.diff_reference_s)
    )

    if args.diff_reference_s is not None and args.diff is None:
        diff_enabled = True

    return replace(
        options,
        diff=diff_enabled,
        diff_reference_s=reference_s,
    )

if __name__ == "__main__":
    # ================================================================
    # Settings
    # ================================================================
    RESULT_DIR = Path("/home/acg17237wy/projects/FlowAD/artifacts/design_ablations/rawdata/framework")
    SAVE_DIR = Path("/home/acg17237wy/projects/FlowAD/artifacts/design_ablations/output/framework")

    DATASET_MODES: dict[str, Mode] = {
        "ddim": "only-average",
        "fm": "only-average",
        "edm": "only-average",
    }
    # Publication-facing labels. Keys must match DATASET_MODES; values are the
    # concise names displayed in the lower-left legend.
    DATASET_DISPLAY_NAMES = {
        "fm": "Rectified Flow",
        "ddim": "DDIM", 
        "edm": "EDM",
    }

    # Explicit per-backbone colors (Okabe-Ito palette: categorical,
    # colorblind-safe, and print-friendly). Keys must match DATASET_MODES.
    # Any Matplotlib color spec works, e.g. "tab:orange" or (0.2, 0.4, 0.6).
    DATASET_COLORS = {
        "ddim": "#0072B2",   # blue
        "fm": "#E69F00",    # orange
        "edm": "#009E73",    # green
    }

    S_VALUES = [1, 2, 4, 8, 20, 40, 80, 200]
    METRIC = "mad"

    BASE_OPTIONS = PlotOptions(
        median_color="#FF0000",  # "red", "orange", "#0072B2", etc.
        # Fallback palette used only when dataset_colors is None.
        individual_colormap="tab10",
        # Explicit per-curve (per-backbone) colors. Set to None to fall back
        # to individual_colormap sampling.
        dataset_colors=DATASET_COLORS,
        individual_linewidth=1.0,
        individual_alpha=1.0,         # normal curves; larger alpha = less transparent
        positive_diff_alpha=1.0,      # curves with any diff > 0 are emphasized
        individual_markersize=5.0,
        median_linewidth=2.4,
        median_markersize=3.0,
        iqr_alpha=0.16,
        reference_line_color="0.30",
        reference_linewidth=0.9,
        reference_line_alpha=0.90,
        reference_linestyle="-",
        xscale="log",         # "linear" or "log"
        yscale="linear",      # use "linear" or "symlog" in diff mode
        xlim=None,             # e.g. (1, 32)
        # Examples: original metric=(0.80, 1.00), diff=(-0.05, 0.05).
        ylim=None,
        show_all_s_ticks=True,

        # ---------------------- DATASET LEGEND ----------------------
        show_dataset_legend=True,
        dataset_display_names=DATASET_DISPLAY_NAMES,
        legend_loc="lower left",
        legend_bbox_to_anchor=(0.015, 0.020),
        legend_ncol=2,
        legend_fontsize=6.4,
        # ------------------------------------------------------------

        # ------------------------ DIFF MODE -------------------------
        # Set True to plot metric(s) - metric(s=1) for every curve.
        diff=True,
        diff_reference_s=1.0,
        hide_zero_diff_curves=True,  # hide curves that are exactly zero after reference subtraction
        # ------------------------------------------------------------
    )
    cli_args = _parse_cli_args()
    OPTIONS = _apply_cli_overrides(BASE_OPTIONS, cli_args)

    curves = load_curves(
        result_dir=RESULT_DIR,
        dataset_modes=DATASET_MODES,
        s_values=S_VALUES,
        metric=METRIC,
    )

    # Combined plot across all selected trajectories.
    fig, ax, summary = plot_curves(
        curves,
        SAVE_DIR / f"combined_{METRIC}.pdf",
        options=OPTIONS,
        strict_font=False,
        scale=100
    )
    summary.to_csv(
        SAVE_DIR / f"combined_{METRIC}_summary.csv"
    )
    plt.close(fig)

    # Optional: one plot per dataset.
    # for dataset in DATASET_MODES:
    #     dataset_curves = curves.xs(dataset, level="dataset")
    #     fig, _, _ = plot_curves(
    #         dataset_curves,
    #         SAVE_DIR / "per_dataset" / f"{dataset}_{METRIC}.pdf",
    #         options=OPTIONS,
    #         strict_font=False,
    #     )
    #     plt.close(fig)

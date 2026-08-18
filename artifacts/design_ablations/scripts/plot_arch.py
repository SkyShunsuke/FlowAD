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
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd


SCRIPT_VERSION = "2026-08-17-architecture-styles-v8"
Mode = Literal["only-average", "no-average"]


@dataclass(frozen=True)
class PlotOptions:
    """Visual and aggregation options for :func:`plot_curves`."""

    median_color: str = "#0072B2"
    individual_colormap: str = "turbo"
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
    # Explicit dataset-level styles override the fallback colormap and marker
    # size. This permits related architectures to use one hue family while
    # retaining a perceptually ordered encoding such as marker size.
    dataset_colors: Mapping[str, str] | None = None
    dataset_markersizes: Mapping[str, float] | None = None
    dataset_legend_order: Sequence[str] | None = None
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

    # Avoid the extreme ends of continuous colormaps, which are often too
    # dark or too light for thin paper-figure trajectories.
    positions = (
        np.array([0.50])
        if count == 1
        else np.linspace(0.04, 0.96, count)
    )
    return np.asarray(cmap(positions), dtype=float)


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


def _resolve_curve_colors(
    color_keys: np.ndarray,
    *,
    colormap: str,
    dataset_colors: Mapping[str, str] | None,
) -> np.ndarray:
    """Resolve one RGBA color per curve with optional dataset overrides."""
    unique_keys = list(dict.fromkeys(color_keys.tolist()))
    configured_colors = dataset_colors or {}

    color_by_key: dict[object, np.ndarray] = {}
    fallback_keys: list[object] = []

    for key in unique_keys:
        lookup_key = str(key)
        if lookup_key not in configured_colors:
            fallback_keys.append(key)
            continue

        configured_color = configured_colors[lookup_key]
        try:
            rgba = mpl.colors.to_rgba(configured_color)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid dataset color for {lookup_key!r}: "
                f"{configured_color!r}."
            ) from exc
        color_by_key[key] = np.asarray(rgba, dtype=float)

    if fallback_keys:
        sampled_colors = _sample_curve_colors(colormap, len(fallback_keys))
        color_by_key.update(
            zip(fallback_keys, sampled_colors, strict=True)
        )

    return np.asarray(
        [color_by_key[key] for key in color_keys],
        dtype=float,
    )


def _resolve_curve_markersizes(
    dataset_keys: np.ndarray | None,
    curve_count: int,
    *,
    default_markersize: float,
    dataset_markersizes: Mapping[str, float] | None,
) -> np.ndarray:
    """Resolve one marker diameter in points per curve."""
    configured_sizes = dataset_markersizes or {}
    normalized_sizes: dict[str, float] = {}

    for dataset, value in configured_sizes.items():
        try:
            size = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid marker size for {dataset!r}: {value!r}."
            ) from exc
        if not np.isfinite(size) or size < 0:
            raise ValueError(
                f"Marker size for {dataset!r} must be finite and "
                f"non-negative; got {value!r}."
            )
        normalized_sizes[str(dataset)] = size

    if dataset_keys is None:
        return np.full(curve_count, default_markersize, dtype=float)

    return np.asarray(
        [
            normalized_sizes.get(str(dataset), default_markersize)
            for dataset in dataset_keys
        ],
        dtype=float,
    )


def _ordered_visible_datasets(
    visible_dataset_keys: np.ndarray,
    preferred_order: Sequence[str] | None,
) -> list[str]:
    """Return unique visible datasets, optionally in a configured order."""
    observed = list(dict.fromkeys(map(str, visible_dataset_keys.tolist())))
    if preferred_order is None:
        return observed

    preferred = [str(dataset) for dataset in preferred_order]
    if len(preferred) != len(set(preferred)):
        raise ValueError("dataset_legend_order contains duplicate entries.")

    observed_set = set(observed)
    ordered = [dataset for dataset in preferred if dataset in observed_set]
    ordered.extend(dataset for dataset in observed if dataset not in preferred)
    return ordered


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
    if (
        not np.isfinite(options.individual_markersize)
        or options.individual_markersize < 0
    ):
        raise ValueError(
            "individual_markersize must be finite and non-negative."
        )
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

    all_curve_colors = _resolve_curve_colors(
        color_keys,
        colormap=options.individual_colormap,
        dataset_colors=(
            options.dataset_colors if dataset_keys is not None else None
        ),
    )
    all_curve_markersizes = _resolve_curve_markersizes(
        dataset_keys,
        len(plotted_curves),
        default_markersize=options.individual_markersize,
        dataset_markersizes=options.dataset_markersizes,
    )
    curve_colors = all_curve_colors[visible_mask]
    curve_markersizes = all_curve_markersizes[visible_mask]
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

        # Same-color circles show every sampled s point. Dataset-specific
        # marker diameters can encode model scale. Larger markers are drawn
        # first so smaller-model markers remain visible at overlapping points.
        if np.any(curve_markersizes > 0):
            marker_draw_order = np.argsort(curve_markersizes)[::-1]
            for position in marker_draw_order:
                markersize = float(curve_markersizes[position])
                if markersize <= 0:
                    continue

                row = visible_y[position]
                color = curve_colors[position]
                alpha = float(curve_alphas[position])
                ax.plot(
                    x,
                    row,
                    linestyle="None",
                    marker="o",
                    markersize=markersize,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.0,
                    alpha=alpha,
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
            visible_datasets = _ordered_visible_datasets(
                visible_dataset_keys,
                options.dataset_legend_order,
            )
            display_names = options.dataset_display_names or {}
            legend_handles: list[Line2D] = []

            for dataset in visible_datasets:
                first_position = int(
                    np.flatnonzero(visible_dataset_keys == dataset)[0]
                )
                color = curve_colors[first_position]
                markersize = float(curve_markersizes[first_position])
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        color=color,
                        linewidth=max(1.15, options.individual_linewidth),
                        marker="o" if markersize > 0 else None,
                        markersize=markersize,
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
    RESULT_DIR = Path("/home/haselab/projects/sakai_ssd/FlowAD/artifacts/design_ablations/rawdata/arch")
    SAVE_DIR = Path("/home/haselab/projects/sakai_ssd/FlowAD/artifacts/design_ablations/output/arch")

    DATASET_MODES: dict[str, Mode] = {
        "dit_base": "only-average",
        "dit_large": "only-average",
        "dit_huge": "only-average",
        # "dit_gigant": "only-average",
        "mlp": "only-average",
        "unet": "only-average",
    }

    # Publication-facing labels. Keys must match DATASET_MODES; values are the
    # concise names displayed in the lower-left legend.
    DATASET_DISPLAY_NAMES = {
        "dit_base": "DiT-B",
        "dit_large": "DiT-L",
        "dit_huge": "DiT-H",
        # "dit_gigant": "DiT-G",
        "mlp": "MLP",
        "unet": "U-Net",
    }

    # Architecture-level color encoding:
    #   - all DiT variants use ordered shades from one blue family;
    #   - MLP and U-Net use independent vermillion and green families.
    # The same color is used for each architecture's curve and markers.
    DATASET_COLORS = {
        "dit_base": "#6BAED6",
        "dit_large": "#4292C6",
        "dit_huge": "#2171B5",
        # "dit_gigant": "#084594",
        "mlp": "#D55E00",
        "unet": "#009E73",
    }

    # Marker diameter in points. DiT marker size grows monotonically with
    # model scale while remaining compact enough for the 3.35 x 2.35 in plot.
    DATASET_MARKERSIZES = {
        "dit_base": 4,
        "dit_large": 6,
        "dit_huge": 8,
        # "dit_gigant": 8,
        "mlp": 6,
        "unet": 6,
    }

    DATASET_LEGEND_ORDER = (
        "dit_base",
        "dit_large",
        "dit_huge",
        # "dit_gigant",
        "mlp",
        "unet",
    )

    S_VALUES = [1, 2, 4, 8, 20, 40, 80, 200]
    METRIC = "mad"
    
    BASE_OPTIONS = PlotOptions(
        median_color="#FF0000",  # "red", "orange", "#0072B2", etc.
        # Fallback for any dataset omitted from DATASET_COLORS.
        individual_colormap="tab10",
        individual_linewidth=1.0,
        individual_alpha=1.0,         # normal curves; larger alpha = less transparent
        positive_diff_alpha=1.0,      # curves with any diff > 0 are emphasized
        # Fallback for any dataset omitted from DATASET_MARKERSIZES.
        individual_markersize=4.6,
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
        dataset_colors=DATASET_COLORS,
        dataset_markersizes=DATASET_MARKERSIZES,
        dataset_legend_order=DATASET_LEGEND_ORDER,
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
    #     # Keep the dataset level so its configured color and marker size are
    #     # retained in the per-dataset figure.
    #     dataset_curves = curves.xs(
    #         dataset,
    #         level="dataset",
    #         drop_level=False,
    #     )
    #     fig, _, _ = plot_curves(
    #         dataset_curves,
    #         SAVE_DIR / "per_dataset" / f"{dataset}_{METRIC}.pdf",
    #         options=OPTIONS,
    #         strict_font=False,
    #     )
    #     plt.close(fig)

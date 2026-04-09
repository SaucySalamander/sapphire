#!/usr/bin/env python3
"""Analyze molding telemetry JSONL files and generate plots.

Run this script with the repository virtual environment when possible:
    .venv/bin/python scripts/analyze_molding.py metrics.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CATEGORY_COLORS = {
    "p_neg1": "#b23a48",
    "p_zero": "#2a9d8f",
    "p_pos1": "#f4a261",
}
RUN_COLORS = ["#1d3557", "#6c5ce7"]
SERIES_LINE_STYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
DEFAULT_OUTPUT_DIR_NAME = "molding_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn molding telemetry JSONL into convergence, sparsity, and efficiency plots.",
    )
    parser.add_argument(
        "metrics_jsonl",
        type=Path,
        help="Primary telemetry JSONL file.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional second telemetry JSONL file to overlay against the primary run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots will be written. Defaults to a sibling analysis directory next to the primary JSONL.",
    )
    return parser.parse_args()


def load_metrics(path: Path, run_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    frame = pd.read_json(path, lines=True)
    if frame.empty:
        raise ValueError(f"Telemetry file is empty: {path}")

    frame = frame.copy()
    frame["run_label"] = run_label
    frame["source_path"] = str(path)

    required_columns = ["layer_idx", "step_idx", "mse_loss", "p_neg1", "p_zero", "p_pos1", "io_ms", "compute_ms"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Telemetry file {path} is missing required columns: {', '.join(missing)}")

    numeric_columns = required_columns + [
        "resume_step_idx",
        "tps",
        "grad_norm",
        "raw_grad_norm",
        "clipped_grad_norm",
        "clip_scale",
        "latent_saturation",
        "hessian_proxy_mean",
        "hessian_proxy_max",
        "hessian_proxy_source",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "raw_grad_norm" not in frame.columns:
        frame["raw_grad_norm"] = pd.to_numeric(frame.get("grad_norm"), errors="coerce")
    if "clipped_grad_norm" not in frame.columns:
        frame["clipped_grad_norm"] = pd.to_numeric(frame.get("grad_norm"), errors="coerce")
    if "clip_scale" not in frame.columns:
        frame["clip_scale"] = 1.0
    if "latent_saturation" not in frame.columns:
        frame["latent_saturation"] = float("nan")
    if "hessian_proxy_mean" not in frame.columns:
        frame["hessian_proxy_mean"] = float("nan")
    if "hessian_proxy_max" not in frame.columns:
        frame["hessian_proxy_max"] = float("nan")
    if "hessian_proxy_source" not in frame.columns:
        frame["hessian_proxy_source"] = float("nan")

    if "resume_step_idx" not in frame.columns:
        frame["resume_step_idx"] = 0.0
    frame["resume_step_idx"] = frame["resume_step_idx"].fillna(0.0)

    if "config_hash" not in frame.columns:
        frame["config_hash"] = "unknown"
    else:
        frame["config_hash"] = frame["config_hash"].astype("string").fillna("unknown")

    frame["series_key"] = frame["config_hash"]
    frame.loc[frame["series_key"] == "unknown", "series_key"] = frame.loc[frame["series_key"] == "unknown", "run_label"]
    frame["series_label"] = frame["run_label"]
    known_hash = frame["config_hash"] != "unknown"
    frame.loc[known_hash, "series_label"] = (
        frame.loc[known_hash, "run_label"] + " [" + frame.loc[known_hash, "config_hash"].astype(str) + "]"
    )

    frame["analysis_step"] = frame["resume_step_idx"] + frame["step_idx"]

    if "tps" in frame.columns:
        tps_series = frame["tps"].copy()
    else:
        tps_series = pd.Series(float("nan"), index=frame.index)

    derived_tps = pd.Series(float("nan"), index=frame.index)
    valid_compute = frame["compute_ms"].notna() & (frame["compute_ms"] > 0)
    derived_tps.loc[valid_compute] = 1000.0 / frame.loc[valid_compute, "compute_ms"]
    frame["tps_value"] = tps_series.fillna(derived_tps).fillna(0.0)

    frame = frame.sort_values(["analysis_step", "layer_idx", "step_idx"]).reset_index(drop=True)
    return frame


def build_output_dir(primary_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        resolved = output_dir
    else:
        resolved = primary_path.parent / DEFAULT_OUTPUT_DIR_NAME / primary_path.stem
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def unique_layers(*frames: pd.DataFrame) -> list[int]:
    layers: set[int] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        layers.update(int(layer_idx) for layer_idx in frame["layer_idx"].dropna().astype(int).tolist())
    return sorted(layers)


def iter_series(frame: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    if frame is None or frame.empty:
        return []

    series: list[tuple[str, str, pd.DataFrame]] = []
    for _, subset in frame.groupby("series_key", sort=False):
        grouped = subset.copy()
        series_key = str(grouped.iloc[0]["series_key"])
        series_label = str(grouped.iloc[0]["series_label"])
        series.append((series_key, series_label, grouped))
    return series


def build_series_style_map(*frames: pd.DataFrame) -> dict[str, str | tuple[float, tuple[float, ...]]]:
    style_map: dict[str, str | tuple[float, tuple[float, ...]]] = {}
    for frame in frames:
        for series_key, _, _ in iter_series(frame):
            if series_key in style_map:
                continue
            style_map[series_key] = SERIES_LINE_STYLES[len(style_map) % len(SERIES_LINE_STYLES)]
    return style_map


def grouped_mean(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected = frame[["analysis_step", *columns]].dropna(subset=["analysis_step"]).copy()
    if selected.empty:
        return pd.DataFrame(columns=list(columns))
    return selected.groupby("analysis_step", as_index=True)[list(columns)].mean().sort_index()


def normalize_probability_columns(grouped: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    normalized = grouped.copy()
    columns = list(columns)
    if normalized.empty:
        return normalized

    totals = normalized[columns].sum(axis=1)
    needs_normalization = totals > 1.0 + 1e-6
    if needs_normalization.any():
        normalized.loc[needs_normalization, columns] = normalized.loc[needs_normalization, columns].div(
            totals[needs_normalization], axis=0
        )
    return normalized


def single_point_bar_width(x_value: float) -> float:
    if not pd.notna(x_value):
        return 1.0
    magnitude = abs(float(x_value))
    if magnitude <= 0.0:
        return 1.0
    return max(1.0, magnitude * 0.02)


def plot_convergence(primary: pd.DataFrame, compare: pd.DataFrame | None, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    layers = unique_layers(primary, compare) if compare is not None else unique_layers(primary)
    series_styles = build_series_style_map(primary, compare) if compare is not None else build_series_style_map(primary)

    for frame, alpha in ((primary, 0.92), (compare, 0.65)) if compare is not None else ((primary, 0.92),):
        if frame is None or frame.empty:
            continue
        for series_key, series_label, series_frame in iter_series(frame):
            series_style = series_styles.get(series_key, "-")
            for index, layer_idx in enumerate(layers):
                color = plt.get_cmap("tab20")(index % 20)
                layer_frame = series_frame[series_frame["layer_idx"] == layer_idx].dropna(subset=["analysis_step", "mse_loss"])
                if layer_frame.empty:
                    continue
                layer_frame = layer_frame.sort_values("analysis_step")
                ax.plot(
                    layer_frame["analysis_step"],
                    layer_frame["mse_loss"],
                    color=color,
                    linestyle=series_style,
                    linewidth=2.0 if frame is primary else 1.8,
                    alpha=alpha,
                    marker="o",
                    markersize=5,
                    label=f"{series_label} layer {layer_idx}",
                )

    ax.set_title("Convergence: MSE Loss vs Step")
    ax.set_xlabel("analysis step")
    ax.set_ylabel("mse_loss")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sparsity(primary: pd.DataFrame, compare: pd.DataFrame | None, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    series_styles = build_series_style_map(primary, compare) if compare is not None else build_series_style_map(primary)

    series_index = 0

    def _stack(series_frame: pd.DataFrame, alpha: float) -> None:
        nonlocal series_index
        grouped = normalize_probability_columns(
            grouped_mean(series_frame, ["p_neg1", "p_zero", "p_pos1"]),
            ["p_neg1", "p_zero", "p_pos1"],
        )
        if grouped.empty:
            return
        x = grouped.index.to_numpy()
        values = [grouped[column].to_numpy() for column in ["p_neg1", "p_zero", "p_pos1"]]
        series_label = str(series_frame.iloc[0]["series_label"])
        series_key = str(series_frame.iloc[0]["series_key"])
        series_style = series_styles.get(series_key, "-")

        if len(x) == 1:
            width = single_point_bar_width(float(x[0]))
            bottom = 0.0
            for value, color in zip(values, [CATEGORY_COLORS["p_neg1"], CATEGORY_COLORS["p_zero"], CATEGORY_COLORS["p_pos1"]]):
                ax.bar(
                    x,
                    value,
                    width=width,
                    bottom=bottom,
                    color=color,
                    alpha=alpha,
                    align="center",
                )
                bottom = bottom + float(value[0])
        else:
            ax.stackplot(
                x,
                *values,
                colors=[CATEGORY_COLORS["p_neg1"], CATEGORY_COLORS["p_zero"], CATEGORY_COLORS["p_pos1"]],
                alpha=alpha,
            )

        ax.plot(x, grouped["p_neg1"], color=CATEGORY_COLORS["p_neg1"], linestyle=series_style, alpha=alpha, linewidth=1.2, marker="o", markersize=4)
        ax.plot(x, grouped["p_zero"], color=CATEGORY_COLORS["p_zero"], linestyle=series_style, alpha=alpha, linewidth=1.2, marker="o", markersize=4)
        ax.plot(x, grouped["p_pos1"], color=CATEGORY_COLORS["p_pos1"], linestyle=series_style, alpha=alpha, linewidth=1.2, marker="o", markersize=4)

        label_y = 0.95 - (0.05 * series_index)
        if label_y < 0.08:
            label_y = 0.08
        ax.text(
            0.01,
            label_y,
            series_label,
            transform=ax.transAxes,
            fontsize=9,
            color="black",
            alpha=0.8,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none", "pad": 2.5},
        )
        series_index += 1

    for series_key, series_label, series_frame in iter_series(primary):
        _stack(series_frame, 0.72)
    if compare is not None:
        for series_key, series_label, series_frame in iter_series(compare):
            _stack(series_frame, 0.28)

    category_handles = [Patch(facecolor=color, label=label) for label, color in CATEGORY_COLORS.items()]
    category_legend = ax.legend(handles=category_handles, loc="upper left", fontsize=9, title="distribution")

    if compare is not None:
        run_handles = [
            Patch(facecolor="#777777", alpha=0.72, label=primary.iloc[0]["run_label"]),
            Patch(facecolor="#777777", alpha=0.28, label=compare.iloc[0]["run_label"]),
        ]
        ax.add_artist(category_legend)
        ax.legend(handles=run_handles, loc="upper right", fontsize=9, title="run opacity")

    ax.set_title("Sparsity Health: p_neg1 / p_zero / p_pos1 Over Time")
    ax.set_xlabel("analysis step")
    ax.set_ylabel("probability mass")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_efficiency(primary: pd.DataFrame, compare: pd.DataFrame | None, output_path: Path) -> None:
    fig, ax_tps = plt.subplots(figsize=(12, 7))
    ax_io = ax_tps.twinx()
    series_styles = build_series_style_map(primary, compare) if compare is not None else build_series_style_map(primary)

    for frame, alpha in ((primary, 0.92), (compare, 0.65)) if compare is not None else ((primary, 0.92),):
        if frame is None or frame.empty:
            continue
        for series_key, series_label, series_frame in iter_series(frame):
            grouped = grouped_mean(series_frame, ["tps_value", "io_ms"])
            if grouped.empty:
                continue
            series_style = series_styles.get(series_key, "-")

            ax_tps.plot(
                grouped.index,
                grouped["tps_value"],
                color=RUN_COLORS[0],
                linestyle=series_style,
                linewidth=2.2 if frame is primary else 1.9,
                alpha=alpha,
                marker="o",
                markersize=5,
                label=f"{series_label} tps",
            )
            ax_io.plot(
                grouped.index,
                grouped["io_ms"],
                color=RUN_COLORS[1],
                linestyle=series_style,
                linewidth=2.2 if frame is primary else 1.9,
                alpha=alpha,
                marker="o",
                markersize=5,
                label=f"{series_label} io_ms",
            )

    ax_tps.set_title("Efficiency: tps vs io_ms")
    ax_tps.set_xlabel("analysis step")
    ax_tps.set_ylabel("tps")
    ax_io.set_ylabel("io_ms")
    ax_tps.grid(True, alpha=0.28)

    handles_tps, labels_tps = ax_tps.get_legend_handles_labels()
    handles_io, labels_io = ax_io.get_legend_handles_labels()
    ax_tps.legend(handles_tps + handles_io, labels_tps + labels_io, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_stability(primary: pd.DataFrame, compare: pd.DataFrame | None, output_path: Path) -> None:
    fig, (ax_norms, ax_saturation) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax_clip = ax_norms.twinx()
    ax_proxy = ax_saturation.twinx()
    series_styles = build_series_style_map(primary, compare) if compare is not None else build_series_style_map(primary)

    for frame, alpha in ((primary, 0.92), (compare, 0.65)) if compare is not None else ((primary, 0.92),):
        if frame is None or frame.empty:
            continue
        for series_key, series_label, series_frame in iter_series(frame):
            grouped = grouped_mean(
                series_frame,
                ["raw_grad_norm", "clipped_grad_norm", "clip_scale", "latent_saturation", "hessian_proxy_mean", "hessian_proxy_max"],
            )
            if grouped.empty:
                continue
            series_style = series_styles.get(series_key, "-")

            ax_norms.plot(
                grouped.index,
                grouped["raw_grad_norm"],
                color="#264653",
                linestyle=series_style,
                linewidth=2.1 if frame is primary else 1.8,
                alpha=alpha,
                marker="o",
                markersize=4,
                label=f"{series_label} raw_grad_norm",
            )
            ax_norms.plot(
                grouped.index,
                grouped["clipped_grad_norm"],
                color="#e76f51",
                linestyle=series_style,
                linewidth=2.1 if frame is primary else 1.8,
                alpha=alpha,
                marker="s",
                markersize=4,
                label=f"{series_label} clipped_grad_norm",
            )
            ax_clip.plot(
                grouped.index,
                grouped["clip_scale"],
                color="#2a9d8f",
                linestyle=series_style,
                linewidth=1.8 if frame is primary else 1.6,
                alpha=alpha,
                marker="^",
                markersize=4,
                label=f"{series_label} clip_scale",
            )

            ax_saturation.plot(
                grouped.index,
                grouped["latent_saturation"],
                color="#7f5539",
                linestyle=series_style,
                linewidth=2.1 if frame is primary else 1.8,
                alpha=alpha,
                marker="o",
                markersize=4,
                label=f"{series_label} latent_saturation",
            )
            ax_proxy.plot(
                grouped.index,
                grouped["hessian_proxy_mean"],
                color="#577590",
                linestyle=series_style,
                linewidth=1.8 if frame is primary else 1.6,
                alpha=alpha,
                marker="d",
                markersize=4,
                label=f"{series_label} hessian_proxy_mean",
            )
            ax_proxy.plot(
                grouped.index,
                grouped["hessian_proxy_max"],
                color="#f4a261",
                linestyle=series_style,
                linewidth=1.8 if frame is primary else 1.6,
                alpha=alpha,
                marker="x",
                markersize=4,
                label=f"{series_label} hessian_proxy_max",
            )

    ax_norms.set_title("Stability: Gradient Norms, Clipping, and Latent Saturation")
    ax_norms.set_ylabel("gradient norm")
    ax_clip.set_ylabel("clip_scale")
    ax_norms.grid(True, alpha=0.28)

    ax_saturation.set_xlabel("analysis step")
    ax_saturation.set_ylabel("latent_saturation")
    ax_proxy.set_ylabel("proxy summary")
    ax_saturation.grid(True, alpha=0.28)

    handles_norms, labels_norms = ax_norms.get_legend_handles_labels()
    handles_clip, labels_clip = ax_clip.get_legend_handles_labels()
    handles_sat, labels_sat = ax_saturation.get_legend_handles_labels()
    handles_proxy, labels_proxy = ax_proxy.get_legend_handles_labels()
    ax_norms.legend(handles_norms + handles_clip, labels_norms + labels_clip, loc="upper right", fontsize=8)
    ax_saturation.legend(handles_sat + handles_proxy, labels_sat + labels_proxy, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_plots(primary: pd.DataFrame, compare: pd.DataFrame | None, output_dir: Path) -> list[Path]:
    compare_suffix = "_compare" if compare is not None else ""
    outputs = [
        output_dir / f"convergence{compare_suffix}.png",
        output_dir / f"sparsity{compare_suffix}.png",
        output_dir / f"efficiency{compare_suffix}.png",
        output_dir / f"stability{compare_suffix}.png",
    ]
    plot_convergence(primary, compare, outputs[0])
    plot_sparsity(primary, compare, outputs[1])
    plot_efficiency(primary, compare, outputs[2])
    plot_stability(primary, compare, outputs[3])
    return outputs


def main() -> int:
    args = parse_args()
    primary_path = args.metrics_jsonl
    compare_path = args.compare
    output_dir = build_output_dir(primary_path, args.output_dir)

    primary = load_metrics(primary_path, primary_path.stem)
    compare = load_metrics(compare_path, compare_path.stem) if compare_path is not None else None

    outputs = write_plots(primary, compare, output_dir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

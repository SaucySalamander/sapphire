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

    numeric_columns = required_columns + ["resume_step_idx", "tps"]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "resume_step_idx" not in frame.columns:
        frame["resume_step_idx"] = 0.0
    frame["resume_step_idx"] = frame["resume_step_idx"].fillna(0.0)

    frame["analysis_step"] = frame["resume_step_idx"] + frame["step_idx"]

    if "tps" in frame.columns:
        tps_series = frame["tps"].copy()
    else:
        tps_series = pd.Series(float("nan"), index=frame.index)

    derived_tps = pd.Series(float("nan"), index=frame.index)
    valid_compute = frame["compute_ms"].notna() & (frame["compute_ms"] > 0)
    derived_tps.loc[valid_compute] = 1000.0 / frame.loc[valid_compute, "compute_ms"]
    frame["tps_value"] = tps_series.fillna(derived_tps)

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


def grouped_mean(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected = frame[["analysis_step", *columns]].dropna(subset=["analysis_step"]).copy()
    if selected.empty:
        return pd.DataFrame(columns=list(columns))
    return selected.groupby("analysis_step", as_index=True)[list(columns)].mean().sort_index()


def plot_convergence(primary: pd.DataFrame, compare: pd.DataFrame | None, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    layers = unique_layers(primary, compare) if compare is not None else unique_layers(primary)

    for index, layer_idx in enumerate(layers):
        color = plt.get_cmap("tab20")(index % 20)
        primary_layer = primary[primary["layer_idx"] == layer_idx].dropna(subset=["analysis_step", "mse_loss"])
        if not primary_layer.empty:
            primary_layer = primary_layer.sort_values("analysis_step")
            ax.plot(
                primary_layer["analysis_step"],
                primary_layer["mse_loss"],
                color=color,
                linewidth=2.0,
                label=f"{primary.iloc[0]['run_label']} layer {layer_idx}",
            )

        if compare is not None:
            compare_layer = compare[compare["layer_idx"] == layer_idx].dropna(subset=["analysis_step", "mse_loss"])
            if not compare_layer.empty:
                compare_layer = compare_layer.sort_values("analysis_step")
                ax.plot(
                    compare_layer["analysis_step"],
                    compare_layer["mse_loss"],
                    color=color,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.85,
                    label=f"{compare.iloc[0]['run_label']} layer {layer_idx}",
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

    def _stack(frame: pd.DataFrame, alpha: float, label: str) -> None:
        grouped = grouped_mean(frame, ["p_neg1", "p_zero", "p_pos1"])
        if grouped.empty:
            return
        x = grouped.index.to_numpy()
        values = [grouped[column].to_numpy() for column in ["p_neg1", "p_zero", "p_pos1"]]
        ax.stackplot(
            x,
            *values,
            colors=[CATEGORY_COLORS["p_neg1"], CATEGORY_COLORS["p_zero"], CATEGORY_COLORS["p_pos1"]],
            alpha=alpha,
        )
        ax.plot(x, grouped["p_neg1"], color=CATEGORY_COLORS["p_neg1"], alpha=alpha, linewidth=1.2)
        ax.plot(x, grouped["p_zero"], color=CATEGORY_COLORS["p_zero"], alpha=alpha, linewidth=1.2)
        ax.plot(x, grouped["p_pos1"], color=CATEGORY_COLORS["p_pos1"], alpha=alpha, linewidth=1.2)
        ax.text(
            0.01,
            0.95 if label == primary.iloc[0]["run_label"] else 0.90,
            f"{label}",
            transform=ax.transAxes,
            fontsize=9,
            color="black",
            alpha=0.8,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none", "pad": 2.5},
        )

    _stack(primary, 0.72, primary.iloc[0]["run_label"])
    if compare is not None:
        _stack(compare, 0.28, compare.iloc[0]["run_label"])

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

    primary_grouped = grouped_mean(primary, ["tps_value", "io_ms"])
    if not primary_grouped.empty:
        ax_tps.plot(
            primary_grouped.index,
            primary_grouped["tps_value"],
            color=RUN_COLORS[0],
            linewidth=2.2,
            label=f"{primary.iloc[0]['run_label']} tps",
        )
        ax_io.plot(
            primary_grouped.index,
            primary_grouped["io_ms"],
            color=RUN_COLORS[1],
            linewidth=2.2,
            label=f"{primary.iloc[0]['run_label']} io_ms",
        )

    if compare is not None:
        compare_grouped = grouped_mean(compare, ["tps_value", "io_ms"])
        if not compare_grouped.empty:
            ax_tps.plot(
                compare_grouped.index,
                compare_grouped["tps_value"],
                color=RUN_COLORS[0],
                linestyle="--",
                linewidth=1.9,
                alpha=0.85,
                label=f"{compare.iloc[0]['run_label']} tps",
            )
            ax_io.plot(
                compare_grouped.index,
                compare_grouped["io_ms"],
                color=RUN_COLORS[1],
                linestyle="--",
                linewidth=1.9,
                alpha=0.85,
                label=f"{compare.iloc[0]['run_label']} io_ms",
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


def write_plots(primary: pd.DataFrame, compare: pd.DataFrame | None, output_dir: Path) -> list[Path]:
    compare_suffix = "_compare" if compare is not None else ""
    outputs = [
        output_dir / f"convergence{compare_suffix}.png",
        output_dir / f"sparsity{compare_suffix}.png",
        output_dir / f"efficiency{compare_suffix}.png",
    ]
    plot_convergence(primary, compare, outputs[0])
    plot_sparsity(primary, compare, outputs[1])
    plot_efficiency(primary, compare, outputs[2])
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

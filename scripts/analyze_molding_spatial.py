#!/usr/bin/env python3
"""Generate spatial molding plots from Sapphire spatial telemetry JSONL.

Run with the repository virtual environment when possible:
    .venv/bin/python scripts/analyze_molding_spatial.py out/ternary_telemetry_20260414_120000_spatial.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR_NAME = "molding_spatial_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn spatial molding telemetry JSONL into topography, leak, and distribution plots.",
    )
    parser.add_argument("spatial_jsonl", type=Path, help="Spatial telemetry JSONL file.")
    parser.add_argument("--tensor", type=str, default=None, help="Optional substring filter for tensor_name.")
    parser.add_argument("--layer", type=int, default=None, help="Optional layer_idx filter.")
    parser.add_argument("--step", type=int, default=None, help="Optional step_idx filter.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots will be written. Defaults to a sibling analysis directory next to the JSONL.",
    )
    return parser.parse_args()


def build_output_dir(primary_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        resolved = output_dir
    else:
        resolved = primary_path.parent / DEFAULT_OUTPUT_DIR_NAME / primary_path.stem
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_records(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Spatial telemetry file not found: {path}")

    frame = pd.read_json(path, lines=True)
    if frame.empty:
        raise ValueError(f"Spatial telemetry file is empty: {path}")

    if "record_type" not in frame.columns:
        raise ValueError(f"Spatial telemetry file is missing record_type: {path}")

    meta = frame[frame["record_type"] == "spatial_snapshot_meta"].copy()
    blocks = frame[frame["record_type"] == "spatial_snapshot_block"].copy()
    hist = frame[frame["record_type"] == "spatial_snapshot_histogram"].copy()
    if meta.empty or blocks.empty or hist.empty:
        raise ValueError(f"Spatial telemetry file {path} is missing one or more required record types")

    return meta, blocks, hist


def select_snapshot(meta: pd.DataFrame, tensor_filter: str | None, layer: int | None, step: int | None) -> pd.Series:
    filtered = meta.copy()

    if tensor_filter:
        filtered = filtered[filtered["tensor_name"].astype(str).str.contains(tensor_filter, regex=False)]
    if layer is not None:
        filtered = filtered[pd.to_numeric(filtered["layer_idx"], errors="coerce") == layer]
    if step is not None:
        filtered = filtered[pd.to_numeric(filtered["step_idx"], errors="coerce") == step]
    if filtered.empty:
        raise ValueError("No spatial snapshot matches the requested filters")

    filtered = filtered.sort_values(["step_idx", "tensor_name", "config_hash"], kind="stable")
    return filtered.iloc[-1]


def snapshot_blocks(blocks: pd.DataFrame, snapshot: pd.Series) -> pd.DataFrame:
    filtered = blocks[
        (pd.to_numeric(blocks["config_hash"], errors="coerce") == int(snapshot["config_hash"]))
        & (pd.to_numeric(blocks["layer_idx"], errors="coerce") == int(snapshot["layer_idx"]))
        & (pd.to_numeric(blocks["step_idx"], errors="coerce") == int(snapshot["step_idx"]))
    ].copy()
    if filtered.empty:
        raise ValueError("No spatial block records found for selected snapshot")
    return filtered


def snapshot_histogram(hist: pd.DataFrame, snapshot: pd.Series) -> pd.Series:
    filtered = hist[
        (pd.to_numeric(hist["config_hash"], errors="coerce") == int(snapshot["config_hash"]))
        & (pd.to_numeric(hist["layer_idx"], errors="coerce") == int(snapshot["layer_idx"]))
        & (pd.to_numeric(hist["step_idx"], errors="coerce") == int(snapshot["step_idx"]))
    ]
    if filtered.empty:
        raise ValueError("No spatial histogram record found for selected snapshot")
    return filtered.iloc[-1]


def build_matrix(blocks: pd.DataFrame, snapshot: pd.Series, column: str) -> np.ndarray:
    row_bucket_count = int(snapshot["row_bucket_count"])
    groups_per_row = int(snapshot["groups_per_row"])
    matrix = np.full((row_bucket_count, groups_per_row), np.nan, dtype=float)

    for _, row in blocks.iterrows():
        row_bucket_idx = int(row["row_bucket_idx"])
        group_idx = int(row["group_idx"])
        if 0 <= row_bucket_idx < row_bucket_count and 0 <= group_idx < groups_per_row:
            matrix[row_bucket_idx, group_idx] = pd.to_numeric(row[column], errors="coerce")
    return matrix


def snapshot_slug(snapshot: pd.Series) -> str:
    config_hash = int(snapshot["config_hash"])
    tensor_name = str(snapshot["tensor_name"]).split(".")[-2:]
    tensor_slug = "_".join(tensor_name) if tensor_name else "snapshot"
    return f"{tensor_slug}_{config_hash:08x}"


def heatmap_with_colorbar(ax: plt.Axes, matrix: np.ndarray, title: str, cmap: str, colorbar_label: str) -> None:
    if not np.isfinite(matrix).any():
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("group_idx")
        ax.set_ylabel("row_bucket_idx")
        return

    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("group_idx")
    ax.set_ylabel("row_bucket_idx")
    colorbar = plt.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)


def plot_topography(blocks: pd.DataFrame, snapshot: pd.Series, output_path: Path) -> None:
    gamma_map = build_matrix(blocks, snapshot, "gamma_mean")
    hessian_map = build_matrix(blocks, snapshot, "hessian_group_max")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    heatmap_with_colorbar(axes[0], gamma_map, "Gamma Landscape", "viridis", "gamma_mean")
    heatmap_with_colorbar(axes[1], hessian_map, "Hessian Ridge Map", "magma", "hessian_group_max")
    fig.suptitle(f"Topography: {snapshot['tensor_name']} step={int(snapshot['step_idx'])}")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_leak_map(blocks: pd.DataFrame, snapshot: pd.Series, output_path: Path) -> None:
    mse_map = build_matrix(blocks, snapshot, "block_weight_mse")
    zero_map = build_matrix(blocks, snapshot, "p_zero_fraction")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    heatmap_with_colorbar(ax, mse_map, "Leak Map", "Reds", "block_weight_mse")
    if np.isfinite(zero_map).sum() >= 4 and zero_map.shape[0] > 1 and zero_map.shape[1] > 1:
        safe_zero = np.nan_to_num(zero_map, nan=0.0)
        ax.contour(safe_zero, levels=[0.25, 0.5, 0.75], colors="#ffffff", linewidths=0.8, alpha=0.65)
    fig.suptitle(f"Leak Map: {snapshot['tensor_name']} step={int(snapshot['step_idx'])}")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(hist_row: pd.Series, output_path: Path) -> None:
    teacher_counts = np.asarray(hist_row["teacher_counts"], dtype=float)
    student_counts = np.asarray(hist_row["student_counts"], dtype=float)
    student_bulk_counts_raw = hist_row.get("student_bulk_counts")
    student_bulk_counts = None
    if isinstance(student_bulk_counts_raw, list):
        student_bulk_counts = np.asarray(student_bulk_counts_raw, dtype=float)

    histogram_min = float(hist_row["histogram_min"])
    histogram_max = float(hist_row["histogram_max"])
    bin_count = int(hist_row["histogram_bin_count"])
    edges = np.linspace(histogram_min, histogram_max, bin_count + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.step(centers, teacher_counts, where="mid", linewidth=2.0, color="#1d3557", label="teacher")
    ax.step(centers, student_counts, where="mid", linewidth=2.0, color="#d62828", label="student")
    if student_bulk_counts is not None:
        ax.step(centers, student_bulk_counts, where="mid", linewidth=1.8, color="#2a9d8f", label="student_bulk")
    ax.set_title(f"Weight Distribution: {hist_row['tensor_name']} step={int(hist_row['step_idx'])}")
    ax.set_xlabel("weight value")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = build_output_dir(args.spatial_jsonl, args.output_dir)
    meta, blocks, hist = load_records(args.spatial_jsonl)
    snapshot = select_snapshot(meta, args.tensor, args.layer, args.step)
    selected_blocks = snapshot_blocks(blocks, snapshot)
    hist_row = snapshot_histogram(hist, snapshot)
    slug = snapshot_slug(snapshot)

    outputs = [
        output_dir / f"topography_{slug}.png",
        output_dir / f"leak_map_{slug}.png",
        output_dir / f"distribution_{slug}.png",
    ]

    plot_topography(selected_blocks, snapshot, outputs[0])
    plot_leak_map(selected_blocks, snapshot, outputs[1])
    plot_distribution(hist_row, outputs[2])

    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
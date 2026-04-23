#!/usr/bin/env python3
"""Targeted runtime smoke test for packaged hybrid anchor-bearing models."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import tempfile
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

import repack_ternary_model as repack


ANCHOR_BUDGET_PPM = 1000


def _write_safetensors(path: Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]], metadata: OrderedDict | None = None) -> None:
    header = OrderedDict()
    header["__metadata__"] = metadata or OrderedDict()

    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = OrderedDict(
            (
                ("dtype", dtype),
                ("shape", [int(dim) for dim in shape]),
                ("data_offsets", [offset, offset + len(data)]),
            )
        )
        offset += len(data)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for _name, _dtype, _shape, data in tensors:
            handle.write(data)


def _select_anchor_tensor(base_tensors: OrderedDict[str, repack.BaseTensorRef], preferred: str | None) -> tuple[str, tuple[int, ...]]:
    if preferred:
        tensor = base_tensors.get(preferred)
        if tensor is None:
            raise KeyError(f"Preferred tensor not found in base model: {preferred}")
        if len(tensor.shape) != 2:
            raise ValueError(f"Preferred tensor must be rank-2, got {tensor.shape}")
        return preferred, tensor.shape

    for name, tensor in base_tensors.items():
        if name.endswith(".mlp.down_proj.weight") and len(tensor.shape) == 2:
            return name, tensor.shape

    for name, tensor in base_tensors.items():
        if len(tensor.shape) == 2:
            return name, tensor.shape

    raise RuntimeError("No rank-2 tensor available for hybrid runtime smoke")


def _safe_tensor_file_name(tensor_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tensor_name)


def _create_hybrid_ternary_output(ternary_dir: Path, tensor_name: str, rows: int, cols: int) -> tuple[str, str]:
    ternary_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_tensor_file_name(tensor_name)
    packed_file_name = f"{safe_name}.safetensors"
    anchor_file_name = f"{safe_name}.anchors.safetensors"

    packed_cols = (cols + repack.TERNARY_PACKED_WEIGHTS_PER_BYTE - 1) // repack.TERNARY_PACKED_WEIGHTS_PER_BYTE
    packed_bytes = b"\x00" * (rows * packed_cols)
    scale_bytes = np.ones((rows,), dtype="<f4").tobytes()
    payload_crc32 = zlib.crc32(packed_bytes) & 0xFFFFFFFF
    payload_crc32 = zlib.crc32(scale_bytes, payload_crc32) & 0xFFFFFFFF

    _write_safetensors(
        ternary_dir / packed_file_name,
        [
            (f"{tensor_name}.packed", "U8", (rows, packed_cols), packed_bytes),
            (f"{tensor_name}.scales", "F32", (rows,), scale_bytes),
        ],
        metadata=OrderedDict(
            (
                (f"{tensor_name}.groups_per_row", 1),
                (f"{tensor_name}.scale_group_size", cols),
            )
        ),
    )

    anchor_entries = np.asarray(
        [
            [0, min(1, cols - 1), 0x3F80, 0],
            [rows - 1, cols - 1, 0x4000, 0],
        ],
        dtype="<u2",
    )
    entry_bytes = anchor_entries.tobytes(order="C")
    entry_crc32 = zlib.crc32(entry_bytes) & 0xFFFFFFFF
    _write_safetensors(
        ternary_dir / anchor_file_name,
        [(f"{tensor_name}.anchor_entries", "U16", (2, 4), entry_bytes)],
        metadata=OrderedDict(
            (
                ("sapphire_quant", "ternary-anchor-v2"),
                ("format_version", str(repack.ANCHOR_SAFETENSORS_VERSION)),
                ("tensor_name", tensor_name),
                ("rows", str(rows)),
                ("cols", str(cols)),
                ("anchor_count", "2"),
                ("scale_group_size", str(cols)),
                ("groups_per_row", "1"),
                ("value_dtype", "BF16"),
                ("saliency_mode", "2"),
                ("budget_ppm", str(ANCHOR_BUDGET_PPM)),
                ("entry_layout", "row,col,value_bf16,pad"),
                ("entry_order", "row_col"),
                ("entry_crc32", f"{entry_crc32:08x}"),
                ("saliency_cutoff", "0.5"),
                ("anchor_value_rms", "1.0"),
                ("bulk_gamma_mean", "0.0"),
                ("max_row_nnz", "1"),
            )
        ),
    )

    (ternary_dir / "manifest.tsv").write_text(
        f"{tensor_name}\t{packed_file_name}\t{rows}\t{cols}\t{len(packed_bytes)}\t{payload_crc32:08x}\tanchor\t{anchor_file_name}\t2\n",
        encoding="utf-8",
    )
    return packed_file_name, anchor_file_name


def _run_command(command: list[str], cwd: Path, env: dict[str, str], expect_success: bool, expected_text: str | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"Command failed unexpectedly ({result.returncode}): {' '.join(command)}\n{output}")
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"Command succeeded unexpectedly: {' '.join(command)}\n{output}")
    if expected_text and expected_text not in output:
        raise RuntimeError(f"Expected text not found in command output: {expected_text!r}\n{output}")
    return output


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run CPU and Vulkan smoke checks against a temporary hybrid model package")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=repo_root / "models" / "gemma-3-270m-it",
        help="Base model package used to build the temporary hybrid repack",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=repo_root / "out" / "sapphire",
        help="Sapphire inference binary to execute",
    )
    parser.add_argument(
        "--tensor-name",
        type=str,
        default=None,
        help="Optional tensor to convert into a synthetic hybrid payload",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name exposed under the smoke workspace models directory (defaults to the base model directory name)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello",
        help="Prompt used for the CPU runtime smoke",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=1,
        help="Number of tokens to generate during the CPU smoke run",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the temporary repacked model and workspace for inspection",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_model_dir = args.base_model_dir.resolve()
    binary_path = args.binary.resolve()
    model_name = args.model_name or base_model_dir.name

    if not base_model_dir.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    if not binary_path.is_file():
        raise FileNotFoundError(f"Sapphire binary not found: {binary_path}")

    base_tensors = repack._collect_base_tensors(base_model_dir)
    tensor_name, tensor_shape = _select_anchor_tensor(base_tensors, args.tensor_name)
    rows, cols = (int(tensor_shape[0]), int(tensor_shape[1]))

    if args.keep_artifacts:
        work_root = Path(tempfile.mkdtemp(prefix="sapphire_hybrid_runtime_"))
        cleanup = False
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="sapphire_hybrid_runtime_")
        work_root = Path(temp_dir.name)
        cleanup = True

    try:
        ternary_dir = work_root / "ternary-output"
        packed_model_dir = work_root / "packed-model"
        runtime_workspace = work_root / "runtime-workspace"
        models_dir = runtime_workspace / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        _create_hybrid_ternary_output(ternary_dir, tensor_name, rows, cols)
        repack.repack_ternary_model(
            base_model_dir=base_model_dir,
            ternary_dir=ternary_dir,
            output_dir=packed_model_dir,
            max_shard_size=repack.DEFAULT_MAX_SHARD_SIZE,
            overwrite=False,
        )

        exposed_model_dir = models_dir / model_name
        os.symlink(packed_model_dir, exposed_model_dir)

        base_env = os.environ.copy()
        cpu_env = dict(base_env)
        cpu_env["SAPPHIRE_BACKEND"] = "cpu"
        cpu_output = _run_command(
            [
                str(binary_path),
                "-m",
                model_name,
                "-p",
                args.prompt,
                "-n",
                str(args.tokens),
                "-t",
                "0.0",
            ],
            cwd=runtime_workspace,
            env=cpu_env,
            expect_success=True,
            expected_text="[Inference time:",
        )

        vulkan_env = dict(base_env)
        vulkan_env["SAPPHIRE_BACKEND"] = "vulkan"
        vulkan_output = _run_command(
            [
                str(binary_path),
                "-m",
                model_name,
                "-p",
                args.prompt,
                "-n",
                "1",
                "-t",
                "0.0",
            ],
            cwd=runtime_workspace,
            env=vulkan_env,
            expect_success=False,
            expected_text="rerun with SAPPHIRE_BACKEND=cpu",
        )

        print(f"CPU runtime smoke passed for {model_name} using tensor {tensor_name} ({rows}x{cols})")
        print("Verified Vulkan rejection output:")
        print(vulkan_output.strip().splitlines()[-1])
        if args.keep_artifacts:
            print(f"Kept smoke artifacts at: {work_root}")
        return 0
    finally:
        if cleanup:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
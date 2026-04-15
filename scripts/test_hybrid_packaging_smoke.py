#!/usr/bin/env python3
"""Synthetic smoke test for hybrid and non-hybrid repack packaging."""

from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

import repack_ternary_model as repack
import verify_model_storage_bits as verify


PACKED_TENSOR_NAME = "layer.weight"
PACKED_FILE_NAME = "layer_weight.safetensors"
ANCHOR_FILE_NAME = "layer_weight.anchors.safetensors"
ANCHOR_BUDGET_PPM = 5000


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


def _bf16_bytes(values: list[float]) -> bytes:
    return repack._float32_to_bf16_words(np.asarray(values, dtype=np.float32)).tobytes()


def _create_base_model(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "config.json").write_text(
        json.dumps(
            OrderedDict(
                (
                    ("model_type", "gemma3_text"),
                    ("torch_dtype", "bfloat16"),
                    ("vocab_size", 16),
                )
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    tensors = [
        (PACKED_TENSOR_NAME, "BF16", (2, 4), _bf16_bytes([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0])),
        ("passthrough.bias", "BF16", (4,), _bf16_bytes([0.5, 0.25, -0.25, -0.5])),
    ]
    _write_safetensors(base_dir / "model.safetensors", tensors)


def _create_ternary_payload(ternary_dir: Path, include_anchor: bool) -> None:
    ternary_dir.mkdir(parents=True, exist_ok=True)

    packed_bytes = bytes((0x12, 0x84))
    scale_bytes = np.asarray([1.0, 0.5], dtype="<f4").tobytes()
    payload_crc32 = zlib.crc32(packed_bytes) & 0xFFFFFFFF
    payload_crc32 = zlib.crc32(scale_bytes, payload_crc32) & 0xFFFFFFFF

    payload_metadata = OrderedDict(
        (
            (f"{PACKED_TENSOR_NAME}.groups_per_row", 1),
            (f"{PACKED_TENSOR_NAME}.scale_group_size", 4),
        )
    )
    _write_safetensors(
        ternary_dir / PACKED_FILE_NAME,
        [
            (f"{PACKED_TENSOR_NAME}.packed", "U8", (2, 1), packed_bytes),
            (f"{PACKED_TENSOR_NAME}.scales", "F32", (2,), scale_bytes),
        ],
        metadata=payload_metadata,
    )

    if include_anchor:
        entries = np.asarray(
            [
                [0, 1, 0x3F80, 0],
                [1, 3, 0x4000, 0],
            ],
            dtype="<u2",
        )
        entry_bytes = entries.tobytes(order="C")
        entry_crc32 = zlib.crc32(entry_bytes) & 0xFFFFFFFF
        anchor_metadata = OrderedDict(
            (
                ("sapphire_quant", "ternary-anchor-v2"),
                ("format_version", str(repack.ANCHOR_SAFETENSORS_VERSION)),
                ("tensor_name", PACKED_TENSOR_NAME),
                ("rows", "2"),
                ("cols", "4"),
                ("anchor_count", "2"),
                ("scale_group_size", "4"),
                ("groups_per_row", "1"),
                ("value_dtype", "BF16"),
                ("saliency_mode", "2"),
                ("budget_ppm", str(ANCHOR_BUDGET_PPM)),
                ("entry_layout", "row,col,value_bf16,pad"),
                ("entry_order", "row_col"),
                ("entry_crc32", f"{entry_crc32:08x}"),
                ("saliency_cutoff", "0.5"),
                ("anchor_value_rms", "1.5"),
                ("bulk_gamma_mean", "0.25"),
                ("max_row_nnz", "1"),
            )
        )
        _write_safetensors(
            ternary_dir / ANCHOR_FILE_NAME,
            [(f"{PACKED_TENSOR_NAME}.anchor_entries", "U16", (2, 4), entry_bytes)],
            metadata=anchor_metadata,
        )
        manifest_line = (
            f"{PACKED_TENSOR_NAME}\t{PACKED_FILE_NAME}\t2\t4\t2\t{payload_crc32:08x}\tanchor\t{ANCHOR_FILE_NAME}\t2\n"
        )
    else:
        manifest_line = f"{PACKED_TENSOR_NAME}\t{PACKED_FILE_NAME}\t2\t4\t2\t{payload_crc32:08x}\n"

    (ternary_dir / "manifest.tsv").write_text(manifest_line, encoding="utf-8")


def _capture_mixed_report(output_dir: Path) -> str:
    entries = list(verify._read_manifest(output_dir).values())
    model_stats = verify._load_model_tensor_stats(output_dir)
    ternary_stats = verify._collect_ternary_tensor_stats(output_dir, entries, show_progress=False)
    hybrid_metadata = verify._validate_hybrid_package_metadata(output_dir, entries)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = verify._print_mixed_model_report(output_dir, model_stats, ternary_stats, hybrid_metadata, None)
    if rc != 0:
        raise AssertionError(f"verify_model_storage_bits report failed for {output_dir}")
    return buffer.getvalue()


def _assert_hybrid_output(output_dir: Path) -> None:
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["sapphire_mixed_precision_anchors"] is True
    assert config["sapphire_anchor_budget_ppm"] == ANCHOR_BUDGET_PPM
    assert config[repack.PACKAGE_FORMAT_VERSION_KEY] == repack.PACKAGE_FORMAT_VERSION_VALUE
    assert config[repack.PACKAGE_FEATURE_CONTRACT_KEY] == repack.PACKAGE_FEATURE_CONTRACT_VALUE
    assert config[repack.PACKAGE_STORAGE_ENCODING_KEY] == repack.PACKAGE_STORAGE_ENCODING
    assert (output_dir / ANCHOR_FILE_NAME).is_file()

    report = _capture_mixed_report(output_dir)
    assert "artifact_type: anchor_bearing_mixed_safetensors_model_package" in report
    assert f"sapphire_anchor_budget_ppm: {ANCHOR_BUDGET_PPM}" in report
    assert f"{repack.PACKAGE_FEATURE_CONTRACT_KEY}: {repack.PACKAGE_FEATURE_CONTRACT_VALUE}" in report


def _assert_non_hybrid_output(output_dir: Path) -> None:
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    for key in (
        "sapphire_mixed_precision_anchors",
        "sapphire_anchor_budget_ppm",
        repack.PACKAGE_FORMAT_VERSION_KEY,
        repack.PACKAGE_FEATURE_CONTRACT_KEY,
        repack.PACKAGE_STORAGE_ENCODING_KEY,
    ):
        assert key not in config
    assert not (output_dir / ANCHOR_FILE_NAME).exists()

    report = _capture_mixed_report(output_dir)
    assert "artifact_type: mixed_safetensors_model_package" in report
    assert "anchor_bearing_mixed_safetensors_model_package" not in report


def _run_case(work_root: Path, case_name: str, include_anchor: bool) -> None:
    base_dir = work_root / f"{case_name}-base"
    ternary_dir = work_root / f"{case_name}-ternary"
    output_dir = work_root / f"{case_name}-packed"

    _create_base_model(base_dir)
    _create_ternary_payload(ternary_dir, include_anchor=include_anchor)
    repack.repack_ternary_model(base_dir, ternary_dir, output_dir, max_shard_size=1 << 20, overwrite=False)

    if include_anchor:
        _assert_hybrid_output(output_dir)
    else:
        _assert_non_hybrid_output(output_dir)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sapphire_hybrid_packaging_") as temp_dir:
        work_root = Path(temp_dir)
        _run_case(work_root, "hybrid", include_anchor=True)
        _run_case(work_root, "plain", include_anchor=False)

    print("PASS: hybrid and non-hybrid repack smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
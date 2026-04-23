#!/usr/bin/env python3
"""Verify the effective storage bits per weight for Sapphire model artifacts.

This script understands two directory layouts:

1. Sapphire ternary conversion output directories containing `manifest.tsv`
   plus one safetensors payload per converted tensor.
2. Standard safetensors model packages containing `model.safetensors` or
    `model.safetensors.index.json`.
3. Mixed BF16 plus packed ternary model packages containing both model shards
    and `manifest.tsv`.

For ternary outputs it reports three different quantities:

- packed_symbol_bits_per_weight: actual bits used by the packed symbol stream
- total_bits_per_weight: packed symbols plus per-row scale storage
- empirical_symbol_entropy_bits: Shannon entropy of the decoded ternary symbols

The first two describe real on-disk storage. The entropy figure is a lower bound
that can approach log2(3) ~= 1.585 for well-balanced ternary weights, but it is
not the same thing as the bytes written on disk.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TERNARY_PACKED_WEIGHTS_PER_BYTE = 4
TERNARY_SYMBOL_CODES = (0, 1, 2)
ANCHOR_MAGIC = 0x414E4331
ANCHOR_VERSION = 1
ANCHOR_SAFETENSORS_VERSION = 2
ANCHOR_METADATA_STRUCT = struct.Struct("<IIIIIIIIIfffII")
PACKAGE_STORAGE_ENCODING = "mixed_bf16_ternary_2bit"
PACKAGE_FEATURE_CONTRACT_KEY = "sapphire_hybrid_feature_contract"
PACKAGE_FEATURE_CONTRACT_VALUE = "anchor-bearing-v1"
PACKAGE_FORMAT_VERSION_KEY = "sapphire_hybrid_package_format_version"
PACKAGE_FORMAT_VERSION_VALUE = 1
PACKAGE_STORAGE_ENCODING_KEY = "sapphire_hybrid_storage_encoding"
MANIFEST_FORM_BULK = 1 << 0
MANIFEST_FORM_MOLD = 1 << 1
MANIFEST_FORM_LEGACY_ANCHOR = 1 << 2
MANIFEST_FORM_UNIFIED_ANCHOR = 1 << 3
SUPPORTED_MODEL_DTYPES = {
    "U8": 8,
    "I8": 8,
    "F16": 16,
    "BF16": 16,
    "F32": 32,
    "I32": 32,
    "U32": 32,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


@dataclass
class TernaryManifestEntry:
    name: str
    file_name: str
    rows: int
    cols: int
    packed_weight_bytes: int
    crc32: int
    kind: str = "ternary"
    groups_per_row: int = 1
    scale_group_size: int = 0
    anchor_file_name: str | None = None
    anchor_count: int = 0

    @property
    def weight_count(self) -> int:
        return self.rows * self.cols

    @property
    def packed_cols(self) -> int:
        return (self.cols + TERNARY_PACKED_WEIGHTS_PER_BYTE - 1) // TERNARY_PACKED_WEIGHTS_PER_BYTE

    @property
    def packed_shape(self) -> tuple[int, ...]:
        return (self.rows, self.packed_cols)

    @property
    def scale_shape(self) -> tuple[int, ...]:
        return (self.rows,) if self.groups_per_row == 1 else (self.rows, self.groups_per_row)

    @property
    def scale_bytes(self) -> int:
        return self.rows * self.groups_per_row * 4


@dataclass(frozen=True)
class TernaryTensorStats:
    name: str
    weight_count: int
    packed_bytes: int
    scale_bytes: int
    packed_symbol_bits_per_weight: float
    total_bits_per_weight: float
    empirical_symbol_entropy_bits: float
    symbol_counts: Counter[int]


@dataclass(frozen=True)
class ModelTensorStats:
    name: str
    dtype: str
    element_count: int
    data_bytes: int
    bits_per_element: float


@dataclass(frozen=True)
class HybridPackageMetadata:
    mixed_precision_anchors: bool
    anchor_budget_ppm: int
    format_version: int
    feature_contract: str
    storage_encoding: str


def _read_json(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=OrderedDict)


def _metadata_int(metadata: object, key: str) -> int | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid metadata value for {key}: {value!r}")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError(f"Invalid metadata value for {key}: {value!r}")


def _metadata_bool(metadata: object, key: str) -> bool | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"Invalid metadata value for {key}: {value!r}")


def _metadata_str(metadata: object, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Invalid metadata value for {key}: {value!r}")


def _read_safetensors_header(path: Path) -> tuple[OrderedDict, int]:
    with path.open("rb") as handle:
        header_len_bytes = handle.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors header prefix: {path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Truncated safetensors header: {path}")

    header = json.loads(header_bytes, object_pairs_hook=OrderedDict)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {path}")
    return header, 8 + int(header_len)


def _tensor_shape(meta: dict, path: Path) -> tuple[int, ...]:
    shape = meta.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"Invalid tensor shape metadata in {path}: {meta}")
    return tuple(int(dim) for dim in shape)


def _tensor_offsets(path: Path, meta: dict, data_start: int) -> tuple[int, int]:
    offsets = meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"Invalid tensor offsets in {path}: {meta}")
    start = data_start + int(offsets[0])
    end = data_start + int(offsets[1])
    if end < start:
        raise ValueError(f"Negative tensor data range in {path}")
    return start, end


def _payload_meta(path: Path, header: OrderedDict, tensor_name: str, suffix: str) -> dict:
    meta = header.get(f"{tensor_name}.{suffix}")
    if not isinstance(meta, dict):
        raise KeyError(f"Missing {tensor_name}.{suffix} in {path}")
    return meta


def _read_ternary_payload_layout(payload_path: Path,
                                 tensor_name: str,
                                 rows: int,
                                 cols: int) -> tuple[int, int]:
    header, _ = _read_safetensors_header(payload_path)
    scales_meta = _payload_meta(payload_path, header, tensor_name, "scales")
    scales_shape = _tensor_shape(scales_meta, payload_path)
    metadata = header.get("__metadata__")

    if len(scales_shape) == 1:
        if scales_shape != (rows,):
            raise ValueError(f"Scale payload shape mismatch for {tensor_name}: {scales_shape}")
        groups_per_row = 1
    elif len(scales_shape) == 2:
        if scales_shape[0] != rows or scales_shape[1] <= 0:
            raise ValueError(f"Grouped scale payload shape mismatch for {tensor_name}: {scales_shape}")
        groups_per_row = int(scales_shape[1])
    else:
        raise ValueError(f"Unsupported scale payload rank for {tensor_name}: {scales_shape}")

    metadata_groups = _metadata_int(metadata, f"{tensor_name}.groups_per_row")
    if metadata_groups is not None:
        if metadata_groups != groups_per_row:
            raise ValueError(
                f"Grouped scale metadata mismatch for {tensor_name}: metadata={metadata_groups} shape={groups_per_row}"
            )
        groups_per_row = metadata_groups

    scale_group_size = _metadata_int(metadata, f"{tensor_name}.scale_group_size")
    if scale_group_size is None:
        scale_group_size = math.ceil(cols / groups_per_row)
    if scale_group_size <= 0 or math.ceil(cols / scale_group_size) != groups_per_row:
        raise ValueError(
            f"Invalid grouped scale layout for {tensor_name}: cols={cols} groups={groups_per_row} group_size={scale_group_size}"
        )

    return groups_per_row, scale_group_size


def _read_manifest(artifact_dir: Path) -> OrderedDict[str, TernaryManifestEntry]:
    manifest_path = artifact_dir / "manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing ternary manifest: {manifest_path}")

    entries: OrderedDict[str, TernaryManifestEntry] = OrderedDict()
    entry_forms: dict[str, int] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) == 6:
                name, file_name, rows, cols, packed_bytes, crc32 = parts
                kind = "ternary"
                anchor_file_name = None
                anchor_count = 0
                form = MANIFEST_FORM_BULK
            elif len(parts) == 7:
                name, file_name, rows, cols, packed_bytes, crc32, kind = parts
                if kind not in {"mold"}:
                    raise ValueError(
                        f"Invalid manifest row at {manifest_path}:{line_number}: {raw_line.rstrip()}"
                    )
                anchor_file_name = None
                anchor_count = 0
                form = MANIFEST_FORM_MOLD
            elif len(parts) == 8:
                name, anchor_file_name, rows, cols, packed_bytes, crc32, kind, anchor_count = parts
                if kind != "anchor":
                    raise ValueError(
                        f"Invalid manifest row at {manifest_path}:{line_number}: {raw_line.rstrip()}"
                    )
                file_name = ""
                form = MANIFEST_FORM_LEGACY_ANCHOR
            elif len(parts) == 9:
                name, file_name, rows, cols, packed_bytes, crc32, kind, anchor_file_name, anchor_count = parts
                if kind != "anchor":
                    raise ValueError(
                        f"Invalid manifest row at {manifest_path}:{line_number}: {raw_line.rstrip()}"
                    )
                form = MANIFEST_FORM_UNIFIED_ANCHOR
            else:
                raise ValueError(
                    f"Invalid manifest row at {manifest_path}:{line_number}: {raw_line.rstrip()}"
                )

            entry = TernaryManifestEntry(
                name=name,
                file_name=file_name,
                rows=int(rows),
                cols=int(cols),
                packed_weight_bytes=int(packed_bytes),
                crc32=int(crc32, 16),
                kind=kind,
                anchor_file_name=anchor_file_name,
                anchor_count=int(anchor_count),
            )

            existing = entries.get(name)
            if existing is None:
                entries[name] = entry
                entry_forms[name] = form
                continue

            existing_forms = entry_forms[name]
            if (existing_forms & MANIFEST_FORM_MOLD) or (form & MANIFEST_FORM_MOLD):
                raise ValueError(f"Invalid mixed or duplicate mold manifest row at {manifest_path}:{line_number}: {raw_line.rstrip()}")
            if (existing_forms & MANIFEST_FORM_UNIFIED_ANCHOR) or (form & MANIFEST_FORM_UNIFIED_ANCHOR):
                raise ValueError(
                    f"Unified anchor manifest rows cannot be mixed with duplicates for {name} at {manifest_path}:{line_number}"
                )
            if form == MANIFEST_FORM_BULK and (existing_forms & MANIFEST_FORM_BULK):
                raise ValueError(f"Duplicate ternary bulk manifest row for {name} at {manifest_path}:{line_number}")
            if form == MANIFEST_FORM_LEGACY_ANCHOR and (existing_forms & MANIFEST_FORM_LEGACY_ANCHOR):
                raise ValueError(f"Duplicate legacy anchor manifest row for {name} at {manifest_path}:{line_number}")
            if entry.file_name:
                if existing.file_name and existing.file_name != entry.file_name:
                    raise ValueError(f"Conflicting bulk payload for {name} at {manifest_path}:{line_number}")
                existing.file_name = entry.file_name
                existing.crc32 = entry.crc32
                existing.packed_weight_bytes = entry.packed_weight_bytes
            if entry.anchor_file_name:
                if existing.anchor_file_name and existing.anchor_file_name != entry.anchor_file_name:
                    raise ValueError(f"Conflicting anchor payload for {name} at {manifest_path}:{line_number}")
                existing.anchor_file_name = entry.anchor_file_name
                existing.anchor_count = entry.anchor_count
            if form == MANIFEST_FORM_LEGACY_ANCHOR or existing.kind == "anchor":
                existing.kind = "anchor"
            entry_forms[name] = existing_forms | form

    if not entries:
        raise RuntimeError(f"No ternary tensors found in {manifest_path}")

    for entry in entries.values():
        forms = entry_forms[entry.name]
        if forms == MANIFEST_FORM_MOLD:
            entry.kind = "mold"
        elif forms == MANIFEST_FORM_BULK:
            entry.kind = "ternary"
        elif forms in {MANIFEST_FORM_UNIFIED_ANCHOR, MANIFEST_FORM_BULK | MANIFEST_FORM_LEGACY_ANCHOR}:
            entry.kind = "anchor"
        else:
            raise ValueError(f"Incomplete or ambiguous manifest migration contract for {entry.name} in {manifest_path}")

        if entry.kind in {"ternary", "anchor"}:
            if not entry.file_name:
                raise ValueError(f"Missing ternary bulk payload for {entry.name} in {manifest_path}")
            payload_path = artifact_dir / entry.file_name
            entry.groups_per_row, entry.scale_group_size = _read_ternary_payload_layout(
                payload_path,
                entry.name,
                entry.rows,
                entry.cols,
            )
        if entry.kind == "anchor":
            if not entry.anchor_file_name or entry.anchor_count <= 0:
                raise ValueError(f"Missing anchor payload metadata for {entry.name} in {manifest_path}")

    return entries


def _count_ternary_symbols(packed_bytes: bytes, weight_count: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    seen = 0

    for packed_value in packed_bytes:
        for shift in (0, 2, 4, 6):
            if seen >= weight_count:
                break
            code = (packed_value >> shift) & 0x3
            counts[code] += 1
            seen += 1

    if seen != weight_count:
        raise ValueError(f"Decoded {seen} weights but expected {weight_count}")
    return counts


def _symbol_entropy_bits(symbol_counts: Counter[int], total_count: int) -> float:
    if total_count <= 0:
        return 0.0

    entropy = 0.0
    for code in TERNARY_SYMBOL_CODES:
        count = symbol_counts.get(code, 0)
        if count <= 0:
            continue
        probability = count / total_count
        entropy -= probability * math.log2(probability)
    return entropy


def _load_ternary_tensor_stats(artifact_dir: Path, entry: TernaryManifestEntry) -> TernaryTensorStats:
    payload_path = artifact_dir / entry.file_name
    if not payload_path.is_file():
        raise FileNotFoundError(f"Missing ternary payload: {payload_path}")

    header, data_start = _read_safetensors_header(payload_path)
    packed_meta = _payload_meta(payload_path, header, entry.name, "packed")
    scales_meta = _payload_meta(payload_path, header, entry.name, "scales")

    if packed_meta.get("dtype") != "U8":
        raise ValueError(f"Unexpected packed dtype for {entry.name}: {packed_meta.get('dtype')}")

    scales_dtype = scales_meta.get("dtype")
    if scales_dtype not in {"F32", "F16", "BF16"}:
        raise ValueError(f"Unexpected scale dtype for {entry.name}: {scales_dtype}")

    packed_shape = _tensor_shape(packed_meta, payload_path)
    scales_shape = _tensor_shape(scales_meta, payload_path)
    if packed_shape != entry.packed_shape:
        raise ValueError(
            f"Packed payload shape mismatch for {entry.name}: expected {entry.packed_shape}, got {packed_shape}"
        )
    if scales_shape != entry.scale_shape:
        raise ValueError(
            f"Scale payload shape mismatch for {entry.name}: expected {entry.scale_shape}, got {scales_shape}"
        )

    packed_start, packed_end = _tensor_offsets(payload_path, packed_meta, data_start)
    scales_start, scales_end = _tensor_offsets(payload_path, scales_meta, data_start)
    packed_size = packed_end - packed_start
    scale_size = scales_end - scales_start

    if packed_size != entry.packed_weight_bytes:
        raise ValueError(
            f"Packed byte count mismatch for {entry.name}: expected {entry.packed_weight_bytes}, got {packed_size}"
        )
    if scale_size != entry.scale_bytes:
        raise ValueError(
            f"Scale byte count mismatch for {entry.name}: expected {entry.scale_bytes}, got {scale_size}"
        )

    with payload_path.open("rb") as handle:
        handle.seek(packed_start)
        packed_bytes = handle.read(packed_size)
        if len(packed_bytes) != packed_size:
            raise ValueError(f"Truncated packed payload for {entry.name}: {payload_path}")
        handle.seek(scales_start)
        scale_bytes = handle.read(scale_size)
        if len(scale_bytes) != scale_size:
            raise ValueError(f"Truncated scale payload for {entry.name}: {payload_path}")

    actual_crc32 = zlib.crc32(packed_bytes) & 0xFFFFFFFF
    actual_crc32 = zlib.crc32(scale_bytes, actual_crc32) & 0xFFFFFFFF
    if actual_crc32 != entry.crc32:
        raise ValueError(
            f"CRC mismatch for {entry.name}: manifest={entry.crc32:08x} actual={actual_crc32:08x}"
        )

    symbol_counts = _count_ternary_symbols(packed_bytes, entry.weight_count)
    packed_symbol_bits_per_weight = (packed_size * 8.0) / entry.weight_count
    total_bits_per_weight = ((packed_size + scale_size) * 8.0) / entry.weight_count
    empirical_symbol_entropy_bits = _symbol_entropy_bits(symbol_counts, entry.weight_count)

    return TernaryTensorStats(
        name=entry.name,
        weight_count=entry.weight_count,
        packed_bytes=packed_size,
        scale_bytes=scale_size,
        packed_symbol_bits_per_weight=packed_symbol_bits_per_weight,
        total_bits_per_weight=total_bits_per_weight,
        empirical_symbol_entropy_bits=empirical_symbol_entropy_bits,
        symbol_counts=symbol_counts,
    )


def _read_anchor_budget_ppm(artifact_dir: Path, entry: TernaryManifestEntry) -> int:
    if entry.kind != "anchor" or not entry.anchor_file_name:
        return 0

    anchor_path = artifact_dir / entry.anchor_file_name
    if not anchor_path.is_file():
        raise FileNotFoundError(f"Missing anchor payload: {anchor_path}")

    if anchor_path.suffix == ".safetensors":
        header, data_start = _read_safetensors_header(anchor_path)
        metadata = header.get("__metadata__")
        entries_meta = header.get(f"{entry.name}.anchor_entries")

        if not isinstance(metadata, dict):
            raise ValueError(f"Missing anchor safetensors metadata: {anchor_path}")
        if not isinstance(entries_meta, dict):
            raise ValueError(f"Missing anchor_entries tensor for {entry.name}: {anchor_path}")
        if entries_meta.get("dtype") != "U16":
            raise ValueError(f"Unsupported anchor_entries dtype in {anchor_path}: {entries_meta.get('dtype')}")
        if _tensor_shape(entries_meta, anchor_path) != (entry.anchor_count, 4):
            raise ValueError(
                f"Anchor entries shape mismatch for {entry.name}: expected {(entry.anchor_count, 4)}, got {_tensor_shape(entries_meta, anchor_path)}"
            )

        start, end = _tensor_offsets(anchor_path, entries_meta, data_start)
        if end - start != entry.anchor_count * 8:
            raise ValueError(
                f"Anchor entries byte count mismatch for {entry.name}: expected {entry.anchor_count * 8}, got {end - start}"
            )

        format_version = _metadata_int(metadata, "format_version")
        rows = _metadata_int(metadata, "rows")
        cols = _metadata_int(metadata, "cols")
        anchor_count = _metadata_int(metadata, "anchor_count")
        budget_ppm = _metadata_int(metadata, "budget_ppm")
        value_dtype = _metadata_str(metadata, "value_dtype")
        tensor_name = _metadata_str(metadata, "tensor_name")

        if format_version != ANCHOR_SAFETENSORS_VERSION:
            raise ValueError(f"Unsupported anchor safetensors format version in {anchor_path}: {format_version}")
        if rows != entry.rows or cols != entry.cols or anchor_count != entry.anchor_count:
            raise ValueError(f"Anchor safetensors metadata mismatch for {entry.name}: {anchor_path}")
        if value_dtype != "BF16":
            raise ValueError(f"Unsupported anchor value dtype in {anchor_path}: {value_dtype!r}")
        if tensor_name != entry.name:
            raise ValueError(f"Anchor safetensors tensor_name mismatch for {entry.name}: {anchor_path}")
        if budget_ppm is None:
            raise ValueError(f"Missing anchor budget metadata in {anchor_path}")

        return int(budget_ppm)

    with anchor_path.open("rb") as anchor_file:
        header = anchor_file.read(ANCHOR_METADATA_STRUCT.size)
    if len(header) != ANCHOR_METADATA_STRUCT.size:
        raise ValueError(f"Truncated anchor payload header: {anchor_path}")

    (
        magic,
        version,
        rows,
        cols,
        anchor_count,
        _scale_group_size,
        _groups_per_row,
        _saliency_mode,
        budget_ppm,
        _saliency_cutoff,
        _anchor_value_rms,
        _bulk_gamma_mean,
        _max_row_nnz,
        _crc32,
    ) = ANCHOR_METADATA_STRUCT.unpack(header)

    if magic != ANCHOR_MAGIC or version != ANCHOR_VERSION:
        raise ValueError(f"Invalid anchor payload header: {anchor_path}")
    if rows != entry.rows or cols != entry.cols or anchor_count != entry.anchor_count:
        raise ValueError(f"Anchor payload metadata mismatch for {entry.name}: {anchor_path}")
    return int(budget_ppm)


def _load_package_sidecar_metadata(model_dir: Path) -> OrderedDict:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index_data = _read_json(index_path)
        metadata = index_data.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid metadata in {index_path}")
        return OrderedDict(metadata)

    shard_path = model_dir / "model.safetensors"
    if not shard_path.is_file():
        shard_path = _iter_model_shards(model_dir)[0]

    header, _ = _read_safetensors_header(shard_path)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing __metadata__ in {shard_path}")
    return OrderedDict(metadata)


def _load_package_config(model_dir: Path) -> OrderedDict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config.json payload: {config_path}")
    return config


def _parse_hybrid_config_metadata(metadata: OrderedDict, source: str) -> HybridPackageMetadata:
    mixed_precision_anchors = _metadata_bool(metadata, "sapphire_mixed_precision_anchors")
    anchor_budget_ppm = _metadata_int(metadata, "sapphire_anchor_budget_ppm")
    format_version = _metadata_int(metadata, PACKAGE_FORMAT_VERSION_KEY)
    feature_contract = _metadata_str(metadata, PACKAGE_FEATURE_CONTRACT_KEY)
    storage_encoding = _metadata_str(metadata, PACKAGE_STORAGE_ENCODING_KEY)

    if mixed_precision_anchors is not True:
        raise ValueError(f"{source} is missing sapphire_mixed_precision_anchors=true")
    if anchor_budget_ppm is None or anchor_budget_ppm <= 0:
        raise ValueError(f"{source} is missing a positive sapphire_anchor_budget_ppm")
    if format_version != PACKAGE_FORMAT_VERSION_VALUE:
        raise ValueError(f"{source} has unsupported {PACKAGE_FORMAT_VERSION_KEY}: {format_version!r}")
    if feature_contract != PACKAGE_FEATURE_CONTRACT_VALUE:
        raise ValueError(f"{source} has unsupported {PACKAGE_FEATURE_CONTRACT_KEY}: {feature_contract!r}")
    if storage_encoding != PACKAGE_STORAGE_ENCODING:
        raise ValueError(f"{source} has unsupported {PACKAGE_STORAGE_ENCODING_KEY}: {storage_encoding!r}")

    return HybridPackageMetadata(
        mixed_precision_anchors=True,
        anchor_budget_ppm=anchor_budget_ppm,
        format_version=format_version,
        feature_contract=feature_contract,
        storage_encoding=storage_encoding,
    )


def _parse_hybrid_sidecar_metadata(metadata: OrderedDict, source: str) -> HybridPackageMetadata:
    package_metadata = _parse_hybrid_config_metadata(metadata, source)
    package_storage_encoding = _metadata_str(metadata, "storage_encoding")
    ternary_manifest = _metadata_str(metadata, "ternary_manifest")

    if package_storage_encoding != PACKAGE_STORAGE_ENCODING:
        raise ValueError(f"{source} has unsupported storage_encoding: {package_storage_encoding!r}")
    if ternary_manifest != "manifest.tsv":
        raise ValueError(f"{source} has unsupported ternary_manifest: {ternary_manifest!r}")
    return package_metadata


def _validate_hybrid_package_metadata(model_dir: Path,
                                      entries: list[TernaryManifestEntry]) -> HybridPackageMetadata | None:
    anchor_entries = [entry for entry in entries if entry.kind == "anchor"]
    config = _load_package_config(model_dir)

    if not anchor_entries:
        stray_keys = [
            key
            for key in (
                "sapphire_mixed_precision_anchors",
                "sapphire_anchor_budget_ppm",
                PACKAGE_FORMAT_VERSION_KEY,
                PACKAGE_FEATURE_CONTRACT_KEY,
                PACKAGE_STORAGE_ENCODING_KEY,
            )
            if key in config
        ]
        if stray_keys:
            raise ValueError(
                "Non-anchor package advertises hybrid capability metadata in config.json: " + ", ".join(stray_keys)
            )
        return None

    budgets = sorted({_read_anchor_budget_ppm(model_dir, entry) for entry in anchor_entries})
    if len(budgets) != 1 or budgets[0] <= 0:
        raise ValueError(f"Anchor-bearing package has inconsistent or missing budget metadata: {budgets}")

    config_metadata = _parse_hybrid_config_metadata(config, f"{model_dir / 'config.json'}")
    sidecar_metadata = _parse_hybrid_sidecar_metadata(_load_package_sidecar_metadata(model_dir), str(model_dir))

    if config_metadata.anchor_budget_ppm != budgets[0]:
        raise ValueError(
            f"config.json sapphire_anchor_budget_ppm={config_metadata.anchor_budget_ppm} does not match payload budget {budgets[0]}"
        )
    if sidecar_metadata.anchor_budget_ppm != budgets[0]:
        raise ValueError(
            f"Model package metadata sapphire_anchor_budget_ppm={sidecar_metadata.anchor_budget_ppm} does not match payload budget {budgets[0]}"
        )
    if config_metadata != sidecar_metadata:
        raise ValueError("config.json hybrid metadata does not match package header/index metadata")

    return config_metadata


def _weighted_average(items: Iterable[tuple[int, float]]) -> float:
    total_weight = 0
    total_value = 0.0
    for weight, value in items:
        total_weight += weight
        total_value += weight * value
    if total_weight == 0:
        return 0.0
    return total_value / total_weight


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0/0"
    return f"{numerator}/{denominator} ({numerator / denominator:.4%})"


def _resolve_progress_setting(show_progress: bool | None) -> bool:
    if show_progress is None:
        return sys.stderr.isatty()
    return show_progress


def _emit_progress_line(message: str, interactive: bool) -> None:
    if interactive:
        sys.stderr.write(f"\r\x1b[2K{message}")
    else:
        sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def _collect_ternary_tensor_stats(artifact_dir: Path,
                                  entries: list[TernaryManifestEntry],
                                  show_progress: bool) -> list[TernaryTensorStats]:
    tensor_stats: list[TernaryTensorStats] = []
    total_entries = len(entries)
    interactive_progress = show_progress and sys.stderr.isatty()

    if show_progress and total_entries:
        print(f"Scanning {total_entries} ternary tensor payload(s)...", file=sys.stderr, flush=True)

    for index, entry in enumerate(entries, start=1):
        if show_progress:
            _emit_progress_line(f"[{index}/{total_entries}] {entry.name}", interactive_progress)
        tensor_stats.append(_load_ternary_tensor_stats(artifact_dir, entry))

    if show_progress and interactive_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()
    if show_progress and total_entries:
        print(f"Loaded {total_entries} ternary tensor payload(s).", file=sys.stderr, flush=True)

    return tensor_stats


def _print_ternary_report(artifact_dir: Path,
                          tensor_stats: list[TernaryTensorStats],
                          expect_max_bits: float | None,
                          metric: str) -> int:
    total_weights = sum(item.weight_count for item in tensor_stats)
    total_packed_bytes = sum(item.packed_bytes for item in tensor_stats)
    total_scale_bytes = sum(item.scale_bytes for item in tensor_stats)
    avg_packed_bits = _weighted_average((item.weight_count, item.packed_symbol_bits_per_weight) for item in tensor_stats)
    avg_total_bits = _weighted_average((item.weight_count, item.total_bits_per_weight) for item in tensor_stats)
    avg_entropy_bits = _weighted_average((item.weight_count, item.empirical_symbol_entropy_bits) for item in tensor_stats)

    merged_counts: Counter[int] = Counter()
    for item in tensor_stats:
        merged_counts.update(item.symbol_counts)

    print(f"artifact_type: sapphire_ternary_output")
    print(f"artifact_dir: {artifact_dir}")
    print(f"converted_tensors: {len(tensor_stats)}")
    print(f"total_weights: {total_weights}")
    print(f"packed_symbol_bytes: {total_packed_bytes}")
    print(f"scale_bytes: {total_scale_bytes}")
    print(f"packed_symbol_bits_per_weight: {avg_packed_bits:.6f}")
    print(f"total_bits_per_weight_including_scales: {avg_total_bits:.6f}")
    print(f"empirical_symbol_entropy_bits: {avg_entropy_bits:.6f}")
    print("note: native ternary payloads use fixed 2-bit symbol packing plus per-row scales; empirical entropy is a lower bound, not the on-disk encoding")
    print(f"zero_symbols: {_format_ratio(merged_counts.get(0, 0), total_weights)}")
    print(f"positive_symbols: {_format_ratio(merged_counts.get(1, 0), total_weights)}")
    print(f"negative_symbols: {_format_ratio(merged_counts.get(2, 0), total_weights)}")
    if merged_counts.get(3, 0):
        print(f"reserved_code_3_symbols: {_format_ratio(merged_counts[3], total_weights)}")

    hottest = sorted(tensor_stats, key=lambda item: item.total_bits_per_weight, reverse=True)[:5]
    print("top_tensors_by_total_bits_per_weight:")
    for item in hottest:
        print(f"  {item.name}: total={item.total_bits_per_weight:.6f}, packed={item.packed_symbol_bits_per_weight:.6f}, entropy={item.empirical_symbol_entropy_bits:.6f}")

    metric_value = {
        "packed": avg_packed_bits,
        "total": avg_total_bits,
        "entropy": avg_entropy_bits,
    }[metric]

    if expect_max_bits is None:
        return 0

    if metric_value <= expect_max_bits + 1e-9:
        print(f"PASS: {metric}_bits_per_weight={metric_value:.6f} <= {expect_max_bits:.6f}")
        return 0

    print(f"FAIL: {metric}_bits_per_weight={metric_value:.6f} > {expect_max_bits:.6f}", file=sys.stderr)
    return 1


def _iter_model_shards(model_dir: Path) -> list[Path]:
    single_path = model_dir / "model.safetensors"
    if single_path.is_file():
        return [single_path]

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index_data = _read_json(index_path)
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {index_path}")
        shard_names = []
        seen = set()
        for shard_name in weight_map.values():
            if shard_name in seen:
                continue
            seen.add(shard_name)
            shard_names.append(shard_name)
        shard_paths = [model_dir / str(name) for name in shard_names]
        missing = [path for path in shard_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing shard(s) listed in index: {', '.join(str(path) for path in missing)}")
        return shard_paths

    shard_paths = sorted(path for path in model_dir.glob("model-*.safetensors") if path.is_file())
    if shard_paths:
        return shard_paths

    raise FileNotFoundError(
        f"No supported model package found in {model_dir}; expected manifest.tsv or safetensors model files"
    )


def _load_model_tensor_stats(model_dir: Path) -> list[ModelTensorStats]:
    stats: list[ModelTensorStats] = []
    seen_names: set[str] = set()
    for shard_path in _iter_model_shards(model_dir):
        header, data_start = _read_safetensors_header(shard_path)
        for tensor_name, meta in header.items():
            if tensor_name == "__metadata__":
                continue
            if tensor_name in seen_names:
                continue
            if not isinstance(meta, dict):
                raise ValueError(f"Invalid tensor metadata in {shard_path}: {tensor_name}")
            dtype = str(meta.get("dtype"))
            bits_per_element = SUPPORTED_MODEL_DTYPES.get(dtype)
            if bits_per_element is None:
                raise ValueError(f"Unsupported tensor dtype {dtype} in {shard_path}: {tensor_name}")
            shape = _tensor_shape(meta, shard_path)
            element_count = math.prod(shape)
            data_start_offset, data_end_offset = _tensor_offsets(shard_path, meta, data_start)
            data_bytes = data_end_offset - data_start_offset
            expected_bytes = (element_count * bits_per_element) // 8
            if data_bytes != expected_bytes:
                raise ValueError(
                    f"Tensor byte count mismatch for {tensor_name}: expected {expected_bytes}, got {data_bytes}"
                )
            stats.append(
                ModelTensorStats(
                    name=str(tensor_name),
                    dtype=dtype,
                    element_count=element_count,
                    data_bytes=data_bytes,
                    bits_per_element=float(bits_per_element),
                )
            )
            seen_names.add(str(tensor_name))
    return stats


def _print_model_report(model_dir: Path,
                        tensor_stats: list[ModelTensorStats],
                        expect_max_bits: float | None) -> int:
    total_elements = sum(item.element_count for item in tensor_stats)
    total_bytes = sum(item.data_bytes for item in tensor_stats)
    average_bits = (total_bytes * 8.0) / total_elements if total_elements else 0.0
    dtype_totals: Counter[str] = Counter()
    for item in tensor_stats:
        dtype_totals[item.dtype] += item.element_count

    print("artifact_type: safetensors_model_package")
    print(f"artifact_dir: {model_dir}")
    print(f"tensor_count: {len(tensor_stats)}")
    print(f"total_elements: {total_elements}")
    print(f"total_data_bytes: {total_bytes}")
    print(f"average_bits_per_element: {average_bits:.6f}")
    print("note: report reflects physical tensor storage in the loadable model package")
    print("dtype_mix_by_element_count:")
    for dtype, count in sorted(dtype_totals.items()):
        print(f"  {dtype}: {_format_ratio(count, total_elements)}")

    if expect_max_bits is None:
        return 0

    if average_bits <= expect_max_bits + 1e-9:
        print(f"PASS: average_bits_per_element={average_bits:.6f} <= {expect_max_bits:.6f}")
        return 0

    print(f"FAIL: average_bits_per_element={average_bits:.6f} > {expect_max_bits:.6f}", file=sys.stderr)
    return 1


def _print_mixed_model_report(model_dir: Path,
                              tensor_stats: list[ModelTensorStats],
                              ternary_stats: list[TernaryTensorStats],
                              hybrid_metadata: HybridPackageMetadata | None,
                              expect_max_bits: float | None) -> int:
    total_physical_bytes = sum(item.data_bytes for item in tensor_stats)
    passthrough_logical_weights = sum(
        item.element_count
        for item in tensor_stats
        if not item.name.endswith(".packed") and not item.name.endswith(".scales")
    )
    ternary_logical_weights = sum(item.weight_count for item in ternary_stats)
    total_logical_weights = passthrough_logical_weights + ternary_logical_weights
    average_bits = (total_physical_bytes * 8.0) / total_logical_weights if total_logical_weights else 0.0
    total_ternary_bytes = sum(item.packed_bytes + item.scale_bytes for item in ternary_stats)
    avg_ternary_total_bits = _weighted_average((item.weight_count, item.total_bits_per_weight) for item in ternary_stats)
    avg_ternary_entropy_bits = _weighted_average((item.weight_count, item.empirical_symbol_entropy_bits) for item in ternary_stats)

    dtype_totals: Counter[str] = Counter()
    for item in tensor_stats:
        dtype_totals[item.dtype] += item.data_bytes

    if hybrid_metadata is not None:
        print("artifact_type: anchor_bearing_mixed_safetensors_model_package")
    else:
        print("artifact_type: mixed_safetensors_model_package")
    print(f"artifact_dir: {model_dir}")
    print(f"physical_tensor_count: {len(tensor_stats)}")
    print(f"ternary_logical_tensor_count: {len(ternary_stats)}")
    print(f"passthrough_logical_tensor_count: {sum(1 for item in tensor_stats if not item.name.endswith('.packed') and not item.name.endswith('.scales'))}")
    print(f"total_logical_weights: {total_logical_weights}")
    print(f"total_physical_data_bytes: {total_physical_bytes}")
    print(f"average_bits_per_logical_weight: {average_bits:.6f}")
    print(f"ternary_physical_data_bytes: {total_ternary_bytes}")
    print(f"ternary_total_bits_per_weight_including_scales: {avg_ternary_total_bits:.6f}")
    print(f"ternary_empirical_symbol_entropy_bits: {avg_ternary_entropy_bits:.6f}")
    print("note: ternary tensors are stored as fixed 2-bit packed symbols plus scales; the model-wide average includes BF16 passthrough tensors")
    print("physical_dtype_mix_by_byte_count:")
    for dtype, count in sorted(dtype_totals.items()):
        print(f"  {dtype}: {_format_ratio(count, total_physical_bytes)}")
    print(f"logical_ternary_weight_share: {_format_ratio(ternary_logical_weights, total_logical_weights)}")
    if hybrid_metadata is not None:
        print("hybrid_capability_metadata:")
        print(f"  sapphire_mixed_precision_anchors: {str(hybrid_metadata.mixed_precision_anchors).lower()}")
        print(f"  sapphire_anchor_budget_ppm: {hybrid_metadata.anchor_budget_ppm}")
        print(f"  {PACKAGE_FORMAT_VERSION_KEY}: {hybrid_metadata.format_version}")
        print(f"  {PACKAGE_FEATURE_CONTRACT_KEY}: {hybrid_metadata.feature_contract}")
        print(f"  {PACKAGE_STORAGE_ENCODING_KEY}: {hybrid_metadata.storage_encoding}")

    if expect_max_bits is None:
        return 0

    if average_bits <= expect_max_bits + 1e-9:
        print(f"PASS: average_bits_per_logical_weight={average_bits:.6f} <= {expect_max_bits:.6f}")
        return 0

    print(f"FAIL: average_bits_per_logical_weight={average_bits:.6f} > {expect_max_bits:.6f}", file=sys.stderr)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify actual storage bits per weight for Sapphire ternary outputs or repacked model packages"
    )
    parser.add_argument(
        "artifact_dir",
        type=Path,
        help="Directory containing either manifest.tsv or a model.safetensors package",
    )
    parser.add_argument(
        "--expect-max-bits",
        type=float,
        default=None,
        help="Fail if the selected metric exceeds this threshold",
    )
    parser.add_argument(
        "--metric",
        choices=("packed", "total", "entropy"),
        default="total",
        help="Metric used with --expect-max-bits for ternary outputs; model packages always use actual average bits per element",
    )
    parser.add_argument(
        "--progress",
        dest="show_progress",
        action="store_true",
        help="Show per-tensor progress while loading manifest-based ternary payloads",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable per-tensor progress while loading manifest-based ternary payloads",
    )
    parser.set_defaults(show_progress=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {artifact_dir}")

    manifest_path = artifact_dir / "manifest.tsv"
    ternary_entries = None
    if manifest_path.is_file():
        ternary_entries = list(_read_manifest(artifact_dir).values())

    model_tensor_stats = None
    try:
        model_tensor_stats = _load_model_tensor_stats(artifact_dir)
    except FileNotFoundError:
        model_tensor_stats = None

    if ternary_entries is not None and model_tensor_stats is not None:
        ternary_stats = _collect_ternary_tensor_stats(
            artifact_dir,
            ternary_entries,
            show_progress=_resolve_progress_setting(args.show_progress),
        )
        hybrid_metadata = _validate_hybrid_package_metadata(artifact_dir, ternary_entries)
        return _print_mixed_model_report(
            artifact_dir,
            model_tensor_stats,
            ternary_stats,
            hybrid_metadata,
            args.expect_max_bits,
        )

    if ternary_entries is not None:
        entries = list(_read_manifest(artifact_dir).values())
        tensor_stats = _collect_ternary_tensor_stats(
            artifact_dir,
            entries,
            show_progress=_resolve_progress_setting(args.show_progress),
        )
        return _print_ternary_report(artifact_dir, tensor_stats, args.expect_max_bits, args.metric)

    if model_tensor_stats is not None:
        return _print_model_report(artifact_dir, model_tensor_stats, args.expect_max_bits)

    raise FileNotFoundError(
        f"No supported model artifact found in {artifact_dir}; expected manifest.tsv and/or safetensors model files"
    )


if __name__ == "__main__":
    raise SystemExit(main())
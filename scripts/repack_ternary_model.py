#!/usr/bin/env python3
"""Repack Sapphire ternary outputs into a mixed BF16 plus packed ternary model package."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

import numpy as np


BF16_BYTES = np.dtype(np.uint16).itemsize
DEFAULT_MAX_SHARD_SIZE = 2 * 1024**3
TERNARY_PACKED_WEIGHTS_PER_BYTE = 4
METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
    "chat_template.jinja",
    "README.md",
)


WriteTensorData = Callable[[BinaryIO], None]


@dataclass(frozen=True)
class BaseTensorRef:
    name: str
    shape: tuple[int, ...]
    dtype: str
    file_path: Path
    data_start: int
    data_end: int

    @property
    def bf16_byte_count(self) -> int:
        return math.prod(self.shape) * BF16_BYTES


@dataclass(frozen=True)
class TernaryManifestEntry:
    name: str
    file_name: str
    rows: int
    cols: int
    packed_weight_bytes: int
    crc32: int
    kind: str = "ternary"

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.rows, self.cols)

    @property
    def packed_cols(self) -> int:
        return (self.cols + TERNARY_PACKED_WEIGHTS_PER_BYTE - 1) // TERNARY_PACKED_WEIGHTS_PER_BYTE

    @property
    def packed_shape(self) -> tuple[int, ...]:
        return (self.rows, self.packed_cols)

    @property
    def scale_shape(self) -> tuple[int, ...]:
        return (self.rows,)

    @property
    def scale_byte_count(self) -> int:
        return self.rows * np.dtype("<f4").itemsize

    @property
    def stored_byte_count(self) -> int:
        if self.kind == "mold":
            return self.packed_weight_bytes
        return self.packed_weight_bytes + self.scale_byte_count


@dataclass(frozen=True)
class OutputHeaderTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    byte_count: int


@dataclass(frozen=True)
class OutputTensorItem:
    logical_name: str
    header_tensors: tuple[OutputHeaderTensor, ...]
    stored_byte_count: int
    writer: WriteTensorData
    source_kind: str
    manifest_entry: TernaryManifestEntry | None = None


def _parse_shard_size(value: str) -> int:
    pattern = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*$", re.IGNORECASE)
    match = pattern.match(value)
    if not match:
        raise ValueError(f"Invalid shard size: {value}")

    amount = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "kb": 10**3,
        "kib": 2**10,
        "mb": 10**6,
        "mib": 2**20,
        "gb": 10**9,
        "gib": 2**30,
        "tb": 10**12,
        "tib": 2**40,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported shard size unit: {unit}")
    return int(amount * multipliers[unit])


def _float32_to_bf16_words(values: np.ndarray) -> np.ndarray:
    float32_values = np.asarray(values, dtype=np.float32)
    float32_bits = float32_values.view(np.uint32)
    rounded_bits = float32_bits + np.uint32(0x7FFF) + ((float32_bits >> np.uint32(16)) & np.uint32(1))
    bf16_words = (rounded_bits >> np.uint32(16)).astype(np.uint16)
    if sys.byteorder != "little":
        bf16_words = bf16_words.byteswap()
    return bf16_words


def _load_json(path: Path) -> OrderedDict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file, object_pairs_hook=OrderedDict)


def _read_safetensors_header(path: Path) -> tuple[OrderedDict, int]:
    with path.open("rb") as file:
        header_len_bytes = file.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Invalid safetensors header prefix: {path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = file.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError(f"Truncated safetensors header: {path}")

    header = json.loads(header_bytes, object_pairs_hook=OrderedDict)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header object: {path}")
    return header, 8 + int(header_len)


def _tensor_shape(meta: dict) -> tuple[int, ...]:
    shape = meta.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"Invalid tensor shape metadata: {meta}")
    return tuple(int(dim) for dim in shape)


def _tensor_offsets(path: Path, meta: dict, data_start: int) -> tuple[int, int]:
    offsets = meta.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"Invalid tensor offsets for {path}: {meta}")
    start = data_start + int(offsets[0])
    end = data_start + int(offsets[1])
    if end < start:
        raise ValueError(f"Negative tensor data range for {path}")
    return start, end


def _collect_shard_glob_paths(model_dir: Path) -> list[Path]:
    return sorted(path for path in model_dir.glob("model-*.safetensors") if path.is_file())


def _collect_tensors_from_path(path: Path) -> OrderedDict[str, BaseTensorRef]:
    header, data_start = _read_safetensors_header(path)
    tensors: OrderedDict[str, BaseTensorRef] = OrderedDict()

    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid tensor metadata in {path}: {name}")
        dtype = meta.get("dtype")
        if dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(f"Unsupported tensor dtype {dtype} in {path}: {name}")
        data_begin, data_end = _tensor_offsets(path, meta, data_start)
        tensors[name] = BaseTensorRef(
            name=name,
            shape=_tensor_shape(meta),
            dtype=str(dtype),
            file_path=path,
            data_start=data_begin,
            data_end=data_end,
        )

    return tensors


def _collect_base_tensors(model_dir: Path) -> OrderedDict[str, BaseTensorRef]:
    single_path = model_dir / "model.safetensors"
    if single_path.is_file():
        return _collect_tensors_from_path(single_path)

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index_data = _load_json(index_path)
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid weight_map in {index_path}")

        shard_headers: dict[Path, tuple[OrderedDict, int]] = {}
        tensors: OrderedDict[str, BaseTensorRef] = OrderedDict()
        for tensor_name, shard_name in weight_map.items():
            shard_path = model_dir / str(shard_name)
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing shard listed in index: {shard_path}")
            if shard_path not in shard_headers:
                shard_headers[shard_path] = _read_safetensors_header(shard_path)
            header, data_start = shard_headers[shard_path]
            meta = header.get(tensor_name)
            if not isinstance(meta, dict):
                raise KeyError(f"Tensor {tensor_name} missing from {shard_path}")
            dtype = meta.get("dtype")
            if dtype not in {"BF16", "F16", "F32"}:
                raise ValueError(f"Unsupported tensor dtype {dtype} in {shard_path}: {tensor_name}")
            data_begin, data_end = _tensor_offsets(shard_path, meta, data_start)
            tensors[tensor_name] = BaseTensorRef(
                name=str(tensor_name),
                shape=_tensor_shape(meta),
                dtype=str(dtype),
                file_path=shard_path,
                data_start=data_begin,
                data_end=data_end,
            )
        return tensors

    shard_paths = _collect_shard_glob_paths(model_dir)
    if not shard_paths:
        raise FileNotFoundError(
            f"No supported base model package found in {model_dir}; expected model.safetensors or sharded safetensors"
        )

    tensors: OrderedDict[str, BaseTensorRef] = OrderedDict()
    for shard_path in shard_paths:
        shard_tensors = _collect_tensors_from_path(shard_path)
        for name, tensor in shard_tensors.items():
            tensors.setdefault(name, tensor)
    return tensors


def _read_manifest(ternary_dir: Path) -> OrderedDict[str, TernaryManifestEntry]:
    manifest_path = ternary_dir / "manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing ternary manifest: {manifest_path}")

    entries: OrderedDict[str, TernaryManifestEntry] = OrderedDict()
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line_number, raw_line in enumerate(manifest_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 6:
                name, file_name, rows, cols, packed_bytes, crc32 = parts
                kind = "ternary"
            elif len(parts) == 7:
                name, file_name, rows, cols, packed_bytes, crc32, kind = parts
                if kind not in {"mold"}:
                    raise ValueError(f"Invalid manifest row at {manifest_path}:{line_number}: {raw_line.rstrip()}")
            else:
                raise ValueError(f"Invalid manifest row at {manifest_path}:{line_number}: {raw_line.rstrip()}")

            entries[name] = TernaryManifestEntry(
                name=name,
                file_name=file_name,
                rows=int(rows),
                cols=int(cols),
                packed_weight_bytes=int(packed_bytes),
                crc32=int(crc32, 16),
                kind=kind,
            )

    if not entries:
        raise RuntimeError(f"No ternary tensors found in {manifest_path}")
    return entries


def _payload_meta(path: Path, header: OrderedDict, tensor_name: str, suffix: str) -> dict:
    meta = header.get(f"{tensor_name}.{suffix}")
    if not isinstance(meta, dict):
        raise KeyError(f"Missing {tensor_name}.{suffix} in {path}")
    return meta


def _load_ternary_payload_bytes(ternary_dir: Path, entry: TernaryManifestEntry) -> tuple[bytes, bytes]:
    payload_path = ternary_dir / entry.file_name
    if not payload_path.is_file():
        raise FileNotFoundError(f"Missing ternary payload: {payload_path}")

    header, data_start = _read_safetensors_header(payload_path)
    packed_meta = _payload_meta(payload_path, header, entry.name, "packed")
    scales_meta = _payload_meta(payload_path, header, entry.name, "scales")

    if packed_meta.get("dtype") != "U8" or scales_meta.get("dtype") != "F32":
        raise ValueError(f"Unexpected ternary payload dtypes in {payload_path}")

    packed_shape = _tensor_shape(packed_meta)
    scales_shape = _tensor_shape(scales_meta)
    if packed_shape != entry.packed_shape:
        raise ValueError(
            f"Packed payload shape mismatch for {entry.name}: expected {entry.packed_shape}, got {packed_shape}"
        )
    if scales_shape != entry.scale_shape:
        raise ValueError(f"Scale payload shape mismatch for {entry.name}: expected {entry.scale_shape}, got {scales_shape}")

    packed_start, packed_end = _tensor_offsets(payload_path, packed_meta, data_start)
    scales_start, scales_end = _tensor_offsets(payload_path, scales_meta, data_start)
    packed_size = packed_end - packed_start
    scales_size = scales_end - scales_start
    if packed_size != entry.packed_weight_bytes:
        raise ValueError(
            f"Packed byte count mismatch for {entry.name}: expected {entry.packed_weight_bytes}, got {packed_size}"
        )
    if scales_size != entry.scale_byte_count:
        raise ValueError(f"Scale byte count mismatch for {entry.name}: {scales_size}")

    with payload_path.open("rb") as payload_file:
        payload_file.seek(packed_start)
        packed_bytes = payload_file.read(packed_size)
        if len(packed_bytes) != packed_size:
            raise ValueError(f"Truncated packed payload for {entry.name}: {payload_path}")
        payload_file.seek(scales_start)
        scale_bytes = payload_file.read(scales_size)
        if len(scale_bytes) != scales_size:
            raise ValueError(f"Truncated scale payload for {entry.name}: {payload_path}")

    actual_crc32 = zlib.crc32(packed_bytes) & 0xFFFFFFFF
    actual_crc32 = zlib.crc32(scale_bytes, actual_crc32) & 0xFFFFFFFF
    if actual_crc32 != entry.crc32:
        raise ValueError(
            f"CRC mismatch for {entry.name}: manifest={entry.crc32:08x} actual={actual_crc32:08x}"
        )

    return packed_bytes, scale_bytes


def _numpy_dtype_for_tensor(dtype: str) -> np.dtype:
    if dtype == "BF16":
        return np.dtype(np.uint16)
    if dtype == "F16":
        return np.dtype(np.float16)
    if dtype == "F32":
        return np.dtype("<f4")
    raise ValueError(f"Unsupported source tensor dtype: {dtype}")


def _write_base_tensor_data(output_file: BinaryIO, tensor: BaseTensorRef, chunk_elements: int = 1 << 20) -> None:
    if tensor.dtype == "BF16":
        chunk_bytes = chunk_elements * BF16_BYTES
        with tensor.file_path.open("rb") as source_file:
            source_file.seek(tensor.data_start)
            remaining = tensor.data_end - tensor.data_start
            while remaining > 0:
                read_bytes = min(remaining, chunk_bytes)
                chunk = source_file.read(read_bytes)
                if len(chunk) != read_bytes:
                    raise ValueError(f"Truncated tensor data for {tensor.name}: {tensor.file_path}")
                output_file.write(chunk)
                remaining -= read_bytes
        return

    numpy_dtype = _numpy_dtype_for_tensor(tensor.dtype)
    chunk_bytes = chunk_elements * numpy_dtype.itemsize
    with tensor.file_path.open("rb") as source_file:
        source_file.seek(tensor.data_start)
        remaining = tensor.data_end - tensor.data_start
        while remaining > 0:
            read_bytes = min(remaining, chunk_bytes)
            chunk = source_file.read(read_bytes)
            if len(chunk) != read_bytes:
                raise ValueError(f"Truncated tensor data for {tensor.name}: {tensor.file_path}")
            values = np.frombuffer(chunk, dtype=numpy_dtype)
            bf16_words = _float32_to_bf16_words(values)
            output_file.write(bf16_words.tobytes(order="C"))
            remaining -= read_bytes


def _write_ternary_tensor_data(
    output_file: BinaryIO,
    ternary_dir: Path,
    entry: TernaryManifestEntry,
) -> None:
    packed_bytes, scale_bytes = _load_ternary_payload_bytes(ternary_dir, entry)
    output_file.write(packed_bytes)
    output_file.write(scale_bytes)


def _make_base_writer(tensor: BaseTensorRef) -> WriteTensorData:
    return lambda output_file: _write_base_tensor_data(output_file, tensor)


def _make_ternary_writer(ternary_dir: Path, entry: TernaryManifestEntry) -> WriteTensorData:
    return lambda output_file: _write_ternary_tensor_data(output_file, ternary_dir, entry)


def _load_mold_bf16_bytes(ternary_dir: Path, entry: TernaryManifestEntry) -> bytes:
    payload_path = ternary_dir / entry.file_name
    if not payload_path.is_file():
        raise FileNotFoundError(f"Missing mold payload: {payload_path}")

    header, data_start = _read_safetensors_header(payload_path)
    meta = header.get(entry.name)
    if not isinstance(meta, dict):
        raise KeyError(f"Missing {entry.name} in {payload_path}")
    if meta.get("dtype") != "BF16":
        raise ValueError(f"Unexpected mold dtype in {payload_path}: {meta.get('dtype')}")

    start, end = _tensor_offsets(payload_path, meta, data_start)
    size = end - start
    if size != entry.packed_weight_bytes:
        raise ValueError(
            f"Mold byte count mismatch for {entry.name}: expected {entry.packed_weight_bytes}, got {size}"
        )

    with payload_path.open("rb") as payload_file:
        payload_file.seek(start)
        data = payload_file.read(size)
        if len(data) != size:
            raise ValueError(f"Truncated mold payload for {entry.name}: {payload_path}")

    actual_crc32 = zlib.crc32(data) & 0xFFFFFFFF
    if actual_crc32 != entry.crc32:
        raise ValueError(
            f"CRC mismatch for {entry.name}: manifest={entry.crc32:08x} actual={actual_crc32:08x}"
        )
    return data


def _write_mold_bf16_data(output_file: BinaryIO, ternary_dir: Path, entry: TernaryManifestEntry) -> None:
    data = _load_mold_bf16_bytes(ternary_dir, entry)
    output_file.write(data)


def _make_mold_writer(ternary_dir: Path, entry: TernaryManifestEntry) -> WriteTensorData:
    return lambda output_file: _write_mold_bf16_data(output_file, ternary_dir, entry)


def _remove_existing_pack(output_dir: Path) -> None:
    candidates = [
        output_dir / "model.safetensors",
        output_dir / "model.safetensors.index.json",
        output_dir / "manifest.tsv",
    ]
    candidates.extend(output_dir.glob("model-*.safetensors"))

    for path in candidates:
        if path.is_file():
            path.unlink()


def _copy_metadata_files(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    for name in METADATA_FILES:
        source_path = source_dir / name
        if not source_path.is_file():
            continue

        destination_path = output_dir / name
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {destination_path}")
        shutil.copy2(source_path, destination_path)


def _pack_shards(tensor_items: list[OutputTensorItem], max_shard_size: int) -> list[list[OutputTensorItem]]:
    shards: list[list[OutputTensorItem]] = []
    current_shard: list[OutputTensorItem] = []
    current_size = 0

    for tensor in tensor_items:
        if current_shard and current_size + tensor.stored_byte_count > max_shard_size:
            shards.append(current_shard)
            current_shard = []
            current_size = 0

        current_shard.append(tensor)
        current_size += tensor.stored_byte_count

    if current_shard:
        shards.append(current_shard)

    return shards


def _write_safetensors_file(
    output_path: Path,
    tensor_items: list[OutputTensorItem],
    total_size: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    header = OrderedDict()
    header["__metadata__"] = OrderedDict(
        [
            ("total_size", str(total_size)),
            ("storage_encoding", "mixed_bf16_ternary_2bit"),
            ("ternary_manifest", "manifest.tsv"),
        ]
    )

    data_offset = 0
    for tensor in tensor_items:
        for header_tensor in tensor.header_tensors:
            header[header_tensor.name] = OrderedDict(
                (
                    ("dtype", header_tensor.dtype),
                    ("shape", [int(dim) for dim in header_tensor.shape]),
                    ("data_offsets", [data_offset, data_offset + header_tensor.byte_count]),
                )
            )
            data_offset += header_tensor.byte_count

    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output_file:
        output_file.write(struct.pack("<Q", len(header_bytes)))
        output_file.write(header_bytes)

        for tensor in tensor_items:
            tensor.writer(output_file)

        output_file.flush()
        os.fsync(output_file.fileno())


def _ensure_output_dir_ready(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _remove_existing_pack(output_dir)
        return

    if (output_dir / "model.safetensors").exists() or (output_dir / "model.safetensors.index.json").exists():
        raise FileExistsError(f"Output directory already contains a model package: {output_dir}")
    if (output_dir / "manifest.tsv").exists():
        raise FileExistsError(f"Output directory already contains a ternary manifest: {output_dir}")
    if any(output_dir.glob("model-*.safetensors")):
        raise FileExistsError(f"Output directory already contains sharded model weights: {output_dir}")


def _build_output_tensor_items(
    base_tensors: OrderedDict[str, BaseTensorRef],
    manifest: OrderedDict[str, TernaryManifestEntry],
    ternary_dir: Path,
) -> tuple[list[OutputTensorItem], int, int, int]:
    tensor_items: list[OutputTensorItem] = []
    converted_count = 0
    passthrough_count = 0
    mold_count = 0

    for name, tensor in base_tensors.items():
        entry = manifest.get(name)
        if entry is None:
            tensor_items.append(
                OutputTensorItem(
                    logical_name=name,
                    header_tensors=(
                        OutputHeaderTensor(
                            name=name,
                            shape=tensor.shape,
                            dtype="BF16",
                            byte_count=tensor.bf16_byte_count,
                        ),
                    ),
                    stored_byte_count=tensor.bf16_byte_count,
                    writer=_make_base_writer(tensor),
                    source_kind="base",
                )
            )
            passthrough_count += 1
            continue

        if entry.kind == "mold":
            tensor_items.append(
                OutputTensorItem(
                    logical_name=name,
                    header_tensors=(
                        OutputHeaderTensor(
                            name=name,
                            shape=tensor.shape,
                            dtype="BF16",
                            byte_count=entry.packed_weight_bytes,
                        ),
                    ),
                    stored_byte_count=entry.packed_weight_bytes,
                    writer=_make_mold_writer(ternary_dir, entry),
                    source_kind="mold",
                )
            )
            mold_count += 1
            continue

        if tensor.shape != entry.shape:
            raise ValueError(f"Shape mismatch for {name}: base={tensor.shape} ternary={entry.shape}")
        tensor_items.append(
            OutputTensorItem(
                logical_name=name,
                header_tensors=(
                    OutputHeaderTensor(
                        name=f"{name}.packed",
                        shape=entry.packed_shape,
                        dtype="U8",
                        byte_count=entry.packed_weight_bytes,
                    ),
                    OutputHeaderTensor(
                        name=f"{name}.scales",
                        shape=entry.scale_shape,
                        dtype="F32",
                        byte_count=entry.scale_byte_count,
                    ),
                ),
                stored_byte_count=entry.stored_byte_count,
                writer=_make_ternary_writer(ternary_dir, entry),
                source_kind="ternary",
                manifest_entry=entry,
            )
        )
        converted_count += 1

    for name, entry in manifest.items():
        if name in base_tensors:
            continue
        if entry.kind == "mold":
            shape = (entry.rows,) if entry.cols == 1 else (entry.rows, entry.cols)
            tensor_items.append(
                OutputTensorItem(
                    logical_name=name,
                    header_tensors=(
                        OutputHeaderTensor(
                            name=name,
                            shape=shape,
                            dtype="BF16",
                            byte_count=entry.packed_weight_bytes,
                        ),
                    ),
                    stored_byte_count=entry.packed_weight_bytes,
                    writer=_make_mold_writer(ternary_dir, entry),
                    source_kind="mold",
                )
            )
            mold_count += 1
            continue

        tensor_items.append(
            OutputTensorItem(
                logical_name=name,
                header_tensors=(
                    OutputHeaderTensor(
                        name=f"{name}.packed",
                        shape=entry.packed_shape,
                        dtype="U8",
                        byte_count=entry.packed_weight_bytes,
                    ),
                    OutputHeaderTensor(
                        name=f"{name}.scales",
                        shape=entry.scale_shape,
                        dtype="F32",
                        byte_count=entry.scale_byte_count,
                    ),
                ),
                stored_byte_count=entry.stored_byte_count,
                writer=_make_ternary_writer(ternary_dir, entry),
                source_kind="ternary-extra",
                manifest_entry=entry,
            )
        )
        converted_count += 1

    if not tensor_items:
        raise RuntimeError("No tensors selected for repack")
    return tensor_items, converted_count, passthrough_count, mold_count


def _write_manifest(output_dir: Path,
                    tensor_items: list[OutputTensorItem],
                    logical_weight_map: dict[str, str]) -> None:
    manifest_path = output_dir / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for tensor in tensor_items:
            entry = tensor.manifest_entry
            if entry is None:
                continue
            shard_name = logical_weight_map.get(tensor.logical_name)
            if shard_name is None:
                raise KeyError(f"Missing shard assignment for ternary tensor {tensor.logical_name}")
            manifest_file.write(
                f"{tensor.logical_name}\t{shard_name}\t{entry.rows}\t{entry.cols}\t{entry.packed_weight_bytes}\t{entry.crc32:08x}\n"
            )


def repack_ternary_model(
    base_model_dir: Path,
    ternary_dir: Path,
    output_dir: Path,
    max_shard_size: int,
    overwrite: bool,
) -> None:
    if not base_model_dir.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base_model_dir}")
    if not ternary_dir.is_dir():
        raise FileNotFoundError(f"Ternary output directory not found: {ternary_dir}")
    if output_dir.resolve() == base_model_dir.resolve():
        raise ValueError("Refusing to overwrite the base model directory in place")

    manifest = _read_manifest(ternary_dir)
    base_tensors = _collect_base_tensors(base_model_dir)
    tensor_items, converted_count, passthrough_count, mold_count = _build_output_tensor_items(
        base_tensors,
        manifest,
        ternary_dir,
    )
    total_size = sum(tensor.stored_byte_count for tensor in tensor_items)
    shards = _pack_shards(tensor_items, max_shard_size=max_shard_size)

    _ensure_output_dir_ready(output_dir, overwrite)
    _copy_metadata_files(base_model_dir, output_dir, overwrite=overwrite)

    weight_map: dict[str, str] = {}
    logical_weight_map: dict[str, str] = {}
    shard_count = len(shards)
    if shard_count == 1:
        shard_path = output_dir / "model.safetensors"
        _write_safetensors_file(shard_path, shards[0], total_size, overwrite)
        for tensor in shards[0]:
            if tensor.manifest_entry is not None:
                logical_weight_map[tensor.logical_name] = shard_path.name
            for header_tensor in tensor.header_tensors:
                weight_map[header_tensor.name] = shard_path.name
    else:
        shard_name_width = 5
        for shard_idx, shard in enumerate(shards, start=1):
            shard_name = f"model-{shard_idx:0{shard_name_width}d}-of-{shard_count:0{shard_name_width}d}.safetensors"
            shard_path = output_dir / shard_name
            _write_safetensors_file(shard_path, shard, total_size, overwrite)
            for tensor in shard:
                if tensor.manifest_entry is not None:
                    logical_weight_map[tensor.logical_name] = shard_name
                for header_tensor in tensor.header_tensors:
                    weight_map[header_tensor.name] = shard_name

        _write_manifest(output_dir, tensor_items, logical_weight_map)

        index_path = output_dir / "model.safetensors.index.json"
        with index_path.open("w", encoding="utf-8") as index_file:
            json.dump(
                {
                    "metadata": {
                        "total_size": total_size,
                        "storage_encoding": "mixed_bf16_ternary_2bit",
                        "ternary_manifest": "manifest.tsv",
                    },
                    "weight_map": weight_map,
                },
                index_file,
                indent=2,
                sort_keys=True,
            )
            index_file.write("\n")

    if shard_count == 1:
        _write_manifest(output_dir, tensor_items, logical_weight_map)

    print(
        f"Repacked {converted_count} ternary tensor(s), {mold_count} mold BF16 tensor(s), "
        f"and {passthrough_count} passthrough tensor(s) "
        f"into {shard_count} mixed BF16 plus ternary-2bit safetensors file(s) at {output_dir}"
    )
    print("Note: ternary tensors remain stored as fixed 2-bit packed symbols plus F32 row scales; mold and passthrough tensors are stored as BF16.")
    if shard_count > 1:
        print(f"Wrote shard index: {output_dir / 'model.safetensors.index.json'}")
    print(f"Wrote ternary manifest: {output_dir / 'manifest.tsv'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repack Sapphire ternary outputs into a mixed BF16 plus packed ternary safetensors model package"
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        required=True,
        help="Directory containing the dense source model package to use for passthrough tensors",
    )
    parser.add_argument(
        "--ternary-dir",
        type=Path,
        required=True,
        help="Directory containing Sapphire ternary conversion outputs (manifest.tsv + layer payloads)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the repacked model package",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="2GiB",
        help="Maximum shard size, using bytes or size suffixes like 2GiB",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing repacked model package in --output-dir",
    )
    args = parser.parse_args()

    repack_ternary_model(
        base_model_dir=args.base_model_dir,
        ternary_dir=args.ternary_dir,
        output_dir=args.output_dir,
        max_shard_size=_parse_shard_size(args.max_shard_size),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
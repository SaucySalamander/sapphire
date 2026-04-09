#!/usr/bin/env python3
"""Import teacher-run curvature statistics into Sapphire's Hessian sidecar format.

The importer consumes two offline artifacts:

* a JSON manifest with the sidecar metadata and tensor entry list
* an NPZ archive containing one 1D float array per primary tensor entry

The generated file matches include/ternary_hessian_sidecar.h and is written in
the host's native endianness, just like the runtime tape and checkpoint files.

Manifest schema:

{
  "teacher_model_name": "gemma-3-1b-it",
  "tape_crc32": "0x12345678",
  "sample_count": 64,
  "entries": [
    {
      "tensor_name": "language_model.model.layers.0.self_attn.q_proj.weight",
      "data_key": "language_model.model.layers.0.self_attn.q_proj.weight",
      "layer_type": 0
    },
    {
      "tensor_name": "language_model.model.layers.0.self_attn.k_proj.weight",
      "alias_of": "language_model.model.layers.0.self_attn.q_proj.weight"
    }
  ]
}

`alias_of` may also be an integer entry index.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ENDIAN = "<" if sys.byteorder == "little" else ">"
HEADER_STRUCT = struct.Struct(f"{ENDIAN}6I128s2Q2I")
ENTRY_STRUCT = struct.Struct(f"{ENDIAN}256s2I2f2Q4I")
UINT32_MAX = 0xFFFFFFFF
UINT64_MAX = 0xFFFFFFFFFFFFFFFF
TENSOR_NAME_MAX = 256
TEACHER_MODEL_MAX = 128


@dataclass
class EntrySpec:
    tensor_name: str
    alias_of: str | int | None
    data_key: str | None
    vector_dim: int | None
    sample_count: int | None
    layer_type: int | None
    array: np.ndarray | None = None
    mean: float | None = None
    max_value: float | None = None
    primary_index: int | None = None
    data_offset: int = 0
    data_bytes: int = 0


def _parse_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer value") from exc
    raise ValueError(f"{field_name} must be an integer value")


def _require_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"Missing or empty required field: {key}")
    return value.strip()


def _normalize_text(value: str, field_name: str, max_bytes: int) -> bytes:
    if "\x00" in value:
        raise ValueError(f"{field_name} cannot contain NUL bytes")
    encoded = value.encode("utf-8")
    if len(encoded) >= max_bytes:
        raise ValueError(f"{field_name} exceeds maximum length of {max_bytes - 1} bytes")
    return encoded


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Top-level manifest must be a JSON object")
    return manifest


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Array archive not found: {path}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            arrays[key] = archive[key]
    return arrays


def _parse_entry_specs(manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> tuple[list[EntrySpec], str, int, int]:
    teacher_model_name = _require_text(manifest, "teacher_model_name")
    tape_crc32 = _parse_int(manifest.get("tape_crc32"), "tape_crc32")
    sample_count = _parse_int(manifest.get("sample_count"), "sample_count")

    if tape_crc32 < 0 or tape_crc32 > UINT32_MAX:
        raise ValueError("tape_crc32 must fit in uint32")
    if sample_count <= 0 or sample_count > UINT32_MAX:
        raise ValueError("sample_count must be in the uint32 range and greater than zero")

    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Manifest must contain a non-empty 'entries' array")

    entries: list[EntrySpec] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Entry {index} must be a JSON object")

        tensor_name = _require_text(raw_entry, "tensor_name")
        alias_of = raw_entry.get("alias_of")
        data_key = raw_entry.get("data_key")

        if alias_of is None:
            if data_key is None:
                data_key = tensor_name
            if not isinstance(data_key, str) or data_key.strip() == "":
                raise ValueError(f"Entry {index} data_key must be a non-empty string")
            if data_key not in arrays:
                raise ValueError(f"Entry {index} tensor '{tensor_name}' is missing array key '{data_key}'")
        else:
            if data_key is not None:
                raise ValueError(f"Alias entry {index} must not define data_key")

        vector_dim = raw_entry.get("vector_dim")
        sample_count_override = raw_entry.get("sample_count")
        layer_type_raw = raw_entry.get("layer_type")
        layer_type = None
        if layer_type_raw is not None:
            layer_type = _parse_int(layer_type_raw, f"entries[{index}].layer_type")
            if layer_type < 0 or layer_type > UINT32_MAX:
                raise ValueError(f"entries[{index}].layer_type must fit in uint32")
        elif alias_of is None:
            layer_type = 0

        entry = EntrySpec(
            tensor_name=tensor_name,
            alias_of=alias_of,
            data_key=data_key.strip() if isinstance(data_key, str) else None,
            vector_dim=_parse_int(vector_dim, f"entries[{index}].vector_dim") if vector_dim is not None else None,
            sample_count=_parse_int(sample_count_override, f"entries[{index}].sample_count") if sample_count_override is not None else sample_count,
            layer_type=layer_type,
        )
        entries.append(entry)

    tensor_to_index: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if entry.tensor_name in tensor_to_index:
            raise ValueError(f"Duplicate tensor name in manifest: {entry.tensor_name}")
        tensor_to_index[entry.tensor_name] = index

    primary_cache: list[int | None] = [None] * len(entries)

    def resolve_primary(entry_index: int, stack: set[int] | None = None) -> int:
        cached = primary_cache[entry_index]
        if cached is not None:
            return cached

        if stack is None:
            stack = set()
        if entry_index in stack:
            raise ValueError(f"Alias cycle detected in manifest at entry {entry_index}")
        stack.add(entry_index)

        entry = entries[entry_index]
        if entry.alias_of is None:
            resolved = entry_index
        else:
            alias_target = entry.alias_of
            if isinstance(alias_target, int):
                target_index = alias_target
            elif isinstance(alias_target, str):
                if alias_target not in tensor_to_index:
                    raise ValueError(f"Entry {entry_index} aliases unknown tensor '{alias_target}'")
                target_index = tensor_to_index[alias_target]
            else:
                raise ValueError(f"Entry {entry_index} alias_of must be a string tensor name or integer index")

            if target_index < 0 or target_index >= len(entries):
                raise ValueError(f"Entry {entry_index} aliases out-of-range entry {target_index}")
            resolved = resolve_primary(target_index, stack)

        stack.remove(entry_index)
        primary_cache[entry_index] = resolved
        return resolved

    for index, entry in enumerate(entries):
        primary_index = resolve_primary(index)
        entry.primary_index = primary_index

    primary_cursor = 0
    for index, entry in enumerate(entries):
        if entry.primary_index != index:
            continue

        array = np.asarray(arrays[entry.data_key], dtype=np.float32)
        if array.ndim != 1:
            if array.size == 0:
                raise ValueError(f"Primary entry '{entry.tensor_name}' has an empty array")
            array = np.reshape(array, (-1,))
        if array.size == 0:
            raise ValueError(f"Primary entry '{entry.tensor_name}' has an empty array")
        if not np.isfinite(array).all():
            raise ValueError(f"Primary entry '{entry.tensor_name}' contains non-finite values")

        inferred_dim = int(array.size)
        if entry.vector_dim is None:
            entry.vector_dim = inferred_dim
        elif entry.vector_dim != inferred_dim:
            raise ValueError(
                f"Primary entry '{entry.tensor_name}' vector_dim mismatch: manifest={entry.vector_dim} array={inferred_dim}"
            )

        if entry.sample_count is None:
            entry.sample_count = sample_count
        elif entry.sample_count != sample_count:
            raise ValueError(
                f"Primary entry '{entry.tensor_name}' sample_count mismatch: manifest={entry.sample_count} header={sample_count}"
            )

        if entry.layer_type is None:
            entry.layer_type = 0

        entry.mean = float(np.mean(array, dtype=np.float64))
        entry.max_value = float(np.max(array))
        entry.data_offset = primary_cursor
        entry.data_bytes = int(entry.vector_dim) * 4
        entry.array = np.ascontiguousarray(array, dtype=np.float32)
        primary_cursor += entry.data_bytes

    for index, entry in enumerate(entries):
        primary_index = entry.primary_index
        if primary_index is None:
            raise ValueError(f"Failed to resolve primary entry for {entry.tensor_name}")
        if primary_index == index:
            continue

        primary = entries[primary_index]
        if primary.vector_dim is None or primary.sample_count is None or primary.mean is None or primary.max_value is None or primary.layer_type is None:
            raise ValueError(f"Primary entry '{primary.tensor_name}' is not fully resolved")

        if entry.vector_dim is not None and entry.vector_dim != primary.vector_dim:
            raise ValueError(
                f"Alias entry '{entry.tensor_name}' vector_dim mismatch: alias={entry.vector_dim} primary={primary.vector_dim}"
            )
        if entry.sample_count is not None and entry.sample_count != primary.sample_count:
            raise ValueError(
                f"Alias entry '{entry.tensor_name}' sample_count mismatch: alias={entry.sample_count} primary={primary.sample_count}"
            )
        if entry.layer_type is not None and entry.layer_type != primary.layer_type:
            raise ValueError(
                f"Alias entry '{entry.tensor_name}' layer_type mismatch: alias={entry.layer_type} primary={primary.layer_type}"
            )

        entry.vector_dim = primary.vector_dim
        entry.sample_count = primary.sample_count
        entry.mean = primary.mean
        entry.max_value = primary.max_value
        entry.layer_type = primary.layer_type
        entry.data_offset = primary.data_offset
        entry.data_bytes = primary.data_bytes

    if primary_cursor <= 0:
        raise ValueError("Manifest must contain at least one primary tensor entry")

    return entries, teacher_model_name, tape_crc32, sample_count


def _pack_header(teacher_model_name: str,
                 entry_count: int,
                 sample_count: int,
                 tape_crc32: int,
                 data_section_offset: int,
                 data_section_size: int,
                 crc32: int) -> bytes:
    teacher_bytes = _normalize_text(teacher_model_name, "teacher_model_name", TEACHER_MODEL_MAX)
    padded_teacher = teacher_bytes.ljust(TEACHER_MODEL_MAX, b"\0")
    if entry_count < 0 or entry_count > UINT32_MAX:
        raise ValueError("entry_count must fit in uint32")
    if data_section_offset < 0 or data_section_offset > UINT64_MAX:
        raise ValueError("data_section_offset must fit in uint64")
    if data_section_size < 0 or data_section_size > UINT64_MAX:
        raise ValueError("data_section_size must fit in uint64")

    return HEADER_STRUCT.pack(
        0x48534331,
        1,
        int(entry_count),
        int(sample_count),
        int(tape_crc32),
        ENTRY_STRUCT.size,
        padded_teacher,
        int(data_section_offset),
        int(data_section_size),
        0,
        int(crc32),
    )


def _pack_entry(entry: EntrySpec, entry_index: int) -> bytes:
    if entry.vector_dim is None or entry.sample_count is None or entry.mean is None or entry.max_value is None or entry.primary_index is None:
        raise ValueError(f"Entry '{entry.tensor_name}' is missing resolved metadata")
    if entry.layer_type is None:
        raise ValueError(f"Entry '{entry.tensor_name}' is missing layer_type metadata")

    tensor_name_bytes = _normalize_text(entry.tensor_name, "tensor_name", TENSOR_NAME_MAX)
    tensor_name_padded = tensor_name_bytes.ljust(TENSOR_NAME_MAX, b"\0")
    alias_of_entry = UINT32_MAX if entry.primary_index == entry_index else int(entry.primary_index)
    if alias_of_entry < 0 or alias_of_entry > UINT32_MAX:
        raise ValueError(f"Alias index for '{entry.tensor_name}' must fit in uint32")

    return ENTRY_STRUCT.pack(
        tensor_name_padded,
        int(entry.vector_dim),
        int(entry.sample_count),
        float(entry.mean),
        float(entry.max_value),
        int(entry.data_offset),
        int(entry.data_bytes),
        alias_of_entry,
        int(entry.layer_type),
        0,
        0,
    )


def _write_sidecar(output_path: Path,
                   entries: list[EntrySpec],
                   teacher_model_name: str,
                   sample_count: int,
                   tape_crc32: int) -> int:
    entry_count = len(entries)
    manifest_bytes = bytearray()
    for index, entry in enumerate(entries):
        manifest_bytes.extend(_pack_entry(entry, index))

    header_size = HEADER_STRUCT.size
    data_section_offset = header_size + len(manifest_bytes)
    data_section_size = sum(entry.data_bytes for index, entry in enumerate(entries) if entry.primary_index == index)

    header_without_crc = _pack_header(
        teacher_model_name=teacher_model_name,
        entry_count=entry_count,
        sample_count=sample_count,
        tape_crc32=tape_crc32,
        data_section_offset=data_section_offset,
        data_section_size=data_section_size,
        crc32=0,
    )

    crc32 = zlib.crc32(header_without_crc[:-4]) & 0xFFFFFFFF

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w+b",
                                     delete=False,
                                     dir=output_path.parent,
                                     prefix=f"{output_path.name}.",
                                     suffix=".tmp") as temp_file:
        temp_path = Path(temp_file.name)
        try:
            temp_file.write(header_without_crc)
            temp_file.write(manifest_bytes)
            crc32 = zlib.crc32(manifest_bytes, crc32) & 0xFFFFFFFF

            for index, entry in enumerate(entries):
                if entry.primary_index != index:
                    continue
                if entry.array is None:
                    raise ValueError(f"Primary entry '{entry.tensor_name}' is missing array data")
                data_bytes = entry.array.tobytes(order="C")
                temp_file.write(data_bytes)
                crc32 = zlib.crc32(data_bytes, crc32) & 0xFFFFFFFF

            temp_file.flush()
            os.fsync(temp_file.fileno())
        except Exception:
            temp_file.close()
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    header = _pack_header(
        teacher_model_name=teacher_model_name,
        entry_count=entry_count,
        sample_count=sample_count,
        tape_crc32=tape_crc32,
        data_section_offset=data_section_offset,
        data_section_size=data_section_size,
        crc32=crc32,
    )

    with temp_path.open("r+b") as temp_file:
        temp_file.seek(0)
        temp_file.write(header)
        temp_file.flush()
        os.fsync(temp_file.fileno())

    os.chmod(temp_path, 0o644)
    os.replace(temp_path, output_path)
    return crc32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert external teacher curvature statistics into Sapphire's binary Hessian sidecar format.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest describing the sidecar metadata and tensor entries.",
    )
    parser.add_argument(
        "--arrays",
        type=Path,
        required=True,
        help="NPZ archive containing primary diagonal arrays keyed by data_key or tensor_name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output sidecar path to write.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    arrays = _load_arrays(args.arrays)
    entries, teacher_model_name, tape_crc32, sample_count = _parse_entry_specs(manifest, arrays)
    crc32 = _write_sidecar(args.output, entries, teacher_model_name, sample_count, tape_crc32)

    primary_count = sum(1 for index, entry in enumerate(entries) if entry.primary_index == index)
    alias_count = len(entries) - primary_count
    print(
        f"Wrote Hessian sidecar: {args.output} (entries={len(entries)} primaries={primary_count} aliases={alias_count} crc32={crc32:08x})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
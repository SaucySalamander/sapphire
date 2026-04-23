#!/usr/bin/env python3
"""Patch converted Sapphire BF16 interface tensors from teacher weights.

This rewrites the embedding and final norm tensors inside a converted model
directory such as out/gemma-3-7b-q1.58b-ternary. The embedding is width-
expanded from the teacher hidden size using Sapphire's existing deterministic
policy, then both rewritten safetensors files have their manifest CRC entries
updated in place.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CONVERTED_DIR = Path(__file__).resolve().parents[1] / "out" / "gemma-3-7b-q1.58b-ternary"
DEFAULT_TEACHER_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "gemma-3-1b-it"
DEFAULT_CHUNK_ROWS = 64
TEACHER_EMBED_TENSOR_NAMES = (
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
)
TEACHER_FINAL_NORM_TENSOR_NAMES = (
    "model.norm.weight",
    "language_model.model.norm.weight",
)


@dataclass(frozen=True)
class PatchTarget:
    cli_name: str
    converted_tensor_name: str
    teacher_candidate_names: tuple[str, ...]


@dataclass(frozen=True)
class ManifestEntry:
    tensor_name: str
    file_name: str
    rows: int
    cols: int
    byte_count: int
    crc32: int
    kind: str | None


@dataclass(frozen=True)
class BaseTensorRef:
    name: str
    shape: tuple[int, ...]
    dtype: str
    file_path: Path
    data_start: int
    data_end: int


PATCH_TARGETS = {
    "embedding": PatchTarget(
        cli_name="embedding",
        converted_tensor_name="language_model.model.embed_tokens.weight",
        teacher_candidate_names=TEACHER_EMBED_TENSOR_NAMES,
    ),
    "final-norm": PatchTarget(
        cli_name="final-norm",
        converted_tensor_name="language_model.model.norm.weight",
        teacher_candidate_names=TEACHER_FINAL_NORM_TENSOR_NAMES,
    ),
}


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
        tensors[str(name)] = BaseTensorRef(
            name=str(name),
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
    if not index_path.is_file():
        raise FileNotFoundError(
            f"No supported teacher model package found in {model_dir}; expected model.safetensors or sharded safetensors"
        )

    with index_path.open("r", encoding="utf-8") as index_file:
        index_data = json.load(index_file)

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
        tensors[str(tensor_name)] = BaseTensorRef(
            name=str(tensor_name),
            shape=_tensor_shape(meta),
            dtype=str(dtype),
            file_path=shard_path,
            data_start=data_begin,
            data_end=data_end,
        )

    return tensors


def _numpy_dtype_for_tensor(dtype: str) -> np.dtype:
    if dtype == "BF16":
        return np.dtype(np.uint16)
    if dtype == "F16":
        return np.dtype(np.float16)
    if dtype == "F32":
        return np.dtype("<f4")
    raise ValueError(f"Unsupported source tensor dtype: {dtype}")


def _resolve_width_strategy(requested: str, source_dim: int, target_dim: int) -> str:
    if requested == "interpolation":
        return "interpolation"
    if requested == "block-replication":
        if target_dim >= source_dim and target_dim % source_dim == 0:
            return "block-replication"
        raise ValueError(
            "block-replication requested but target_dim is not an integer multiple of source_dim "
            f"({target_dim} vs {source_dim})"
        )
    if source_dim > 0 and target_dim >= source_dim and target_dim % source_dim == 0:
        return "block-replication"
    return "interpolation"


def _bf16_words_to_float32(values: np.ndarray) -> np.ndarray:
    words = np.asarray(values, dtype=np.uint16)
    bits = words.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def _read_tensor_chunk(ref: BaseTensorRef,
                       row_start: int | None = None,
                       row_end: int | None = None) -> np.ndarray:
    raw_dtype = _numpy_dtype_for_tensor(ref.dtype)
    tensor_view = np.memmap(ref.file_path, dtype=raw_dtype, mode="r", offset=ref.data_start, shape=ref.shape)
    if row_start is None or row_end is None:
        chunk = np.asarray(tensor_view)
    else:
        chunk = np.asarray(tensor_view[row_start:row_end])
    if ref.dtype == "BF16":
        return _bf16_words_to_float32(chunk)
    return np.asarray(chunk, dtype=np.float32)


def _get_required_teacher_tensor(teacher_tensors: OrderedDict[str, BaseTensorRef],
                                 candidate_names: tuple[str, ...]) -> BaseTensorRef:
    for name in candidate_names:
        tensor_ref = teacher_tensors.get(name)
        if tensor_ref is not None:
            return tensor_ref
    raise KeyError(f"Teacher model is missing required tensor(s): {', '.join(candidate_names)}")


def _float32_to_bf16_words(values: np.ndarray) -> np.ndarray:
    float32_values = np.asarray(values, dtype=np.float32)
    float32_bits = float32_values.view(np.uint32)
    rounded_bits = float32_bits + np.uint32(0x7FFF) + ((float32_bits >> np.uint32(16)) & np.uint32(1))
    bf16_words = (rounded_bits >> np.uint32(16)).astype(np.uint16)
    if sys.byteorder != "little":
        bf16_words = bf16_words.byteswap()
    return bf16_words


def _safe_tensor_file_name(tensor_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in tensor_name) + ".safetensors"


def _build_mold_header_bytes(tensor_name: str, rows: int, cols: int, crc32: int) -> bytes:
    byte_count = rows * cols * np.dtype(np.uint16).itemsize
    if cols == 1:
        header = (
            f'{{"__metadata__":{{"sapphire_quant":"mold-bf16","crc32":"{crc32:08x}"}},'
            f'"{tensor_name}":{{"dtype":"BF16","shape":[{rows}],"data_offsets":[0,{byte_count}]}}}}'
        )
    else:
        header = (
            f'{{"__metadata__":{{"sapphire_quant":"mold-bf16","crc32":"{crc32:08x}"}},'
            f'"{tensor_name}":{{"dtype":"BF16","shape":[{rows},{cols}],"data_offsets":[0,{byte_count}]}}}}'
        )
    return header.encode("utf-8")


def _parse_manifest_entry(line: str) -> ManifestEntry | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None

    fields = text.split("\t")
    if len(fields) < 6:
        return None

    kind = fields[6] if len(fields) >= 7 else None
    return ManifestEntry(
        tensor_name=fields[0],
        file_name=fields[1],
        rows=int(fields[2]),
        cols=int(fields[3]),
        byte_count=int(fields[4]),
        crc32=int(fields[5], 16),
        kind=kind,
    )


def _find_manifest_entry(manifest_path: Path, tensor_name: str) -> ManifestEntry | None:
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            entry = _parse_manifest_entry(line)
            if entry and entry.tensor_name == tensor_name:
                return entry
    return None


def _replace_manifest_entry(manifest_path: Path,
                            tensor_name: str,
                            file_name: str,
                            rows: int,
                            cols: int,
                            crc32: int) -> None:
    byte_count = rows * cols * np.dtype(np.uint16).itemsize
    replacement = f"{tensor_name}\t{file_name}\t{rows}\t{cols}\t{byte_count}\t{crc32:08x}\tmold\n"
    target_prefix = f"{tensor_name}\t"
    updated = False

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        lines = manifest_file.readlines()

    for index, line in enumerate(lines):
        if line.startswith(target_prefix):
            lines[index] = replacement
            updated = True
            break

    if not updated:
        lines.append(replacement)

    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as manifest_file:
        manifest_file.writelines(lines)
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    os.replace(tmp_path, manifest_path)


def _read_output_tensor_shape(output_path: Path, tensor_name: str) -> tuple[int, int]:
    header, _ = _read_safetensors_header(output_path)
    meta = header.get(tensor_name)
    if not isinstance(meta, dict):
        raise KeyError(f"Tensor {tensor_name} missing from {output_path}")

    shape = _tensor_shape(meta)
    if len(shape) == 1:
        return int(shape[0]), 1
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"Unsupported tensor rank for {tensor_name}: {shape}")


def _expand_vector(source: np.ndarray, target_dim: int, width_strategy: str) -> np.ndarray:
    source_dim = int(source.shape[0])
    strategy = _resolve_width_strategy(width_strategy, source_dim, target_dim)
    if source_dim == target_dim:
        return np.asarray(source, dtype=np.float32)
    if strategy == "block-replication":
        return np.repeat(np.asarray(source, dtype=np.float32), target_dim // source_dim).astype(np.float32, copy=False)
    if source_dim == 1 or target_dim == 1:
        return np.full((target_dim,), float(source[0]), dtype=np.float32)

    source_positions = np.linspace(0.0, 1.0, source_dim, dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target_dim, dtype=np.float64)
    return np.interp(target_positions, source_positions, np.asarray(source, dtype=np.float64)).astype(np.float32)


def _write_mold_file_from_chunks(output_path: Path,
                                 tensor_name: str,
                                 rows: int,
                                 cols: int,
                                 chunk_iter) -> int:
    placeholder_header = _build_mold_header_bytes(tensor_name, rows, cols, 0)
    final_crc32 = 0
    tmp_path = output_path.with_name(output_path.name + ".tmp")

    try:
        with tmp_path.open("wb") as output_file:
            output_file.write(struct.pack("<Q", len(placeholder_header)))
            output_file.write(placeholder_header)

            for chunk in chunk_iter:
                bf16_words = _float32_to_bf16_words(chunk)
                chunk_bytes = bf16_words.tobytes(order="C")
                output_file.write(chunk_bytes)
                final_crc32 = zlib.crc32(chunk_bytes, final_crc32) & 0xFFFFFFFF

            final_header = _build_mold_header_bytes(tensor_name, rows, cols, final_crc32)
            if len(final_header) != len(placeholder_header):
                raise ValueError(f"Header length changed while patching {tensor_name}")

            output_file.flush()
            os.fsync(output_file.fileno())
            output_file.seek(8)
            output_file.write(final_header)
            output_file.flush()
            os.fsync(output_file.fileno())

        os.replace(tmp_path, output_path)
        return final_crc32
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _patch_embedding(output_path: Path,
                     tensor_name: str,
                     teacher_ref: BaseTensorRef,
                     rows: int,
                     cols: int,
                     width_strategy: str,
                     chunk_rows: int) -> int:
    if len(teacher_ref.shape) != 2:
        raise ValueError(f"Teacher tensor {teacher_ref.name} is not a matrix: {teacher_ref.shape}")
    if int(teacher_ref.shape[0]) != rows:
        raise ValueError(
            f"Teacher row count mismatch for {teacher_ref.name}: {teacher_ref.shape[0]} vs {rows}"
        )

    source_cols = int(teacher_ref.shape[1])
    strategy = _resolve_width_strategy(width_strategy, source_cols, cols)
    repeat = 1
    interp_lo = None
    interp_hi = None
    interp_frac = None

    if source_cols != cols:
        if strategy == "block-replication":
            repeat = cols // source_cols
        else:
            positions = np.linspace(0.0, float(source_cols - 1), cols, dtype=np.float32)
            interp_lo = np.floor(positions).astype(np.int64)
            interp_hi = np.clip(interp_lo + 1, 0, source_cols - 1)
            interp_frac = (positions - interp_lo.astype(np.float32))[np.newaxis, :]

    def chunk_iter():
        for row_start in range(0, rows, chunk_rows):
            row_end = min(rows, row_start + chunk_rows)
            source_chunk = _read_tensor_chunk(teacher_ref, row_start, row_end)
            if source_cols == cols:
                yield source_chunk.astype(np.float32, copy=False)
            elif strategy == "block-replication":
                yield np.repeat(source_chunk, repeat, axis=1).astype(np.float32, copy=False)
            else:
                left = source_chunk[:, interp_lo]
                right = source_chunk[:, interp_hi]
                yield (left + (right - left) * interp_frac).astype(np.float32, copy=False)

    return _write_mold_file_from_chunks(output_path, tensor_name, rows, cols, chunk_iter())


def _patch_vector(output_path: Path,
                  tensor_name: str,
                  teacher_ref: BaseTensorRef,
                  rows: int,
                  width_strategy: str) -> int:
    if len(teacher_ref.shape) != 1:
        raise ValueError(f"Teacher tensor {teacher_ref.name} is not a vector: {teacher_ref.shape}")

    source = _read_tensor_chunk(teacher_ref).reshape(-1)
    expanded = _expand_vector(source, rows, width_strategy)
    return _write_mold_file_from_chunks(output_path, tensor_name, rows, 1, [expanded])


def _patch_target(converted_dir: Path,
                  manifest_path: Path,
                  teacher_tensors,
                  target: PatchTarget,
                  width_strategy: str,
                  chunk_rows: int,
                  dry_run: bool) -> None:
    manifest_entry = _find_manifest_entry(manifest_path, target.converted_tensor_name)
    file_name = manifest_entry.file_name if manifest_entry else _safe_tensor_file_name(target.converted_tensor_name)
    output_path = converted_dir / file_name

    if output_path.is_file():
        rows, cols = _read_output_tensor_shape(output_path, target.converted_tensor_name)
    elif manifest_entry:
        rows, cols = manifest_entry.rows, manifest_entry.cols
    else:
        raise FileNotFoundError(f"Cannot resolve output tensor for {target.converted_tensor_name}: {output_path}")

    teacher_ref = _get_required_teacher_tensor(teacher_tensors, target.teacher_candidate_names)

    if dry_run:
        print(
            f"would patch {target.converted_tensor_name} -> {file_name} "
            f"using {teacher_ref.name} ({rows}x{cols})"
        )
        return

    if target.cli_name == "embedding":
        crc32 = _patch_embedding(output_path,
                                 target.converted_tensor_name,
                                 teacher_ref,
                                 rows,
                                 cols,
                                 width_strategy,
                                 chunk_rows)
    else:
        crc32 = _patch_vector(output_path,
                              target.converted_tensor_name,
                              teacher_ref,
                              rows,
                              width_strategy)

    _replace_manifest_entry(manifest_path,
                            target.converted_tensor_name,
                            file_name,
                            rows,
                            cols,
                            crc32)
    print(
        f"patched {target.converted_tensor_name} -> {file_name} "
        f"using {teacher_ref.name} (crc32={crc32:08x})"
    )


def patch_converted_model(converted_dir: Path,
                          teacher_model_dir: Path,
                          targets: list[str],
                          width_strategy: str,
                          chunk_rows: int,
                          dry_run: bool) -> None:
    if not converted_dir.is_dir():
        raise FileNotFoundError(f"Converted model directory not found: {converted_dir}")
    if not teacher_model_dir.is_dir():
        raise FileNotFoundError(f"Teacher model directory not found: {teacher_model_dir}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    manifest_path = converted_dir / "manifest.tsv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.tsv: {manifest_path}")

    teacher_tensors = _collect_base_tensors(teacher_model_dir)

    for target_name in targets:
        target = PATCH_TARGETS[target_name]
        _patch_target(converted_dir,
                      manifest_path,
                      teacher_tensors,
                      target,
                      width_strategy,
                      chunk_rows,
                      dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch converted Sapphire BF16 interface tensors from teacher weights"
    )
    parser.add_argument(
        "--converted-dir",
        type=Path,
        default=DEFAULT_CONVERTED_DIR,
        help="Converted model directory to patch in place",
    )
    parser.add_argument(
        "--teacher-model-dir",
        type=Path,
        default=DEFAULT_TEACHER_MODEL_DIR,
        help="Teacher model package used as the BF16 source",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=tuple(PATCH_TARGETS.keys()),
        default=list(PATCH_TARGETS.keys()),
        help="Which tensors to patch",
    )
    parser.add_argument(
        "--width-strategy",
        choices=("auto", "interpolation", "block-replication"),
        default="auto",
        help="Width expansion policy for teacher sourced tensors",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help="Row chunk size used while rewriting large matrices",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without modifying files",
    )
    args = parser.parse_args()

    patch_converted_model(args.converted_dir,
                          args.teacher_model_dir,
                          args.targets,
                          args.width_strategy,
                          args.chunk_rows,
                          args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
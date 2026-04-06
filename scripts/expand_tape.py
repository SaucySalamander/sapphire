#!/usr/bin/env python3
"""Expand a raw Sapphire activation tape into a student-aligned tape.

This mirrors the deterministic activation-alignment logic used by Sapphire's
full-model conversion path:

* depth mapping uses either bucket or repeat assignment
* width mapping uses block replication when possible, otherwise interpolation
* aliased weights (K/V, up_proj) reuse the primary entry's aligned data
* the output tape uses the student tensor prefix so the student can read it

The script writes:

* an aligned ``.tape`` file with Sapphire's native binary tape format
* an optional TSV manifest for inspection/debugging

The default workflow is tuned for the 1B teacher -> 7B student experiment.
"""

from __future__ import annotations

import argparse
import binascii
import json
import mmap
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ENDIAN = "<" if sys.byteorder == "little" else ">"
TAPE_MAGIC = 0x54415045
TAPE_VERSION = 1
TAPE_TENSOR_NAME_MAX = 256
TAPE_NO_ALIAS = 0xFFFFFFFF
HEADER_STRUCT = struct.Struct(f"{ENDIAN}5I3I2QI3I")
ENTRY_STRUCT = struct.Struct(f"{ENDIAN}256s2I2Q4I")
HEADER_CRC_OFFSET = struct.calcsize(f"{ENDIAN}5I3I2Q")

ALIGNMENT_WEIGHT_SPECS = (
    ("self_attn.q_proj.weight", "qkv_input", -1),
    ("self_attn.k_proj.weight", "qkv_input", 0),
    ("self_attn.v_proj.weight", "qkv_input", 0),
    ("self_attn.o_proj.weight", "out_input", -1),
    ("mlp.gate_proj.weight", "ffn_input", -1),
    ("mlp.up_proj.weight", "ffn_input", 4),
    ("mlp.down_proj.weight", "down_input", -1),
)


@dataclass(frozen=True)
class TapeHeader:
    magic: int
    version: int
    entry_count: int
    sample_count: int
    hidden_size: int
    data_section_offset: int
    data_section_size: int
    crc32: int


@dataclass(frozen=True)
class TapeEntry:
    tensor_name: str
    vector_dim: int
    sample_count: int
    data_offset: int
    data_bytes: int
    alias_of_entry: int
    layer_type: int


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int


@dataclass(frozen=True)
class AlignmentEntry:
    student_entry_idx: int
    student_layer_idx: int
    teacher_entry_idx: int
    teacher_layer_idx: int
    capture_target: str
    source_dim: int
    target_dim: int
    sample_count: int
    alias_of_entry: int
    teacher_layer_type: int
    student_layer_type: int
    teacher_tensor_name: str
    student_tensor_name: str


@dataclass
class ParsedTape:
    path: Path
    mm: mmap.mmap
    header: TapeHeader
    entries: list[TapeEntry]
    data: np.ndarray
    prefix: str
    layer_count: int
    entry_index_by_name: dict[str, int]

    def close(self) -> None:
        self.data = None  # Release exported NumPy views before closing the mmap.
        self.mm.close()


def _normalize_prefix(prefix: str) -> str:
    if prefix.endswith("."):
        return prefix
    return prefix + "."


def _capture_target_label(target: str) -> str:
    return target


def _infer_prefix(tensor_name: str) -> str:
    marker = ".layers."
    marker_idx = tensor_name.find(marker)
    if marker_idx < 0:
        raise ValueError(f"Unable to infer tensor prefix from {tensor_name!r}")
    return tensor_name[: marker_idx + len(marker)]


def _extract_layer_index(tensor_name: str) -> int | None:
    marker = ".layers."
    marker_idx = tensor_name.find(marker)
    if marker_idx < 0:
        return None

    cursor = marker_idx + len(marker)
    end = cursor
    while end < len(tensor_name) and tensor_name[end].isdigit():
        end += 1

    if end == cursor:
        return None
    return int(tensor_name[cursor:end])


def _infer_layer_count(entries: Iterable[TapeEntry]) -> int:
    max_layer = -1
    for entry in entries:
        layer_idx = _extract_layer_index(entry.tensor_name)
        if layer_idx is None:
            continue
        if layer_idx > max_layer:
            max_layer = layer_idx
    return max_layer + 1 if max_layer >= 0 else 0


def _load_model_config(model_dir: Path, model_id: str) -> ModelConfig:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = json.load(config_file)

    config = raw_config.get("text_config", raw_config)

    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_hidden_layers = int(config["num_hidden_layers"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or (hidden_size // num_attention_heads))

    if hidden_size <= 0 or intermediate_size <= 0 or num_hidden_layers <= 0:
        raise ValueError(f"Invalid model dimensions in {config_path}")
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError(f"Invalid attention dimensions in {config_path}")
    if num_attention_heads * head_dim != hidden_size:
        raise ValueError(
            "hidden_size must equal num_attention_heads * head_dim for the alignment script "
            f"({hidden_size} != {num_attention_heads} * {head_dim})"
        )

    return ModelConfig(
        model_id=model_id,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
    )


def _target_dim_for_capture(target: str, config: ModelConfig) -> int:
    if target == "qkv_input":
        return config.hidden_size
    if target == "out_input":
        return config.num_attention_heads * config.head_dim
    if target == "ffn_input":
        return config.hidden_size
    if target == "down_input":
        return config.intermediate_size
    raise ValueError(f"Unknown capture target: {target}")


def _map_teacher_layer(student_layer: int,
                       teacher_layer_count: int,
                       student_layer_count: int,
                       depth_strategy: str) -> int:
    if teacher_layer_count <= 0 or student_layer_count <= 0:
        raise ValueError("Layer counts must be positive")

    if depth_strategy == "repeat":
        return student_layer % teacher_layer_count

    return (student_layer * teacher_layer_count) // student_layer_count


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


def _resample_vector(source: np.ndarray,
                     source_dim: int,
                     target_dim: int,
                     strategy: str) -> np.ndarray:
    if source_dim == target_dim:
        return np.asarray(source, dtype=np.float32)

    if strategy == "block-replication":
        repeat = target_dim // source_dim
        return np.repeat(np.asarray(source, dtype=np.float32), repeat).astype(np.float32, copy=False)

    if source_dim == 1 or target_dim == 1:
        return np.full((target_dim,), float(source[0]), dtype=np.float32)

    source_positions = np.linspace(0.0, 1.0, source_dim, dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, target_dim, dtype=np.float64)
    interpolated = np.interp(target_positions, source_positions, np.asarray(source, dtype=np.float64))
    return interpolated.astype(np.float32)


def _parse_tape_entry(raw_entry: tuple[bytes, ...]) -> TapeEntry:
    tensor_name = raw_entry[0].split(b"\0", 1)[0].decode("utf-8")
    return TapeEntry(
        tensor_name=tensor_name,
        vector_dim=int(raw_entry[1]),
        sample_count=int(raw_entry[2]),
        data_offset=int(raw_entry[3]),
        data_bytes=int(raw_entry[4]),
        alias_of_entry=int(raw_entry[5]),
        layer_type=int(raw_entry[6]),
    )


def _parse_tape(path: Path) -> ParsedTape:
    if not path.is_file():
        raise FileNotFoundError(f"Input tape not found: {path}")

    file_handle = path.open("rb")
    mm = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
    file_handle.close()

    if mm.size() < HEADER_STRUCT.size:
        mm.close()
        raise ValueError(f"Tape file is too small: {path}")

    header_values = HEADER_STRUCT.unpack_from(mm, 0)
    header = TapeHeader(
        magic=int(header_values[0]),
        version=int(header_values[1]),
        entry_count=int(header_values[2]),
        sample_count=int(header_values[3]),
        hidden_size=int(header_values[4]),
        data_section_offset=int(header_values[8]),
        data_section_size=int(header_values[9]),
        crc32=int(header_values[10]),
    )

    if header.magic != TAPE_MAGIC:
        mm.close()
        raise ValueError(f"Invalid tape magic in {path}")
    if header.version != TAPE_VERSION:
        mm.close()
        raise ValueError(f"Unsupported tape version {header.version} in {path}")
    if header.entry_count <= 0:
        mm.close()
        raise ValueError(f"Tape has no manifest entries: {path}")
    if header.data_section_offset + header.data_section_size > mm.size():
        mm.close()
        raise ValueError(f"Tape data section is truncated: {path}")
    if header.data_section_size % 4 != 0:
        mm.close()
        raise ValueError(f"Tape data section is not float32-aligned: {path}")

    entries: list[TapeEntry] = []
    manifest_offset = HEADER_STRUCT.size
    expected_manifest_size = header.entry_count * ENTRY_STRUCT.size
    if manifest_offset + expected_manifest_size > header.data_section_offset:
        mm.close()
        raise ValueError(f"Tape manifest overlaps data section: {path}")

    for entry_idx in range(header.entry_count):
        raw_entry = ENTRY_STRUCT.unpack_from(mm, manifest_offset + entry_idx * ENTRY_STRUCT.size)
        entry = _parse_tape_entry(raw_entry)
        if len(entry.tensor_name.encode("utf-8")) >= TAPE_TENSOR_NAME_MAX:
            mm.close()
            raise ValueError(f"Tensor name too long in {path}: {entry.tensor_name}")
        if entry.sample_count != header.sample_count:
            mm.close()
            raise ValueError(
                f"Manifest sample count mismatch for {entry.tensor_name}: "
                f"{entry.sample_count} vs {header.sample_count}"
            )
        entries.append(entry)

    prefix = _infer_prefix(entries[0].tensor_name)
    layer_count = _infer_layer_count(entries)
    if layer_count <= 0:
        mm.close()
        raise ValueError(f"Could not infer layer count from tape: {path}")

    entry_index_by_name = {entry.tensor_name: index for index, entry in enumerate(entries)}
    data = np.frombuffer(mm, dtype=np.float32, count=header.data_section_size // 4, offset=header.data_section_offset)

    return ParsedTape(
        path=path,
        mm=mm,
        header=header,
        entries=entries,
        data=data,
        prefix=prefix,
        layer_count=layer_count,
        entry_index_by_name=entry_index_by_name,
    )


def _teacher_target_name(teacher_prefix: str, teacher_layer_idx: int, suffix: str) -> str:
    return f"{teacher_prefix}{teacher_layer_idx}.{suffix}"


def _student_target_name(student_prefix: str, student_layer_idx: int, suffix: str) -> str:
    return f"{student_prefix}{student_layer_idx}.{suffix}"


def _build_alignment_entries(raw_tape: ParsedTape,
                             student_config: ModelConfig,
                             teacher_model_id: str,
                             student_model_id: str,
                             teacher_prefix: str,
                             student_prefix: str,
                             depth_strategy: str,
                             width_strategy: str) -> list[AlignmentEntry]:
    teacher_layer_count = raw_tape.layer_count
    student_layer_count = student_config.num_hidden_layers
    entries: list[AlignmentEntry] = []

    if teacher_layer_count <= 0 or student_layer_count <= 0:
        raise ValueError("Teacher/student layer counts must be positive")
    if len(raw_tape.entries) % len(ALIGNMENT_WEIGHT_SPECS) != 0:
        raise ValueError("Teacher tape does not contain a whole number of 7-entry layers")

    for student_layer_idx in range(student_layer_count):
        teacher_layer_idx = _map_teacher_layer(student_layer_idx,
                                               teacher_layer_count,
                                               student_layer_count,
                                               depth_strategy)
        if teacher_layer_idx >= teacher_layer_count:
            raise ValueError(f"Invalid depth mapping for student layer {student_layer_idx}")

        for wt_idx, (suffix, capture_target, alias_idx) in enumerate(ALIGNMENT_WEIGHT_SPECS):
            teacher_tensor_name = _teacher_target_name(teacher_prefix, teacher_layer_idx, suffix)
            student_tensor_name = _student_target_name(student_prefix, student_layer_idx, suffix)

            teacher_entry_idx = raw_tape.entry_index_by_name.get(teacher_tensor_name)
            if teacher_entry_idx is None:
                raise KeyError(f"Teacher tape does not contain tensor {teacher_tensor_name}")

            source_entry = raw_tape.entries[teacher_entry_idx]
            target_dim = _target_dim_for_capture(capture_target, student_config)
            if target_dim <= 0:
                raise ValueError(f"Invalid target dimension for {student_tensor_name}")

            _resolve_width_strategy(width_strategy, source_entry.vector_dim, target_dim)

            entries.append(
                AlignmentEntry(
                    student_entry_idx=student_layer_idx * 7 + wt_idx,
                    student_layer_idx=student_layer_idx,
                    teacher_entry_idx=teacher_entry_idx,
                    teacher_layer_idx=teacher_layer_idx,
                    capture_target=capture_target,
                    source_dim=source_entry.vector_dim,
                    target_dim=target_dim,
                    sample_count=raw_tape.header.sample_count,
                    alias_of_entry=TAPE_NO_ALIAS if alias_idx < 0 else student_layer_idx * 7 + alias_idx,
                    teacher_layer_type=1 if (teacher_layer_idx % 6) == 5 else 0,
                    student_layer_type=1 if (student_layer_idx % 6) == 5 else 0,
                    teacher_tensor_name=teacher_tensor_name,
                    student_tensor_name=student_tensor_name,
                )
            )

    return entries


def _pack_manifest_entry(name: str,
                         vector_dim: int,
                         sample_count: int,
                         data_offset: int,
                         data_bytes: int,
                         alias_of_entry: int,
                         layer_type: int) -> bytes:
    encoded_name = name.encode("utf-8")
    if len(encoded_name) >= TAPE_TENSOR_NAME_MAX:
        raise ValueError(f"Tensor name too long: {name}")

    padded_name = encoded_name + b"\0" * (TAPE_TENSOR_NAME_MAX - len(encoded_name))
    return ENTRY_STRUCT.pack(
        padded_name,
        int(vector_dim),
        int(sample_count),
        int(data_offset),
        int(data_bytes),
        int(alias_of_entry),
        int(layer_type),
        0,
        0,
    )


def _write_alignment_manifest(path: Path,
                              teacher_model_id: str,
                              student_model_id: str,
                              teacher_prefix: str,
                              student_prefix: str,
                              teacher_layer_count: int,
                              student_layer_count: int,
                              sample_count: int,
                              depth_strategy: str,
                              width_strategy: str,
                              entries: list[AlignmentEntry],
                              overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")

    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("# Sapphire activation alignment manifest v1\n")
        output_file.write(f"# teacher_model_id\t{teacher_model_id}\n")
        output_file.write(f"# student_model_id\t{student_model_id}\n")
        output_file.write(f"# teacher_prefix\t{teacher_prefix}\n")
        output_file.write(f"# student_prefix\t{student_prefix}\n")
        output_file.write(f"# depth_strategy\t{depth_strategy}\n")
        output_file.write(f"# width_strategy\t{width_strategy}\n")
        output_file.write(f"# teacher_layer_count\t{teacher_layer_count}\n")
        output_file.write(f"# student_layer_count\t{student_layer_count}\n")
        output_file.write(f"# sample_count\t{sample_count}\n")
        output_file.write(
            "student_entry_idx\tstudent_layer_idx\tstudent_target\tteacher_entry_idx\t"
            "teacher_layer_idx\tteacher_target\tteacher_tensor_name\tstudent_tensor_name\t"
            "source_dim\ttarget_dim\tteacher_layer_type\tstudent_layer_type\talias_of_entry\n"
        )

        for entry in entries:
            alias_text = "-" if entry.alias_of_entry == TAPE_NO_ALIAS else str(entry.alias_of_entry)
            output_file.write(
                f"{entry.student_entry_idx}\t"
                f"{entry.student_layer_idx}\t"
                f"{_capture_target_label(entry.capture_target)}\t"
                f"{entry.teacher_entry_idx}\t"
                f"{entry.teacher_layer_idx}\t"
                f"{_capture_target_label(entry.capture_target)}\t"
                f"{entry.teacher_tensor_name}\t"
                f"{entry.student_tensor_name}\t"
                f"{entry.source_dim}\t"
                f"{entry.target_dim}\t"
                f"{entry.teacher_layer_type}\t"
                f"{entry.student_layer_type}\t"
                f"{alias_text}\n"
            )


def _write_aligned_tape(output_path: Path,
                        raw_tape: ParsedTape,
                        student_config: ModelConfig,
                        entries: list[AlignmentEntry],
                        width_strategy: str,
                        overwrite: bool) -> None:
    entry_count = len(entries)
    sample_count = raw_tape.header.sample_count
    header = {
        "magic": TAPE_MAGIC,
        "version": TAPE_VERSION,
        "entry_count": entry_count,
        "sample_count": sample_count,
        "hidden_size": student_config.hidden_size,
        "_pad": (0, 0, 0),
        "data_section_offset": HEADER_STRUCT.size + entry_count * ENTRY_STRUCT.size,
        "data_section_size": 0,
        "crc32": 0,
        "_pad2": (0, 0, 0),
    }

    primary_entries: list[AlignmentEntry] = [entry for entry in entries if entry.alias_of_entry == TAPE_NO_ALIAS]
    data_offset = 0
    manifest_entries: list[bytes] = []
    primary_offset_by_student_entry_idx: dict[int, int] = {}

    for entry in entries:
        if entry.alias_of_entry == TAPE_NO_ALIAS:
            data_bytes = entry.target_dim * sample_count * 4
            primary_offset_by_student_entry_idx[entry.student_entry_idx] = data_offset
            manifest_entries.append(
                _pack_manifest_entry(
                    entry.student_tensor_name,
                    entry.target_dim,
                    sample_count,
                    data_offset,
                    data_bytes,
                    TAPE_NO_ALIAS,
                    entry.student_layer_type,
                )
            )
            data_offset += data_bytes
        else:
            primary_offset = primary_offset_by_student_entry_idx[entry.alias_of_entry]
            primary_entry = entries[entry.alias_of_entry]
            manifest_entries.append(
                _pack_manifest_entry(
                    entry.student_tensor_name,
                    primary_entry.target_dim,
                    sample_count,
                    primary_offset,
                    primary_entry.target_dim * sample_count * 4,
                    entry.alias_of_entry,
                    entry.student_layer_type,
                )
            )

    header["data_section_size"] = data_offset
    header_bytes = HEADER_STRUCT.pack(
        header["magic"],
        header["version"],
        header["entry_count"],
        header["sample_count"],
        header["hidden_size"],
        *header["_pad"],
        header["data_section_offset"],
        header["data_section_size"],
        header["crc32"],
        *header["_pad2"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing tape: {output_path}")

    with output_path.open("wb") as output_file:
        output_file.write(header_bytes)
        for manifest_entry in manifest_entries:
            output_file.write(manifest_entry)

        for entry in primary_entries:
            source_entry = raw_tape.entries[entry.teacher_entry_idx]
            source_offset = raw_tape.header.data_section_offset + source_entry.data_offset
            source_values = np.frombuffer(
                raw_tape.mm,
                dtype=np.float32,
                count=source_entry.vector_dim * sample_count,
                offset=source_offset,
            ).reshape(sample_count, source_entry.vector_dim)

            aligned = np.empty((sample_count, entry.target_dim), dtype=np.float32)
            effective_strategy = _resolve_width_strategy(width_strategy,
                                                         source_entry.vector_dim,
                                                         entry.target_dim)
            for sample_idx in range(sample_count):
                aligned[sample_idx] = _resample_vector(source_values[sample_idx],
                                                       source_entry.vector_dim,
                                                       entry.target_dim,
                                                       effective_strategy)

            output_file.write(aligned.tobytes(order="C"))

        output_file.flush()
        os.fsync(output_file.fileno())

        crc32 = binascii.crc32(header_bytes[:HEADER_CRC_OFFSET]) & 0xFFFFFFFF
        output_file.seek(HEADER_CRC_OFFSET)
        output_file.write(struct.pack(f"{ENDIAN}I", crc32))
        output_file.flush()
        os.fsync(output_file.fileno())


def expand_tape(input_tape: Path,
                output_tape: Path,
                manifest_out: Path,
                student_model_dir: Path,
                teacher_model_id: str,
                student_model_id: str,
                teacher_prefix: str | None,
                student_prefix: str,
                depth_strategy: str,
                width_strategy: str,
                overwrite: bool) -> None:
    raw_tape = _parse_tape(input_tape)
    try:
        student_config = _load_model_config(student_model_dir, student_model_id)
        resolved_teacher_prefix = _normalize_prefix(teacher_prefix) if teacher_prefix else _normalize_prefix(raw_tape.prefix)
        resolved_student_prefix = _normalize_prefix(student_prefix)
        entries = _build_alignment_entries(raw_tape,
                                           student_config,
                                           teacher_model_id,
                                           student_model_id,
                                           resolved_teacher_prefix,
                                           resolved_student_prefix,
                                           depth_strategy,
                                           width_strategy)

        _write_alignment_manifest(manifest_out,
                                  teacher_model_id,
                                  student_model_id,
                                  resolved_teacher_prefix,
                                  resolved_student_prefix,
                                  raw_tape.layer_count,
                                  student_config.num_hidden_layers,
                                  raw_tape.header.sample_count,
                                  depth_strategy,
                                  width_strategy,
                                  entries,
                                  overwrite)
        _write_aligned_tape(output_tape,
                            raw_tape,
                            student_config,
                            entries,
                            width_strategy,
                            overwrite)
    finally:
        raw_tape.close()


def _default_student_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "gemma-3-7b-q1.58b"


def _default_teacher_id() -> str:
    return "gemma-3-1b-it"


def _default_student_id() -> str:
    return "gemma-3-7b-q1.58b"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand a raw Sapphire teacher tape into a student-aligned tape",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_tape", type=Path, help="Raw teacher tape produced by --record-tape")
    parser.add_argument("output_tape", type=Path, help="Aligned student tape to write")
    parser.add_argument("--manifest-out", type=Path, default=None,
                        help="Path to the TSV alignment manifest")
    parser.add_argument("--student-model-dir", type=Path, default=_default_student_model_dir(),
                        help="Directory containing the student config.json")
    parser.add_argument("--teacher-model-id", type=str, default=_default_teacher_id(),
                        help="Teacher model identifier written to the manifest")
    parser.add_argument("--student-model-id", type=str, default=_default_student_id(),
                        help="Student model identifier written to the manifest")
    parser.add_argument("--teacher-prefix", type=str, default=None,
                        help="Override the teacher tensor prefix inferred from the tape")
    parser.add_argument("--student-prefix", type=str, default="language_model.model.layers.",
                        help="Tensor prefix to write into the aligned tape")
    parser.add_argument("--depth-strategy", type=str, choices=("bucket", "repeat"), default="bucket",
                        help="Teacher-to-student layer mapping strategy")
    parser.add_argument("--width-strategy", type=str, choices=("auto", "interpolation", "block-replication"),
                        default="auto",
                        help="Width expansion strategy for activation vectors")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing output files")
    args = parser.parse_args()

    manifest_out = args.manifest_out or args.output_tape.with_suffix(".tsv")
    expand_tape(args.input_tape,
                args.output_tape,
                manifest_out,
                args.student_model_dir,
                args.teacher_model_id,
                args.student_model_id,
                args.teacher_prefix,
                args.student_prefix,
                args.depth_strategy,
                args.width_strategy,
                args.overwrite)

    print(f"Expanded tape: {args.input_tape} -> {args.output_tape}")
    print(f"Alignment manifest: {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
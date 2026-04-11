#!/usr/bin/env python3
"""Expand a raw Sapphire Hessian sidecar into a student-aligned sidecar.

This mirrors the deterministic teacher-to-student alignment used by
scripts/expand_tape.py, but applies it to the diagonal vectors stored in a raw
teacher Hessian sidecar.

Workflow:

1. record a raw teacher tape and raw teacher Hessian sidecar
2. run scripts/expand_tape.py to create the aligned student tape
3. run this script to create the aligned student Hessian sidecar keyed to the
   aligned tape CRC
4. run conversion against the aligned tape + aligned sidecar directly
"""

from __future__ import annotations

import argparse
import binascii
import mmap
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from expand_tape import (
    TAPE_NO_ALIAS,
    StructuralMapping,
    _build_alignment_entries,
    _default_student_id,
    _default_student_model_dir,
    _default_teacher_id,
    _load_model_config,
    _normalize_prefix,
    _parse_structural_map,
    _parse_tape,
    _resolve_width_strategy,
    _resample_vector,
)
from import_hessian_sidecar import (
    ENTRY_STRUCT as SIDECAR_ENTRY_STRUCT,
    HEADER_STRUCT as SIDECAR_HEADER_STRUCT,
    EntrySpec,
    _write_sidecar,
)


SIDE_CAR_MAGIC = 0x48534331
SIDE_CAR_VERSION = 1
SIDE_CAR_NAME_MAX = 256


@dataclass(frozen=True)
class SidecarHeader:
    magic: int
    version: int
    entry_count: int
    sample_count: int
    tape_crc32: int
    manifest_entry_size: int
    teacher_model_name: str
    data_section_offset: int
    data_section_size: int
    reserved: int
    crc32: int


@dataclass(frozen=True)
class SidecarEntry:
    tensor_name: str
    vector_dim: int
    sample_count: int
    mean: float
    max_value: float
    data_offset: int
    data_bytes: int
    alias_of_entry: int
    layer_type: int


@dataclass
class ParsedSidecar:
    path: Path
    mm: mmap.mmap
    header: SidecarHeader
    entries: list[SidecarEntry]
    data: np.ndarray
    entry_index_by_name: dict[str, int]

    def close(self) -> None:
        self.data = None
        self.mm.close()


def _compute_file_crc32(path: Path) -> int:
    crc32 = 0

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            crc32 = binascii.crc32(chunk, crc32)

    return crc32 & 0xFFFFFFFF


def _parse_sidecar_entry(raw_entry: tuple[bytes, ...]) -> SidecarEntry:
    tensor_name = raw_entry[0].split(b"\0", 1)[0].decode("utf-8")
    return SidecarEntry(
        tensor_name=tensor_name,
        vector_dim=int(raw_entry[1]),
        sample_count=int(raw_entry[2]),
        mean=float(raw_entry[3]),
        max_value=float(raw_entry[4]),
        data_offset=int(raw_entry[5]),
        data_bytes=int(raw_entry[6]),
        alias_of_entry=int(raw_entry[7]),
        layer_type=int(raw_entry[8]),
    )


def _sidecar_compute_crc32(mm: mmap.mmap) -> int:
    header_prefix_size = SIDECAR_HEADER_STRUCT.size - 4
    crc32 = zlib.crc32(mm[:header_prefix_size]) & 0xFFFFFFFF

    if mm.size() > SIDECAR_HEADER_STRUCT.size:
        crc32 = zlib.crc32(mm[SIDECAR_HEADER_STRUCT.size :], crc32) & 0xFFFFFFFF

    return crc32


def _parse_sidecar(path: Path) -> ParsedSidecar:
    if not path.is_file():
        raise FileNotFoundError(f"Input sidecar not found: {path}")

    file_handle = path.open("rb")
    mm = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
    file_handle.close()

    if mm.size() < SIDECAR_HEADER_STRUCT.size:
        mm.close()
        raise ValueError(f"Sidecar file is too small: {path}")

    header_values = SIDECAR_HEADER_STRUCT.unpack_from(mm, 0)
    teacher_model_name = header_values[6].split(b"\0", 1)[0].decode("utf-8")
    header = SidecarHeader(
        magic=int(header_values[0]),
        version=int(header_values[1]),
        entry_count=int(header_values[2]),
        sample_count=int(header_values[3]),
        tape_crc32=int(header_values[4]),
        manifest_entry_size=int(header_values[5]),
        teacher_model_name=teacher_model_name,
        data_section_offset=int(header_values[7]),
        data_section_size=int(header_values[8]),
        reserved=int(header_values[9]),
        crc32=int(header_values[10]),
    )

    if header.magic != SIDE_CAR_MAGIC:
        mm.close()
        raise ValueError(f"Invalid sidecar magic in {path}")
    if header.version != SIDE_CAR_VERSION:
        mm.close()
        raise ValueError(f"Unsupported sidecar version {header.version} in {path}")
    if header.entry_count <= 0 or header.sample_count <= 0:
        mm.close()
        raise ValueError(f"Sidecar manifest is empty: {path}")
    if header.manifest_entry_size != SIDECAR_ENTRY_STRUCT.size:
        mm.close()
        raise ValueError(f"Unexpected sidecar manifest entry size in {path}")
    if not header.teacher_model_name:
        mm.close()
        raise ValueError(f"Sidecar teacher model name is missing in {path}")
    if header.reserved != 0:
        mm.close()
        raise ValueError(f"Sidecar reserved header field must be zero in {path}")
    if header.data_section_offset + header.data_section_size > mm.size():
        mm.close()
        raise ValueError(f"Sidecar data section is truncated: {path}")
    if header.data_section_size % 4 != 0:
        mm.close()
        raise ValueError(f"Sidecar data section is not float32-aligned: {path}")

    computed_crc32 = _sidecar_compute_crc32(mm)
    if computed_crc32 != header.crc32:
        mm.close()
        raise ValueError(
            f"Sidecar checksum mismatch in {path}: expected={header.crc32:08x} actual={computed_crc32:08x}"
        )

    manifest_offset = SIDECAR_HEADER_STRUCT.size
    expected_manifest_size = header.entry_count * SIDECAR_ENTRY_STRUCT.size
    if manifest_offset + expected_manifest_size > header.data_section_offset:
        mm.close()
        raise ValueError(f"Sidecar manifest overlaps data section: {path}")

    entries: list[SidecarEntry] = []
    entry_index_by_name: dict[str, int] = {}
    for entry_idx in range(header.entry_count):
        raw_entry = SIDECAR_ENTRY_STRUCT.unpack_from(mm, manifest_offset + entry_idx * SIDECAR_ENTRY_STRUCT.size)
        entry = _parse_sidecar_entry(raw_entry)
        if not entry.tensor_name:
            mm.close()
            raise ValueError(f"Empty tensor name in sidecar entry {entry_idx}: {path}")
        if len(entry.tensor_name.encode("utf-8")) >= SIDE_CAR_NAME_MAX:
            mm.close()
            raise ValueError(f"Tensor name too long in {path}: {entry.tensor_name}")
        if entry.sample_count != header.sample_count:
            mm.close()
            raise ValueError(
                f"Sidecar sample count mismatch for {entry.tensor_name}: "
                f"{entry.sample_count} vs {header.sample_count}"
            )
        if entry.tensor_name in entry_index_by_name:
            mm.close()
            raise ValueError(f"Duplicate sidecar tensor name in {path}: {entry.tensor_name}")
        entry_index_by_name[entry.tensor_name] = entry_idx
        entries.append(entry)

    data = np.frombuffer(mm, dtype=np.float32, count=header.data_section_size // 4, offset=header.data_section_offset)
    return ParsedSidecar(path=path, mm=mm, header=header, entries=entries, data=data, entry_index_by_name=entry_index_by_name)


def _sidecar_primary_index(sidecar: ParsedSidecar, entry_idx: int) -> int:
    cursor = entry_idx

    for _ in range(len(sidecar.entries)):
        entry = sidecar.entries[cursor]
        if entry.alias_of_entry == 0xFFFFFFFF:
            return cursor
        if entry.alias_of_entry < 0 or entry.alias_of_entry >= len(sidecar.entries):
            break
        cursor = entry.alias_of_entry

    raise ValueError(f"Invalid sidecar alias chain at entry {entry_idx} in {sidecar.path}")


def _sidecar_diagonal(sidecar: ParsedSidecar, tensor_name: str) -> np.ndarray:
    entry_idx = sidecar.entry_index_by_name.get(tensor_name)
    if entry_idx is None:
        raise KeyError(f"Sidecar does not contain tensor {tensor_name}")

    primary_idx = _sidecar_primary_index(sidecar, entry_idx)
    primary = sidecar.entries[primary_idx]
    start = primary.data_offset // 4
    count = primary.data_bytes // 4
    return np.asarray(sidecar.data[start : start + count], dtype=np.float32)


def _build_aligned_sidecar_entries(sidecar: ParsedSidecar,
                                   alignment_entries: list,
                                   width_strategy: str) -> list[EntrySpec]:
    entries: list[EntrySpec] = []
    next_data_offset = 0

    for entry_index, alignment_entry in enumerate(alignment_entries):
        if alignment_entry.alias_of_entry == TAPE_NO_ALIAS:
            source_entry_idx = sidecar.entry_index_by_name.get(alignment_entry.teacher_tensor_name)
            if source_entry_idx is None:
                raise KeyError(f"Sidecar does not contain tensor {alignment_entry.teacher_tensor_name}")

            source_entry = sidecar.entries[source_entry_idx]
            if source_entry.vector_dim != alignment_entry.source_dim:
                raise ValueError(
                    f"Sidecar dimension mismatch for {alignment_entry.teacher_tensor_name}: "
                    f"{source_entry.vector_dim} vs {alignment_entry.source_dim}"
                )

            source_diagonal = _sidecar_diagonal(sidecar, alignment_entry.teacher_tensor_name)
            effective_strategy = _resolve_width_strategy(width_strategy,
                                                         source_entry.vector_dim,
                                                         alignment_entry.target_dim)
            aligned_diagonal = np.ascontiguousarray(
                _resample_vector(source_diagonal,
                                 source_entry.vector_dim,
                                 alignment_entry.target_dim,
                                 effective_strategy),
                dtype=np.float32,
            )

            entries.append(
                EntrySpec(
                    tensor_name=alignment_entry.student_tensor_name,
                    alias_of=None,
                    data_key=None,
                    vector_dim=alignment_entry.target_dim,
                    sample_count=sidecar.header.sample_count,
                    layer_type=alignment_entry.student_layer_type,
                    array=aligned_diagonal,
                    mean=float(np.mean(aligned_diagonal, dtype=np.float64)),
                    max_value=float(np.max(aligned_diagonal)),
                    primary_index=entry_index,
                    data_offset=next_data_offset,
                    data_bytes=alignment_entry.target_dim * 4,
                )
            )
            next_data_offset += alignment_entry.target_dim * 4
            continue

        if alignment_entry.alias_of_entry < 0 or alignment_entry.alias_of_entry >= len(entries):
            raise ValueError(f"Invalid aligned alias index for {alignment_entry.student_tensor_name}")

        primary = entries[alignment_entry.alias_of_entry]
        entries.append(
            EntrySpec(
                tensor_name=alignment_entry.student_tensor_name,
                alias_of=alignment_entry.alias_of_entry,
                data_key=None,
                vector_dim=primary.vector_dim,
                sample_count=primary.sample_count,
                layer_type=primary.layer_type,
                array=None,
                mean=primary.mean,
                max_value=primary.max_value,
                primary_index=alignment_entry.alias_of_entry,
                data_offset=primary.data_offset,
                data_bytes=primary.data_bytes,
            )
        )

    return entries


def expand_hessian_sidecar(input_tape: Path,
                          aligned_tape: Path,
                          input_sidecar: Path,
                          output_sidecar: Path,
                          student_model_dir: Path,
                          teacher_model_id: str,
                          student_model_id: str,
                          teacher_prefix: str | None,
                          student_prefix: str,
                          depth_strategy: str,
                          width_strategy: str,
                          structural_map_path: Path | None = None) -> int:
    raw_tape = _parse_tape(input_tape)
    raw_sidecar = _parse_sidecar(input_sidecar)
    aligned_tape_view = _parse_tape(aligned_tape)

    structural_map: StructuralMapping | None = None
    if structural_map_path is not None:
        structural_map = _parse_structural_map(structural_map_path)
        print(f"Using structural map: {structural_map_path}")
        print(f"  Source: {structural_map.source_name or 'unspecified'}")
        print(f"  Target: {structural_map.target_name or 'unspecified'}")
        print(f"  Student layers: {structural_map.student_layer_count}")

    try:
        student_config = _load_model_config(student_model_dir, student_model_id)
        resolved_teacher_prefix = _normalize_prefix(teacher_prefix) if teacher_prefix else _normalize_prefix(raw_tape.prefix)
        resolved_student_prefix = _normalize_prefix(student_prefix)
        alignment_entries = _build_alignment_entries(raw_tape,
                                                     student_config,
                                                     teacher_model_id,
                                                     student_model_id,
                                                     resolved_teacher_prefix,
                                                     resolved_student_prefix,
                                                     depth_strategy,
                                                     width_strategy,
                                                     structural_map)

        raw_tape_crc32 = _compute_file_crc32(input_tape)
        if raw_sidecar.header.tape_crc32 != raw_tape_crc32:
            raise ValueError(
                f"Input sidecar tape CRC mismatch: sidecar={raw_sidecar.header.tape_crc32:08x} "
                f"raw_tape={raw_tape_crc32:08x}"
            )
        if raw_sidecar.header.sample_count != raw_tape.header.sample_count:
            raise ValueError(
                f"Input sidecar sample count mismatch: sidecar={raw_sidecar.header.sample_count} "
                f"raw_tape={raw_tape.header.sample_count}"
            )
        if aligned_tape_view.header.sample_count != raw_sidecar.header.sample_count:
            raise ValueError(
                f"Aligned tape sample count mismatch: aligned_tape={aligned_tape_view.header.sample_count} "
                f"sidecar={raw_sidecar.header.sample_count}"
            )

        aligned_entries = _build_aligned_sidecar_entries(raw_sidecar,
                                                         alignment_entries,
                                                         width_strategy)
        aligned_tape_crc32 = _compute_file_crc32(aligned_tape)
        return _write_sidecar(output_sidecar,
                              aligned_entries,
                              raw_sidecar.header.teacher_model_name,
                              raw_sidecar.header.sample_count,
                              aligned_tape_crc32)
    finally:
        aligned_tape_view.close()
        raw_sidecar.close()
        raw_tape.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expand a raw Sapphire Hessian sidecar into a student-aligned sidecar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_sidecar", type=Path, help="Raw teacher sidecar produced by --record-hessian-sidecar")
    parser.add_argument("output_sidecar", type=Path, help="Aligned student sidecar to write")
    parser.add_argument("--input-tape", type=Path, required=True,
                        help="Raw teacher tape used to derive the alignment mapping")
    parser.add_argument("--aligned-tape", type=Path, required=True,
                        help="Aligned student tape whose CRC should key the output sidecar")
    parser.add_argument("--student-model-dir", type=Path, default=_default_student_model_dir(),
                        help="Directory containing the student config.json")
    parser.add_argument("--teacher-model-id", type=str, default=_default_teacher_id(),
                        help="Teacher model identifier used for the alignment mapping")
    parser.add_argument("--student-model-id", type=str, default=_default_student_id(),
                        help="Student model identifier used for the alignment mapping")
    parser.add_argument("--teacher-prefix", type=str, default=None,
                        help="Override the teacher tensor prefix inferred from the raw tape")
    parser.add_argument("--student-prefix", type=str, default="language_model.model.layers.",
                        help="Tensor prefix expected in the aligned student tape")
    parser.add_argument("--depth-strategy", type=str, choices=("bucket", "repeat"), default="bucket",
                        help="Teacher-to-student layer mapping strategy (ignored if --structural-map is provided)")
    parser.add_argument("--width-strategy", type=str, choices=("auto", "interpolation", "block-replication"),
                        default="auto",
                        help="Width expansion strategy for Hessian diagonals")
    parser.add_argument("--structural-map", type=Path, default=None,
                        help="Explicit student-to-teacher layer mapping TSV file (overrides --depth-strategy)")
    args = parser.parse_args()

    crc32 = expand_hessian_sidecar(args.input_tape,
                                   args.aligned_tape,
                                   args.input_sidecar,
                                   args.output_sidecar,
                                   args.student_model_dir,
                                   args.teacher_model_id,
                                   args.student_model_id,
                                   args.teacher_prefix,
                                   args.student_prefix,
                                   args.depth_strategy,
                                   args.width_strategy,
                                   args.structural_map)

    print(f"Expanded Hessian sidecar: {args.input_sidecar} -> {args.output_sidecar} (crc32={crc32:08x})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
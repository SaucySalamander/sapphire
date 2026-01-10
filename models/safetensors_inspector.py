#!/usr/bin/env python3
"""Utility for inspecting safetensors files used by Sapphire.

The script helps debug tensor metadata and values without needing to load the
model inside the C runtime. It can list all tensors defined inside a
`.safetensors` file, filter them by name or regex, and optionally compute basic
statistics plus sample values. It defaults to the Gemma 3 270M file that this
workspace uses, but any safetensors file path can be supplied.
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    print("numpy is required for safetensors_inspector.py", file=sys.stderr)
    print("install with: pip install numpy", file=sys.stderr)
    raise SystemExit(1) from exc

# Mapping from safetensors dtype string to numpy dtype plus a conversion hint.
DTYPE_MAP: Dict[str, Dict[str, object]] = {
    "BF16": {"numpy": np.uint16, "bytes": 2, "kind": "bf16"},
    "F16": {"numpy": np.float16, "bytes": 2, "kind": "float"},
    "F32": {"numpy": np.float32, "bytes": 4, "kind": "float"},
    "F64": {"numpy": np.float64, "bytes": 8, "kind": "float"},
    "I8": {"numpy": np.int8, "bytes": 1, "kind": "int"},
    "I16": {"numpy": np.int16, "bytes": 2, "kind": "int"},
    "I32": {"numpy": np.int32, "bytes": 4, "kind": "int"},
    "I64": {"numpy": np.int64, "bytes": 8, "kind": "int"},
    "U8": {"numpy": np.uint8, "bytes": 1, "kind": "int"},
    "U16": {"numpy": np.uint16, "bytes": 2, "kind": "int"},
    "U32": {"numpy": np.uint32, "bytes": 4, "kind": "int"},
    "U64": {"numpy": np.uint64, "bytes": 8, "kind": "int"},
}


def parse_args() -> argparse.Namespace:
    default_path = Path(__file__).with_name("gemma") / "270m-it" / "model.safetensors"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=default_path,
        help="Path to the .safetensors file (default: models/gemma/270m-it/model.safetensors)",
    )
    parser.add_argument(
        "-t",
        "--tensor",
        action="append",
        help="Exact tensor name to inspect (can be repeated)",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        action="append",
        help="Substring filter applied case-insensitively (can be repeated)",
    )
    parser.add_argument(
        "-r",
        "--regex",
        help="Regular expression applied to tensor names",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every tensor (potentially slow and memory heavy)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list tensor names that match the provided filters",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Compute stats and print sample values for each selected tensor",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=8,
        help="Number of flattened values to sample when --stats is set (default: 8)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Skip tensors larger than this many bytes when computing stats (default: 64 MiB)",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        help="Stop after processing this many tensors",
    )
    return parser.parse_args()


def load_header(buffer: mmap.mmap) -> Tuple[Dict[str, Dict[str, object]], int]:
    header_len = int.from_bytes(buffer[:8], "little")
    header_end = 8 + header_len
    header = json.loads(buffer[8:header_end])
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return tensors, header_end


def filter_tensors(
    tensors: Dict[str, Dict[str, object]],
    names: Iterable[str] | None,
    patterns: Iterable[str] | None,
    regex: str | None,
    include_all: bool,
) -> List[Tuple[str, Dict[str, object]]]:
    compiled = re.compile(regex) if regex else None
    wanted = set(names) if names else None
    lowered_patterns = [p.lower() for p in patterns] if patterns else None

    selected: List[Tuple[str, Dict[str, object]]] = []
    for name, meta in tensors.items():
        if not include_all:
            if wanted and name not in wanted:
                continue
            if lowered_patterns and not any(pattern in name.lower() for pattern in lowered_patterns):
                continue
            if compiled and not compiled.search(name):
                continue
            if not wanted and not lowered_patterns and not compiled:
                continue
        selected.append((name, meta))
    return sorted(selected, key=lambda item: item[0])


def fmt_list(values: Iterable[float]) -> str:
    formatted = []
    for value in values:
        if isinstance(value, (float, np.floating)):
            formatted.append(f"{float(value):.6g}")
        else:
            formatted.append(str(value))
    return ", ".join(formatted)


def bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    uptyped = raw.astype(np.uint32) << 16
    return uptyped.view(np.float32)


def load_numeric_view(raw_bytes: memoryview, dtype_key: str) -> np.ndarray:
    try:
        dtype_info = DTYPE_MAP[dtype_key]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype_key}") from exc

    array = np.frombuffer(raw_bytes, dtype=dtype_info["numpy"])  # type: ignore[index]
    kind = dtype_info["kind"]  # type: ignore[index]
    if kind == "bf16":
        return bf16_to_float32(array.view(np.uint16))
    if kind == "float" and array.dtype != np.float32:
        return array.astype(np.float32)
    if kind == "int":
        return array.astype(np.float32)
    return array


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def inspect_vocab_and_embedding(safetensors_path: Path) -> None:
    """Quickly inspect embedding layer to determine vocab size."""
    with safetensors_path.open("rb") as handle:
        buffer = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            tensors, data_start = load_header(buffer)
            
            print("\n=== VOCAB & EMBEDDING INSPECTION ===")
            for name in sorted(tensors.keys()):
                if "embed" in name.lower() or "vocab" in name.lower():
                    meta = tensors[name]
                    shape = meta.get("shape", [])
                    dtype = meta.get("dtype", "?")
                    if shape and len(shape) >= 2:
                        vocab_size, d_model = shape[0], shape[1]
                        print(f"{name}: vocab_size={vocab_size}, d_model={d_model}, dtype={dtype}")
                        
                        # Check for multimodal indicators
                        added_tokens_path = safetensors_path.parent / "added_tokens.json"
                        if added_tokens_path.exists():
                            with added_tokens_path.open("r") as f:
                                added = json.load(f)
                                if added:
                                    print(f"\n⚠ VARIANT INFO from added_tokens.json:")
                                    for token_name, token_id in added.items():
                                        print(f"    {token_name}: ID {token_id}")
                                        if "image" in token_name.lower():
                                            print(f"    → MULTIMODAL variant detected (image support)")
                    else:
                        print(f"{name}: shape={shape}, dtype={dtype}")
        finally:
            buffer.close()


def main() -> None:
    args = parse_args()
    safetensors_path = args.file.expanduser().resolve()
    if not safetensors_path.exists():
        raise SystemExit(f"file not found: {safetensors_path}")

    with safetensors_path.open("rb") as handle:
        buffer = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            tensors, data_start = load_header(buffer)

            has_selector = bool(args.tensor or args.pattern or args.regex)
            include_all = args.all or (args.list and not has_selector)
            
            # Quick vocab check if no specific selection
            if not has_selector and not include_all:
                inspect_vocab_and_embedding(safetensors_path)
                print("\nNo filters provided. Use --tensor/--pattern/--regex or pass --all/--list to inspect everything.")
                return

            selection = filter_tensors(
                tensors,
                args.tensor,
                args.pattern,
                args.regex,
                include_all,
            )

            if not selection:
                print("No tensors matched. Use --list to see available names.")
                return

            if args.list and not args.stats:
                for name, meta in selection:
                    shape = meta.get("shape", [])
                    dtype = meta.get("dtype", "?")
                    offsets = meta.get("data_offsets", [0, 0])
                    print(f"{name}: dtype={dtype} shape={shape} data_offsets={offsets}")
                return

            processed = 0
            for name, meta in selection:
                dtype = meta.get("dtype")
                shape = meta.get("shape", [])
                offsets = meta.get("data_offsets")
                if dtype not in DTYPE_MAP or offsets is None:
                    print(f"{name}: missing metadata (dtype={dtype}, offsets={offsets})")
                    continue

                start_offset, end_offset = offsets
                abs_start = data_start + start_offset
                abs_end = data_start + end_offset
                byte_count = abs_end - abs_start
                element_count = math.prod(shape) if shape else 1
                expected_bytes = DTYPE_MAP[dtype]["bytes"] * element_count  # type: ignore[index]

                print(f"{name}")
                print(
                    f"  dtype={dtype} shape={shape} count={element_count} "
                    f"file_offset={abs_start} bytes={byte_count}"
                )
                if expected_bytes and expected_bytes != byte_count:
                    print(
                        f"  warning: byte count mismatch (expected {expected_bytes}, got {byte_count})"
                    )

                if not args.stats:
                    continue
                if byte_count > args.max_bytes:
                    print(
                        f"  skipped stats: tensor uses {byte_count} bytes (set --max-bytes to override)"
                    )
                    continue

                raw_view = memoryview(buffer)[abs_start:abs_end]
                try:
                    try:
                        values = load_numeric_view(raw_view, dtype)
                    except ValueError as exc:
                        print(f"  failed to decode tensor: {exc}")
                        continue

                    stats = compute_stats(values)
                    print(
                        "  stats: "
                        f"min={stats['min']:.6g} max={stats['max']:.6g} "
                        f"mean={stats['mean']:.6g} std={stats['std']:.6g}"
                    )

                    if args.head > 0 and values.size > 0:
                        sample = values.reshape(-1)[: args.head]
                        print(f"  sample[{args.head}]: {fmt_list(sample)}")
                        del sample

                finally:
                    # Explicitly release the memoryview and delete reference to numpy array
                    # before the next iteration or closing the buffer.
                    if 'values' in locals():
                        del values
                    if 'raw_view' in locals():
                        raw_view.release()
                        del raw_view

                processed += 1
                if args.max_tensors and processed >= args.max_tensors:
                    break
        finally:
            buffer.close()


if __name__ == "__main__":
    main()

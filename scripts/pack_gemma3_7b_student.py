#!/usr/bin/env python3
"""Pack a clean-slate Gemma 3 7B student into BF16 safetensors shards.

This script expects the bootstrap layout produced by
scripts/bootstrap_gemma3_7b_student.py:

  - config.json
  - embedding.bin
  - norm_final.bin
  - blk.<n>.<tensor>.bin

It writes either a single dense BF16 model.safetensors file or a sharded BF16
safetensors package with model.safetensors.index.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import os
import shutil
import struct
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "gemma-3-7b-q1.58b"
DEFAULT_MAX_SHARD_SIZE = 2 * 1024**3
STAGING_BYTES = np.dtype(np.float16).itemsize
BF16_BYTES = np.dtype(np.uint16).itemsize
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


@dataclass(frozen=True)
class TensorSpec:
    hf_name: str
    source_name: str
    shape: tuple[int, ...]
    required: bool = True


def _load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    return config.get("text_config", config)


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


def _build_tensor_specs(cfg: dict, include_lm_head: bool) -> list[TensorSpec]:
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_hidden_layers = int(cfg["num_hidden_layers"])
    num_attention_heads = int(cfg["num_attention_heads"])
    num_key_value_heads = int(cfg["num_key_value_heads"])
    vocab_size = int(cfg["vocab_size"])

    if "head_dim" in cfg and cfg["head_dim"] is not None:
        head_dim = int(cfg["head_dim"])
    else:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "head_dim is missing and hidden_size is not divisible by num_attention_heads "
                f"({hidden_size} / {num_attention_heads})"
            )
        head_dim = hidden_size // num_attention_heads

    if num_attention_heads * head_dim != hidden_size:
        raise ValueError(
            "num_attention_heads * head_dim must equal hidden_size for this bootstrap layout "
            f"({num_attention_heads} * {head_dim} != {hidden_size})"
        )

    specs: list[TensorSpec] = [
        TensorSpec(
            "language_model.model.embed_tokens.weight",
            "embedding",
            (vocab_size, hidden_size),
        ),
    ]

    for layer_idx in range(num_hidden_layers):
        specs.extend(
            [
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.input_layernorm.weight",
                    f"blk.{layer_idx}.norm_attn",
                    (hidden_size,),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.self_attn.q_proj.weight",
                    f"blk.{layer_idx}.q_proj",
                    (num_attention_heads * head_dim, hidden_size),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.self_attn.q_norm.weight",
                    f"blk.{layer_idx}.q_norm",
                    (head_dim,),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.self_attn.k_proj.weight",
                    f"blk.{layer_idx}.k_proj",
                    (num_key_value_heads * head_dim, hidden_size),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.self_attn.k_norm.weight",
                    f"blk.{layer_idx}.k_norm",
                    (head_dim,),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.self_attn.v_proj.weight",
                    f"blk.{layer_idx}.v_proj",
                    (num_key_value_heads * head_dim, hidden_size),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.self_attn.o_proj.weight",
                    f"blk.{layer_idx}.o_proj",
                    (hidden_size, num_attention_heads * head_dim),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.post_attention_layernorm.weight",
                    f"blk.{layer_idx}.norm_attn_post",
                    (hidden_size,),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.pre_feedforward_layernorm.weight",
                    f"blk.{layer_idx}.norm_ffn",
                    (hidden_size,),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.post_feedforward_layernorm.weight",
                    f"blk.{layer_idx}.norm_ffn_post",
                    (hidden_size,),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.mlp.gate_proj.weight",
                    f"blk.{layer_idx}.gate_proj",
                    (intermediate_size, hidden_size),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.mlp.up_proj.weight",
                    f"blk.{layer_idx}.up_proj",
                    (intermediate_size, hidden_size),
                ),
                TensorSpec(
                    f"language_model.model.layers.{layer_idx}.mlp.down_proj.weight",
                    f"blk.{layer_idx}.down_proj",
                    (hidden_size, intermediate_size),
                ),
            ]
        )

    specs.append(
        TensorSpec(
            "language_model.model.norm.weight",
            "norm_final",
            (hidden_size,),
        )
    )

    if include_lm_head:
        specs.append(
            TensorSpec(
                "language_model.lm_head.weight",
                "lm_head",
                (vocab_size, hidden_size),
                required=False,
            )
        )

    return specs


def _source_path(model_dir: Path, source_name: str) -> Path:
    return model_dir / f"{source_name}.bin"


def _load_tensor(model_dir: Path, spec: TensorSpec) -> np.ndarray:
    source_path = _source_path(model_dir, spec.source_name)
    if not source_path.is_file():
        if spec.required:
            raise FileNotFoundError(f"Missing tensor file: {source_path}")
        raise FileNotFoundError(source_path)

    expected_elements = math.prod(spec.shape)
    expected_size = expected_elements * STAGING_BYTES
    actual_size = source_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"Size mismatch for {source_path.name}: expected {expected_size} bytes, got {actual_size}"
        )

    return np.memmap(source_path, dtype=np.float16, mode="r", shape=spec.shape)


def _float32_to_bf16_words(values: np.ndarray) -> np.ndarray:
    float32_values = np.asarray(values, dtype=np.float32)
    float32_bits = float32_values.view(np.uint32)
    rounded_bits = float32_bits + np.uint32(0x7FFF) + ((float32_bits >> np.uint32(16)) & np.uint32(1))
    bf16_words = (rounded_bits >> np.uint32(16)).astype(np.uint16)
    if sys.byteorder != "little":
        bf16_words = bf16_words.byteswap()
    return bf16_words


def _write_bf16_tensor_data(output_file, tensor: np.ndarray, chunk_elements: int = 1 << 20) -> None:
    flat_tensor = np.asarray(tensor, dtype=np.float16).reshape(-1)

    for start_idx in range(0, flat_tensor.size, chunk_elements):
        chunk = flat_tensor[start_idx:start_idx + chunk_elements]
        bf16_words = _float32_to_bf16_words(chunk)
        output_file.write(bf16_words.tobytes(order="C"))


def _write_safetensors_file(output_path: Path,
                            tensor_items: list[tuple[TensorSpec, np.ndarray]],
                            total_size: int,
                            overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    header = OrderedDict()
    header["__metadata__"] = OrderedDict(
        [
            ("total_size", str(total_size)),
            ("storage_encoding", "dense_bf16"),
        ]
    )

    data_offset = 0
    for spec, tensor in tensor_items:
        tensor_bytes = int(tensor.size) * BF16_BYTES
        header[spec.hf_name] = OrderedDict(
            (
                ("dtype", "BF16"),
                ("shape", [int(dim) for dim in spec.shape]),
                ("data_offsets", [data_offset, data_offset + tensor_bytes]),
            )
        )
        data_offset += tensor_bytes

    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as output_file:
        output_file.write(struct.pack("<Q", len(header_bytes)))
        output_file.write(header_bytes)

        for _spec, tensor in tensor_items:
            _write_bf16_tensor_data(output_file, tensor)

        output_file.flush()
        os.fsync(output_file.fileno())


def _copy_metadata_files(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if source_dir.resolve() == output_dir.resolve():
        return

    for name in METADATA_FILES:
        source_path = source_dir / name
        if not source_path.is_file():
            continue

        destination_path = output_dir / name
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {destination_path}")

        shutil.copy2(source_path, destination_path)


def _remove_existing_pack(output_dir: Path) -> None:
    candidates = [
        output_dir / "model.safetensors",
        output_dir / "model.safetensors.index.json",
    ]
    candidates.extend(output_dir.glob("model-*.safetensors"))

    for path in candidates:
        if path.is_file():
            path.unlink()


def _pack_shards(tensor_items: list[tuple[TensorSpec, np.ndarray]], max_shard_size: int) -> list[list[tuple[TensorSpec, np.ndarray]]]:
    shards: list[list[tuple[TensorSpec, np.ndarray]]] = []
    current_shard: list[tuple[TensorSpec, np.ndarray]] = []
    current_size = 0

    for spec, tensor in tensor_items:
        tensor_size = int(tensor.nbytes)
        if current_shard and current_size + tensor_size > max_shard_size:
            shards.append(current_shard)
            current_shard = []
            current_size = 0

        current_shard.append((spec, tensor))
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    return shards


def pack_student(source_dir: Path, output_dir: Path, max_shard_size: int, include_lm_head: bool,
                 overwrite: bool, cleanup_bins: bool) -> None:
    cfg = _load_config(source_dir)
    specs = _build_tensor_specs(cfg, include_lm_head=include_lm_head or not bool(cfg.get("tie_word_embeddings", True)))

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _remove_existing_pack(output_dir)
    else:
        if (output_dir / "model.safetensors").exists() or (output_dir / "model.safetensors.index.json").exists():
            raise FileExistsError(f"Output directory already contains packed weights: {output_dir}")
        if any(output_dir.glob("model-*.safetensors")):
            raise FileExistsError(f"Output directory already contains sharded packed weights: {output_dir}")
        if source_dir.resolve() != output_dir.resolve():
            for name in METADATA_FILES:
                source_path = source_dir / name
                destination_path = output_dir / name
                if source_path.is_file() and destination_path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing file: {destination_path}")

    tensor_items: list[tuple[TensorSpec, np.ndarray]] = []
    total_size = 0
    for spec in specs:
        try:
            tensor = _load_tensor(source_dir, spec)
        except FileNotFoundError:
            if spec.required:
                raise
            continue
        tensor_items.append((spec, tensor))
        total_size += int(tensor.nbytes)

    if not tensor_items:
        raise RuntimeError(f"No tensors loaded from {source_dir}")

    shards = _pack_shards(tensor_items, max_shard_size=max_shard_size)
    shard_count = len(shards)
    weight_map: dict[str, str] = {}

    if shard_count == 1:
        shard_path = output_dir / "model.safetensors"
        _write_safetensors_file(shard_path, shards[0], total_size, overwrite)

        for spec, _tensor in shards[0]:
            weight_map[spec.hf_name] = shard_path.name
    else:
        shard_name_width = 5
        for shard_idx, shard in enumerate(shards, start=1):
            shard_name = f"model-{shard_idx:0{shard_name_width}d}-of-{shard_count:0{shard_name_width}d}.safetensors"
            shard_path = output_dir / shard_name
            _write_safetensors_file(shard_path, shard, total_size, overwrite)

            for spec, _tensor in shard:
                weight_map[spec.hf_name] = shard_name

        index_path = output_dir / "model.safetensors.index.json"
        with index_path.open("w", encoding="utf-8") as index_file:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, index_file, indent=2, sort_keys=True)
            index_file.write("\n")

    _copy_metadata_files(source_dir, output_dir, overwrite=overwrite)

    if cleanup_bins:
        for bin_path in source_dir.glob("*.bin"):
            if bin_path.is_file():
                bin_path.unlink()

    print(f"Packed {len(tensor_items)} tensors into {shard_count} dense BF16 safetensors file(s) at {output_dir}")
    print("Note: this export writes dense BF16 weights for current Sapphire loaders; it does not preserve compact ternary storage.")
    if shard_count > 1:
        print(f"Wrote shard index: {output_dir / 'model.safetensors.index.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack a Gemma 3 7B student directory into dense BF16 safetensors")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Directory containing the bootstrap .bin tensors and config.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write the safetensors package (defaults to --source-dir)")
    parser.add_argument("--max-shard-size", type=str, default="2GiB",
                        help="Maximum shard size, using bytes or size suffixes like 2GiB")
    parser.add_argument("--include-lm-head", action="store_true",
                        help="Include lm_head.weight instead of relying on tie_word_embeddings")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing safetensors files in the output directory")
    parser.add_argument("--cleanup-bins", action="store_true",
                        help="Delete the source .bin files after a successful pack")
    args = parser.parse_args()

    output_dir = args.output_dir or args.source_dir
    max_shard_size = _parse_shard_size(args.max_shard_size)
    pack_student(args.source_dir, output_dir, max_shard_size, args.include_lm_head,
                 args.overwrite, args.cleanup_bins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
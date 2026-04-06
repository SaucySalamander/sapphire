#!/usr/bin/env python3
"""Bootstrap a clean-slate Gemma 3 7B student directory."""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "gemma-3-7b-q1.58b"


def _load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _write_array(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output_file:
        np.asarray(array, dtype=np.float16).tofile(output_file)
    print(f"Created {path.name}: {tuple(array.shape)}")


def _write_noise_tensor(path: Path, shape: tuple[int, ...], rng: np.random.Generator,
                        noise_std: float, overwrite: bool) -> np.ndarray:
    tensor = rng.normal(0.0, noise_std, size=shape).astype(np.float16)
    _write_array(path, tensor, overwrite)
    return tensor


def _write_ones_tensor(path: Path, shape: tuple[int, ...], overwrite: bool) -> np.ndarray:
    tensor = np.ones(shape, dtype=np.float16)
    _write_array(path, tensor, overwrite)
    return tensor


def bootstrap_student(model_dir: Path, seed: int, noise_std: float, overwrite: bool) -> None:
    cfg = _load_config(model_dir)
    rng = np.random.default_rng(seed)

    hidden_size = int(cfg["hidden_size"])
    intermediate_size = int(cfg["intermediate_size"])
    num_hidden_layers = int(cfg["num_hidden_layers"])
    num_attention_heads = int(cfg["num_attention_heads"])
    num_key_value_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    vocab_size = int(cfg["vocab_size"])

    embedding = _write_noise_tensor(
        model_dir / "embedding.bin",
        (vocab_size, hidden_size),
        rng,
        noise_std,
        overwrite,
    )
    _write_array(model_dir / "lm_head.bin", embedding, overwrite)
    _write_ones_tensor(model_dir / "norm_final.bin", (hidden_size,), overwrite)

    q_shape = (num_attention_heads * head_dim, hidden_size)
    kv_shape = (num_key_value_heads * head_dim, hidden_size)
    o_shape = (hidden_size, num_attention_heads * head_dim)
    mlp_up_shape = (intermediate_size, hidden_size)
    mlp_down_shape = (hidden_size, intermediate_size)
    norm_shape = (hidden_size,)

    for layer_idx in range(num_hidden_layers):
        _write_ones_tensor(model_dir / f"blk.{layer_idx}.norm_attn.bin", norm_shape, overwrite)
        _write_noise_tensor(model_dir / f"blk.{layer_idx}.q_proj.bin", q_shape, rng, noise_std, overwrite)
        _write_ones_tensor(model_dir / f"blk.{layer_idx}.q_norm.bin", norm_shape, overwrite)
        _write_noise_tensor(model_dir / f"blk.{layer_idx}.k_proj.bin", kv_shape, rng, noise_std, overwrite)
        _write_ones_tensor(model_dir / f"blk.{layer_idx}.k_norm.bin", norm_shape, overwrite)
        _write_noise_tensor(model_dir / f"blk.{layer_idx}.v_proj.bin", kv_shape, rng, noise_std, overwrite)
        _write_noise_tensor(model_dir / f"blk.{layer_idx}.o_proj.bin", o_shape, rng, noise_std, overwrite)
        _write_ones_tensor(model_dir / f"blk.{layer_idx}.norm_attn_post.bin", norm_shape, overwrite)
        _write_ones_tensor(model_dir / f"blk.{layer_idx}.norm_ffn.bin", norm_shape, overwrite)
        _write_ones_tensor(model_dir / f"blk.{layer_idx}.norm_ffn_post.bin", norm_shape, overwrite)
        _write_noise_tensor(model_dir / f"blk.{layer_idx}.gate_proj.bin", mlp_up_shape, rng, noise_std, overwrite)
        _write_noise_tensor(model_dir / f"blk.{layer_idx}.up_proj.bin", mlp_up_shape, rng, noise_std, overwrite)
        _write_noise_tensor(model_dir / f"blk.{layer_idx}.down_proj.bin", mlp_down_shape, rng, noise_std, overwrite)

    print("\nStudent model bootstrapped. Ready for Sapphire loader.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap a Gemma 3 7B student directory")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Path to the student model directory")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the small-noise initializer")
    parser.add_argument("--noise-std", type=float, default=0.002,
                        help="Standard deviation for the initialization noise")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing tensor files")
    args = parser.parse_args()

    bootstrap_student(args.model_dir, args.seed, args.noise_std, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
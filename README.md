# Sapphire

Sapphire is a C17 playground for Large Language Model building blocks. It includes
quantized matrix-vector kernels (Q4_0/Q8_0), a small tensor abstraction, key/value
cache management, and transformer components such as RoPE, ALiBi, activations,
normalization layers, and pluggable attention strategies. Everything is built with
straightforward C and comes with focused tests and micro-benchmarks.

## Highlights
- Quantized GEMV kernels with Q4_0 and Q8_0 block formats, plus a thread-pool
  context for batched inference (`include/sapphire.h`, `src/sapphire/`).
- Tensor abstraction that supports float and quantized types with reference
  counting (`include/tensor.h`, `src/tensor/`).
- Transformer primitives: rotary positional embeddings, ALiBi, attention strategy
  hooks, activation functions, and normalization layers (`src/transformer/`).
- KV cache utilities for autoregressive decoding (`src/kv_cache/`).
- BitNet ternary linear reference implementation (`src/bitnet/`).
- Demo/validation harness for transformer components (`src/main.c`).

## Repository layout
- `include/` - Public headers for the tensor, transformer, kv_cache, GEMV, and
  sapphire kernels.
- `src/transformer/` - RoPE/ALiBi positional encodings, attention strategies,
  activations, normalization, and their tests.
- `src/sapphire/` - Quantized GEMV kernels, GGML-compatible loader, thread-pool
  context, and benchmarks.
- `src/tensor/` - Tensor core implementation and tests.
- `src/gemv/` - GEMV helpers and tests built on the tensor layer and sapphire
  kernels.
- `src/kv_cache/` - KV cache implementation and tests for attention contexts.
- `src/bitnet/` - BitNet ternary linear layer and its test.
- `src/main.c` - Standalone transformer component demo invoked by
  `make test-transformer`.
- `out/` - Build artifacts (created automatically).

## Building and running
Prerequisites: `make` and a compiler with AVX2/FMA support (the Makefile uses
`-mavx2 -mfma`). Optional HIP targets require `hipcc` and ROCm headers.

Common targets:
```bash
# Build and run the full test suite (bitnet, sapphire kernels, tensor, GEMV,
# transformer components, KV cache)
make test

# Only run the transformer component demo in src/main.c
make test-transformer

# Individual suites if you want a quicker pass
make test-tensor
make test-activations
make test-normalization
make test-kv-cache
make test-tensor-gemv
make test-sapphire

# Run quantized GEMV benchmarks
make bench

# Optional HIP build (requires ROCm/hipcc)
make hip

# Remove build artifacts
make clean
```

All binaries are written to `out/`. After running `make test` you can re-run any
binary directly (for example, `./out/transformer_test` or `./out/test_tensor`).

## Usage notes
- Public APIs are declared under `include/`; link against the objects produced in
  `out/` or incorporate the sources directly with the provided Makefile.
- The transformer demo prints softmax, RoPE/ALiBi, and attention strategy results
  to illustrate the modular interfaces.
- For HIP/ROCm setups, `make check-hip-setup` can help locate `hipcc` and headers.

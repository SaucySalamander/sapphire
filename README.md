# Sapphire

Sapphire is a C18 codebase and a test bed for small, high-performance models.
It currently targets the Gemma-3 family of models only and is not yet a
fully reusable library — several components are experimental and intended for
rapid iteration. It provides quantized matrix-vector kernels, a compact tensor
abstraction with reference counting, KV-cache utilities, and transformer
primitives (RoPE, ALiBi, activations, normalization) together with benchmark
tools.

## Highlights
- Quantized GEMV kernels with Q4_0 and Q8_0 block formats in `src/kernels/`.
- A compact tensor implementation with float and quantized types in `src/tensor/`.
- Transformer primitives in `src/transformer/` (RoPE, ALiBi, activations,
  normalization, attention strategies).
- KV cache utilities in `src/memory/`.
- Loader and model-reader utilities in `src/io/` and `src/loader/` for safetensors
  and GGML-like formats.
- A small inference/demo harness in `src/inference/` and benchmark tools.
- Note: current kernels are CPU-only; GPU support is not available yet.

## Repository layout
- `include/` - Public headers for tensors, transformer, KV cache, kernels, and
  model/loader interfaces.
- `src/kernels/` - Quantized GEMV kernels and architecture-specific implementations.
- `src/tensor/` - Tensor core implementation.
- `src/transformer/` - Transformer blocks, activations, normalization, RoPE,
  attention strategies.
- `src/inference/` - Inference orchestration and demo harness.
- `src/io/` - Model-reader and safetensors helper implementations.
- `src/loader/` - Model spec loader and model-format helpers.
- `src/memory/` - KV cache implementation and related utilities.
- `src/tokenizer/` - Tokenizer implementation.
- `models/` - Example model artifacts and helper scripts (e.g. `models/gemma/270m-it`).
- `scripts/` - Utility scripts for weight dumping and comparisons.
- `out/` - Build artifacts (created automatically).

## Building and running
Prerequisites: `make` and a compiler with AVX2/FMA support (the Makefile uses
`-mavx2 -mfma`). Optional HIP targets require `hipcc` and ROCm headers.

Common targets:

```bash
# Build the project
make all

# Clean build artifacts
make clean
```

All build artifacts are produced in `out/`.

## Usage notes
- Public APIs are declared under `include/`; link against built objects in `out/`
  or build in-tree via the Makefile.
- The repository serves as a test bed for the Gemma-3 family; components are
  experimental and not guaranteed to be reusable as stable library APIs.
- Example model artifacts and tokenizer files are in `models/gemma/270m-it`.

### Model artifacts

- Model weights and tokenizer files for Gemma-3 models are NOT included. Download a Gemma-3 model (for example `gemma-3-270m-it`) from Hugging Face or another provider and place the required files under `models/<model-name>/`.

Required files (typical):

```
models/<model-name>/
  model.safetensors    # or model.gguf / model.bin
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json  # optional
```

The runtime looks for `./models/<model-name>` as passed to `-m/--model`. If you keep models in another location, create a symlink under `models/` pointing to the external directory.

### Build targets

- `make bin` builds the main runtime binary and produces `out/sapphire` (this is a convenient alias for building the non-test runtime). `make all` will also build the runtime.

### CLI usage and examples

The runtime exposes a small CLI and an interactive REPL. Primary flags:

- `-m, --model <name>`   : Model directory name under `./models/` (required).
- `-c, --context <N>`    : Context length (default: 2048).
- `-t, --temp <val>`     : Sampling temperature (default: 1.0).
- `-n, --max-tokens <N>` : Maximum tokens to generate (default: 100).
- `-p, --prompt <str>`   : Run a single non-interactive prompt and exit.

Examples:

```bash
# Build runtime
make bin

# Interactive mode (loads model from ./models/gemma-3-270m-it)
./out/sapphire -m gemma3-270m-it

# Non-interactive one-shot prompt
./out/sapphire -m gemma3-270m-it -p "Write a haiku about compiler optimizations" -n 80 -t 0.7
```

Interactive REPL commands:

- `/exit` or `/quit` : Exit the program
- `/clear`           : Clear conversation history (resets session)
- `/info`            : Show current model/config
- `/help`            : Show command help

If you want help for the runtime itself, run `./out/sapphire -h`.

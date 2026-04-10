# Sapphire

Sapphire is a C18 codebase for Gemma 3 inference and model tooling. It supports
CPU reference execution, Vulkan compute, a compact tensor abstraction with
reference counting, KV-cache utilities, transformer primitives (RoPE, ALiBi,
activations, normalization), and workflows for activation tapes, ternary
conversion, validation, and benchmarking. Several components are still
experimental and intended for rapid iteration.

## Highlights

- Quantized GEMV kernels with Q4_0 and Q8_0 block formats in `src/kernels/`.
- CPU reference execution and Vulkan compute backend code in `src/inference/`
  and `src/kernels/backends/vulkan/`.
- A compact tensor implementation with float and quantized types in `src/tensor/`.
- Transformer primitives in `src/transformer/` (RoPE, ALiBi, activations,
  normalization, attention strategies).
- KV cache utilities in `src/memory/`.
- Loader and model-reader utilities in `src/io/` and `src/loader/` for safetensors
  and GGML-like formats.
- Inference orchestration, session lifecycle, and demo harness code in `src/inference/`.
- Benchmark tools and workflow scripts in `scripts/`.

## Repository layout

- `include/` - Public headers for tensors, transformer, KV cache, kernels, and
  model/loader interfaces.
- `src/kernels/` - Quantized GEMV kernels and architecture-specific implementations.
- `src/kernels/backends/vulkan/` - Vulkan buffers, pipelines, streaming, and
  synchronization helpers.
- `src/tensor/` - Tensor core implementation.
- `src/transformer/` - Transformer blocks, activations, normalization, RoPE,
  attention strategies.
- `src/inference/` - Inference orchestration and demo harness.
- `src/io/` - Model-reader and safetensors helper implementations.
- `src/loader/` - Model spec loader and model-format helpers.
- `src/memory/` - KV cache implementation and related utilities.
- `src/tokenizer/` - Tokenizer implementation.
- `models/` - Example model artifacts and helper scripts (e.g. `models/gemma-3-270m-it`).
- `scripts/` - Utility scripts for weight dumping and comparisons.
- `out/` - Build artifacts (created automatically).

## Building and running

Prerequisites: `make` and a compiler with AVX2/FMA support (the Makefile uses
`-mavx2 -mfma`). Vulkan SDK support is required for the Vulkan backend and
shader builds. Optional HIP targets require `hipcc` and ROCm headers.

Common targets:

```bash
# Build the project
make bin

# Clean build artifacts
make clean

# Rebuild runtime + shaders after Vulkan or shader changes
make clean bin shaders
```

All build artifacts are produced in `out/`.

## Usage notes

- Public APIs are declared under `include/`; link against built objects in `out/`
  or build in-tree via the Makefile.
- The repository serves as a test bed for the Gemma-3 family; components are
  experimental and not guaranteed to be reusable as stable library APIs.
- Select the backend with `SAPPHIRE_BACKEND=cpu` or `SAPPHIRE_BACKEND=vulkan`.
  CPU is the reference path.
- Example model artifacts and tokenizer files are in `models/gemma-3-270m-it`.

### Model artifacts

- Model weights and tokenizer files for Gemma 3 models are NOT included. Download
  a Gemma 3 model (for example `gemma-3-270m-it`, `gemma-3-1b-it`, or
  `gemma-3-27b-it`) from Hugging Face or another provider and place the required
  files under `models/<model-name>/`.

Required files (typical):

```text
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
./out/sapphire -m gemma-3-270m-it

# Non-interactive one-shot prompt
./out/sapphire -m gemma-3-270m-it -p "Write a haiku about compiler optimizations" -n 80 -t 0.7
```

Interactive REPL commands:

- `/exit` or `/quit` : Exit the program
- `/clear`           : Clear conversation history (resets session)
- `/info`            : Show current model/config
- `/help`            : Show command help

If you want help for the runtime itself, run `./out/sapphire -h`.

### Workflow notes

- `--record-tape` runs on the CPU backend and requires `--calib-manifest`.
- `--convert-ternary` requires `--output`.
- `--teacher-model` requires `--activation-tape` and only applies to full-model
  conversion.

### Ternary conversion corpora

The ternary conversion path accepts either a single text corpus source or a local
manifest file via `--calib-manifest` and `--validation-manifest`.

- The manifest file itself must be local.
- Each manifest entry points at a plain-text source that Sapphire can read.
- Hugging Face dataset pages are not directly usable as manifest entries unless
  they resolve to raw text; most dataset repos expose Parquet/JSON shards instead.

This repository includes a 27B-oriented “high-signal” corpus preset:

- [configs/corpus/27b_high_signal_calib_manifest.csv](configs/corpus/27b_high_signal_calib_manifest.csv)
- [configs/corpus/27b_high_signal_validation_manifest.csv](configs/corpus/27b_high_signal_validation_manifest.csv)
- [configs/corpus/README.md](configs/corpus/README.md)
- [scripts/prepare_ternary_corpus.py](scripts/prepare_ternary_corpus.py)

Prepare the local text shards first:

```bash
./.venv/bin/python -m pip install -U -r scripts/requirements-ternary-corpus.txt
./.venv/bin/python scripts/prepare_ternary_corpus.py
```

That command writes plain-text corpora under `./corpora/` matching the checked-in
manifest files.

Note: `bigcode/the-stack-v2` is gated on Hugging Face. Accept the dataset terms and
authenticate the environment (for example with `huggingface-cli login` or `HF_TOKEN`)
before running the corpus export helper.

Also note: `HF_TOKEN` only unlocks The Stack v2 metadata on Hugging Face. The helper
now tries public unsigned Software Heritage object access first and falls back to
the public HTTPS content endpoint. Explicit AWS credentials are optional.

To avoid rescanning The Stack metadata stream on every export, you can build a local
metadata cache once and sample from that cache on subsequent runs:

```bash
./.venv/bin/python scripts/prepare_ternary_corpus.py --stack-cache-only
./.venv/bin/python scripts/prepare_ternary_corpus.py
```

Example full-model conversion run:

```bash
./out/sapphire \
  -m gemma-3-27b-it \
  --convert-ternary \
  --output ./out/gemma-3-27b-it-ternary \
  --calib-manifest ./configs/corpus/27b_high_signal_calib_manifest.csv \
  --validation-manifest ./configs/corpus/27b_high_signal_validation_manifest.csv \
  --calibration-samples 8 \
  --validation-samples 64 \
  --validate-every 32 \
  --kl-weight 0.05
```

The conversion output directory is a Sapphire workflow artifact, not a directly
loadable runtime model package. To turn it into a normal safetensors model the
inference loader can consume, repack it against the base dense model:

```bash
./.venv/bin/python scripts/repack_ternary_model.py \
  --base-model-dir ./models/gemma-3-7b-q1.58b \
  --ternary-dir ./out/gemma-3-7b-q1.58b-ternary \
  --output-dir ./models/gemma-3-7b-q1.58b-ternary-infer \
  --overwrite

./out/sapphire -m gemma-3-7b-q1.58b-ternary-infer -t 0.0 -p "The capital of France is" -n 10
```

The repacker reconstructs every tensor listed in `manifest.tsv` from the ternary
payloads, copies any remaining tensors from `--base-model-dir`, and writes a
standard `model.safetensors` or sharded `model-00001-of-NNNNN.safetensors`
package plus tokenizer/config assets. This means partial conversion outputs can
still be exported for inference, with missing tensors falling back to the base model.

Recommended 27B mix rationale:

- 50% FineWeb-Edu prose to preserve language coherence.
- 25% C/C++/Python code to preserve systems reasoning.
- 25% Proof-Pile-2 scientific/math text to preserve technical intuition.

For validation, the preset uses GSM8K and HumanEval prompts so checkpoint KL/NLL
stays anchored to math and coding behavior rather than the same replay corpus.

### Vulkan KV paging perf matrix

To measure the Phase 11 KV paging perf matrix (VRAM hit-rate plus p50/p95/p99 token latency):

```bash
make kv-paging-matrix
```

Outputs:

- `reports/vk_kv_paging_matrix.csv`
- `reports/vk_kv_paging_matrix.md`

You can customize matrix dimensions and run count directly:

```bash
python3 scripts/benchmark_vk_kv_paging_matrix.py \
  --model gemma-3-270m-it \
  --runs 9 \
  --page-tokens 64,128 \
  --vram-budgets 2048,1024,512 \
  --max-tokens 32
```

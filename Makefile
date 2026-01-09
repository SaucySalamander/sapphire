CC = gcc
SRCDIR = src
INCDIR = include
OUTDIR = out

# Compilation flags
CFLAGS = -O3 -Wall -I. -I$(INCDIR) -mavx2 -mfma
LDFLAGS = -lm -pthread

# HIP configuration (optional ROCm support)
HIPCC = hipcc
HIP_INCLUDES = $(shell for p in /opt/rocm/include /opt/rocm/hip/include /usr/include/hip /usr/local/include/hip; do [ -d $$p ] && { echo -I$$p; break; }; done)
HIP_LIBDIRS = $(shell for p in /opt/rocm/lib /opt/rocm/lib64 /usr/lib64 /usr/lib; do [ -d $$p ] && { echo -L$$p; break; }; done)
HIPCFLAGS_COMPILE = -O3 -fPIC -D__HIP_PLATFORM_AMD__ -x hip -std=c++14 $(HIP_INCLUDES)
HIPCFLAGS_LINK = -O3 -fPIC -D__HIP_PLATFORM_AMD__ -std=c++14 $(HIP_LIBDIRS)

# Target binaries
# NOTE: bench targets (sapphire_end_to_end_bench, bench_q4, bench_q8) are temporarily
# excluded - they reference old GGML_TYPE_* enums that were unified into tensor_dtype_t.
# They will be migrated in a follow-up task.
TARGETS = \
	$(OUTDIR)/test_bitnet \
	$(OUTDIR)/test_sapphire \
	$(OUTDIR)/transformer_test \
	$(OUTDIR)/sapphire \
	$(OUTDIR)/test_tensor \
	$(OUTDIR)/test_activations \
	$(OUTDIR)/test_normalization \
	$(OUTDIR)/test_transformer_block \
	$(OUTDIR)/test_e2e_transformer \
	$(OUTDIR)/test_kv_cache \
	$(OUTDIR)/test_tensor_gemv \
	$(OUTDIR)/test_ggml_reader \
	$(OUTDIR)/test_inference \
	$(OUTDIR)/test_safetensors_reader \
	$(OUTDIR)/test_tensor_mapper

.PHONY: all bench test test-sapphire test-transformer test-transformer-block test-e2e test-tensor test-activations test-normalization test-kv-cache test-tensor-gemv test-ggml-reader test-inference test-safetensors test-tensor-mapper hip check-hip check-hip-setup clean asan-test asan-all run

all: $(TARGETS)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generic compilation rule for all C files
$(OUTDIR)/%.o: $(SRCDIR)/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/transformer/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/memory/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/tensor/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/gemv/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/bitnet/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/io/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/loader/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/kernels/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/kernels/bench/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rename main.c to transformer_main.o for disambiguation
$(OUTDIR)/transformer_main.o: $(SRCDIR)/main.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile test_normalization from transformer module (not consolidated)
$(OUTDIR)/test_normalization.o: $(SRCDIR)/transformer/test_normalization.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# ============================================================================
# Benchmark Targets
# ============================================================================

$(OUTDIR)/bench_q4: $(OUTDIR)/bench_q4.o $(OUTDIR)/q4_0_avx.o $(OUTDIR)/q8_0_avx.o $(OUTDIR)/dispatch.o $(OUTDIR)/pool.o $(OUTDIR)/bf16_avx.o $(OUTDIR)/f32_avx.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/bench_q8: $(OUTDIR)/bench_q8.o $(OUTDIR)/q8_0_avx.o $(OUTDIR)/q4_0_avx.o $(OUTDIR)/dispatch.o $(OUTDIR)/pool.o $(OUTDIR)/bf16_avx.o $(OUTDIR)/f32_avx.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/sapphire_end_to_end_bench: $(OUTDIR)/bench_sapphire_end_to_end.o $(OUTDIR)/dispatch.o $(OUTDIR)/pool.o $(OUTDIR)/q4_0_avx.o $(OUTDIR)/q8_0_avx.o $(OUTDIR)/bf16_avx.o $(OUTDIR)/f32_avx.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# ============================================================================
# Test Targets
# ============================================================================

$(OUTDIR)/test_bitnet: $(OUTDIR)/test_bitnet.o $(OUTDIR)/bitnet.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_sapphire: $(OUTDIR)/test_dispatch.o $(OUTDIR)/dispatch.o $(OUTDIR)/pool.o $(OUTDIR)/q4_0_avx.o $(OUTDIR)/q8_0_avx.o $(OUTDIR)/bf16_avx.o $(OUTDIR)/f32_avx.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/transformer_test: $(OUTDIR)/transformer_main.o $(OUTDIR)/rope.o $(OUTDIR)/positional_encoding.o $(OUTDIR)/attention.o $(OUTDIR)/attention_strategy.o $(OUTDIR)/activations.o $(OUTDIR)/normalization.o $(OUTDIR)/utils.o $(OUTDIR)/inference.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o $(OUTDIR)/kv_cache.o $(OUTDIR)/dispatch.o $(OUTDIR)/dispatch.o $(OUTDIR)/pool.o $(OUTDIR)/q4_0_avx.o $(OUTDIR)/q8_0_avx.o $(OUTDIR)/bf16_avx.o $(OUTDIR)/f32_avx.o $(OUTDIR)/safetensors_reader.o $(OUTDIR)/tensor_mapper.o $(OUTDIR)/tensor_mapper_safetensors.o $(OUTDIR)/tensor_mapper_ggml.o $(OUTDIR)/llm_model.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Alias: sapphire is the main interactive inference engine
$(OUTDIR)/sapphire: $(OUTDIR)/transformer_test
	cp $< $@

$(OUTDIR)/test_tensor: $(OUTDIR)/test_tensor.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_activations: $(OUTDIR)/test_activations.o $(OUTDIR)/activations.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_normalization: $(OUTDIR)/test_normalization.o $(OUTDIR)/normalization.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_transformer_block: $(OUTDIR)/test_transformer_block.o $(OUTDIR)/transformer.o $(OUTDIR)/normalization.o $(OUTDIR)/activations.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_e2e_transformer: $(OUTDIR)/test_e2e_transformer.o $(OUTDIR)/transformer.o $(OUTDIR)/normalization.o $(OUTDIR)/activations.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_kv_cache: $(OUTDIR)/test_kv_cache.o $(OUTDIR)/kv_cache.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_tensor_gemv: $(OUTDIR)/test_tensor_gemv.o $(OUTDIR)/dispatch.o $(OUTDIR)/tensor.o $(OUTDIR)/pool.o $(OUTDIR)/q4_0_avx.o $(OUTDIR)/q8_0_avx.o $(OUTDIR)/bf16_avx.o $(OUTDIR)/f32_avx.o $(OUTDIR)/ggml_reader.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_ggml_reader: $(OUTDIR)/test_ggml_reader.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_inference: $(OUTDIR)/test_inference.o $(OUTDIR)/inference.o $(OUTDIR)/tensor.o $(OUTDIR)/kv_cache.o $(OUTDIR)/rope.o $(OUTDIR)/attention.o $(OUTDIR)/attention_strategy.o $(OUTDIR)/activations.o $(OUTDIR)/normalization.o $(OUTDIR)/dispatch.o $(OUTDIR)/positional_encoding.o $(OUTDIR)/utils.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/dispatch.o $(OUTDIR)/pool.o $(OUTDIR)/q4_0_avx.o $(OUTDIR)/q8_0_avx.o $(OUTDIR)/bf16_avx.o $(OUTDIR)/f32_avx.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_safetensors_reader: $(OUTDIR)/test_safetensors_reader.o $(OUTDIR)/safetensors_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_tensor_mapper: $(OUTDIR)/test_tensor_mapper.o $(OUTDIR)/tensor_mapper.o $(OUTDIR)/tensor_mapper_safetensors.o $(OUTDIR)/tensor_mapper_ggml.o $(OUTDIR)/safetensors_reader.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o $(OUTDIR)/llm_model.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# ============================================================================
# Test & Benchmark Rules
# ============================================================================

bench: $(OUTDIR)/bench_q4 $(OUTDIR)/bench_q8 $(OUTDIR)/sapphire_end_to_end_bench
	@echo "Running Q4 benchmark..." && $(OUTDIR)/bench_q4
	@echo "\nRunning Q8 benchmark..." && $(OUTDIR)/bench_q8
	@echo "\nRunning Sapphire end-to-end benchmark..." && $(OUTDIR)/sapphire_end_to_end_bench

test: $(TARGETS)
	@echo "Running bitnet test..." && $(OUTDIR)/test_bitnet
	@echo "\nRunning sapphire test..." && $(OUTDIR)/test_sapphire
	@echo "\nRunning tensor test..." && $(OUTDIR)/test_tensor
	@echo "\nRunning activations test..." && $(OUTDIR)/test_activations
	@echo "\nRunning normalization test..." && $(OUTDIR)/test_normalization
	@echo "\nRunning transformer block test..." && $(OUTDIR)/test_transformer_block
	@echo "\nRunning end-to-end transformer test..." && $(OUTDIR)/test_e2e_transformer
	@echo "\nRunning kv_cache test..." && $(OUTDIR)/test_kv_cache
	@echo "\nRunning tensor_gemv test..." && $(OUTDIR)/test_tensor_gemv
	@echo "\nRunning ggml_reader test..." && $(OUTDIR)/test_ggml_reader
	@echo "\nRunning inference test..." && $(OUTDIR)/test_inference
	@echo "\nRunning safetensors_reader test..." && $(OUTDIR)/test_safetensors_reader
	@echo "\nRunning tensor_mapper test..." && $(OUTDIR)/test_tensor_mapper

test-transformer: $(OUTDIR)/transformer_test
	@echo "Running transformer components test..." && $(OUTDIR)/transformer_test

test-tensor: $(OUTDIR)/test_tensor
	@echo "Running tensor test..." && $(OUTDIR)/test_tensor

test-activations: $(OUTDIR)/test_activations
	@echo "Running activations test..." && $(OUTDIR)/test_activations

test-normalization: $(OUTDIR)/test_normalization
	@echo "Running normalization test..." && $(OUTDIR)/test_normalization

test-kv-cache: $(OUTDIR)/test_kv_cache
	@echo "Running kv_cache test..." && $(OUTDIR)/test_kv_cache

test-tensor-gemv: $(OUTDIR)/test_tensor_gemv
	@echo "Running tensor_gemv test..." && $(OUTDIR)/test_tensor_gemv

test-sapphire: $(OUTDIR)/test_sapphire
	@echo "Running sapphire test..." && $(OUTDIR)/test_sapphire

test-ggml-model:
	@echo "NOTE: ggml_model test excluded (functionality moved to tensor_mapper)"

test-ggml-reader: $(OUTDIR)/test_ggml_reader
	@echo "Running ggml_reader test..." && $(OUTDIR)/test_ggml_reader

test-inference: $(OUTDIR)/test_inference
	@echo "Running inference test..." && $(OUTDIR)/test_inference

test-safetensors: $(OUTDIR)/test_safetensors_reader
	@echo "Running safetensors_reader test..." && $(OUTDIR)/test_safetensors_reader

test-tensor-mapper: $(OUTDIR)/test_tensor_mapper
	@echo "Running tensor_mapper test..." && $(OUTDIR)/test_tensor_mapper

# ============================================================================
# Run Targets
# ============================================================================

run: $(OUTDIR)/sapphire
	@echo "========================================================================"
	@echo "                  Sapphire Inference Engine"
	@echo "========================================================================"
	@echo ""
	@echo "Usage: make run MODEL=/path/to/model.gguf [ARGS=\"-c 4096 -t 0.7\"]"
	@echo ""
	@echo "Examples:"
	@echo "  make run MODEL=./models/gemma-3-2b.gguf"
	@echo "  make run MODEL=./models/gemma-3-2b.safetensors ARGS=\"-c 4096 -t 0.7\""
	@echo ""
	@echo "Supported model formats:"
	@echo "  - GGML/GGUF (.gguf, .ggml, .bin)"
	@echo "  - Safetensors (.safetensors)"
	@echo ""
	@echo "Interactive commands:"
	@echo "  /exit          - Exit the program"
	@echo "  /clear         - Clear conversation history"
	@echo "  /info          - Show model information"
	@echo "  /help          - Show command help"
	@echo ""
	@if [ -z "$(MODEL)" ]; then \
		echo "ERROR: MODEL argument required"; \
		echo ""; \
		echo "Usage: make run MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi
	@echo "Launching: $(OUTDIR)/sapphire $(MODEL) $(ARGS)"
	@echo ""
	$(OUTDIR)/sapphire $(MODEL) $(ARGS)
# ============================================================================

$(OUTDIR)/bitnet_hip.o: $(SRCDIR)/bitnet_hip.c | $(OUTDIR)
	@which $(HIPCC) >/dev/null 2>&1 || { echo "ERROR: hipcc not found. Install ROCm/hip-sdk"; exit 1; }
	$(HIPCC) $(HIPCFLAGS_COMPILE) -c $< -o $@

$(OUTDIR)/bitnet_hip: $(OUTDIR)/bitnet_hip.o
	$(HIPCC) $(HIPCFLAGS_LINK) $^ -o $@

hip: $(OUTDIR)/bitnet_hip

check-hip: hip
	@echo "Attempting to run HIP binary (may fail if ROCm runtime unavailable)..." && $(OUTDIR)/bitnet_hip || echo "HIP binary ran but returned non-zero or ROCm runtime not available"

check-hip-setup:
	@echo "hipcc ->" $$(which hipcc 2>/dev/null || echo "NOT FOUND")
	@echo "hip_runtime.h locations:"
	@find /opt /usr -name hip_runtime.h 2>/dev/null || echo "no hip_runtime.h found in /opt or /usr"
	@echo "Detected HIP include flags: $(HIP_INCLUDES)"
	@echo "Detected HIP lib flags: $(HIP_LIBDIRS)"
	@echo "If ROCm installed under custom prefix, either:"
	@echo " - export PATH to include hipcc bin dir"
	@echo " - or run: source /opt/rocm/hip/bin/hipvars.sh (if present)"

# ============================================================================
# Sanitizer Targets
# ============================================================================

asan-test:
	@echo "Building and running tensor_gemv test with AddressSanitizer..."
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS) -fsanitize=address -fno-omit-frame-pointer -g -O1" LDFLAGS="$(LDFLAGS) -fsanitize=address" test-tensor-gemv
	@echo "Running ASan-enabled tensor_gemv test" && $(OUTDIR)/test_tensor_gemv

asan-all:
	@echo "Building and running full test suite with AddressSanitizer (may be slow)..."
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS) -fsanitize=address -fno-omit-frame-pointer -g -O1" LDFLAGS="$(LDFLAGS) -fsanitize=address" test

clean:
	rm -rf $(OUTDIR)

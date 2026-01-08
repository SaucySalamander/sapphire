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
TARGETS = \
	$(OUTDIR)/sapphire_end_to_end_bench \
	$(OUTDIR)/bench_q4 \
	$(OUTDIR)/bench_q8 \
	$(OUTDIR)/test_bitnet \
	$(OUTDIR)/test_sapphire \
	$(OUTDIR)/transformer_test \
	$(OUTDIR)/test_tensor \
	$(OUTDIR)/test_activations \
	$(OUTDIR)/test_normalization \
	$(OUTDIR)/test_kv_cache \
	$(OUTDIR)/test_tensor_gemv \
	$(OUTDIR)/test_ggml_model \
	$(OUTDIR)/test_ggml_reader \
	$(OUTDIR)/test_inference

.PHONY: all bench test test-sapphire test-transformer test-tensor test-activations test-normalization test-kv-cache test-tensor-gemv test-ggml-model test-ggml-reader test-inference hip check-hip check-hip-setup clean asan-test asan-all

all: $(TARGETS)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generic compilation rule for all C files
$(OUTDIR)/%.o: $(SRCDIR)/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/transformer/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/kv_cache/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/tensor/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/gemv/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/bitnet/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/sapphire/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/sapphire/bench/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Explicit rules for quantization implementations
$(OUTDIR)/sapphire_q4_0.o: $(SRCDIR)/sapphire/q4_0.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/sapphire_q8_0.o: $(SRCDIR)/sapphire/q8_0.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rename main.c to transformer_main.o for disambiguation
$(OUTDIR)/transformer_main.o: $(SRCDIR)/main.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile test_activations from transformer module
$(OUTDIR)/test_activations.o: $(SRCDIR)/transformer/test_activations.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile test_normalization from transformer module
$(OUTDIR)/test_normalization.o: $(SRCDIR)/transformer/test_normalization.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# ============================================================================
# Benchmark Targets
# ============================================================================

$(OUTDIR)/bench_q4: $(OUTDIR)/bench_q4.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/bench_q8: $(OUTDIR)/bench_q8.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/sapphire_end_to_end_bench: $(OUTDIR)/bench_sapphire_end_to_end.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# ============================================================================
# Test Targets
# ============================================================================

$(OUTDIR)/test_bitnet: $(OUTDIR)/test_bitnet.o $(OUTDIR)/bitnet.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_sapphire: $(OUTDIR)/test_sapphire.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/transformer_test: $(OUTDIR)/transformer_main.o $(OUTDIR)/rope.o $(OUTDIR)/positional_encoding.o $(OUTDIR)/attention.o $(OUTDIR)/attention_strategy.o $(OUTDIR)/activations.o $(OUTDIR)/normalization.o $(OUTDIR)/utils.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_tensor: $(OUTDIR)/test_tensor.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_activations: $(OUTDIR)/test_activations.o $(OUTDIR)/activations.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_normalization: $(OUTDIR)/test_normalization.o $(OUTDIR)/normalization.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_kv_cache: $(OUTDIR)/test_kv_cache.o $(OUTDIR)/kv_cache.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_tensor_gemv: $(OUTDIR)/test_tensor_gemv.o $(OUTDIR)/tensor_gemv.o $(OUTDIR)/tensor.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/ggml_reader.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_ggml_model: $(OUTDIR)/test_ggml_model.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_ggml_reader: $(OUTDIR)/test_ggml_reader.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OUTDIR)/test_inference: $(OUTDIR)/test_inference.o $(OUTDIR)/inference.o $(OUTDIR)/tensor.o $(OUTDIR)/kv_cache.o $(OUTDIR)/rope.o $(OUTDIR)/attention.o $(OUTDIR)/attention_strategy.o $(OUTDIR)/activations.o $(OUTDIR)/normalization.o $(OUTDIR)/tensor_gemv.o $(OUTDIR)/positional_encoding.o $(OUTDIR)/utils.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o
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
	@echo "\nRunning kv_cache test..." && $(OUTDIR)/test_kv_cache
	@echo "\nRunning tensor_gemv test..." && $(OUTDIR)/test_tensor_gemv
	@echo "\nRunning ggml_model test..." && $(OUTDIR)/test_ggml_model
	@echo "\nRunning ggml_reader test..." && $(OUTDIR)/test_ggml_reader
	@echo "\nRunning inference test..." && $(OUTDIR)/test_inference

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

test-ggml-model: $(OUTDIR)/test_ggml_model
	@echo "Running ggml_model test..." && $(OUTDIR)/test_ggml_model

test-ggml-reader: $(OUTDIR)/test_ggml_reader
	@echo "Running ggml_reader test..." && $(OUTDIR)/test_ggml_reader

test-inference: $(OUTDIR)/test_inference
	@echo "Running inference test..." && $(OUTDIR)/test_inference

# ============================================================================
# HIP/ROCm Targets (Optional)
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

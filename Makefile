CC = gcc
# project layout
SRCDIR = src
INCDIR = include

CFLAGS = -O3 -Wall -I. -I$(INCDIR) -mavx2 -mfma
LDFLAGS = -lm -pthread

HIPCC = hipcc
# Compile HIP sources as C++ and ensure AMD platform macro is defined so hip headers work
HIPCFLAGS_COMPILE = -O3 -fPIC -D__HIP_PLATFORM_AMD__ -x hip -std=c++14
HIPCFLAGS_LINK = -O3 -fPIC -D__HIP_PLATFORM_AMD__ -std=c++14

# detect common ROCm include/lib locations and add to flags if present
HIP_INCLUDES = $(shell for p in /opt/rocm/include /opt/rocm/hip/include /usr/include/hip /usr/local/include/hip; do [ -d $$p ] && { echo -I$$p; break; }; done)
HIP_LIBDIRS = $(shell for p in /opt/rocm/lib /opt/rocm/lib64 /usr/lib64 /usr/lib; do [ -d $$p ] && { echo -L$$p; break; }; done)

HIPCFLAGS_COMPILE += $(HIP_INCLUDES)
HIPCFLAGS_LINK += $(HIP_LIBDIRS)

# try common include locations for ROCm/HIP
HIP_INCLUDES = $(shell for p in /opt/rocm/include /opt/rocm/hip/include /usr/include/hip /usr/local/include/hip; do [ -d $$p ] && { echo -I$$p; break; }; done)
HIP_LIBDIRS = $(shell for p in /opt/rocm/lib /opt/rocm/lib64 /usr/lib64 /usr/lib; do [ -d $$p ] && { echo -L$$p; break; }; done)

HIPCFLAGS += $(HIP_INCLUDES) $(HIP_LIBDIRS)

OUTDIR = out

# Core benchmark and test targets
SAPP_END_TO_END_BIN = $(OUTDIR)/sapphire_end_to_end_bench
BENCH_Q4_BIN = $(OUTDIR)/bench_q4
BENCH_Q8_BIN = $(OUTDIR)/bench_q8
TEST_BITNET_BIN = $(OUTDIR)/test_bitnet
TEST_SAPPHIRE_BIN = $(OUTDIR)/test_sapphire
TRANSFORMER_BIN = $(OUTDIR)/transformer_test
TENSOR_TEST_BIN = $(OUTDIR)/test_tensor
ACTIVATIONS_TEST_BIN = $(OUTDIR)/test_activations
NORMALIZATION_TEST_BIN = $(OUTDIR)/test_normalization
KV_CACHE_TEST_BIN = $(OUTDIR)/test_kv_cache
TENSOR_GEMV_TEST_BIN = $(OUTDIR)/test_tensor_gemv
PHASE4_TEST_BIN = $(OUTDIR)/test_phase4

# HIP targets (optional)
BITNET_HIP_BIN = $(OUTDIR)/bitnet_hip

.PHONY: all bench test test-sapphire test-transformer test-tensor test-activations test-normalization test-kv-cache test-tensor-gemv test-phase4 hip check-hip check-hip-setup clean


all: $(SAPP_END_TO_END_BIN) $(BENCH_Q4_BIN) $(BENCH_Q8_BIN) $(TEST_BITNET_BIN) $(TEST_SAPPHIRE_BIN) $(TRANSFORMER_BIN) $(TENSOR_TEST_BIN) $(ACTIVATIONS_TEST_BIN) $(NORMALIZATION_TEST_BIN) $(KV_CACHE_TEST_BIN) $(TENSOR_GEMV_TEST_BIN) $(PHASE4_TEST_BIN)

# ensure output directory exists
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Core sapphire library object files
$(OUTDIR)/sapphire.o: $(SRCDIR)/sapphire/sapphire.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/sapphire_pool.o: $(SRCDIR)/sapphire/sapphire_pool.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/sapphire_q4_0.o: $(SRCDIR)/sapphire/q4_0.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/sapphire_q8_0.o: $(SRCDIR)/sapphire/q8_0.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/ggml_reader.o: $(SRCDIR)/ggml_reader.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Bitnet object files
$(OUTDIR)/bitnet.o: $(SRCDIR)/bitnet/bitnet.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/test_bitnet.o: $(SRCDIR)/bitnet/test_bitnet.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Sapphire test
$(OUTDIR)/test_sapphire.o: $(SRCDIR)/sapphire/test_sapphire.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_SAPPHIRE_BIN): $(OUTDIR)/test_sapphire.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/ggml_reader.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Bitnet test
$(TEST_BITNET_BIN): $(OUTDIR)/bitnet.o $(OUTDIR)/test_bitnet.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Module-aware source lists
TRANSFORMER_SRCS := $(wildcard $(SRCDIR)/transformer/*.c)
TRANSFORMER_OBJS := $(patsubst $(SRCDIR)/transformer/%.c,$(OUTDIR)/%.o,$(TRANSFORMER_SRCS))

# Other module source lists
KV_CACHE_SRCS := $(wildcard $(SRCDIR)/kv_cache/*.c)
KV_CACHE_OBJS := $(patsubst $(SRCDIR)/kv_cache/%.c,$(OUTDIR)/%.o,$(KV_CACHE_SRCS))

TENSOR_SRCS := $(wildcard $(SRCDIR)/tensor/*.c)
TENSOR_OBJS := $(patsubst $(SRCDIR)/tensor/%.c,$(OUTDIR)/%.o,$(TENSOR_SRCS))

GEMV_SRCS := $(wildcard $(SRCDIR)/gemv/*.c)
GEMV_OBJS := $(patsubst $(SRCDIR)/gemv/%.c,$(OUTDIR)/%.o,$(GEMV_SRCS))

BITNET_SRCS := $(wildcard $(SRCDIR)/bitnet/*.c)
BITNET_OBJS := $(patsubst $(SRCDIR)/bitnet/%.c,$(OUTDIR)/%.o,$(BITNET_SRCS))

# Module pattern rules (compile per-module sources to flat objects in $(OUTDIR))
$(OUTDIR)/%.o: $(SRCDIR)/transformer/%.c | $(OUTDIR)
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/kv_cache/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/tensor/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/gemv/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/bitnet/%.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/utils.o: $(SRCDIR)/utils.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/positional_encoding.o: $(SRCDIR)/transformer/positional_encoding.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/rope.o: $(SRCDIR)/transformer/rope.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/attention.o: $(SRCDIR)/transformer/attention.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/attention_strategy.o: $(SRCDIR)/transformer/attention_strategy.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/transformer_main.o: $(SRCDIR)/main.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TRANSFORMER_BIN): $(OUTDIR)/transformer_main.o $(TRANSFORMER_OBJS) $(OUTDIR)/utils.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Tensor test
$(OUTDIR)/tensor.o: $(SRCDIR)/tensor/tensor.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/test_tensor.o: $(SRCDIR)/tensor/test_tensor.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TENSOR_TEST_BIN): $(OUTDIR)/test_tensor.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Activations test (moved into transformer module)
$(OUTDIR)/test_activations.o: $(SRCDIR)/transformer/test_activations.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(ACTIVATIONS_TEST_BIN): $(OUTDIR)/test_activations.o $(OUTDIR)/activations.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Normalization test (moved into transformer module)
$(OUTDIR)/test_normalization.o: $(SRCDIR)/transformer/test_normalization.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(NORMALIZATION_TEST_BIN): $(OUTDIR)/test_normalization.o $(OUTDIR)/normalization.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# KV Cache test
$(OUTDIR)/test_kv_cache.o: $(SRCDIR)/kv_cache/test_kv_cache.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(KV_CACHE_TEST_BIN): $(OUTDIR)/test_kv_cache.o $(OUTDIR)/kv_cache.o $(OUTDIR)/tensor.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Tensor GEMV test
$(OUTDIR)/test_tensor_gemv.o: $(SRCDIR)/gemv/test_tensor_gemv.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TENSOR_GEMV_TEST_BIN): $(OUTDIR)/test_tensor_gemv.o $(OUTDIR)/tensor_gemv.o $(OUTDIR)/tensor.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/ggml_reader.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Sapphire end-to-end benchmark
$(OUTDIR)/sapphire_end_to_end_bench.o: $(SRCDIR)/sapphire/bench/bench_sapphire_end_to_end.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(SAPP_END_TO_END_BIN): $(OUTDIR)/sapphire_end_to_end_bench.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/ggml_reader.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Q4 benchmark
$(OUTDIR)/bench_q4.o: $(SRCDIR)/sapphire/bench/bench_q4.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BENCH_Q4_BIN): $(OUTDIR)/bench_q4.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/ggml_reader.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Q8 benchmark
$(OUTDIR)/bench_q8.o: $(SRCDIR)/sapphire/bench/bench_q8.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BENCH_Q8_BIN): $(OUTDIR)/bench_q8.o $(OUTDIR)/sapphire_q8_0.o $(OUTDIR)/sapphire_q4_0.o $(OUTDIR)/sapphire.o $(OUTDIR)/sapphire_pool.o $(OUTDIR)/ggml_reader.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Phase 4: GGML Model Loading & Inference
$(OUTDIR)/ggml_model.o: $(SRCDIR)/ggml_model.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/inference.o: $(SRCDIR)/inference.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/test_phase4.o: $(SRCDIR)/test_phase4.c | $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(PHASE4_TEST_BIN): $(OUTDIR)/test_phase4.o $(OUTDIR)/ggml_model.o $(OUTDIR)/inference.o $(OUTDIR)/ggml_reader.o $(OUTDIR)/tensor.o $(OUTDIR)/kv_cache.o $(OUTDIR)/activations.o $(OUTDIR)/attention.o $(OUTDIR)/attention_strategy.o $(OUTDIR)/normalization.o $(OUTDIR)/positional_encoding.o $(OUTDIR)/rope.o $(OUTDIR)/utils.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: bench
bench: $(BENCH_Q4_BIN) $(BENCH_Q8_BIN) $(SAPP_END_TO_END_BIN)
	@echo "Running Q4 benchmark..."
	@$(BENCH_Q4_BIN)
	@echo "\nRunning Q8 benchmark..."
	@$(BENCH_Q8_BIN)
	@echo "\nRunning Sapphire end-to-end benchmark..."
	@$(SAPP_END_TO_END_BIN)

test: $(TEST_BITNET_BIN) $(TEST_SAPPHIRE_BIN) $(TENSOR_TEST_BIN) $(ACTIVATIONS_TEST_BIN) $(NORMALIZATION_TEST_BIN) $(KV_CACHE_TEST_BIN) $(TENSOR_GEMV_TEST_BIN) $(PHASE4_TEST_BIN)
	@echo "Running bitnet test..."
	@$(TEST_BITNET_BIN)
	@echo "\nRunning sapphire test..."
	@$(TEST_SAPPHIRE_BIN)
	@echo "\nRunning tensor test..."
	@$(TENSOR_TEST_BIN)
	@echo "\nRunning activations test..."
	@$(ACTIVATIONS_TEST_BIN)
	@echo "\nRunning normalization test..."
	@$(NORMALIZATION_TEST_BIN)
	@echo "\nRunning kv_cache test..."
	@$(KV_CACHE_TEST_BIN)
	@echo "\nRunning tensor_gemv test..."
	@$(TENSOR_GEMV_TEST_BIN)
	@echo "\nRunning phase4 (GGML & inference) test..."
	@$(PHASE4_TEST_BIN)

test-transformer: $(TRANSFORMER_BIN)
	@echo "Running transformer components test..."
	@$(TRANSFORMER_BIN)

test-tensor: $(TENSOR_TEST_BIN)
	@echo "Running tensor test..."
	@$(TENSOR_TEST_BIN)

test-activations: $(ACTIVATIONS_TEST_BIN)
	@echo "Running activations test..."
	@$(ACTIVATIONS_TEST_BIN)

test-normalization: $(NORMALIZATION_TEST_BIN)
	@echo "Running normalization test..."
	@$(NORMALIZATION_TEST_BIN)

test-kv-cache: $(KV_CACHE_TEST_BIN)
	@echo "Running kv_cache test..."
	@$(KV_CACHE_TEST_BIN)

test-tensor-gemv: $(TENSOR_GEMV_TEST_BIN)
	@echo "Running tensor_gemv test..."
	@$(TENSOR_GEMV_TEST_BIN)

test-sapphire: $(TEST_SAPPHIRE_BIN)
	$(TEST_SAPPHIRE_BIN)

test-phase4: $(PHASE4_TEST_BIN)
	@echo "Running phase4 (GGML & inference) test..."
	@$(PHASE4_TEST_BIN)

# HIP build: compile bitnet_hip.c into object and link with hipcc
$(OUTDIR)/bitnet_hip.o: $(SRCDIR)/bitnet_hip.c | $(OUTDIR)
	@which $(HIPCC) >/dev/null 2>&1 || { echo "ERROR: hipcc not found in PATH. Install ROCm/hip-sdk or add hipcc to PATH."; exit 1; }
	$(HIPCC) $(HIPCFLAGS_COMPILE) -c $< -o $(OUTDIR)/bitnet_hip.o

$(BITNET_HIP_BIN): $(OUTDIR)/bitnet_hip.o
	$(HIPCC) $(HIPCFLAGS_LINK) $(OUTDIR)/bitnet_hip.o -o $(BITNET_HIP_BIN)

hip: $(BITNET_HIP_BIN)

# optionally run a tiny smoke test if hip binary exists
check-hip: hip
	@echo "Attempting to run hip binary (may fail if ROCm runtime not available)..."
	@$(OUTDIR)/bitnet_hip || echo "hip binary ran but returned non-zero or ROCm runtime not available"

# diagnostic target to help locate hipcc and headers
check-hip-setup:
	@echo "hipcc ->" $$(which hipcc 2>/dev/null || echo "NOT FOUND")
	@echo "hip_runtime.h locations:"
	@find /opt /usr -name hip_runtime.h 2>/dev/null || echo "no hip_runtime.h found in /opt or /usr"
	@echo "Detected HIP include flags: $(HIP_INCLUDES)"
	@echo "Detected HIP lib flags: $(HIP_LIBDIRS)"
	@echo "If ROCm is installed under a custom prefix, either:"
	@echo " - export PATH to include the hipcc bin dir, e.g. export PATH=/opt/rocm/bin:$$PATH"
	@echo " - or run: source /opt/rocm/hip/bin/hipvars.sh (if present) to set env vars"

clean:
	rm -rf $(OUTDIR)

# -------------------------
# AddressSanitizer (ASan) targets
# -------------------------
.PHONY: asan-test asan-all
asan-test:
	@echo "Building and running tensor_gemv test with AddressSanitizer (ASan)..."
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS) -fsanitize=address -fno-omit-frame-pointer -g -O1" LDFLAGS="$(LDFLAGS) -fsanitize=address" test-tensor-gemv
	@echo "Running ASan-enabled tensor_gemv test"
	@$(TENSOR_GEMV_TEST_BIN)

asan-all:
	@echo "Building and running full test suite with AddressSanitizer (may be slow)..."
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS) -fsanitize=address -fno-omit-frame-pointer -g -O1" LDFLAGS="$(LDFLAGS) -fsanitize=address" test

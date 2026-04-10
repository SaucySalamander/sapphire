CC = gcc
CXX = g++
SRCDIR = src
INCDIR = include
OUTDIR = out
ASAN_OUTDIR = out/asan

# Vulkan SDK detection (find vulkan headers and libraries)
VK_INCLUDE_PATHS = $(shell for p in /usr/include /usr/local/include /opt/vulkan/include; do [ -f $$p/vulkan/vulkan.h ] && { echo -I$$p; break; }; done)
VK_LIB_PATHS = $(shell for p in /usr/lib /usr/lib64 /usr/local/lib /opt/vulkan/lib; do [ -f $$p/libvulkan.so ] && { echo -L$$p; break; }; done)
VMA_INCLUDE_PATHS = $(shell for p in $(INCDIR) $(INCDIR)/third_party/vma /usr/include /usr/local/include; do [ -f $$p/vk_mem_alloc.h ] && { echo -I$$p; break; }; done)

# Optional libcurl detection for remote calibration/validation corpus ingestion
CURL_PKG_CFLAGS = $(shell pkg-config --cflags libcurl 2>/dev/null)
CURL_PKG_LIBS = $(shell pkg-config --libs libcurl 2>/dev/null)
CURL_INCLUDE_PATHS = $(shell if [ -z "$(CURL_PKG_CFLAGS)$(CURL_PKG_LIBS)" ]; then for p in /usr/include /usr/local/include; do [ -f $$p/curl/curl.h ] && { echo -I$$p; break; }; done; fi)
CURL_LIB_PATHS = $(shell if [ -z "$(CURL_PKG_CFLAGS)$(CURL_PKG_LIBS)" ]; then for p in /usr/lib /usr/lib64 /usr/local/lib; do [ -f $$p/libcurl.so ] && { echo -L$$p -lcurl; break; }; done; fi)
CURL_DEFS = $(if $(strip $(CURL_PKG_LIBS)$(CURL_LIB_PATHS)),-DSAPPHIRE_HAVE_LIBCURL=1,)

# Compilation flags
CFLAGS = -O3 -Wall -I. -I$(INCDIR) -mavx2 -mfma $(VK_INCLUDE_PATHS) $(VMA_INCLUDE_PATHS) $(CURL_PKG_CFLAGS) $(CURL_INCLUDE_PATHS) $(CURL_DEFS)
LDFLAGS = -lm -pthread -lvulkan -lstdc++ $(VK_LIB_PATHS) $(CURL_PKG_LIBS) $(CURL_LIB_PATHS)
DEPFLAGS = -MMD -MP

# AddressSanitizer + UndefinedBehaviorSanitizer flags
# Use -g for debug info (better error messages), -O1 for reasonable speed
# Include paths (-I. -I$(INCDIR)) must be present for sanitizer builds
# IMPORTANT: must include -mavx2 -mfma for AVX/FMA intrinsics in kernel code
SANITIZER_FLAGS = -g -O1 -I. -I$(INCDIR) -mavx2 -mfma -fsanitize=address,undefined -fno-omit-frame-pointer $(VK_INCLUDE_PATHS) $(VMA_INCLUDE_PATHS) $(CURL_PKG_CFLAGS) $(CURL_INCLUDE_PATHS) $(CURL_DEFS)
SANITIZER_LDFLAGS = -lm -pthread -fsanitize=address,undefined -lvulkan -lstdc++ $(VK_LIB_PATHS) $(CURL_PKG_LIBS) $(CURL_LIB_PATHS)

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
	$(OUTDIR)/sapphire \


DEPS := $(shell test -d $(OUTDIR) && find $(OUTDIR) -name '*.d' -print; \
	test -d $(ASAN_OUTDIR) && find $(ASAN_OUTDIR) -name '*.d' -print)
-include $(DEPS)

.PHONY: all bench check-bench bench_f32 bench_bf16 kv-paging-matrix test clean shaders

# ============================================================================
# SPIR-V Shader Compilation (Phase 11-04)
# ============================================================================

GLSLANG ?= glslangValidator
SHADER_DIR = src/kernels/backends/vulkan/shaders
SHADER_SRCS := $(wildcard $(SHADER_DIR)/*.comp)
SHADER_SPVS := $(patsubst %.comp,%.comp.spv,$(SHADER_SRCS))

# Rule for GLSL -> SPIR-V compilation
$(SHADER_DIR)/%.comp.spv: $(SHADER_DIR)/%.comp
	$(GLSLANG) -V $< -o $@

# Compile all shaders
shaders: $(SHADER_SPVS)
	@echo "Compiled $(words $(SHADER_SPVS)) shaders to SPIR-V"

all: $(TARGETS) $(SHADER_SPVS)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generic compilation rule for all C files
$(OUTDIR)/%.o: $(SRCDIR)/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/inference/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/io/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/kernels/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/kernels/backends/vulkan/%.o: $(SRCDIR)/kernels/backends/vulkan/%.cpp | $(OUTDIR)
	mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/loader/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/memory/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/tensor/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/tokenizer/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/transformer/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/utils/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# Build non-test binary from non-test sources
# Discover all non-test .c files under $(SRCDIR) (exclude test/)
/* Discover non-test sources: exclude src/test/ directory and common test filename patterns */
NON_TEST_SRCS := $(shell find $(SRCDIR) -type f -name '*.c' ! -path '$(SRCDIR)/test/*' ! -name 'test_*.c' ! -name '*_test.c' -print)
NON_TEST_OBJS := $(patsubst $(SRCDIR)/%.c,$(OUTDIR)/%.o,$(NON_TEST_SRCS))
NON_TEST_OBJS += $(OUTDIR)/kernels/backends/vulkan/vma_impl.o

# Library objects (non-test objects excluding main.o)
LIB_OBJS := $(filter-out $(OUTDIR)/main.o, $(NON_TEST_OBJS))

$(OUTDIR)/sapphire: $(NON_TEST_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)


# ============================================================================
# Test Build Rules
# ============================================================================

# Test sources
TEST_SRCS := $(shell find $(SRCDIR)/test -name 'test_*.c' ! -name 'test_tokenizer_debug.c')
TEST_BINS := $(patsubst $(SRCDIR)/test/%.c, $(OUTDIR)/test/%, $(TEST_SRCS))

# Rule for test objects
$(OUTDIR)/test/%.o: $(SRCDIR)/test/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

# Rule for test binaries
$(OUTDIR)/test/%: $(OUTDIR)/test/%.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

# Test target
test: $(TEST_BINS)
	@echo "Running tests..."
	@for test in $(TEST_BINS); do \
		echo "----------------------------------------------------------------"; \
		echo "Running $$test..."; \
		$$test || exit 1; \
	done
	@echo "----------------------------------------------------------------"
	@echo "All tests passed!"


# ============================================================================
# Run Targets
# ============================================================================

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

bin: $(OUTDIR)/sapphire

# ============================================================================
# AddressSanitizer + UndefinedBehaviorSanitizer (Phase 1 Hardening)
# ============================================================================

# Build with sanitizer instrumentation (separate output directory)
$(ASAN_OUTDIR):
	mkdir -p $(ASAN_OUTDIR)

# Generic rules for sanitizer builds (mirrors normal rules but uses SANITIZER_FLAGS)
$(ASAN_OUTDIR)/%.o: $(SRCDIR)/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/inference/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/io/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/kernels/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/kernels/backends/vulkan/%.o: $(SRCDIR)/kernels/backends/vulkan/%.cpp | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CXX) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/loader/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/memory/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/tensor/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/tokenizer/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/transformer/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/utils/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) $(DEPFLAGS) -c $< -o $@

# Reuse NON_TEST_SRCS and NON_TEST_OBJS but map to asan directory
ASAN_TEST_OBJS := $(patsubst $(SRCDIR)/%.c,$(ASAN_OUTDIR)/%.o,$(NON_TEST_SRCS))
ASAN_TEST_OBJS += $(ASAN_OUTDIR)/kernels/backends/vulkan/vma_impl.o

$(ASAN_OUTDIR)/sapphire: $(ASAN_TEST_OBJS)
	$(CC) $(SANITIZER_FLAGS) $^ -o $@ $(SANITIZER_LDFLAGS)

# Build and run sanitizer tests
.PHONY: bin-asan
bin-asan: $(ASAN_OUTDIR)/sapphire
	@echo "Built ASan/UBSan binary at $(ASAN_OUTDIR)/sapphire"

bin: $(OUTDIR)/sapphire

# ============================================================================
# Reports Directory
# ============================================================================

REPORTS_DIR = reports

$(REPORTS_DIR):
	mkdir -p $(REPORTS_DIR)

# ============================================================================
# Generate compile_commands.json for static analysis tools
# ============================================================================
# This allows tools like cppcheck and clang-tidy to use the exact compile commands.
.PHONY: compile_commands.json compile_commands
compile_commands.json:
	@bash scripts/gen_compile_commands.sh

compile_commands: compile_commands.json

# Static analysis with cppcheck
CPPCHECK ?= cppcheck
CPPCHECK_HTMLREPORT ?= cppcheck-htmlreport
CPPCHECK_FLAGS ?= --enable=all --inconclusive --std=c11 -I $(INCDIR) -j4 --suppressions-list=.cppcheck_suppressions --check-level=exhaustive

.PHONY: cppcheck-report

# Generate HTML report from cppcheck analysis. Runs cppcheck in non-strict mode.
cppcheck-report: compile_commands $(REPORTS_DIR)
	@echo "Running cppcheck and generating HTML report..."
	@if [ -f compile_commands.json ]; then \
	  $(CPPCHECK) $(CPPCHECK_FLAGS) --xml-version=2 --project=compile_commands.json 2> $(REPORTS_DIR)/cppcheck.xml || true; \
	else \
	  $(CPPCHECK) $(CPPCHECK_FLAGS) --xml-version=2 src 2> $(REPORTS_DIR)/cppcheck.xml || true; \
	fi
	@{ $(CPPCHECK_HTMLREPORT) --file=$(REPORTS_DIR)/cppcheck.xml --report-dir=$(REPORTS_DIR)/cppcheck-report --source-dir=. || python3 /usr/share/cppcheck/cppcheck-htmlreport.py --file=$(REPORTS_DIR)/cppcheck.xml --report-dir=$(REPORTS_DIR)/cppcheck-report --source-dir=. ; } >/dev/null 2>&1 || true
	@echo "HTML report available at $(REPORTS_DIR)/cppcheck-report/index.html"

# ============================================================================
# Strict Quality Checks (for CI/Pre-commit)
# ============================================================================

# Check: Cppcheck in strict mode (fails on errors/warnings)
.PHONY: check-cppcheck
check-cppcheck: compile_commands $(REPORTS_DIR)
	@echo "Running cppcheck (strict mode - fails on errors)..."
	@if [ -f compile_commands.json ]; then \
	  $(CPPCHECK) $(CPPCHECK_FLAGS) --project=compile_commands.json --error-exitcode=1 || (echo "❌ Cppcheck found issues"; exit 1); \
	else \
	  $(CPPCHECK) $(CPPCHECK_FLAGS) src --error-exitcode=1 || (echo "❌ Cppcheck found issues"; exit 1); \
	fi
	@echo "✓ Cppcheck passed"

# Check: Lizard complexity in strict mode (fails on violations)
.PHONY: check-complexity
check-complexity: $(REPORTS_DIR)
	@echo "Running Lizard (strict mode - CC: $(LIZARD_THRESHOLD_CC), NLOC: $(LIZARD_THRESHOLD_NLOC), Params: $(LIZARD_THRESHOLD_PARAM), Tokens: $(LIZARD_THRESHOLD_TOKEN))..."
	@command -v $(LIZARD) >/dev/null 2>&1 || { echo "❌ Lizard not found: $(LIZARD)"; exit 1; }
	@$(LIZARD) -l c -m -C $(LIZARD_THRESHOLD_CC) -L $(LIZARD_THRESHOLD_NLOC) -a $(LIZARD_THRESHOLD_PARAM) -T token_count=$(LIZARD_THRESHOLD_TOKEN) -x '*/test/*' src 2>&1 | tee $(REPORTS_DIR)/lizard-strict.txt || true
	@if grep -q "!!!!.*Warnings" $(REPORTS_DIR)/lizard-strict.txt 2>/dev/null; then \
	  echo "❌ Complexity checks failed. See $(REPORTS_DIR)/lizard-strict.txt"; \
	  exit 1; \
	fi
	@echo "✓ Complexity checks passed"

# Check: ASan/UBSan in strict mode (fails on violations)
.PHONY: check-sanitize
check-sanitize: bin-asan
	@echo "Running sanitizer checks (strict mode)..."
	@bash scripts/run_sanitizer_tests.sh $(ASAN_OUTDIR)/sapphire 2>&1 || (echo "❌ Sanitizer violations detected"; exit 1)

# Check: Run all checks (cppcheck + lizard + sanitize)
# Use -k to continue on errors so all checks run and report all issues
.PHONY: check-all
check-all:
	@make -k check-cppcheck check-complexity check-sanitize || exit 1

clean:
	rm -rf $(OUTDIR) $(ASAN_OUTDIR) $(REPORTS_DIR) compile_commands.json
	rm -f $(SHADER_DIR)/*.comp.spv

# ============================================================================
# Complexity Analysis with Lizard (Phase 2 Quality Metrics)
# ============================================================================

LIZARD ?= .venv/bin/lizard
LIZARD_THRESHOLD_CC ?= 30
LIZARD_THRESHOLD_NLOC ?= 150
LIZARD_THRESHOLD_PARAM ?= 6
LIZARD_THRESHOLD_TOKEN ?= 800

.PHONY: lizard-csv lizard-report

# Generate CSV report only (for monitoring/scripting - no browser)
lizard-csv: $(REPORTS_DIR)
	@$(LIZARD) -l c -m -C $(LIZARD_THRESHOLD_CC) -L $(LIZARD_THRESHOLD_NLOC) -a $(LIZARD_THRESHOLD_PARAM) -T token_count=$(LIZARD_THRESHOLD_TOKEN) -x '*/test/*' src --csv > $(REPORTS_DIR)/lizard-report.csv || true

# Generate CSV/HTML reports from lizard analysis (excluding tests)
lizard-report: $(REPORTS_DIR)
	@echo "Running Lizard complexity analysis and generating reports..."
	@$(LIZARD) -l c -m -C $(LIZARD_THRESHOLD_CC) -L $(LIZARD_THRESHOLD_NLOC) -a $(LIZARD_THRESHOLD_PARAM) -T token_count=$(LIZARD_THRESHOLD_TOKEN) -x '*/test/*' src --csv > $(REPORTS_DIR)/lizard-report.csv || true
	@$(LIZARD) -l c -m -C $(LIZARD_THRESHOLD_CC) -L $(LIZARD_THRESHOLD_NLOC) -a $(LIZARD_THRESHOLD_PARAM) -T token_count=$(LIZARD_THRESHOLD_TOKEN) -x '*/test/*' src -H > $(REPORTS_DIR)/lizard-report.html || true
	@echo "Complexity reports generated: $(REPORTS_DIR)/lizard-report.csv, $(REPORTS_DIR)/lizard-report.html"
	@if [ -f $(REPORTS_DIR)/lizard-report.html ]; then \
	  xdg-open $(REPORTS_DIR)/lizard-report.html 2>/dev/null || open $(REPORTS_DIR)/lizard-report.html 2>/dev/null || firefox $(REPORTS_DIR)/lizard-report.html 2>/dev/null || echo "Open $(REPORTS_DIR)/lizard-report.html manually"; \
	fi

BENCH_BINS := $(OUTDIR)/bench_q4 $(OUTDIR)/bench_q8 $(OUTDIR)/bench_end_to_end $(OUTDIR)/bench_f32 $(OUTDIR)/bench_bf16

bench: $(BENCH_BINS) check-bench

check-bench: $(BENCH_BINS)
	@bash scripts/check_benchmarks.sh

$(OUTDIR)/bench_q4: $(OUTDIR)/test/bench/bench_q4.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(OUTDIR)/bench_q8: $(OUTDIR)/test/bench/bench_q8.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(OUTDIR)/bench_end_to_end: $(OUTDIR)/test/bench/bench_sapphire_end_to_end.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)


bench_f32: $(OUTDIR)/bench_f32
bench_bf16: $(OUTDIR)/bench_bf16

KV_MATRIX_ARGS ?=

kv-paging-matrix: $(OUTDIR)/sapphire
	@python3 scripts/benchmark_vk_kv_paging_matrix.py $(KV_MATRIX_ARGS)

$(OUTDIR)/bench_f32: $(OUTDIR)/test/bench/bench_f32.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)

$(OUTDIR)/bench_bf16: $(OUTDIR)/test/bench/bench_bf16.o $(LIB_OBJS)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) -o $@ $(LDFLAGS)


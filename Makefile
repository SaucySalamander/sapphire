CC = gcc
SRCDIR = src
INCDIR = include
OUTDIR = out
ASAN_OUTDIR = out/asan

# Compilation flags
CFLAGS = -O3 -Wall -I. -I$(INCDIR) -mavx2 -mfma
LDFLAGS = -lm -pthread

# AddressSanitizer + UndefinedBehaviorSanitizer flags
# Use -g for debug info (better error messages), -O1 for reasonable speed
# Include paths (-I. -I$(INCDIR)) must be present for sanitizer builds
# IMPORTANT: must include -mavx2 -mfma for AVX/FMA intrinsics in kernel code
SANITIZER_FLAGS = -g -O1 -I. -I$(INCDIR) -mavx2 -mfma -fsanitize=address,undefined -fno-omit-frame-pointer
SANITIZER_LDFLAGS = -lm -pthread -fsanitize=address,undefined

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


.PHONY: all bench test clean

all: $(TARGETS)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generic compilation rule for all C files
$(OUTDIR)/%.o: $(SRCDIR)/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/inference/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/io/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/kernels/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/loader/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/memory/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/tensor/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/tokenizer/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/transformer/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OUTDIR)/%.o: $(SRCDIR)/utils/%.c | $(OUTDIR)
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

# Build non-test binary from non-test sources
# Discover all non-test .c files under $(SRCDIR) (exclude test/)
/* Discover non-test sources: exclude src/test/ directory and common test filename patterns */
NON_TEST_SRCS := $(shell find $(SRCDIR) -type f -name '*.c' ! -path '$(SRCDIR)/test/*' ! -name 'test_*.c' ! -name '*_test.c' -print)
NON_TEST_OBJS := $(patsubst $(SRCDIR)/%.c,$(OUTDIR)/%.o,$(NON_TEST_SRCS))

$(OUTDIR)/sapphire: $(NON_TEST_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)


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
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/inference/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/io/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/kernels/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/loader/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/memory/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/tensor/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/tokenizer/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/transformer/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

$(ASAN_OUTDIR)/%.o: $(SRCDIR)/utils/%.c | $(ASAN_OUTDIR)
	mkdir -p $(@D)
	$(CC) $(SANITIZER_FLAGS) -c $< -o $@

# Reuse NON_TEST_SRCS and NON_TEST_OBJS but map to asan directory
ASAN_TEST_OBJS := $(patsubst $(SRCDIR)/%.c,$(ASAN_OUTDIR)/%.o,$(NON_TEST_SRCS))

$(ASAN_OUTDIR)/sapphire: $(ASAN_TEST_OBJS)
	$(CC) $(SANITIZER_FLAGS) $^ -o $@ $(SANITIZER_LDFLAGS)

# Build and run sanitizer tests
.PHONY: bin-asan sanitize
bin-asan: $(ASAN_OUTDIR)/sapphire
	@echo "Built ASan/UBSan binary at $(ASAN_OUTDIR)/sapphire"

sanitize: bin-asan
	@echo "Running sanitizer checks (ASan + UBSan)..."
	@bash scripts/run_sanitizer_tests.sh $(ASAN_OUTDIR)/sapphire

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
CPPCHECK_FLAGS ?= --enable=all --inconclusive --std=c11 -I $(INCDIR) --xml-version=2 -j4 --suppressions-list=.cppcheck_suppressions --check-level=exhaustive

.PHONY: cppcheck cppcheck-report

# Run cppcheck and write XML to reports/. This target will first ensure
# a compile_commands.json exists so cppcheck sees the real compile flags.
cppcheck: compile_commands $(REPORTS_DIR)
	@echo "Running cppcheck (this may take a while)..."
	@# Prefer using compile_commands.json for accurate flags; fall back to scanning src/
	@if [ -f compile_commands.json ]; then \
	  $(CPPCHECK) $(CPPCHECK_FLAGS) --project=compile_commands.json 2> $(REPORTS_DIR)/cppcheck.xml || true; \
	else \
	  $(CPPCHECK) $(CPPCHECK_FLAGS) src 2> $(REPORTS_DIR)/cppcheck.xml || true; \
	fi
	@echo "cppcheck: xml written to $(REPORTS_DIR)/cppcheck.xml"

# Generate an HTML report from the XML output; non-fatal if report script is absent.
cppcheck-report: cppcheck
	@echo "Generating HTML report..."
	@{ $(CPPCHECK_HTMLREPORT) --file=$(REPORTS_DIR)/cppcheck.xml --report-dir=$(REPORTS_DIR)/cppcheck-report --source-dir=. || python3 /usr/share/cppcheck/cppcheck-htmlreport.py --file=$(REPORTS_DIR)/cppcheck.xml --report-dir=$(REPORTS_DIR)/cppcheck-report --source-dir=. ; } >/dev/null 2>&1 || true
	@echo "HTML report available at $(REPORTS_DIR)/cppcheck-report/index.html (if generated)"

clean:
	rm -rf $(OUTDIR) $(ASAN_OUTDIR) $(REPORTS_DIR) compile_commands.json

# ============================================================================
# Complexity Analysis with Lizard (Phase 2 Quality Metrics)
# ============================================================================

LIZARD ?= .venv/bin/lizard
LIZARD_THRESHOLD_CC ?= 15
LIZARD_THRESHOLD_LENGTH ?= 1000

.PHONY: complexity lizard-report

# Run lizard on src/ directory and generate CSV/HTML reports
# -C: Cyclomatic Complexity threshold (default 15)
# -L: Maximum function length threshold (default 1000)
# -m: Modified CCN (switch/case counts as 1)
complexity: $(REPORTS_DIR)
	@echo "Running Lizard complexity analysis..."
	@$(LIZARD) -l c -m -C $(LIZARD_THRESHOLD_CC) -L $(LIZARD_THRESHOLD_LENGTH) --csv > $(REPORTS_DIR)/lizard-report.csv || true
	@$(LIZARD) -l c -m -C $(LIZARD_THRESHOLD_CC) -L $(LIZARD_THRESHOLD_LENGTH) -H > $(REPORTS_DIR)/lizard-report.html || true
	@echo "Complexity report generated: $(REPORTS_DIR)/lizard-report.csv, $(REPORTS_DIR)/lizard-report.html"

# Generate and open HTML report
lizard-report: complexity
	@echo "Opening Lizard HTML report..."
	@if [ -f $(REPORTS_DIR)/lizard-report.html ]; then \
	  xdg-open $(REPORTS_DIR)/lizard-report.html 2>/dev/null || open $(REPORTS_DIR)/lizard-report.html 2>/dev/null || firefox $(REPORTS_DIR)/lizard-report.html 2>/dev/null || echo "Open $(REPORTS_DIR)/lizard-report.html manually"; \
	else \
	  echo "Report not generated"; \
	fi

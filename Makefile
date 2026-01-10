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

clean:
	rm -rf $(OUTDIR)

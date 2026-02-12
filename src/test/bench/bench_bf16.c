// bench_bf16.c - Benchmark for BF16 kernels

#define _POSIX_C_SOURCE 200809L
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Helper to convert float to bf16 (truncate)
static uint16_t float_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(float));
    return (uint16_t)(x >> 16);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    const int rows = 1024;
    const int blocks_per_row = 8;
    const int block_size = 32;
    const int cols = blocks_per_row * block_size;

    // Allocate W (rows * cols elements of uint16_t)
    uint16_t *W = aligned_alloc(32, sizeof(uint16_t) * rows * cols);
    // x is F32
    float *x = aligned_alloc(32, sizeof(float) * (cols + 16)); // extra for unaligned test
    float *y_ref = aligned_alloc(32, sizeof(float) * rows);
    float *y_avx = aligned_alloc(32, sizeof(float) * rows);

    if (!W || !x || !y_ref || !y_avx) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    // Init x
    for (int i = 0; i < cols; ++i) x[i] = 0.01f * ((i & 31) - 16);

    // Init W with BF16 values
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float val = 0.01f * ((c + r) & 31);
            W[r * cols + c] = float_to_bf16(val);
        }
    }

    // Warmup and reference using scalar implementation
    for (int r = 0; r < rows; ++r) {
        y_ref[r] = quantized_gemv_bf16_scalar(&W[r * cols], x, blocks_per_row, block_size);
    }

    // Benchmark AVX2
    const int repeats = 50;
    double t0 = now_s();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int r = 0; r < rows; ++r) {
            y_avx[r] = quantized_gemv_bf16_avx2(&W[r * cols], x, blocks_per_row, block_size);
        }
    }
    double t1 = now_s();
    double elapsed_aligned = (t1 - t0) / repeats;
    double wops_aligned = ((double)rows * (double)cols) / elapsed_aligned;

    // Verify
    double max_diff = 0.0;
    for (int r = 0; r < rows; ++r) {
        double d = fabs((double)y_ref[r] - (double)y_avx[r]);
        if (d > max_diff) max_diff = d;
    }

    // throughput: ops = rows * cols * 2 (mul + add) + bf16 conversion overhead
    double gflops = (2.0 * rows * cols) / elapsed_aligned / 1e9;

        printf("BF16 AVX2: rows=%d cols=%d elapsed=%.6f s wops/s=%.2f GFLOPS=%.2f max_diff=%.9f\n",
            rows, cols, elapsed_aligned, wops_aligned, gflops, max_diff);

    // Benchmark unaligned input x (force unaligned by off-by-one float)
    double t2 = now_s();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int r = 0; r < rows; ++r) {
            y_avx[r] = quantized_gemv_bf16_avx2(&W[r * cols], x + 1, blocks_per_row, block_size);
        }
    }
    double t3 = now_s();
    double elapsed_unaligned = (t3 - t2) / repeats;
    double wops_unaligned = ((double)rows * (double)cols) / elapsed_unaligned;

    printf("BF16 Unaligned (x+1): elapsed=%.6f s wops/s=%.2f\n", elapsed_unaligned, wops_unaligned);

    free(W); free(x); free(y_ref); free(y_avx);
    return 0;
}

// bench_q8.c - small benchmark for Q8 AVX2 kernels

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

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    const int rows = 1024;
    const int blocks_per_row = 8; // 8*32 = 256 cols
    const int block_size = 32;
    const int cols = blocks_per_row * block_size;

    // allocate W (rows * blocks_per_row blocks)
    ggml_block_q8_0 *W = aligned_alloc(32, sizeof(ggml_block_q8_0) * rows * blocks_per_row);
    float *x = aligned_alloc(32, sizeof(float) * cols);
    float *y_ref = aligned_alloc(32, sizeof(float) * rows);
    float *y_avx = aligned_alloc(32, sizeof(float) * rows);

    if (!W || !x || !y_ref || !y_avx) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    // init x and W with deterministic values
    for (int i = 0; i < cols; ++i) x[i] = (float)((i & 31) - 16);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            ggml_block_q8_0 *blk = &W[r * blocks_per_row + b];
            blk->scale = 0.02f * (1.0f + ((r + b) & 7));
            for (int i = 0; i < 32; ++i) blk->q_data[i] = (uint8_t)((int8_t)((i & 31) - 16));
        }
    }

    // warmup and reference (scalar or naive) using our q8 unaligned (which is AVX2 now)
    for (int r = 0; r < rows; ++r) {
        y_ref[r] = quantized_gemv_q8_0_unaligned(&W[r * blocks_per_row], x, blocks_per_row, block_size);
    }

    // benchmark aligned
    const int repeats = 10;
    double t0 = now_s();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int r = 0; r < rows; ++r) {
            y_avx[r] = quantized_gemv_q8_0_aligned(&W[r * blocks_per_row], x, blocks_per_row, block_size);
        }
    }
    double t1 = now_s();
    double elapsed_aligned = (t1 - t0) / repeats;
    double wops_aligned = ((double)rows * (double)blocks_per_row * (double)block_size) / elapsed_aligned;

    // verify
    double max_diff = 0.0;
    for (int r = 0; r < rows; ++r) {
        double d = fabs((double)y_ref[r] - (double)y_avx[r]);
        if (d > max_diff) max_diff = d;
    }

        printf("Q8 aligned: rows=%d blocks_per_row=%d elapsed_per_iter=%.6f s wops/s=%.2f max_diff=%.6f\n",
            rows, blocks_per_row, elapsed_aligned, wops_aligned, max_diff);

    // benchmark unaligned (use x+1 to force unaligned loads)
    double t2 = now_s();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int r = 0; r < rows; ++r) {
            y_avx[r] = quantized_gemv_q8_0_unaligned(&W[r * blocks_per_row], x + 1, blocks_per_row, block_size);
        }
    }
    double t3 = now_s();
    double elapsed_unaligned = (t3 - t2) / repeats;
    double wops_unaligned = ((double)rows * (double)blocks_per_row * (double)block_size) / elapsed_unaligned;

    printf("Q8 unaligned (x+1): elapsed_per_iter=%.6f s wops/s=%.2f\n", elapsed_unaligned, wops_unaligned);

    free(W); free(x); free(y_ref); free(y_avx);
    return 0;
}

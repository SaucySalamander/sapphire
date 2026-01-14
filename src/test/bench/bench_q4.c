// bench_q4.c - small benchmark for Q4 kernels

#define _POSIX_C_SOURCE 200809L
#include "sapphire.h"
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
    ggml_block_q4_0 *W = aligned_alloc(32, sizeof(ggml_block_q4_0) * rows * blocks_per_row);
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
            ggml_block_q4_0 *blk = &W[r * blocks_per_row + b];
            blk->scale = 0.02f * (1.0f + ((r + b) & 7));
            // pack 32 4-bit values into 16 bytes
            for (int i = 0; i < 16; ++i) {
                uint8_t lo = (uint8_t)(((i*2 + 0) & 31) - 16) & 0x0F;
                uint8_t hi = (uint8_t)(((i*2 + 1) & 31) - 16) & 0x0F;
                blk->q_data[i] = (hi << 4) | (lo & 0x0F);
            }
        }
    }

    // warmup and reference using scalar implementation
    for (int r = 0; r < rows; ++r) {
        y_ref[r] = quantized_gemv_row_dot_product_scalar(&W[r * blocks_per_row], x, blocks_per_row, block_size);
    }

    // benchmark aligned
    const int repeats = 10;
    double t0 = now_s();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int r = 0; r < rows; ++r) {
            y_avx[r] = quantized_gemv_row_dot_product_aligned(&W[r * blocks_per_row], x, blocks_per_row, block_size);
        }
    }
    double t1 = now_s();
    double elapsed_aligned = (t1 - t0) / repeats;

    // verify
    double max_diff = 0.0;
    for (int r = 0; r < rows; ++r) {
        double d = fabs((double)y_ref[r] - (double)y_avx[r]);
        if (d > max_diff) max_diff = d;
    }

    printf("Q4 aligned: rows=%d blocks_per_row=%d elapsed_per_iter=%.6f s max_diff=%.6f\n",
           rows, blocks_per_row, elapsed_aligned, max_diff);

    // benchmark unaligned (use x+1 to force unaligned loads)
    double t2 = now_s();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int r = 0; r < rows; ++r) {
            y_avx[r] = quantized_gemv_row_dot_product(&W[r * blocks_per_row], x + 1, blocks_per_row, block_size);
        }
    }
    double t3 = now_s();
    double elapsed_unaligned = (t3 - t2) / repeats;

    printf("Q4 unaligned (x+1): elapsed_per_iter=%.6f s\n", elapsed_unaligned);

    free(W); free(x); free(y_ref); free(y_avx);
    return 0;
}

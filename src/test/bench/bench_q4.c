// bench_q4.c - small benchmark for Q4 kernels

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

// Simple scalar implementation for verification
static float quantized_gemv_q4_0_scalar(const void *vp, const float *x, int block_count, int block_size) {
    const ggml_block_q4_0 *W = (const ggml_block_q4_0 *)vp;
    float sum = 0.0f;
    for (int i = 0; i < block_count; i++) {
        float scale = W[i].scale;
        const uint8_t *q = W[i].q_data;
        const float *x_blk = x + i * block_size;
        
        for (int j = 0; j < block_size; j += 2) {
            uint8_t v = q[j/2];
            int8_t lo = (int8_t)(v & 0x0F) - 8;
            int8_t hi = (int8_t)(v >> 4) - 8;
            
            sum += scale * lo * x_blk[j];
            sum += scale * hi * x_blk[j+1];
        }
    }
    return sum;
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
                // Ensure values are in 0..15 range to avoid weirdness with -8 offset
                uint8_t val_lo = ((i*2+0) % 16); 
                uint8_t val_hi = ((i*2+1) % 16);
                blk->q_data[i] = (val_lo) | (val_hi << 4);
            }
        }
    }

    // warmup and reference using scalar implementation
    for (int r = 0; r < rows; ++r) {
        y_ref[r] = quantized_gemv_q4_0_scalar(&W[r * blocks_per_row], x, blocks_per_row, block_size);
    }

    // benchmark aligned
    const int repeats = 10;
    double t0 = now_s();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int r = 0; r < rows; ++r) {
            y_avx[r] = quantized_gemv_q4_0_aligned(&W[r * blocks_per_row], x, blocks_per_row, block_size);
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

        printf("Q4 aligned: rows=%d blocks_per_row=%d elapsed_per_iter=%.6f s wops/s=%.2f max_diff=%.6f\n",
            rows, blocks_per_row, elapsed_aligned, wops_aligned, max_diff);

    // benchmark unaligned (posix aligned_alloc returns aligned memory, but x+1 is definitely unaligned)
    double t2 = now_s();
    for (int rep = 0; rep < repeats; ++rep) {
        for (int r = 0; r < rows; ++r) {
            y_avx[r] = quantized_gemv_q4_0_unaligned(&W[r * blocks_per_row], x + 1, blocks_per_row, block_size);
        }
    }
    double t3 = now_s();
    double elapsed_unaligned = (t3 - t2) / repeats;
    double wops_unaligned = ((double)rows * (double)blocks_per_row * (double)block_size) / elapsed_unaligned;

    printf("Q4 unaligned (x+1): elapsed_per_iter=%.6f s wops/s=%.2f\n", elapsed_unaligned, wops_unaligned);

    free(W); free(x); free(y_ref); free(y_avx);
    return 0;
}

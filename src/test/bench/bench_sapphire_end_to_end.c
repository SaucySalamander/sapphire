#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sapphire.h"

static int run_and_time(sapphire_context* ctx, const ggml_tensor_t* tensors, size_t tensor_count, int rows, int blocks_per_row, const float* x, float* y, const char* label, int repeats) {
    if (!ctx || !tensors || !x || !y || !label || repeats <= 0) return -1;
    // warm-up
    (void)sapphire_batched_gemv(ctx, tensors, tensor_count, rows, blocks_per_row, x, y);

    double total_elapsed = 0.0;
    int rc = 0;
    for (int it = 0; it < repeats; ++it) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        rc = sapphire_batched_gemv(ctx, tensors, tensor_count, rows, blocks_per_row, x, y);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        total_elapsed += elapsed;
    }
    double avg_elapsed = total_elapsed / (double)repeats;

    double sum = 0.0;
    for (int r = 0; r < rows; ++r) sum += (double)y[r];

    double wops = (double)rows * (double)blocks_per_row * 32.0;  // ops per call
    double throughput = wops / avg_elapsed;

    printf("%-25s avg=%10.6fs sum=%15.8f rc=%3d throughput=%15.2f wops/s\n", label, avg_elapsed, sum, rc, throughput);
    return rc;
}

int main(void) {
    const int configs[][2] = {
        {64, 2},
        {128, 4},
        {256, 8},
        {1024, 16},
        {8192, 32},
    };
    const int nconfigs = sizeof(configs) / sizeof(configs[0]);
    const int t4 = 4;
    const int t8 = 8;
    const int t16 = 16;

    for (int ci = 0; ci < nconfigs; ++ci) {
        const int rows = configs[ci][0];
        const int blocks_per_row = configs[ci][1];
        const int block_size = 32;
        const int chunk_size = 8;

        printf("\n=== CONFIG: rows=%d blocks_per_row=%d ===\n", rows, blocks_per_row);

        ggml_block_q4_0* W = (ggml_block_q4_0*)calloc((size_t)rows * blocks_per_row, sizeof(ggml_block_q4_0));
        if (!W) {
            fprintf(stderr, "calloc(W) failed for config %d\n", ci);
            continue;
        }

        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < blocks_per_row; ++b) {
                ggml_block_q4_0* blk = &W[r * blocks_per_row + b];
                blk->scale = 0.01f * (1.0f + ((r + b) & 7));
                for (int i = 0; i < 16; ++i) {
                    uint8_t lo = (uint8_t)((i * 2 + 0 + r + b) & 0x0F);
                    uint8_t hi = (uint8_t)((i * 2 + 1 + r + b) & 0x0F);
                    blk->q_data[i] = (hi << 4) | (lo & 0x0F);
                }
            }
        }

        float* x = NULL;
        if (posix_memalign((void**)&x, 32, sizeof(float) * block_size * blocks_per_row) != 0) x = NULL;
        if (!x) {
            free(W);
            fprintf(stderr, "posix_memalign(x) failed for config %d\n", ci);
            continue;
        }
        for (int i = 0; i < block_size * blocks_per_row; ++i) x[i] = 0.001f * (float)(i + 1);

        float* y4 = (float*)calloc((size_t)rows, sizeof(float));
        float* y8 = (float*)calloc((size_t)rows, sizeof(float));
        float* y16 = (float*)calloc((size_t)rows, sizeof(float));
        float* y8_numa = (float*)calloc((size_t)rows, sizeof(float));
        float* y16_numa = (float*)calloc((size_t)rows, sizeof(float));
        if (!y4 || !y8 || !y8_numa || !y16 || !y16_numa) {
            free(W);
            free(x);
            free(y4);
            free(y8);
            free(y8_numa);
            free(y16);
            free(y16_numa);
            fprintf(stderr, "alloc y failed for config %d\n", ci);
            continue;
        }

        // runs: end-to-end bench for Q4
        ggml_tensor_t q4_tensor = {.type = GGML_TYPE_Q4_0, .rows = rows, .cols = blocks_per_row * 32, .data = (void*)W, .data_size = 0};
        const ggml_tensor_t q4_tensors[1] = {q4_tensor};

        sapphire_context* ctx1 = sapphire_context_create(1, chunk_size);
        if (ctx1) {
            run_and_time(ctx1, q4_tensors, 1, rows, blocks_per_row, x, y4, "Q4 single-thread", 5);
            sapphire_context_destroy(ctx1);
        }

        sapphire_context* ctx4 = sapphire_context_create(t4, chunk_size);
        if (ctx4) {
            run_and_time(ctx4, q4_tensors, 1, rows, blocks_per_row, x, y4, "Q4 multi-thread (4)", 5);
            sapphire_context_destroy(ctx4);
        }

        sapphire_context* ctx8 = sapphire_context_create(t8, chunk_size);
        if (ctx8) {
            run_and_time(ctx8, q4_tensors, 1, rows, blocks_per_row, x, y4, "Q4 multi-thread (8)", 5);
            sapphire_context_destroy(ctx8);
        }

        sapphire_context* ctx16 = sapphire_context_create(t16, chunk_size);
        if (ctx16) {
            run_and_time(ctx16, q4_tensors, 1, rows, blocks_per_row, x, y16, "Q4 multi-thread (16)", 5);
            sapphire_context_destroy(ctx16);
        }

        if (setenv("SAPPHIRE_NUMA", "1", 1) != 0) {
            fprintf(stderr, "setenv(SAPPHIRE_NUMA) failed\n");
        }
        sapphire_context* ctx8_numa = sapphire_context_create(t8, chunk_size);
        if (ctx8_numa) {
            run_and_time(ctx8_numa, q4_tensors, 1, rows, blocks_per_row, x, y8_numa, "Q4 multi-thread (8) NUMA", 5);
            sapphire_context_destroy(ctx8_numa);
        }
        sapphire_context* ctx16_numa = sapphire_context_create(t16, chunk_size);
        if (ctx16_numa) {
            run_and_time(ctx16_numa, q4_tensors, 1, rows, blocks_per_row, x, y16_numa, "Q4 multi-thread (16) NUMA", 5);
            sapphire_context_destroy(ctx16_numa);
        }
        unsetenv("SAPPHIRE_NUMA");

        double max_diff_16_vs_4 = 0.0;
        double max_diff_16_numa_vs_16 = 0.0;
        for (int r = 0; r < rows; ++r) {
            double d1 = fabs((double)y16[r] - (double)y4[r]);
            if (d1 > max_diff_16_vs_4) max_diff_16_vs_4 = d1;
            double d2 = fabs((double)y16_numa[r] - (double)y16[r]);
            if (d2 > max_diff_16_numa_vs_16) max_diff_16_numa_vs_16 = d2;
        }

        printf("verification: max_diff(16 vs 4)=%.8f max_diff(16NUMA vs 16)=%.8f\n", max_diff_16_vs_4, max_diff_16_numa_vs_16);

        // Now run Q8 end-to-end bench using properly allocated Q8 weights
        ggml_block_q8_0* Wq8 = (ggml_block_q8_0*)calloc((size_t)rows * blocks_per_row, sizeof(ggml_block_q8_0));
        if (!Wq8) {
            fprintf(stderr, "calloc(Wq8) failed for config %d\n", ci);
            free(W);
            free(x);
            free(y4);
            free(y8);
            free(y8_numa);
            free(y16);
            free(y16_numa);
            continue;
        }

        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < blocks_per_row; ++b) {
                ggml_block_q8_0* blk = &Wq8[r * blocks_per_row + b];
                blk->scale = 0.01f * (1.0f + ((r + b) & 7));
                for (int i = 0; i < 32; ++i) {
                    blk->q_data[i] = (uint8_t)((int8_t)((i + r + b) & 0x1F) - 16);
                }
            }
        }

        ggml_tensor_t q8_tensor = {.type = GGML_TYPE_Q8_0, .rows = rows, .cols = blocks_per_row * 32, .data = (void*)Wq8, .data_size = 0};
        const ggml_tensor_t q8_tensors[1] = {q8_tensor};

        // reuse x and y buffers
        sapphire_context* ctxq4 = sapphire_context_create(1, chunk_size);
        if (ctxq4) {
            run_and_time(ctxq4, q8_tensors, 1, rows, blocks_per_row, x, y4, "Q8 single-thread", 5);
            sapphire_context_destroy(ctxq4);
        }

        sapphire_context* ctxq4_4 = sapphire_context_create(t4, chunk_size);
        if (ctxq4_4) {
            run_and_time(ctxq4_4, q8_tensors, 1, rows, blocks_per_row, x, y4, "Q8 multi-thread (4)", 5);
            sapphire_context_destroy(ctxq4_4);
        }

        sapphire_context* ctxq4_8 = sapphire_context_create(t8, chunk_size);
        if (ctxq4_8) {
            run_and_time(ctxq4_8, q8_tensors, 1, rows, blocks_per_row, x, y8, "Q8 multi-thread (8)", 5);
            sapphire_context_destroy(ctxq4_8);
        }

        sapphire_context* ctxq4_16 = sapphire_context_create(t16, chunk_size);
        if (ctxq4_16) {
            run_and_time(ctxq4_16, q8_tensors, 1, rows, blocks_per_row, x, y16, "Q8 multi-thread (16)", 5);
            sapphire_context_destroy(ctxq4_16);
        }

        if (setenv("SAPPHIRE_NUMA", "1", 1) != 0) {
            fprintf(stderr, "setenv(SAPPHIRE_NUMA) failed\n");
        }
        sapphire_context* ctxq4_8_numa = sapphire_context_create(t8, chunk_size);
        if (ctxq4_8_numa) {
            run_and_time(ctxq4_8_numa, q8_tensors, 1, rows, blocks_per_row, x, y8_numa, "Q8 multi-thread (8) NUMA", 5);
            sapphire_context_destroy(ctxq4_8_numa);
        }
        sapphire_context* ctxq4_16_numa = sapphire_context_create(t16, chunk_size);
        if (ctxq4_16_numa) {
            run_and_time(ctxq4_16_numa, q8_tensors, 1, rows, blocks_per_row, x, y16_numa, "Q8 multi-thread (16) NUMA", 5);
            sapphire_context_destroy(ctxq4_16_numa);
        }
        unsetenv("SAPPHIRE_NUMA");

        free(Wq8);

        free(W);
        free(x);
        free(y4);
        free(y8);
        free(y16);
        free(y8_numa);
        free(y16_numa);
    }
    return 0;
}

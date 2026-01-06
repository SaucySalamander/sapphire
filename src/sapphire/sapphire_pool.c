#include "../../include/sapphire.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

struct sapphire_context {
    int num_threads;
    int chunk_size;
    pthread_t *threads;
    atomic_int next_row;
    const void *W; // pointer to tensor payload (format depends on tensor.type)
    ggml_type_t type;
    int rows;
    int blocks_per_row;
    const float *x;
    float *y;
    int x_aligned;
    int stop;
};

static void *worker_fn(void *arg) {
    sapphire_context *ctx = (sapphire_context*)arg;
    while (!ctx->stop) {
        int start = atomic_fetch_add(&ctx->next_row, ctx->chunk_size);
        if (start >= ctx->rows) break;
        int end = start + ctx->chunk_size;
        if (end > ctx->rows) end = ctx->rows;
        for (int r = start; r < end; ++r) {
            // Dispatch based on tensor type
            if (ctx->type == GGML_TYPE_Q4_0) {
                const ggml_block_q4_0 *rowptr = (const ggml_block_q4_0*)(((const char*)ctx->W) + (size_t)r * ctx->blocks_per_row * sizeof(ggml_block_q4_0));
                if (ctx->x_aligned) {
                    ctx->y[r] = quantized_gemv_row_dot_product_aligned(rowptr, ctx->x, ctx->blocks_per_row, 32);
                } else {
                    ctx->y[r] = quantized_gemv_row_dot_product(rowptr, ctx->x, ctx->blocks_per_row, 32);
                }
            } else if (ctx->type == GGML_TYPE_Q8_0) {
                const ggml_block_q8_0 *rowptr = (const ggml_block_q8_0*)(((const char*)ctx->W) + (size_t)r * ctx->blocks_per_row * sizeof(ggml_block_q8_0));
                if (ctx->x_aligned) {
                    ctx->y[r] = quantized_gemv_q8_0_aligned(rowptr, ctx->x, ctx->blocks_per_row, 32);
                } else {
                    ctx->y[r] = quantized_gemv_q8_0_unaligned(rowptr, ctx->x, ctx->blocks_per_row, 32);
                }
            } else if (ctx->type == GGML_TYPE_F32) {
                // F32 raw weights: W points to float matrix row-major with rows * blocks_per_row*32 cols
                const float *rowptr = (const float*)(((const char*)ctx->W) + (size_t)r * ctx->blocks_per_row * 32 * sizeof(float));
                // compute dot product
                float acc = 0.0f;
                int cols = ctx->blocks_per_row * 32;
                for (int i = 0; i < cols; ++i) acc += rowptr[i] * ctx->x[i];
                ctx->y[r] = acc;
            } else {
                // unknown type
                ctx->y[r] = 0.0f;
            }
        }
    }
    return NULL;
}

sapphire_context *sapphire_context_create(int num_threads, int chunk_size) {
    if (num_threads <= 0) {
        num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (num_threads <= 0) num_threads = 1;
    }
    if (chunk_size <= 0) chunk_size = 16;

    sapphire_context *ctx = (sapphire_context*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->num_threads = num_threads;
    ctx->chunk_size = chunk_size;
    ctx->threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    if (!ctx->threads) { free(ctx); return NULL; }
    ctx->stop = 0;
    return ctx;
}

void sapphire_context_destroy(sapphire_context *ctx) {
    if (!ctx) return;
    free(ctx->threads);
    free(ctx);
}

int sapphire_batched_gemv(sapphire_context *ctx, const ggml_tensor_t *tensors, size_t tensor_count, int rows, int blocks_per_row, const float *x, float *y) {
    if (!ctx || !tensors || tensor_count == 0 || !x || !y) return 1;
    // For now we assume a single tensor covering rows; in future we may accept per-row tensor arrays
    const ggml_tensor_t *t = &tensors[0];
    ctx->W = t->data;
    ctx->type = t->type;
    ctx->rows = rows;
    ctx->blocks_per_row = blocks_per_row;
    ctx->x = x;
    ctx->y = y;
    ctx->x_aligned = (((uintptr_t)(const void*)x) & 31) == 0;
    atomic_store(&ctx->next_row, 0);
    ctx->stop = 0;

    // create workers
    for (int i = 0; i < ctx->num_threads; ++i) {
        if (pthread_create(&ctx->threads[i], NULL, worker_fn, ctx) != 0) {
            ctx->stop = 1;
            // join already created
            for (int j = 0; j < i; ++j) pthread_join(ctx->threads[j], NULL);
            return 2;
        }
    }

    // join
    for (int i = 0; i < ctx->num_threads; ++i) pthread_join(ctx->threads[i], NULL);
    ctx->stop = 1;  // signal workers to stop
    return 0;
}

#include "../../include/sapphire.h"
#include "../../include/tensor.h"  // For unified tensor_dtype_t

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
    
    // Generic GEMV kernel dispatch
    gemv_kernel_t kernel_fn;  // Function pointer to the selected kernel
    const void *W;            // Opaque weight data pointer
    
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
        
        // Compute row pointers and dispatch through function pointer
        for (int r = start; r < end; ++r) {
            // Calculate row pointer based on block size
            // For Q4_0/Q8_0: blocks are variable-sized, for BF16/F32: fixed size
            const char *W_base = (const char *)ctx->W;
            // Assume contiguous row storage: row_r starts at offset r * blocks_per_row * block_bytes
            // Block bytes depend on dtype (handled by dispatcher when setting W)
            const void *row_ptr = (const void *)(W_base + (size_t)r * ctx->blocks_per_row * 20); // 20 = max(sizeof Q4_0 block)
            
            // Call the kernel through function pointer
            float result = ctx->kernel_fn(row_ptr, ctx->x, ctx->blocks_per_row, 32);
            ctx->y[r] = result;
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

int sapphire_batched_gemv(sapphire_context *ctx, const tensor_t *A, const float *x, float *y) {
    if (!ctx || !A || !x || !y) return 1;
    
    tensor_dtype_t dtype = tensor_dtype(A);
    const void *W_data = tensor_data(A);
    int rows = tensor_shape(A)[0];
    int cols = tensor_shape(A)[tensor_ndim(A) - 1];
    int blocks_per_row = (cols + 31) / 32;
    
    if (!W_data) return 1;
    
    // Dispatcher: Select kernel based on dtype
    int x_aligned = (((uintptr_t)(const void*)x) & 31) == 0;
    
    switch (dtype) {
        case DTYPE_Q4_0: {
            ctx->kernel_fn = x_aligned ? quantized_gemv_q4_0_aligned : quantized_gemv_q4_0_unaligned;
            break;
        }
        case DTYPE_Q8_0: {
            ctx->kernel_fn = x_aligned ? quantized_gemv_q8_0_aligned : quantized_gemv_q8_0_unaligned;
            break;
        }
        case DTYPE_BF16: {
            ctx->kernel_fn = quantized_gemv_bf16_avx2;  // NEW: BF16 support for Gemma 3
            break;
        }
        case DTYPE_F32: {
            ctx->kernel_fn = quantized_gemv_f32_avx2;
            break;
        }
        case DTYPE_F16: {
            // F16 fallback: use F32 kernel (converts on-the-fly if needed)
            fprintf(stderr, "WARNING: F16 not yet optimized, falling back to F32 kernel\n");
            ctx->kernel_fn = quantized_gemv_f32_avx2;
            break;
        }
        default: {
            fprintf(stderr, "ERROR: Unsupported dtype %d\n", (int)dtype);
            return -1;
        }
    }
    
    // Setup context for worker threads
    ctx->W = W_data;
    ctx->rows = rows;
    ctx->blocks_per_row = blocks_per_row;
    ctx->x = x;
    ctx->y = y;
    ctx->x_aligned = x_aligned;
    atomic_store(&ctx->next_row, 0);
    ctx->stop = 0;

    // Create and join worker threads
    for (int i = 0; i < ctx->num_threads; ++i) {
        if (pthread_create(&ctx->threads[i], NULL, worker_fn, ctx) != 0) {
            ctx->stop = 1;
            for (int j = 0; j < i; ++j) pthread_join(ctx->threads[j], NULL);
            return 2;
        }
    }

    for (int i = 0; i < ctx->num_threads; ++i) pthread_join(ctx->threads[i], NULL);
    ctx->stop = 1;
    return 0;
}

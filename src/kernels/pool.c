#include "../../include/kernels.h"
#include "../../include/tensor.h"  // For unified tensor_dtype_t
#include "../../include/log.h"

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
    
    // Persistent thread state
    int initialized;  // 1 if threads have been launched, 0 otherwise
    atomic_int shutdown_flag;  // 1 to signal threads to exit
    pthread_mutex_t work_mutex;
    pthread_cond_t work_cond;
    pthread_mutex_t exec_mutex;  // Serializes kernel_gemv_backend_exec calls
    
    // Task-based synchronization
    atomic_int task_id;          // Incremented by main thread to start new work
    atomic_int threads_done;     // Incremented by workers when they finish a task
    
    // Work queue state
    atomic_int next_row;
    
    // Generic GEMV kernel dispatch
    gemv_kernel_t kernel_fn;  // Function pointer to the selected kernel
    const void *W;            // Opaque weight data pointer
    
    int rows;
    int cols;        // Actual columns per row (for bounds check)
    int blocks_per_row;
    size_t row_stride_bytes;  // Correct byte stride for row pointers
    const float *x;
    float *y;
    int x_aligned;
};

// Internal prototype for dispatch from dispatch.c
// Declared here to ensure implementation matches
int kernel_gemv_backend_exec(kernel_context_t *ctx, const tensor_t *A, const float *x, float *y);

/**
 * Persistent worker thread function.
 * Waits on condition variable for task_id change, processes rows, signals completion.
 * Exits when shutdown_flag is set.
 */
static void *worker_fn(void *arg) {
    kernel_context_t *ctx = (kernel_context_t*)arg;
    int my_last_task_id = 0;
    
    while (!atomic_load(&ctx->shutdown_flag)) {
        // 1. Wait for a new task_id
        pthread_mutex_lock(&ctx->work_mutex);
        while (atomic_load(&ctx->task_id) == my_last_task_id && !atomic_load(&ctx->shutdown_flag)) {
            pthread_cond_wait(&ctx->work_cond, &ctx->work_mutex);
        }
        
        if (atomic_load(&ctx->shutdown_flag)) {
            pthread_mutex_unlock(&ctx->work_mutex);
            break;
        }
        
        // 2. Snapshot task parameters
        my_last_task_id = atomic_load(&ctx->task_id);
        int rows = ctx->rows;
        int cols = ctx->cols;
        int blocks_per_row = ctx->blocks_per_row;
        size_t row_stride_bytes = ctx->row_stride_bytes;
        const void *W = ctx->W;
        const float *x = ctx->x;
        float *y = ctx->y;
        gemv_kernel_t kernel_fn = ctx->kernel_fn;
        int chunk_size = ctx->chunk_size;
        pthread_mutex_unlock(&ctx->work_mutex);
        
        // 3. Process assigned rows
        while (1) {
            int start = atomic_fetch_add(&ctx->next_row, chunk_size);
            if (start >= rows) break;
            int end = start + chunk_size;
            if (end > rows) end = rows;
            
            for (int r = start; r < end; ++r) {
                const char *W_base = (const char *)W;
                const void *row_ptr = (const void *)(W_base + (size_t)r * row_stride_bytes);
                
                int count = blocks_per_row;
                int b_size = 32;
                if (row_stride_bytes == (size_t)cols * 2 || 
                    row_stride_bytes == (size_t)cols * 4) {
                    count = cols;
                    b_size = 1;
                }
                float result = kernel_fn(row_ptr, x, count, b_size);
                y[r] = result;
            }
        }
        
        // 4. Signal completion of this thread
        pthread_mutex_lock(&ctx->work_mutex);
        atomic_fetch_add(&ctx->threads_done, 1);
        pthread_cond_broadcast(&ctx->work_cond); // Notify main thread
        pthread_mutex_unlock(&ctx->work_mutex);
    }
    
    return NULL;
}

kernel_context_t *kernel_ctx_create(int num_threads, int chunk_size) {
    if (num_threads <= 0) {
        num_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (num_threads <= 0) num_threads = 1;
    }
    if (chunk_size <= 0) chunk_size = 16;

    kernel_context_t *ctx = (kernel_context_t*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    
    ctx->num_threads = num_threads;
    ctx->chunk_size = chunk_size;
    ctx->threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    if (!ctx->threads) {
        free(ctx);
        return NULL;
    }
    
    // Initialize synchronization primitives
    if (pthread_mutex_init(&ctx->work_mutex, NULL) != 0) {
        LOG_ERROR("Failed to initialize work mutex");
        free(ctx->threads);
        free(ctx);
        return NULL;
    }
    if (pthread_cond_init(&ctx->work_cond, NULL) != 0) {
        LOG_ERROR("Failed to initialize work condition variable");
        pthread_mutex_destroy(&ctx->work_mutex);
        free(ctx->threads);
        free(ctx);
        return NULL;
    }
    if (pthread_mutex_init(&ctx->exec_mutex, NULL) != 0) {
        LOG_ERROR("Failed to initialize exec mutex");
        pthread_cond_destroy(&ctx->work_cond);
        pthread_mutex_destroy(&ctx->work_mutex);
        free(ctx->threads);
        free(ctx);
        return NULL;
    }
    
    ctx->initialized = 0;
    atomic_store(&ctx->shutdown_flag, 0);
    atomic_store(&ctx->task_id, 0);
    atomic_store(&ctx->threads_done, 0);
    
    LOG_DEBUG("Created kernel context with %d threads, chunk_size=%d", num_threads, chunk_size);
    return ctx;
}

void kernel_ctx_destroy(kernel_context_t *ctx) {
    if (!ctx) return;
    
    // Signal threads to shut down
    if (ctx->initialized) {
        atomic_store(&ctx->shutdown_flag, 1);
        
        pthread_mutex_lock(&ctx->work_mutex);
        pthread_cond_broadcast(&ctx->work_cond);
        pthread_mutex_unlock(&ctx->work_mutex);
        
        for (int i = 0; i < ctx->num_threads; ++i) {
            pthread_join(ctx->threads[i], NULL);
        }
        LOG_DEBUG("All worker threads joined");
    }
    
    // Clean up synchronization primitives
    pthread_cond_destroy(&ctx->work_cond);
    pthread_mutex_destroy(&ctx->work_mutex);
    pthread_mutex_destroy(&ctx->exec_mutex);
    
    free(ctx->threads);
    free(ctx);
}

/**
 * Initialize persistent worker threads.
 * Call this once per inference session to launch threads.
 * Returns 0 on success, -1 on failure (errors logged internally).
 */
int kernel_ctx_init(kernel_context_t *ctx) {
    if (!ctx) {
        LOG_ERROR("kernel_ctx_init: context is NULL");
        return -1;
    }
    
    if (ctx->initialized) {
        return 0;  // Already initialized
    }
    
    // Launch persistent worker threads
    for (int i = 0; i < ctx->num_threads; ++i) {
        if (pthread_create(&ctx->threads[i], NULL, worker_fn, ctx) != 0) {
            LOG_ERROR("Failed to create worker thread %d", i);
            
            atomic_store(&ctx->shutdown_flag, 1);
            pthread_mutex_lock(&ctx->work_mutex);
            pthread_cond_broadcast(&ctx->work_cond);
            pthread_mutex_unlock(&ctx->work_mutex);

            for (int j = 0; j < i; ++j) {
                pthread_join(ctx->threads[j], NULL);
            }
            return -1;
        }
    }
    
    ctx->initialized = 1;
    LOG_INFO("Initialized %d persistent worker threads", ctx->num_threads);
    return 0;
}

int kernel_gemv_backend_exec(kernel_context_t *ctx, const tensor_t *A, const float *x, float *y) {
    if (!ctx || !A || !x || !y) {
        LOG_ERROR("kernel_gemv_backend_exec: null pointer");
        return 1;
    }
    
    if (!ctx->initialized) {
        LOG_ERROR("kernel_gemv_backend_exec: context not initialized");
        return 1;
    }
    
    tensor_dtype_t dtype = tensor_dtype(A);
    const void *W_data = tensor_data(A);
    int rows = tensor_shape(A)[0];
    int cols = tensor_shape(A)[tensor_ndim(A) - 1];
    int blocks_per_row = (cols + 31) / 32;
    size_t row_stride_bytes = 0;
    
    if (!W_data) return 1;
    
    int x_aligned = (((uintptr_t)(const void*)x) & 31) == 0;
    gemv_kernel_t kernel_fn = NULL;
    
    switch (dtype) {
        case DTYPE_Q4_0:
            kernel_fn = x_aligned ? quantized_gemv_q4_0_aligned : quantized_gemv_q4_0_unaligned;
            row_stride_bytes = (size_t)blocks_per_row * sizeof(ggml_block_q4_0);
            break;
        case DTYPE_Q8_0:
            kernel_fn = x_aligned ? quantized_gemv_q8_0_aligned : quantized_gemv_q8_0_unaligned;
            row_stride_bytes = (size_t)blocks_per_row * sizeof(ggml_block_q8_0);
            break;
        case DTYPE_BF16:
            kernel_fn = quantized_gemv_bf16_avx2;
            row_stride_bytes = (size_t)cols * 2;
            break;
        case DTYPE_F32:
            kernel_fn = quantized_gemv_f32_avx2;
            row_stride_bytes = (size_t)cols * 4;
            break;
        default:
            LOG_ERROR("kernel_gemv_backend_exec: unsupported dtype %d", (int)dtype);
            return -1;
    }
    
    // Serialize execution to prevent re-entry corruption
    pthread_mutex_lock(&ctx->exec_mutex);

    // Set up work parameters
    pthread_mutex_lock(&ctx->work_mutex);
    ctx->kernel_fn = kernel_fn;
    ctx->W = W_data;
    ctx->rows = rows;
    ctx->cols = cols;
    ctx->blocks_per_row = blocks_per_row;
    ctx->row_stride_bytes = row_stride_bytes;
    ctx->x = x;
    ctx->y = y;
    ctx->x_aligned = x_aligned;
    atomic_store(&ctx->next_row, 0);
    atomic_store(&ctx->threads_done, 0);
    
    // Trigger task by incrementing task_id
    atomic_fetch_add(&ctx->task_id, 1);
    pthread_cond_broadcast(&ctx->work_cond);
    
    // Wait for all threads to complete
    while (atomic_load(&ctx->threads_done) < ctx->num_threads) {
        pthread_cond_wait(&ctx->work_cond, &ctx->work_mutex);
    }
    pthread_mutex_unlock(&ctx->work_mutex);
    pthread_mutex_unlock(&ctx->exec_mutex);
    
    return 0;
}


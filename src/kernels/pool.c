#include "../../include/kernels.h"
#include "../../include/tensor.h"  // For unified tensor_dtype_t
#include "../../include/ternary_anchor.h"
#include "../../include/log.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include "../../include/tracy_profile.h"

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
    atomic_int worker_name_index;
    
    // Work queue state
    atomic_int next_row;
    
    // Generic GEMV kernel dispatch
    gemv_kernel_t kernel_fn;  // Function pointer to the selected kernel
    gemm_kernel_t gemm_kernel_fn; // Function pointer for batched GEMM
    const void *W;            // Opaque weight data pointer
    
    int rows;
    int cols;        // Actual columns per row (for bounds check)
    int blocks_per_row;
    size_t row_stride_bytes;  // Correct byte stride for row pointers
    const float *x;
    float *y;
    int x_aligned;

    // GEMM (Batched prefill) extension
    int is_gemm;
    int batch_size;
    int d_model;    // input dimension for batching stride
    size_t ternary_packed_cols;
    size_t ternary_row_stride_bytes;
    const float *ternary_scales;
    uint32_t ternary_scale_group_size;
    uint32_t ternary_groups_per_row;
    const ternary_anchor_entry_t *hybrid_anchor_entries;
    const uint32_t *hybrid_anchor_row_offsets;

    // Parallel For Extension
    parallel_for_fn_t parallel_fn;
    void *parallel_arg;
};

// Internal prototype for dispatch from dispatch.c
// Declared here to ensure implementation matches
int kernel_backend_exec(kernel_context_t *ctx, const tensor_t *A, const float *X, float *Y, int is_gemm, int batch_size);

typedef struct {
    const void *row_ptr;
    tensor_ternary_view_t ternary_row;
    tensor_hybrid_view_t hybrid_row;
    uint32_t hybrid_row_offsets[2];
    int block_count;
    int block_size;
} worker_row_state_t;

typedef struct {
    gemv_kernel_t kernel_fn;
    gemm_kernel_t gemm_kernel_fn;
    const void *weight_data;
    size_t row_stride_bytes;
    size_t ternary_packed_cols;
    size_t ternary_row_stride_bytes;
    const float *ternary_scales;
    uint32_t ternary_scale_group_size;
    uint32_t ternary_groups_per_row;
    const ternary_anchor_entry_t *hybrid_anchor_entries;
    const uint32_t *hybrid_anchor_row_offsets;
} backend_exec_plan_t;

typedef struct {
    int cols;
    int blocks_per_row;
    size_t row_stride_bytes;
    size_t ternary_packed_cols;
    size_t ternary_row_stride_bytes;
    const float *ternary_scales;
    uint32_t ternary_scale_group_size;
    uint32_t ternary_groups_per_row;
    const ternary_anchor_entry_t *hybrid_anchor_entries;
    const uint32_t *hybrid_anchor_row_offsets;
} row_prepare_config_t;

static void init_ternary_row_view(tensor_ternary_view_t *row,
                                  const row_prepare_config_t *config,
                                  const uint8_t *packed_weights,
                                  const float *scales)
{
    memset(row, 0, sizeof(*row));
    row->packed_weights = packed_weights;
    row->scales = scales;
    row->rows = 1u;
    row->cols = (uint32_t)config->cols;
    row->packed_cols = (uint32_t)config->ternary_packed_cols;
    row->scale_group_size = config->ternary_scale_group_size;
    row->groups_per_row = config->ternary_groups_per_row;
    row->packed_weight_bytes = config->ternary_row_stride_bytes;
    row->scale_count = config->ternary_groups_per_row;
    row->scale_bytes = (size_t)config->ternary_groups_per_row * sizeof(float);
}

static void init_hybrid_row_view(tensor_hybrid_view_t *row,
                                 uint32_t local_row_offsets[2],
                                 const row_prepare_config_t *config,
                                 const uint8_t *packed_weights,
                                 const float *scales,
                                 int row_index)
{
    uint32_t anchor_start = config->hybrid_anchor_row_offsets[row_index];
    uint32_t anchor_end = config->hybrid_anchor_row_offsets[row_index + 1];

    memset(row, 0, sizeof(*row));
    row->packed_weights = packed_weights;
    row->scales = scales;
    row->rows = 1u;
    row->cols = (uint32_t)config->cols;
    row->packed_cols = (uint32_t)config->ternary_packed_cols;
    row->scale_group_size = config->ternary_scale_group_size;
    row->groups_per_row = config->ternary_groups_per_row;
    row->packed_weight_bytes = config->ternary_row_stride_bytes;
    row->scale_count = config->ternary_groups_per_row;
    row->scale_bytes = (size_t)config->ternary_groups_per_row * sizeof(float);
    row->anchor_entries = config->hybrid_anchor_entries ? (config->hybrid_anchor_entries + anchor_start) : NULL;
    local_row_offsets[0] = 0u;
    local_row_offsets[1] = anchor_end - anchor_start;
    row->anchor_row_offsets = local_row_offsets;
    row->anchor_count = local_row_offsets[1];
    row->owns_anchor_memory = 0;
}

static void prepare_worker_row_state(worker_row_state_t *state,
                                     const row_prepare_config_t *config,
                                     const void *weights,
                                     int row_index)
{
    const uint8_t *row_weights = (const uint8_t *)weights + (size_t)row_index * config->row_stride_bytes;

    state->row_ptr = row_weights;
    state->block_count = config->blocks_per_row;
    state->block_size = 32;

    if (config->row_stride_bytes == (size_t)config->cols * 2u ||
        config->row_stride_bytes == (size_t)config->cols * 4u) {
        state->block_count = config->cols;
        state->block_size = 1;
    }

    if (config->ternary_scales) {
        init_ternary_row_view(&state->ternary_row,
                              config,
                              row_weights,
                              config->ternary_scales + (size_t)row_index * config->ternary_groups_per_row);
        state->row_ptr = &state->ternary_row;
        state->block_count = config->cols;
        state->block_size = 1;
    }

    if (config->hybrid_anchor_row_offsets) {
        init_hybrid_row_view(&state->hybrid_row,
                             state->hybrid_row_offsets,
                             config,
                             row_weights,
                             config->ternary_scales + (size_t)row_index * config->ternary_groups_per_row,
                             row_index);
        state->row_ptr = &state->hybrid_row;
        state->block_count = config->cols;
        state->block_size = 1;
    }
}

static int prepare_ternary_exec_plan(backend_exec_plan_t *plan,
                                     const tensor_t *A,
                                     int rows,
                                     int cols)
{
    const tensor_ternary_view_t *ternary_view = tensor_data_ternary(A);

    if (!ternary_view || !ternary_view->packed_weights || !ternary_view->scales ||
        ternary_view->rows != (uint32_t)rows || ternary_view->cols != (uint32_t)cols ||
        ternary_view->groups_per_row == 0u ||
        ternary_view->scale_count != (size_t)rows * ternary_view->groups_per_row) {
        LOG_ERROR("kernel_backend_exec: invalid ternary tensor payload");
        return -1;
    }

    plan->kernel_fn = quantized_gemv_ternary_scalar;
    plan->gemm_kernel_fn = kernel_gemm_ternary_scalar;
    plan->weight_data = ternary_view->packed_weights;
    plan->row_stride_bytes = ternary_view->packed_cols;
    plan->ternary_packed_cols = ternary_view->packed_cols;
    plan->ternary_row_stride_bytes = ternary_view->packed_cols;
    plan->ternary_scales = ternary_view->scales;
    plan->ternary_scale_group_size = ternary_view->scale_group_size;
    plan->ternary_groups_per_row = ternary_view->groups_per_row;
    return 0;
}

static int prepare_hybrid_exec_plan(backend_exec_plan_t *plan,
                                    const tensor_t *A,
                                    int rows,
                                    int cols)
{
    const tensor_hybrid_view_t *hybrid_view = tensor_data_hybrid(A);

    if (!hybrid_view || !hybrid_view->packed_weights || !hybrid_view->scales ||
        hybrid_view->rows != (uint32_t)rows || hybrid_view->cols != (uint32_t)cols ||
        hybrid_view->groups_per_row == 0u ||
        hybrid_view->scale_count != (size_t)rows * hybrid_view->groups_per_row) {
        LOG_ERROR("kernel_backend_exec: invalid hybrid tensor payload");
        return -1;
    }
    if (hybrid_view->anchor_count > 0u &&
        (!hybrid_view->anchor_entries || !hybrid_view->anchor_row_offsets)) {
        LOG_ERROR("kernel_backend_exec: hybrid tensor is missing anchor metadata");
        return -1;
    }

    plan->kernel_fn = quantized_gemv_ternary_hybrid_avx2;
    plan->gemm_kernel_fn = kernel_gemm_ternary_hybrid_avx2;
    plan->weight_data = hybrid_view->packed_weights;
    plan->row_stride_bytes = hybrid_view->packed_cols;
    plan->ternary_packed_cols = hybrid_view->packed_cols;
    plan->ternary_row_stride_bytes = hybrid_view->packed_cols;
    plan->ternary_scales = hybrid_view->scales;
    plan->ternary_scale_group_size = hybrid_view->scale_group_size;
    plan->ternary_groups_per_row = hybrid_view->groups_per_row;
    plan->hybrid_anchor_entries = (const ternary_anchor_entry_t *)hybrid_view->anchor_entries;
    plan->hybrid_anchor_row_offsets = hybrid_view->anchor_row_offsets;
    return 0;
}

static int build_backend_exec_plan(backend_exec_plan_t *plan,
                                   const tensor_t *A,
                                   int rows,
                                   int cols,
                                   int blocks_per_row,
                                   int x_aligned)
{
    memset(plan, 0, sizeof(*plan));

    switch (tensor_dtype(A)) {
        case DTYPE_Q4_0:
            plan->kernel_fn = x_aligned ? quantized_gemv_q4_0_aligned : quantized_gemv_q4_0_unaligned;
            plan->row_stride_bytes = (size_t)blocks_per_row * sizeof(ggml_block_q4_0);
            return 0;
        case DTYPE_Q8_0:
            plan->kernel_fn = x_aligned ? quantized_gemv_q8_0_aligned : quantized_gemv_q8_0_unaligned;
            plan->row_stride_bytes = (size_t)blocks_per_row * sizeof(ggml_block_q8_0);
            return 0;
        case DTYPE_BF16:
            plan->kernel_fn = quantized_gemv_bf16_avx2;
            plan->gemm_kernel_fn = kernel_gemm_bf16_avx2;
            plan->weight_data = tensor_data(A);
            plan->row_stride_bytes = (size_t)cols * 2u;
            return 0;
        case DTYPE_F32:
            plan->kernel_fn = quantized_gemv_f32_avx2;
            plan->gemm_kernel_fn = kernel_gemm_f32_avx2;
            plan->weight_data = tensor_data(A);
            plan->row_stride_bytes = (size_t)cols * 4u;
            return 0;
        case DTYPE_TERNARY_2BIT:
            return prepare_ternary_exec_plan(plan, A, rows, cols);
        case DTYPE_TERNARY_HYBRID:
            return prepare_hybrid_exec_plan(plan, A, rows, cols);
        default:
            LOG_ERROR("kernel_backend_exec: unsupported dtype %d", (int)tensor_dtype(A));
            return -1;
    }
}

/**
 * Persistent worker thread function.
 * Waits on condition variable for task_id change, processes rows, signals completion.
 * Exits when shutdown_flag is set.
 */
static void *worker_fn(void *arg) {
    kernel_context_t *ctx = (kernel_context_t*)arg;
    int my_last_task_id = 0;
    int worker_index = atomic_fetch_add(&ctx->worker_name_index, 1);
    char worker_name[32];

    snprintf(worker_name, sizeof(worker_name), "cpu-worker-%d", worker_index);
    sapphire_tracy_name_thread(worker_name);
    
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
        gemm_kernel_t gemm_kernel_fn = ctx->gemm_kernel_fn;
        int is_gemm = ctx->is_gemm;
        int batch_size = ctx->batch_size;
        int d_model = ctx->d_model;
        row_prepare_config_t row_config = {
            .cols = cols,
            .blocks_per_row = blocks_per_row,
            .row_stride_bytes = row_stride_bytes,
            .ternary_packed_cols = ctx->ternary_packed_cols,
            .ternary_row_stride_bytes = ctx->ternary_row_stride_bytes,
            .ternary_scales = ctx->ternary_scales,
            .ternary_scale_group_size = ctx->ternary_scale_group_size,
            .ternary_groups_per_row = ctx->ternary_groups_per_row,
            .hybrid_anchor_entries = ctx->hybrid_anchor_entries,
            .hybrid_anchor_row_offsets = ctx->hybrid_anchor_row_offsets,
        };
        int chunk_size = ctx->chunk_size;
        parallel_for_fn_t parallel_fn = ctx->parallel_fn;
        void *parallel_arg = ctx->parallel_arg;
        pthread_mutex_unlock(&ctx->work_mutex);
        
        // 3. Process assigned rows / iterations
        if (parallel_fn) {
            while (1) {
                int idx = atomic_fetch_add(&ctx->next_row, 1);
                if (idx >= rows) break;
                parallel_fn(parallel_arg, idx);
            }
        } else {
            while (1) {
            int start = atomic_fetch_add(&ctx->next_row, chunk_size);
            if (start >= rows) break;
            int end = start + chunk_size;
            if (end > rows) end = rows;
            
            for (int r = start; r < end; ++r) {
                worker_row_state_t row_state;

                prepare_worker_row_state(&row_state,
                                         &row_config,
                                         W,
                                         r);

                if (is_gemm) {
                    // Batched path: Y[token][row]
                    // pass y + r as the start address for this row's column in Y
                    gemm_args_t g_args = {
                        .w_row = row_state.row_ptr,
                        .X = x,
                        .Y = y + r,
                        .batch_size = batch_size,
                        .d_model = d_model,
                        .out_stride = rows,
                        .blocks = row_state.block_count,
                        .block_size = row_state.block_size
                    };
                    gemm_kernel_fn(&g_args);
                } else {
                    float result = kernel_fn(row_state.row_ptr, x, row_state.block_count, row_state.block_size);
                    y[r] = result;
                }
            }
        }
    }
        
    // 4. Signal completion of this thread
        if (atomic_fetch_add(&ctx->threads_done, 1) + 1 == ctx->num_threads) {
            pthread_mutex_lock(&ctx->work_mutex);
            pthread_cond_broadcast(&ctx->work_cond); // Notify main thread
            pthread_mutex_unlock(&ctx->work_mutex);
        }
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
    atomic_store(&ctx->worker_name_index, 0);
    
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

int kernel_backend_exec(kernel_context_t *ctx, const tensor_t *A, const float *X, float *Y, int is_gemm, int batch_size) {
    if (!ctx || !A || !X || !Y) {
        LOG_ERROR("kernel_backend_exec: null pointer");
        return 1;
    }
    
    if (!ctx->initialized) {
        LOG_ERROR("kernel_backend_exec: context not initialized");
        return 1;
    }
    
    tensor_dtype_t dtype = tensor_dtype(A);
    int rows = tensor_shape(A)[0];
    int cols = tensor_shape(A)[tensor_ndim(A) - 1];
    int blocks_per_row = (cols + 31) / 32;
    backend_exec_plan_t plan;
    
    if (!tensor_data(A)) return 1;
    
    int x_aligned = (((uintptr_t)(const void*)X) & 31) == 0;
    if (build_backend_exec_plan(&plan, A, rows, cols, blocks_per_row, x_aligned) != 0) {
        return -1;
    }

    if (is_gemm && plan.gemm_kernel_fn == NULL) {
        LOG_ERROR("kernel_backend_exec: dtype %d does not support batching yet", (int)dtype);
        return -1;
    }
    
    // Serialize execution to prevent re-entry corruption
    pthread_mutex_lock(&ctx->exec_mutex);

    // Dynamic chunk size to ensure all threads get work
    int num_threads = ctx->num_threads;
    int chunk_size = (rows + num_threads - 1) / num_threads;
    if (chunk_size < 1) chunk_size = 1;

    // Set up work parameters
    pthread_mutex_lock(&ctx->work_mutex);
    ctx->is_gemm = is_gemm;
    ctx->batch_size = batch_size;
    ctx->d_model = cols;
    ctx->kernel_fn = plan.kernel_fn;
    ctx->gemm_kernel_fn = plan.gemm_kernel_fn;
    ctx->W = plan.weight_data ? plan.weight_data : tensor_data(A);
    ctx->rows = rows;
    ctx->cols = cols;
    ctx->blocks_per_row = blocks_per_row;
    ctx->row_stride_bytes = plan.row_stride_bytes;
    ctx->ternary_packed_cols = plan.ternary_packed_cols;
    ctx->ternary_row_stride_bytes = plan.ternary_row_stride_bytes;
    ctx->ternary_scales = plan.ternary_scales;
    ctx->ternary_scale_group_size = plan.ternary_scale_group_size;
    ctx->ternary_groups_per_row = plan.ternary_groups_per_row;
    ctx->hybrid_anchor_entries = plan.hybrid_anchor_entries;
    ctx->hybrid_anchor_row_offsets = plan.hybrid_anchor_row_offsets;
    ctx->x = X;
    ctx->y = Y;
    ctx->x_aligned = x_aligned;
    ctx->chunk_size = chunk_size; // Override with optimal chunk size
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

int kernel_parallel_for(kernel_context_t *ctx, parallel_for_fn_t fn, void* arg, int n) {
    if (!ctx || !fn || n <= 0) return -1;
    
    pthread_mutex_lock(&ctx->exec_mutex);
    pthread_mutex_lock(&ctx->work_mutex);
    
    // Clear GEMV state to indicate parallel_for
    ctx->kernel_fn = NULL;
    ctx->gemm_kernel_fn = NULL;
    ctx->W = NULL;
    ctx->parallel_fn = fn;
    ctx->parallel_arg = arg;
    ctx->rows = n;
    
    atomic_store(&ctx->next_row, 0);
    atomic_store(&ctx->threads_done, 0);
    
    // Trigger task
    atomic_fetch_add(&ctx->task_id, 1);
    pthread_cond_broadcast(&ctx->work_cond);
    
    while (atomic_load(&ctx->threads_done) < ctx->num_threads) {
        pthread_cond_wait(&ctx->work_cond, &ctx->work_mutex);
    }
    
    // Reset parallel_fn for next task
    ctx->parallel_fn = NULL;
    
    pthread_mutex_unlock(&ctx->work_mutex);
    pthread_mutex_unlock(&ctx->exec_mutex);
    
    return 0;
}


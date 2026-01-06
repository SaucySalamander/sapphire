#ifndef TENSOR_GEMV_H
#define TENSOR_GEMV_H

/* Forward declaration to avoid exposing tensor internals in this public header */
typedef struct tensor_t tensor_t;

/* Forward declaration for the Sapphire thread-pool context type */
typedef struct sapphire_context sapphire_context;

/**
 * @file tensor_gemv.h
 * @brief Tensor-based GEMV (matrix-vector multiplication) dispatcher.
 *
 * This module bridges Phase 3 tensor abstraction with Phase 1's quantized kernels.
 * It provides a unified interface for matrix-vector multiplication that:
 * - Automatically detects weight tensor data type (F32, Q4_0, Q8_0)
 * - Dispatches to appropriate kernel from Phase 1 (sapphire)
 * - Handles dequantization internally for quantized weights
 * - Uses an explicit Sapphire context passed by the caller (no hidden globals)
 *
 * Breaking change: The previous singleton/global API (e.g. tensor_gemv_init(),
 * tensor_gemv_cleanup(), and convenience wrappers like tensor_gemv()) has been
 * removed. Callers must create and manage an explicit `sapphire_context*` and
 * use the `_with_ctx` variants documented below.
 */

/* Context-managed GEMV API (explicit context, no hidden globals) */

/* Create/destroy explicit context for GEMV operations */
sapphire_context* tensor_gemv_ctx_create(int num_threads, int chunk_size);
void tensor_gemv_ctx_destroy(sapphire_context *ctx);

/*
 * Core context-aware GEMV operations
 *
 * Note: For quantized weight dtypes (DTYPE_Q4_0 / DTYPE_Q8_0) a non-NULL
 * sapphire_context is required because the implementation dispatches to the
 * Sapphire batched kernels which use the provided thread-pool context.
 */
int tensor_gemv_with_ctx(sapphire_context *ctx, float *y, const tensor_t *A, const float *x);
int tensor_gemv_tensor_with_ctx(sapphire_context *ctx, tensor_t *y, const tensor_t *A, const tensor_t *x);
int tensor_gemv_add_with_ctx(sapphire_context *ctx, float *y, const tensor_t *A, const float *x, float alpha);
int tensor_gemv_batch_with_ctx(sapphire_context *ctx, float *Y, const tensor_t *A, const float *X, int batch_size);

/**
 * @example
 *   // Create a context with auto-detected thread count (0) and chunk size of 1024
 *   sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
 *
 *   // Prepare tensors / buffers
 *   float output[m];
 *   const tensor_t *weights = ...; // DTYPE_F32/Q4_0/Q8_0
 *   float input[n];
 *
 *   // Compute: use the context-aware call
 *   int rc = tensor_gemv_with_ctx(ctx, output, weights, input);
 *
 *   // Destroy context when done
 *   tensor_gemv_ctx_destroy(ctx);
 */

#endif // TENSOR_GEMV_H

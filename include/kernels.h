/**
 * @file kernels.h
 * @brief Unified kernel interface for all math and GEMV operations.
 *
 * This header serves as the single public API for all kernel implementations
 * under src/kernels/. It provides:
 *   1. Low-level SIMD kernels for GEMV (F32, BF16, Q4_0, Q8_0).
 *   2. Support for activations, normalization, and vector mathematics.
 *   3. High-level tensor-based GEMV operations with multi-threading support.
 *   4. Modern, domain-consistent naming conventions.
 */

#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>
#include <stddef.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// BLOCK STRUCTURE DEFINITIONS (Memory layout for weights)
// ============================================================================

/** Q4_0: 4-bit quantized weights */
typedef struct {
    float scale;         /**< Scaling factor (f32) */
    uint8_t q_data[16];  /**< 16 bytes * 2 nibbles/byte = 32 packed 4-bit weights */
} ggml_block_q4_0;

/** Q8_0: 8-bit quantized weights */
typedef struct {
    float scale;
    uint8_t q_data[32];  /**< 32 bytes = 32 int8 weights */
} ggml_block_q8_0;

// ============================================================================
// CONTEXT MANAGEMENT (Multi-threaded execution)
// ============================================================================

/** Opaque handle for the kernel execution context/thread pool. */
typedef struct sapphire_context kernel_context_t;

/**
 * Create a kernel execution context.
 * Allocates and initializes the context structure and synchronization primitives.
 * Threads are NOT launched until kernel_ctx_init() is called.
 * 
 * @param num_threads  Number of worker threads (0 = autodetect CPU count)
 * @param chunk_size   Rows per task per worker (default 16)
 * @return Allocated context, or NULL on error
 */
kernel_context_t* kernel_ctx_create(int num_threads, int chunk_size);

/**
 * Initialize persistent worker threads.
 * Must be called after kernel_ctx_create() and before kernel_gemv calls.
 * Threads are created once and reused across all subsequent GEMV operations.
 * 
 * @param ctx  Context created by kernel_ctx_create()
 * @return 0 on success, -1 on failure (errors logged internally)
 */
int kernel_ctx_init(kernel_context_t *ctx);

/**
 * Destroy a kernel execution context.
 * Signals worker threads to exit, joins them, and frees all resources.
 */
void kernel_ctx_destroy(kernel_context_t *ctx);

// ============================================================================
// LOW-LEVEL GEMV KERNELS
// ============================================================================

/** Generic GEMV kernel function signature. */
typedef float (*kernel_fn_t)(const void* w_row, const float* x, int blocks, int block_size);

// Forward compatibility alias
typedef float (*gemv_kernel_t)(const void* w_row, const float* x, int blocks, int block_size);

typedef struct {
    const void* w_row;
    const float* X;
    float* Y;
    int batch_size;
    int d_model;
    int out_stride;
    int blocks;
    int block_size;
} gemm_args_t;

typedef void (*gemm_kernel_t)(const gemm_args_t* args);

// Q4_0 kernels
float quantized_gemv_q4_0_aligned(const void *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_q4_0_unaligned(const void *W_row, const float *x, int block_count, int block_size);

// Q8_0 kernels
float quantized_gemv_q8_0_aligned(const void *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_q8_0_unaligned(const void *W_row, const float *x, int block_count, int block_size);

// BF16 kernels
float quantized_gemv_bf16_avx2(const void *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_bf16_scalar(const void *W_row, const float *x, int block_count, int block_size);

// F32 kernels
float quantized_gemv_f32_avx2(const void *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_f32_scalar(const void *W_row, const float *x, int block_count, int block_size);

// Batched kernels (GEMM)
void kernel_gemm_f32_avx2(const gemm_args_t* args);
void kernel_gemm_bf16_avx2(const gemm_args_t* args);

// ============================================================================
// HIGH-LEVEL TENSOR OPERATIONS (Thread-safe, dtype-aware)
// ============================================================================

/**
 * Core GEMV operation: y = A @ x.
 * Dispatches to best kernel based on A's dtype.
 */
int kernel_gemv(kernel_context_t *ctx, float *y, const tensor_t *A, const float *x);

/**
 * Batched prefill operation (GEMM): Y = A @ X.
 * X is [batch_size, context_len], Y is output buffer [batch_size, hidden_size].
 */
int kernel_gemm(kernel_context_t *ctx, float *Y, const tensor_t *A, const float *X, int batch_size, int out_stride);

/**
 * Tensor-to-tensor GEMV: y = A @ x (where y and x are F32 tensors).
 */
int kernel_gemv_tensor(kernel_context_t *ctx, const tensor_t *y, const tensor_t *A, const tensor_t *x);

/**
 * GEMV with accumulation: y += alpha * (A @ x).
 */
int kernel_gemv_add(kernel_context_t *ctx, float *y, const tensor_t *A, const float *x, float alpha);

/**
 * Batched GEMV for sequence processing.
 */
int kernel_gemv_batch(kernel_context_t *ctx, float *Y, const tensor_t *A, const float *X, int batch_size);

/**
 * Return the preferred SIMD float lane count for a given tensor dtype.
 */
int kernel_get_simd_lane_count(tensor_dtype_t dtype);

// ============================================================================
// ACTIVATIONS & NORMALIZATION
// ============================================================================

/** SiLU (Swish): x / (1 + exp(-x)) */
float silu(float x);
void silu_inplace(float *x, int n);

/** ReLU: max(0, x) */
float relu(float x);

/** GELU (approximate) */
float gelu(float x);
void gelu_inplace(float *x, int n);

/** RMSNorm: out[i] = (in[i] / RMS) * weight[i] */
int rmsnorm(float *out, const float *in, const float *weight, float epsilon, int dim);

/** RMSNorm (Gemma 3 style): out[i] = (in[i] / RMS) * (1.0 + weight[i]) */
int rmsnorm_delta(float *out, const float *in, const float *weight, float epsilon, int dim);

/** Batch RMSNorm */
int rmsnorm_batch(float *out, const float *in, const float *weight, float epsilon, int batch_size, int dim);

/** Softmax with numerical stability */
void softmax(float *scores, int n);

/** Numerical softcapping (tanh clamping) */
void vec_softcap(float *x, int n, float cap);

// ============================================================================
// VECTOR MATHEMATICS
// ============================================================================

void vec_add(float *dst, const float *src, int n);
void vec_scale(float *x, float s, int n);
void vec_copy(float *dst, const float *src, int n);
float vec_dot(const float *a, const float *b, int n);

/** BF16 to F32 conversion for a vector */
void bf16_to_f32_vec(float *dst, const uint16_t *src, int n);

/** Utility to retrieve normalized weights (handles BF16 conversion if needed) */
const float* get_norm_weights(const tensor_t *weight, float *scratch, int n);

// ============================================================================
// LEGACY COMPATIBILITY LAYER (To be removed after full refactor)
// ============================================================================

#define sapphire_context_create kernel_ctx_create
#define sapphire_context_destroy kernel_ctx_destroy
#define tensor_gemv_ctx_create kernel_ctx_create
#define tensor_gemv_ctx_destroy kernel_ctx_destroy
#define tensor_gemv_with_ctx kernel_gemv
#define tensor_gemm_with_ctx kernel_gemm
#define tensor_gemv_tensor_with_ctx kernel_gemv_tensor
#define tensor_gemv_add_with_ctx kernel_gemv_add
#define tensor_gemv_batch_with_ctx kernel_gemv_batch
#define tensor_gemv_simd_lane_count_for_dtype kernel_get_simd_lane_count
#define sapphire_batched_gemv kernel_gemv

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H

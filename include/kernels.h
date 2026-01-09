// kernels.h - Unified kernel interface header
// 
// This is the public API for all kernel implementations under src/kernels/
// It re-exports the essential types and functions from sapphire.h and tensor_gemv.h
// with modern naming conventions.

#ifndef KERNELS_H
#define KERNELS_H

#include "sapphire.h"      // Core kernel infrastructure
#include "tensor_gemv.h"   // High-level tensor operations

// ============================================================================
// UNIFIED KERNEL NAMESPACE ALIASES (Modern naming)
// ============================================================================

// Alias the core kernel function pointer type
typedef gemv_kernel_t kernel_fn_t;

// Alias the thread pool context type
typedef sapphire_context kernel_context_t;

// ============================================================================
// CONVENIENT CONTEXT MANAGEMENT
// ============================================================================

/**
 * Create a kernel execution context.
 * 
 * @param num_threads  Number of worker threads (0 = autodetect)
 * @param chunk_size   Rows per task per worker (0 = default 16)
 * @return Allocated context, or NULL on error
 */
static inline kernel_context_t* kernel_ctx_create(int num_threads, int chunk_size) {
    return sapphire_context_create(num_threads, chunk_size);
}

/**
 * Destroy a kernel execution context.
 */
static inline void kernel_ctx_destroy(kernel_context_t *ctx) {
    sapphire_context_destroy(ctx);
}

// ============================================================================
// KERNEL IMPLEMENTATIONS (Re-exported from sapphire.h)
// ============================================================================

// Q4_0: 4-bit quantized kernels
// These are the core dispatch entry points for 4-bit weights
extern float quantized_gemv_q4_0_aligned(const void *W_row, const float *x, int block_count, int block_size);
extern float quantized_gemv_q4_0_unaligned(const void *W_row, const float *x, int block_count, int block_size);

// Q8_0: 8-bit quantized kernels
extern float quantized_gemv_q8_0_aligned(const void *W_row, const float *x, int block_count, int block_size);
extern float quantized_gemv_q8_0_unaligned(const void *W_row, const float *x, int block_count, int block_size);

// BF16: Brain float 16-bit kernels (Gemma 3 native format)
extern float quantized_gemv_bf16_avx2(const void *W_row, const float *x, int block_count, int block_size);
extern float quantized_gemv_bf16_scalar(const void *W_row, const float *x, int block_count, int block_size);

// F32: Standard 32-bit float kernels (fallback)
extern float quantized_gemv_f32_avx2(const void *W_row, const float *x, int block_count, int block_size);
extern float quantized_gemv_f32_scalar(const void *W_row, const float *x, int block_count, int block_size);

// ============================================================================
// HIGH-LEVEL TENSOR OPERATIONS (Re-exported from tensor_gemv.h)
// ============================================================================

// Core GEMV operation
extern int tensor_gemv_with_ctx(kernel_context_t *ctx, float *y, const tensor_t *A, const float *x);

// Tensor-to-tensor GEMV
extern int tensor_gemv_tensor_with_ctx(kernel_context_t *ctx, tensor_t *y, const tensor_t *A, const tensor_t *x);

// GEMV with accumulation: y += alpha * (A @ x)
extern int tensor_gemv_add_with_ctx(kernel_context_t *ctx, float *y, const tensor_t *A, const float *x, float alpha);

// Batched GEMV
extern int tensor_gemv_batch_with_ctx(kernel_context_t *ctx, float *Y, const tensor_t *A, const float *X, int batch_size);

#endif // KERNELS_H

// sapphire.h - Kernels Interface (formerly sapphire.h)
// Model-agnostic GEMV API using unified tensor_dtype_t
#ifndef SAPPHIRE_H
#define SAPPHIRE_H

#include <stdint.h>
#include <stddef.h>
#include "tensor.h"  // Unified tensor type system

// ============================================================================
// KERNEL FUNCTION POINTERS (Generic dispatch interface)
// ============================================================================

/**
 * Generic GEMV kernel function signature.
 * All kernels conform to this interface for polymorphic dispatch.
 * 
 * @param w_row        Opaque pointer to weight row (Q4_0 blocks, Q8_0 blocks, BF16 array, or F32 array)
 * @param x            Input vector (always F32)
 * @param blocks       Number of blocks/elements to process
 * @param block_size   Size of each block (typically 32 for quantized)
 * @return Dot product result (float)
 */
typedef float (*gemv_kernel_t)(const void* w_row, const float* x, int blocks, int block_size);

// ============================================================================
// BLOCK STRUCTURE DEFINITIONS (Memory layout for weights)
// ============================================================================

// Q4_0: 4-bit quantized weights
typedef struct {
    float scale;         // Scaling factor (f32)
    uint8_t q_data[16];  // 16 bytes * 2 nibbles/byte = 32 packed 4-bit weights
} ggml_block_q4_0;

// Q8_0: 8-bit quantized weights
typedef struct {
    float scale;
    uint8_t q_data[32];  // 32 bytes = 32 int8 weights
} ggml_block_q8_0;

// ============================================================================
// KERNEL IMPLEMENTATIONS (Model-agnostic, use void* for opaque data)
// ============================================================================

// Q4_0 kernels
float quantized_gemv_q4_0_unaligned(const void *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_q4_0_aligned(const void *W_row, const float *x, int block_count, int block_size);

// Q8_0 kernels
float quantized_gemv_q8_0_unaligned(const void *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_q8_0_aligned(const void *W_row, const float *x, int block_count, int block_size);

// BF16 kernels
float quantized_gemv_bf16_scalar(const void *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_bf16_avx2(const void *W_row, const float *x, int block_count, int block_size);

// F32 kernels
float quantized_gemv_f32_scalar(const void *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_f32_avx2(const void *W_row, const float *x, int block_count, int block_size);

// ============================================================================
// THREAD POOL CONTEXT (Persistent batched GEMV dispatcher)
// ============================================================================

typedef struct sapphire_context sapphire_context;

/**
 * Create a thread-pool context for batched GEMV operations.
 * 
 * @param num_threads   Number of worker threads (0 = autodetect CPU count)
 * @param chunk_size    Rows per task per worker (default 16)
 * @return Allocated context, or NULL on error
 */
sapphire_context *sapphire_context_create(int num_threads, int chunk_size);

/**
 * Destroy a thread-pool context and free resources.
 */
void sapphire_context_destroy(sapphire_context *ctx);

/**
 * Batched GEMV: Compute Y = A @ X using thread pool.
 * 
 * The dispatcher automatically selects the best kernel for A's dtype
 * (Q4_0, Q8_0, BF16, F32) and uses aligned/unaligned fast-paths.
 * 
 * @param ctx    Thread pool context
 * @param A      Weight matrix tensor (can be any supported dtype)
 * @param x      Input vector (F32)
 * @param y      Output vector (F32)
 * @return 0 on success, non-zero on error
 */
int sapphire_batched_gemv(sapphire_context *ctx, const tensor_t *A, const float *x, float *y);

#endif // SAPPHIRE_H

// dispatch.c - Central dispatcher and tensor GEMV operations
//
// This file combines two key responsibilities:
// 1. Dispatcher: Routes GEMV operations to appropriate kernel based on tensor dtype
// 2. Tensor GEMV: High-level tensor operations wrapping kernel dispatch
//
// Merged from: sapphire.c (dispatcher) + tensor_gemv.c (tensor operations)
// The actual kernel implementations are in separate files:
// - q4_0_avx.c (4-bit quantized)
// - q8_0_avx.c (8-bit quantized)
// - bf16_avx.c (BF16)
// - f32_avx.c (F32)

#define _POSIX_C_SOURCE 200809L
#include "../../include/sapphire.h"
#include "../../include/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

// ============================================================================
// SECTION 1: DISPATCHER (Routes to appropriate kernel based on tensor dtype)
// ============================================================================

/**
 * gemv_dispatch - Route GEMV to the correct kernel based on weight dtype
 * 
 * This is the main switchboard. Given a weight matrix with a specific dtype,
 * this routes to the appropriate optimized kernel implementation.
 * 
 * Supported paths:
 * - DTYPE_Q4_0: Quantized 4-bit weights -> quantized_gemv_q4_0_aligned/unaligned
 * - DTYPE_Q8_0: Quantized 8-bit weights -> quantized_gemv_q8_0_aligned/unaligned
 * - DTYPE_BF16: Brain float 16-bit -> quantized_gemv_bf16_avx2/scalar
 * - DTYPE_F32:  Standard 32-bit float -> quantized_gemv_f32_avx2/scalar
 * 
 * @param weight_tensor  Weight matrix (can be Q4_0, Q8_0, BF16, or F32)
 * @param input_vector   Activation vector (always F32)
 * @param block_size     Block size for quantized weights (typically 32 for Q4/Q8)
 * 
 * @return Dot product result (float), or 0.0 if dispatch fails
 */
static float gemv_dispatch(const tensor_t *weight_tensor, const float *input_vector, 
                           const int block_size) {
    if (!weight_tensor || !input_vector) {
        fprintf(stderr, "ERROR: gemv_dispatch null pointer\n");
        return 0.0f;
    }

    tensor_dtype_t dtype = tensor_dtype(weight_tensor);
    const void *W_data = (const void *)tensor_data(weight_tensor);

    switch (dtype) {
        case DTYPE_Q4_0: {
            // 4-bit quantized: check alignment for fast-path
            int ndim = tensor_ndim(weight_tensor);
            const int *shape = tensor_shape(weight_tensor);
            int w_cols = shape[ndim - 1];
            int blocks_per_row = (w_cols + 31) / 32;
            
            int x_aligned = ((uintptr_t)input_vector % 32) == 0;
            float result = x_aligned ? 
                quantized_gemv_q4_0_aligned(W_data, input_vector, blocks_per_row, block_size) :
                quantized_gemv_q4_0_unaligned(W_data, input_vector, blocks_per_row, block_size);
            return result;
        }

        case DTYPE_Q8_0: {
            // 8-bit quantized: check alignment for fast-path
            int ndim = tensor_ndim(weight_tensor);
            const int *shape = tensor_shape(weight_tensor);
            int w_cols = shape[ndim - 1];
            int blocks_per_row = (w_cols + 31) / 32;
            
            int x_aligned = ((uintptr_t)input_vector % 32) == 0;
            float result = x_aligned ? 
                quantized_gemv_q8_0_aligned(W_data, input_vector, blocks_per_row, block_size) :
                quantized_gemv_q8_0_unaligned(W_data, input_vector, blocks_per_row, block_size);
            return result;
        }

        case DTYPE_BF16: {
            // BF16 (brain float): 16-bit per value, no blocking
            int ndim = tensor_ndim(weight_tensor);
            const int *shape = tensor_shape(weight_tensor);
            int w_cols = shape[ndim - 1];
            int blocks_per_row = (w_cols + 31) / 32;
            
            // Use AVX2 implementation when available
            return quantized_gemv_bf16_avx2(W_data, input_vector, blocks_per_row, block_size);
        }

        case DTYPE_F32: {
            // Standard F32: use AVX2 when available
            int ndim = tensor_ndim(weight_tensor);
            const int *shape = tensor_shape(weight_tensor);
            int w_cols = shape[ndim - 1];
            int blocks_per_row = (w_cols + 31) / 32;
            
            // Use AVX2 implementation
            return quantized_gemv_f32_avx2(W_data, input_vector, blocks_per_row, block_size);
        }

        default: {
            // Unsupported dtype: log and return zero (safe fallback, no crash)
            fprintf(stderr, "WARNING: gemv_dispatch unsupported dtype %d, returning 0.0\n", (int)dtype);
            return 0.0f;
        }
    }
}

// ============================================================================
// LEGACY COMPATIBILITY WRAPPERS (for backward compatibility with old tests)
// ============================================================================

float quantized_gemv_row_dot_product(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size) {
    return quantized_gemv_q4_0_unaligned((const void *)W_row, x, block_count, block_size);
}

float quantized_gemv_row_dot_product_aligned(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size) {
    return quantized_gemv_q4_0_aligned((const void *)W_row, x, block_count, block_size);
}

float quantized_gemv_row_dot_product_scalar(const void *W_row, const float *x, int block_count, int block_size) {
    return quantized_gemv_q4_0_unaligned(W_row, x, block_count, block_size);
}

// ============================================================================
// SECTION 2: TENSOR GEMV OPERATIONS (High-level tensor operations)
// ============================================================================

/**
 * Convert BF16 (bfloat16) to F32.
 * BF16: [sign | exponent (8) | mantissa (7)]
 * F32:  [sign | exponent (8) | mantissa (23)]
 * Conversion: treat BF16 as high 16 bits of F32, shift left by 16
 */
static float bf16_to_f32(uint16_t bf16_val) {
    uint32_t f32_bits = ((uint32_t)bf16_val) << 16;
    return *(float *)&f32_bits;
}

/**
 * Naive F32 GEMV: y = A @ x
 * 
 * Layout: A is [m, n] row-major
 * y[i] = sum_j(A[i*n + j] * x[j])
 */
static void gemv_f32(float *y, const float *A, const float *x, int m, int n) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * BF16 GEMV: y = A @ x (A is BF16, x and y are F32)
 * 
 * Layout: A is [m, n] row-major in BF16 format
 * y[i] = sum_j(bf16_to_f32(A[i*n + j]) * x[j])
 */
static void gemv_bf16(float *y, const uint16_t *A, const float *x, int m, int n) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            float a_val = bf16_to_f32(A[i * n + j]);
            sum += a_val * x[j];
        }
        y[i] = sum;
    }
}

/* Context-based APIs */
sapphire_context* tensor_gemv_ctx_create(int num_threads, int chunk_size) {
    return sapphire_context_create(num_threads, chunk_size);
}

void tensor_gemv_ctx_destroy(sapphire_context *ctx) {
    sapphire_context_destroy(ctx);
}

int tensor_gemv_with_ctx(sapphire_context *ctx, float *y, const tensor_t *A, const float *x) {
    if (!y || !A || !x) {
        fprintf(stderr, "ERROR: tensor_gemv_with_ctx null pointer\n");
        return -1;
    }

    if (tensor_ndim(A) != 2) {
        fprintf(stderr, "ERROR: tensor_gemv_with_ctx weight must be 2D\n");
        return -1;
    }

    const int *shape = tensor_shape(A);
    if (!shape) {
        fprintf(stderr, "ERROR: tensor_gemv_with_ctx invalid shape\n");
        return -1;
    }

    int m = shape[0];
    int n = shape[1];

    if (m <= 0 || n <= 0) {
        fprintf(stderr, "ERROR: tensor_gemv_with_ctx invalid shape [%d, %d]\n", m, n);
        return -1;
    }

    switch (tensor_dtype(A)) {
        case DTYPE_F32: {
            const float *A_data = (const float *)tensor_data(A);
            gemv_f32(y, A_data, x, m, n);
            return 0;
        }

        case DTYPE_BF16: {
            const uint16_t *A_data = (const uint16_t *)tensor_data(A);
            gemv_bf16(y, A_data, x, m, n);
            return 0;
        }

        case DTYPE_Q4_0:
        case DTYPE_Q8_0: {
            if (!ctx) {
                fprintf(stderr, "ERROR: tensor_gemv_with_ctx requires non-NULL context for quantized weights\n");
                return -1;
            }
            int ret = sapphire_batched_gemv(ctx, A, x, y);
            if (ret != 0) {
                fprintf(stderr, "ERROR: sapphire_batched_gemv failed with code %d\n", ret);
                return -1;
            }
            return 0;
        }

        default:
            fprintf(stderr, "ERROR: tensor_gemv_with_ctx unsupported dtype %d\n", (int)tensor_dtype(A));
            return -1;
    }
}

int tensor_gemv_tensor_with_ctx(sapphire_context *ctx, tensor_t *y, const tensor_t *A, const tensor_t *x) {
    if (!y || !A || !x) {
        fprintf(stderr, "ERROR: tensor_gemv_tensor_with_ctx null pointer\n");
        return -1;
    }

    if (tensor_ndim(y) != 1 || tensor_dtype(y) != DTYPE_F32) {
        fprintf(stderr, "ERROR: tensor_gemv_tensor_with_ctx output must be 1D F32 tensor\n");
        return -1;
    }

    if (tensor_ndim(x) != 1 || tensor_dtype(x) != DTYPE_F32) {
        fprintf(stderr, "ERROR: tensor_gemv_tensor_with_ctx input must be 1D F32 tensor\n");
        return -1;
    }

    const int *shape_A = tensor_shape(A);
    const int *shape_x = tensor_shape(x);
    const int *shape_y = tensor_shape(y);
    if (!shape_A || !shape_x || !shape_y) {
        fprintf(stderr, "ERROR: tensor_gemv_tensor_with_ctx shapes invalid\n");
        return -1;
    }

    if (shape_A[1] != shape_x[0]) {
        fprintf(stderr, "ERROR: tensor_gemv_tensor_with_ctx shape mismatch: A[*,%d] vs x[%d]\n",
                shape_A[1], shape_x[0]);
        return -1;
    }

    if (shape_y[0] != shape_A[0]) {
        fprintf(stderr, "ERROR: tensor_gemv_tensor_with_ctx output size mismatch: y[%d] vs A[%d,*]\n",
                shape_y[0], shape_A[0]);
        return -1;
    }

    float *y_data = tensor_data_f32(y);
    const float *x_data = (const float *)tensor_data(x);

    return tensor_gemv_with_ctx(ctx, y_data, A, x_data);
}

int tensor_gemv_add_with_ctx(sapphire_context *ctx, float *y, const tensor_t *A, const float *x, float alpha) {
    if (!y || !A || !x) {
        fprintf(stderr, "ERROR: tensor_gemv_add_with_ctx null pointer\n");
        return -1;
    }

    if (tensor_ndim(A) != 2) {
        fprintf(stderr, "ERROR: tensor_gemv_add_with_ctx weight must be 2D\n");
        return -1;
    }

    const int *shape = tensor_shape(A);
    int m = shape[0];

    float *temp = (float *)malloc(m * sizeof(float));
    if (!temp) {
        fprintf(stderr, "ERROR: tensor_gemv_add_with_ctx malloc failed\n");
        return -1;
    }

    int ret = tensor_gemv_with_ctx(ctx, temp, A, x);
    if (ret != 0) {
        free(temp);
        return -1;
    }

    for (int i = 0; i < m; i++) {
        y[i] += alpha * temp[i];
    }

    free(temp);
    return 0;
}

int tensor_gemv_batch_with_ctx(sapphire_context *ctx, float *Y, const tensor_t *A, const float *X, int batch_size) {
    if (!Y || !A || !X || batch_size <= 0) {
        fprintf(stderr, "ERROR: tensor_gemv_batch_with_ctx invalid arguments\n");
        return -1;
    }

    if (tensor_ndim(A) != 2) {
        fprintf(stderr, "ERROR: tensor_gemv_batch_with_ctx weight must be 2D\n");
        return -1;
    }

    const int *shape = tensor_shape(A);
    int m = shape[0];  // Output rows
    int n = shape[1];  // Input columns

    for (int k = 0; k < batch_size; k++) {
        const float *x_k = X + k * n;
        float *y_k = Y + k * m;

        int ret = tensor_gemv_with_ctx(ctx, y_k, A, x_k);
        if (ret != 0) {
            return -1;
        }
    }

    return 0;
}

// End of dispatch.c

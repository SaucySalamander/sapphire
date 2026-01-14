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

// ============================================================================
// LEGACY COMPATIBILITY WRAPPERS (DEPRECATED - REMOVED)
// ============================================================================

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
    float f;
    memcpy(&f, &f32_bits, sizeof(float));
    return f;
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
        case DTYPE_F32:
        case DTYPE_BF16:
        case DTYPE_Q4_0:
        case DTYPE_Q8_0: {
            if (!ctx) {
                // Fallback for single-threaded or small models if no context provided
                if (tensor_dtype(A) == DTYPE_F32) {
                    gemv_f32(y, (const float *)tensor_data(A), x, m, n);
                    return 0;
                } else if (tensor_dtype(A) == DTYPE_BF16) {
                    gemv_bf16(y, (const uint16_t *)tensor_data(A), x, m, n);
                    return 0;
                }
                fprintf(stderr, "ERROR: tensor_gemv_with_ctx requires non-NULL context for quantized weights\n");
                return -1;
            }
            // Use the optimized and multithreaded pool implementations
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

/*
 * Return preferred SIMD float lane count for a given dtype.
 * This is a lightweight query used by higher-level code to size scratch
 * buffers (round-up padding). It mirrors assumptions used by the kernels.
 */
int tensor_gemv_simd_lane_count_for_dtype(tensor_dtype_t dtype) {
    switch (dtype) {
        case DTYPE_F32:
        case DTYPE_BF16:
        case DTYPE_F16:
            return 8; // AVX2 256-bit -> 8 x 32-bit floats
        case DTYPE_Q4_0:
        case DTYPE_Q8_0:
            return 8; // quantized kernels operate on blocks but 8 is a safe lane count
        default:
            return 1;
    }
}

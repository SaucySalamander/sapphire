#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor_gemv.h"
#include "tensor.h"
#include "sapphire.h"
#include <pthread.h>

// ============================================================================
// NOTE: The legacy global initialization (tensor_gemv_init /
// tensor_gemv_cleanup) and convenience wrappers that relied on a hidden
// singleton `g_sapphire_ctx` have been removed. Callers must now create and
// manage an explicit `sapphire_context*` via the context APIs provided in
// `tensor_gemv.h` (e.g. `tensor_gemv_ctx_create` / `tensor_gemv_ctx_destroy`) and
// use the `_with_ctx` variants for GEMV operations.
// ============================================================================

// ============================================================================
// Helper: Standard GEMV for F32 tensors
// ============================================================================

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

// ============================================================================
// Helper: Convert tensors to ggml_tensor_t format for Sapphire
// ============================================================================

/**
 * Create an ggml_tensor_t view of a tensor_t for Sapphire dispatch.
 * Does not copy data; just creates a view.
 */
static ggml_tensor_t tensor_to_ggml(const tensor_t *t) {
    ggml_tensor_t ggml_t;
    ggml_t.data = (void *)tensor_data(t);
    ggml_t.data_size = tensor_nbytes(t);

    const int *shape = tensor_shape(t);
    int ndim = tensor_ndim(t);

    if (ndim == 1) {
        ggml_t.rows = 1;
        ggml_t.cols = shape ? shape[0] : 1;
    } else if (ndim >= 2) {
        ggml_t.rows = shape[0];
        ggml_t.cols = shape[1];
    } else {
        ggml_t.rows = 1;
        ggml_t.cols = 1;
    }

    // Map tensor dtype to ggml dtype
    switch (tensor_dtype(t)) {
        case DTYPE_F32:   ggml_t.type = GGML_TYPE_F32; break;
        case DTYPE_Q4_0:  ggml_t.type = GGML_TYPE_Q4_0; break;
        case DTYPE_Q8_0:  ggml_t.type = GGML_TYPE_Q8_0; break;
        default:
            fprintf(stderr, "ERROR: Unsupported dtype in tensor_to_ggml\n");
            ggml_t.type = GGML_TYPE_F32;
    }

    return ggml_t;
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

        case DTYPE_Q4_0:
        case DTYPE_Q8_0: {
            if (!ctx) {
                fprintf(stderr, "ERROR: tensor_gemv_with_ctx requires non-NULL context for quantized weights\n");
                return -1;
            }
            ggml_tensor_t ggml_A = tensor_to_ggml(A);
            int blocks_per_row = (n + 31) / 32;
            int ret = sapphire_batched_gemv(ctx, &ggml_A, 1, m, blocks_per_row, x, y);
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

// End of file

#include <math.h>
#include <string.h>
#include <immintrin.h>
#include "kernels.h"
#include "log.h"
#include "utils.h"

// ============================================================================
// AVX Helper Functions for RMSNorm
// ============================================================================

static inline float sum_sq_avx2(const float *in, int dim) {
    int i = 0;
    __m256 v_sum = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 v_in = _mm256_loadu_ps(in + i);
        v_sum = _mm256_fmadd_ps(v_in, v_in, v_sum);
    }
    
    // Horizontal reduction
    __m128 v_low = _mm256_castps256_ps128(v_sum);
    __m128 v_high = _mm256_extractf128_ps(v_sum, 1);
    v_low = _mm_add_ps(v_low, v_high);
    v_low = _mm_hadd_ps(v_low, v_low);
    v_low = _mm_hadd_ps(v_low, v_low);
    float sum = _mm_cvtss_f32(v_low);

    for (; i < dim; i++) {
        sum += in[i] * in[i];
    }
    return sum;
}

static inline void rmsnorm_apply_avx2(float *out, const float *in, const float *weight, float rms_inv, int dim) {
    int i = 0;
    __m256 v_rms_inv = _mm256_set1_ps(rms_inv);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_in = _mm256_loadu_ps(in + i);
        __m256 v_w = _mm256_loadu_ps(weight + i);
        __m256 v_out = _mm256_mul_ps(_mm256_mul_ps(v_in, v_rms_inv), v_w);
        _mm256_storeu_ps(out + i, v_out);
    }
    for (; i < dim; i++) {
        out[i] = (in[i] * rms_inv) * weight[i];
    }
}

static inline void rmsnorm_delta_apply_avx2(float *out, const float *in, const float *weight, float rms_inv, int dim) {
    int i = 0;
    __m256 v_rms_inv = _mm256_set1_ps(rms_inv);
    __m256 v_one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= dim; i += 8) {
        __m256 v_in = _mm256_loadu_ps(in + i);
        __m256 v_w = _mm256_loadu_ps(weight + i);
        __m256 v_scale = _mm256_add_ps(v_one, v_w);
        __m256 v_out = _mm256_mul_ps(_mm256_mul_ps(v_in, v_rms_inv), v_scale);
        _mm256_storeu_ps(out + i, v_out);
    }
    for (; i < dim; i++) {
        out[i] = (in[i] * rms_inv) * (1.0f + weight[i]);
    }
}

// ============================================================================
// RMSNorm (Root Mean Square Normalization)
// ============================================================================

/**
 * RMSNorm: out[i] = (in[i] / (RMS + eps)) * weight[i]
 * 
 * Non-in-place normalization with separate input/output buffers.
 * Used for transformer blocks and other pipeline stages.
 * 
 * Algorithm:
 *   1. Compute sum of squares: sum = Σ(in[i]²)
 *   2. Compute RMS: rms = sqrt(sum / dim)
 *   3. Normalize and scale: out[i] = (in[i] / (rms + eps)) * weight[i]
 * 
 * Loop unrolling factor: 4
 * - Improves instruction parallelism
 * - Reduces loop overhead
 * - Better cache utilization
 * 
 * Safety:
 * - NULL checks for all pointers
 * - Dimension validation (dim > 0)
 * - Epsilon validation (epsilon >= 0)
 */
int rmsnorm(float *out, const float *in, const float *weight,
            float epsilon, int dim) {
    // Validate inputs
    if (!out || !in || !weight || dim <= 0 || epsilon < 0.0f) {
        return -1;
    }
    
    float sum_sq = sum_sq_avx2(in, dim);
    float rms_inv = 1.0f / sqrtf(sum_sq / (float)dim + epsilon);
    
    rmsnorm_apply_avx2(out, in, weight, rms_inv, dim);
    
    return 0;
}

int rmsnorm_delta(float *out, const float *in, const float *weight,
                 float epsilon, int dim) {
    // Validate inputs
    if (!out || !in || !weight || dim <= 0 || epsilon < 0.0f) {
        return -1;
    }

    float sum_sq = sum_sq_avx2(in, dim);
    float rms_inv = 1.0f / sqrtf(sum_sq / (float)dim + epsilon);

    rmsnorm_delta_apply_avx2(out, in, weight, rms_inv, dim);

    return 0;
}

/**
 * Batch RMSNorm: Process multiple vectors efficiently.
 * 
 * Processes batch_size vectors of dimension dim each.
 * Memory layout: row-major, C-contiguous (as in typical neural networks).
 * 
 * GEMMA 3 UPDATE: Uses direct gamma scaling (weight only)
 * 
 * Matrix layout:
 *   in: [batch_size x dim] matrix, stored row-major
 *   out: [batch_size x dim] output matrix
 *   weight: [dim] shared per-dimension scaling
 * 
 * @param out Output matrix [batch_size x dim] (row-major)
 * @param in Input matrix [batch_size x dim] (row-major)
 * @param weight Per-dimension scaling [dim]
 * @param epsilon Small constant (e.g., 1e-6)
 * @param batch_size Number of vectors
 * @param dim Dimension per vector
 * 
 * @return 0 on success, -1 on error
 * 
 * Performance: Useful for batched token processing in autoregressive generation
 */
int rmsnorm_batch(float *out, const float *in, const float *weight,
                 float epsilon, int batch_size, int dim) {
    if (!out || !in || !weight || batch_size <= 0 || dim <= 0 || epsilon < 0.0f) {
        return -1;
    }
    
    for (int b = 0; b < batch_size; b++) {
        float *row_out = out + b * dim;
        const float *row_in = in + b * dim;
        
        float sum_sq = sum_sq_avx2(row_in, dim);
        float rms_inv = 1.0f / sqrtf(sum_sq / (float)dim + epsilon);
        
        rmsnorm_apply_avx2(row_out, row_in, weight, rms_inv, dim);
    }
    
    return 0;
}

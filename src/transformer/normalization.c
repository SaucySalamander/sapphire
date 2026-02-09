#include <math.h>
#include <string.h>
#include "kernels.h"
#include "log.h"
#include "utils.h"



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
    
    // Step 1: Compute sum of squares with loop unrolling (factor 4)
    float sum_sq = 0.0f;
    
    int unroll_limit = (dim / 4) * 4;
    int i = 0;
    
    // Main unrolled loop (4x per iteration)
    for (i = 0; i < unroll_limit; i += 4) {
        sum_sq += in[i + 0] * in[i + 0];
        sum_sq += in[i + 1] * in[i + 1];
        sum_sq += in[i + 2] * in[i + 2];
        sum_sq += in[i + 3] * in[i + 3];
    }
    
    // Handle remaining elements (0-3)
    for (; i < dim; i++) {
        sum_sq += in[i] * in[i];
    }
    
    // Step 2: Compute RMS
    float rms = sqrtf(sum_sq / (float)dim + epsilon);
    
    // Step 3: Normalize and scale with loop unrolling
    i = 0;
    for (i = 0; i < unroll_limit; i += 4) {
        out[i + 0] = (in[i + 0] / rms) * weight[i + 0];
        out[i + 1] = (in[i + 1] / rms) * weight[i + 1];
        out[i + 2] = (in[i + 2] / rms) * weight[i + 2];
        out[i + 3] = (in[i + 3] / rms) * weight[i + 3];
    }
    
    // Handle remaining elements
    for (; i < dim; i++) {
        out[i] = (in[i] / rms) * weight[i];
    }
    
    return 0;
}

int rmsnorm_delta(float *out, const float *in, const float *weight,
                 float epsilon, int dim) {
    // Validate inputs
    if (!out || !in || !weight || dim <= 0 || epsilon < 0.0f) {
        return -1;
    }

    // Compute sum of squares
    float sum_sq = 0.0f;
    int unroll_limit = (dim / 4) * 4;
    int i = 0;
    for (i = 0; i < unroll_limit; i += 4) {
        sum_sq += in[i + 0] * in[i + 0];
        sum_sq += in[i + 1] * in[i + 1];
        sum_sq += in[i + 2] * in[i + 2];
        sum_sq += in[i + 3] * in[i + 3];
    }
    for (; i < dim; i++) sum_sq += in[i] * in[i];

    float rms = sqrtf(sum_sq / (float)dim + epsilon);

    // Normalize and apply Gemma3-style scaling: (1 + weight[i])
    i = 0;
    for (i = 0; i < unroll_limit; i += 4) {
        out[i + 0] = (in[i + 0] / rms) * (1.0f + weight[i + 0]);
        out[i + 1] = (in[i + 1] / rms) * (1.0f + weight[i + 1]);
        out[i + 2] = (in[i + 2] / rms) * (1.0f + weight[i + 2]);
        out[i + 3] = (in[i + 3] / rms) * (1.0f + weight[i + 3]);
    }
    for (; i < dim; i++) {
        out[i] = (in[i] / rms) * (1.0f + weight[i]);
    }

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
        
        // Compute RMS for this row
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_sq += row_in[i] * row_in[i];
        }
        
        float rms = sqrtf(sum_sq / (float)dim + epsilon);
        
        for (int i = 0; i < dim; i++) {
            row_out[i] = (row_in[i] / rms) * weight[i];
        }
    }
    
    return 0;
}

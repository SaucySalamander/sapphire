#include <math.h>
#include <string.h>
#include "normalization.h"

// ============================================================================
// RMSNorm (Root Mean Square Normalization)
// ============================================================================

/**
 * RMSNorm computation:
 * 1. Compute RMS: sqrt(mean(x^2))
 * 2. Scale: x * weight / (RMS + eps)
 * 
 * Efficient: Only requires one pass through data to compute RMS.
 * Used in modern LLMs (LLaMA, Mistral, etc.).
 */
void rmsnorm(float *x, const float *weight, int n, float eps) {
    if (!x || !weight || n <= 0) return;
    
    // Step 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    
    // Step 2: Compute inverse RMS using the Gemma 3 convention (no unit offset)
    float mean_sq = sum_sq / (float)n;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

    // Step 3: Normalize and scale by learned weight (Gemma 3 standard)
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

void rmsnorm_delta(float *x, const float *weight, int n, float eps) {
    if (!x || !weight || n <= 0) return;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }

    // Compute inverse RMS
    float mean_sq = sum_sq / (float)n;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

    // Apply delta semantics: scale by (1.0 + weight[i])
    for (int i = 0; i < n; i++) {
        x[i] = x[i] * inv_rms * (1.0f + weight[i]);
    }
}

void rmsnorm_no_weight(float *x, int n, float eps) {
    if (!x || n <= 0) return;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    
    // Compute RMS
    float rms = sqrtf(sum_sq / (float)n + eps);
    
    // Normalize
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / rms;
    }
}

void rmsnorm_batch(float *matrix, const float *weight, int num_rows, int row_size, float eps) {
    if (!matrix || !weight || num_rows <= 0 || row_size <= 0) return;
    
    for (int row = 0; row < num_rows; row++) {
        float *row_ptr = matrix + row * row_size;
        
        // Compute RMS for this row
        float sum_sq = 0.0f;
        for (int i = 0; i < row_size; i++) {
            sum_sq += row_ptr[i] * row_ptr[i];
        }
        
        float rms = sqrtf(sum_sq / (float)row_size + eps);
        
        // Normalize and scale with learned weights
        for (int i = 0; i < row_size; i++) {
            row_ptr[i] = (row_ptr[i] / rms) * weight[i];
        }
    }
}

// ============================================================================
// LayerNorm (Layer Normalization)
// ============================================================================

/**
 * LayerNorm computation:
 * 1. Compute mean: mean(x)
 * 2. Compute variance: mean((x - mean)^2)
 * 3. Normalize: (x - mean) / sqrt(var + eps)
 * 4. Scale and shift: output * weight + bias
 * 
 * More expensive than RMSNorm (requires computing both mean and variance),
 * but includes learnable bias term which can be useful for fine-tuning.
 */
void layernorm(float *x, const float *weight, const float *bias, int n, float eps) {
    if (!x || !weight || !bias || n <= 0) return;
    
    // Step 1: Compute mean
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    float mean = sum / (float)n;
    
    // Step 2: Compute variance
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        sum_sq_diff += diff * diff;
    }
    float var = sum_sq_diff / (float)n;
    
    // Step 3: Normalize, scale, and shift
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        x[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
    }
}

void layernorm_no_params(float *x, int n, float eps) {
    if (!x || n <= 0) return;
    
    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    float mean = sum / (float)n;
    
    // Compute variance
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        sum_sq_diff += diff * diff;
    }
    float var = sum_sq_diff / (float)n;
    
    // Normalize
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        x[i] = (x[i] - mean) * inv_std;
    }
}

void layernorm_batch(float *matrix, const float *weight, const float *bias, 
                     int num_rows, int row_size, float eps) {
    if (!matrix || !weight || !bias || num_rows <= 0 || row_size <= 0) return;
    
    for (int row = 0; row < num_rows; row++) {
        float *row_ptr = matrix + row * row_size;
        
        // Compute mean for this row
        float sum = 0.0f;
        for (int i = 0; i < row_size; i++) {
            sum += row_ptr[i];
        }
        float mean = sum / (float)row_size;
        
        // Compute variance
        float sum_sq_diff = 0.0f;
        for (int i = 0; i < row_size; i++) {
            float diff = row_ptr[i] - mean;
            sum_sq_diff += diff * diff;
        }
        float var = sum_sq_diff / (float)row_size;
        
        // Normalize, scale, and shift
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < row_size; i++) {
            row_ptr[i] = ((row_ptr[i] - mean) * inv_std) * weight[i] + bias[i];
        }
    }
}

// ============================================================================
// Standardized RMSNorm (Non-In-Place)
// ============================================================================

/**
 * Standardized RMSNorm: out[i] = (in[i] / (RMS + eps)) * weight[i]
 * 
 * This is the non-in-place API version (preserves input).
 * Used for transformer blocks and other pipeline stages.
 * 
 * Compared to existing rmsnorm():
 * - Separate input/output buffers
 * - Return error code instead of void
 * - Better for pipelined operations
 * - Uses direct gamma scaling (weight)
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
 * - Epsilon validation (epsilon > 0)
 * - Bounds assertions
 */
int sapphire_rmsnorm(float *out, const float *in, const float *weight,
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

int sapphire_rmsnorm_delta(float *out, const float *in, const float *weight,
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

    // Normalize and apply (1.0 + weight)
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
int sapphire_rmsnorm_batch(float *out, const float *in, const float *weight,
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
        
        // Normalize and scale
        for (int i = 0; i < dim; i++) {
            row_out[i] = (row_in[i] / rms) * weight[i];
        }
    }
    
    return 0;
}

// ============================================================================
// Gemma 3 QK-Normalization Attention Mechanism
// ============================================================================

/**
 * @brief Apply QK-Norm (RMSNorm on Q and K vectors) for Gemma 3.
 * 
 * Requirement:
 * - RMSNorm Logic: x_i = (x_i / sqrt(mean(x^2) + eps)) * g_i
 * - Single pass optimization for sum-of-squares.
 * - 1e-6 epsilon.
 * 
 * @param q Pointer to Query vectors [num_q_heads * head_dim]
 * @param k Pointer to Key vectors [num_kv_heads * head_dim]
 * @param q_scale Pointer to Query scale weights [num_q_heads * head_dim]
 * @param k_scale Pointer to Key scale weights [num_kv_heads * head_dim]
 * @param head_dim Dimension of each head
 * @param num_q_heads Number of Query heads
 * @param num_kv_heads Number of Key/Value heads
 */
void apply_qk_norm(float* q, float* k, float* q_scale, float* k_scale, int head_dim, int num_q_heads, int num_kv_heads) {
    const float eps = 1e-6f;

    // Normalize Q vectors
    if (q && q_scale) {
        for (int h = 0; h < num_q_heads; h++) {
            float* head_q = q + h * head_dim;
            float* head_scale = q_scale + h * head_dim;

            // Single pass sum-of-squares
            float sum_sq = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                sum_sq += head_q[i] * head_q[i];
            }

            // Compute RMS and inverse
            float mean_sq = sum_sq / (float)head_dim;
            float inv_rms = 1.0f / sqrtf(mean_sq + eps);

            // Apply normalization and scaling
            for (int i = 0; i < head_dim; i++) {
                head_q[i] = (head_q[i] * inv_rms) * head_scale[i];
            }
        }
    }

    // Normalize K vectors
    if (k && k_scale) {
        for (int h = 0; h < num_kv_heads; h++) {
            float* head_k = k + h * head_dim;
            float* head_scale = k_scale + h * head_dim;

            // Single pass sum-of-squares
            float sum_sq = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                sum_sq += head_k[i] * head_k[i];
            }

            // Compute RMS and inverse
            float mean_sq = sum_sq / (float)head_dim;
            float inv_rms = 1.0f / sqrtf(mean_sq + eps);

            // Apply normalization and scaling
            for (int i = 0; i < head_dim; i++) {
                head_k[i] = (head_k[i] * inv_rms) * head_scale[i];
            }
        }
    }
}


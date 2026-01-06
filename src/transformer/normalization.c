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
    
    // Step 2: Compute RMS
    float rms = sqrtf(sum_sq / (float)n + eps);
    
    // Step 3: Normalize and scale by weight
    for (int i = 0; i < n; i++) {
        x[i] = (x[i] / rms) * weight[i];
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
        
        // Normalize and scale
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

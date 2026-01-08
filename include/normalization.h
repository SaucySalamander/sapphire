#ifndef NORMALIZATION_H
#define NORMALIZATION_H

/**
 * @file normalization.h
 * @brief Layer normalization functions for transformer blocks.
 * 
 * This module provides two popular normalization schemes:
 * - RMSNorm: Root Mean Square normalization (used in modern LLMs like LLaMA)
 * - LayerNorm: Classic layer normalization (used in BERT, GPT-2)
 */

/**
 * @brief Compute RMSNorm (Root Mean Square Normalization).
 * 
 * Formula:
 *   RMS = sqrt(mean(x^2))
 *   output = x * (weight / (RMS + eps))
 * 
 * Used in modern LLMs (LLaMA, etc.). More efficient than LayerNorm and
 * empirically performs well. Does not include learnable bias term.
 * 
 * @param x Input values (modified in-place).
 * @param weight Learnable scale factor (per-dimension). Length: n.
 * @param n Number of elements.
 * @param eps Small constant to prevent division by zero (e.g., 1e-6f).
 * 
 * @note x and weight are both length n.
 * @note Assumes weight is already initialized to 1.0f if no training.
 */
void rmsnorm(float *x, const float *weight, int n, float eps);

/**
 * @brief Compute RMSNorm without learnable weight (scale by 1/RMS only).
 * 
 * Formula: output = x / (RMS(x) + eps)
 * 
 * Useful for testing or when weight scaling is not needed.
 * 
 * @param x Input values (modified in-place).
 * @param n Number of elements.
 * @param eps Small constant to prevent division by zero.
 */
void rmsnorm_no_weight(float *x, int n, float eps);

/**
 * @brief Compute LayerNorm (Classic Layer Normalization).
 * 
 * Formula:
 *   mean = mean(x)
 *   var = var(x) = mean((x - mean)^2)
 *   output = (x - mean) / sqrt(var + eps) * weight + bias
 * 
 * Classic normalization. More expensive than RMSNorm (requires mean and variance
 * computation) but includes learnable shift (bias).
 * 
 * @param x Input values (modified in-place with output).
 * @param weight Learnable scale factor (per-dimension). Length: n.
 * @param bias Learnable shift factor (per-dimension). Length: n.
 * @param n Number of elements.
 * @param eps Small constant to prevent division by zero (e.g., 1e-6f).
 * 
 * @note x, weight, and bias are all length n.
 * @note Assumes weight initialized to 1.0f, bias to 0.0f if no training.
 */
void layernorm(float *x, const float *weight, const float *bias, int n, float eps);

/**
 * @brief Compute LayerNorm without learnable parameters (mean-variance normalization only).
 * 
 * Formula: output = (x - mean(x)) / sqrt(var(x) + eps)
 * 
 * Useful for testing or analysis layers without learned parameters.
 * 
 * @param x Input values (modified in-place).
 * @param n Number of elements.
 * @param eps Small constant to prevent division by zero.
 */
void layernorm_no_params(float *x, int n, float eps);

/**
 * @brief Compute RMSNorm for a 2D matrix (layer-wise normalization).
 * 
 * Applies RMSNorm independently to each row of a matrix.
 * 
 * Formula (per row): output[i,:] = input[i,:] * (weight / RMS(input[i,:]) + eps)
 * 
 * @param matrix Input matrix, row-major layout. Size: [num_rows, row_size].
 *               Modified in-place with normalized output.
 * @param weight Learnable scale per feature. Length: row_size.
 * @param num_rows Number of rows to normalize.
 * @param row_size Number of columns per row (feature dimension).
 * @param eps Small constant to prevent division by zero.
 */
void rmsnorm_batch(float *matrix, const float *weight, int num_rows, int row_size, float eps);

/**
 * @brief Compute LayerNorm for a 2D matrix (layer-wise normalization).
 * 
 * Applies LayerNorm independently to each row of a matrix.
 * 
 * @param matrix Input matrix, row-major layout. Size: [num_rows, row_size].
 *               Modified in-place with normalized output.
 * @param weight Learnable scale per feature. Length: row_size.
 * @param bias Learnable shift per feature. Length: row_size.
 * @param num_rows Number of rows to normalize.
 * @param row_size Number of columns per row (feature dimension).
 * @param eps Small constant to prevent division by zero.
 */
void layernorm_batch(float *matrix, const float *weight, const float *bias, 
                     int num_rows, int row_size, float eps);

/**
 * @brief Standardized RMSNorm (Root Mean Square Normalization).
 *
 * Computes: out[i] = (in[i] / (RMS(in) + epsilon)) * weight[i]
 *
 * This is the standardized API version of RMSNorm with separate output buffer.
 * Used for implementing pre-norm and post-norm layer positioning in transformers.
 *
 * Formula:
 *   RMS = sqrt(mean(in^2))
 *   out[i] = (in[i] / (RMS + eps)) * weight[i]
 *
 * Key differences from in-place rmsnorm():
 * - Separate input and output buffers (non-destructive)
 * - Explicit output parameter
 * - Enables pipelining without temporary buffers
 *
 * Algorithm:
 *   1. Compute sum of squares: sum = Σ(in[i]^2)
 *   2. Compute RMS: rms = sqrt(sum / dim)
 *   3. Normalize and scale: out[i] = (in[i] / (rms + eps)) * weight[i]
 *
 * Optimization: Inner loop is unrolled (factor 4-8) for cache efficiency.
 *
 * @param out Output array where normalized values are written.
 *            Must be pre-allocated and have size >= dim.
 *            Must not overlap with 'in' (unless separate pointers needed).
 * @param in Input array to normalize (not modified).
 *           Must contain dim elements.
 * @param weight Learnable scale factors (per-dimension).
 *               Must contain dim elements.
 *               Common initialization: all 1.0f
 * @param epsilon Small constant to prevent division by zero.
 *                 Typical value: 1e-6f
 *                 Must be positive.
 * @param dim Number of dimensions (elements).
 *            Must be > 0 and match size of all arrays.
 *
 * @return 0 on success, -1 on error (NULL pointers, invalid dim, negative epsilon).
 *
 * @note All arrays must be pre-allocated with size >= dim.
 * @note 'in' is read-only; 'out' receives normalized values.
 * @note NULL checks are performed; function returns -1 on invalid input.
 *
 * Example:
 *   float in[] = {1.0, 2.0, 3.0};
 *   float weight[] = {1.0, 1.0, 1.0};
 *   float out[3];
 *   sapphire_rmsnorm(out, in, weight, 1e-6f, 3);
 *   // out contains RMSNorm(in) * weight
 */
int sapphire_rmsnorm(float *out, const float *in, const float *weight,
                     float epsilon, int dim);

#endif // NORMALIZATION_H

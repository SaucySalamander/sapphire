#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "transformer.h"
#include "llm_model.h"
#include "gemma3_270m_config.h"

/**
 * @file normalization.h
 * @brief Layer normalization functions for transformer blocks.
 * 
 * This module provides RMSNorm (Root Mean Square Normalization),
 * which is efficient and used in modern LLMs like LLaMA, Mistral, and Gemma 3.
 */

/**
 * @brief RMSNorm (Root Mean Square Normalization).
 * 
 * Computes: out[i] = (in[i] / (RMS(in) + epsilon)) * weight[i]
 * 
 * Non-in-place normalization with separate input/output buffers.
 * Used for implementing pre-norm and post-norm layer positioning in transformers.
 * 
 * Formula:
 *   RMS = sqrt(mean(in^2))
 *   out[i] = (in[i] / (RMS + eps)) * weight[i]
 * 
 * Algorithm:
 *   1. Compute sum of squares: sum = Σ(in[i]^2)
 *   2. Compute RMS: rms = sqrt(sum / dim)
 *   3. Normalize and scale: out[i] = (in[i] / (rms + eps)) * weight[i]
 * 
 * Optimization: Inner loop is unrolled (factor 4) for cache efficiency.
 * 
 * @param out Output array where normalized values are written.
 *            Must be pre-allocated and have size >= dim.
 * @param in Input array to normalize (not modified).
 *           Must contain dim elements.
 * @param weight Learnable scale factors (per-dimension).
 *               Must contain dim elements.
 * @param epsilon Small constant to prevent division by zero (e.g., 1e-6f).
 * @param dim Number of dimensions (elements). Must be > 0.
 * 
 * @return 0 on success, -1 on error (NULL pointers, invalid dim, negative epsilon).
 * 
 * @note All arrays must be pre-allocated with size >= dim.
 * @note 'in' is read-only; 'out' receives normalized values.
 */
int rmsnorm(float *out, const float *in, const float *weight,
            float epsilon, int dim);

/**
 * @brief RMSNorm with delta semantics (Gemma 3 style).
 *
 * Applies (1.0 + weight[i]) scaling for zero-centered parameterization.
 *
 * Computes: out[i] = (in[i] / (RMS(in) + epsilon)) * (1.0 + weight[i])
 *
 * @param out Output array.
 * @param in Input array.
 * @param weight Delta weights (applied as 1.0 + weight).
 * @param epsilon Small constant to prevent division by zero.
 * @param dim Number of dimensions.
 *
 * @return 0 on success, -1 on error.
 */
int rmsnorm_delta(float *out, const float *in, const float *weight,
                  float epsilon, int dim);

/**
 * @brief Batch RMSNorm: Process multiple vectors efficiently.
 * 
 * Processes batch_size vectors of dimension dim each.
 * Memory layout: row-major, C-contiguous.
 * 
 * @param out Output matrix [batch_size x dim] (row-major)
 * @param in Input matrix [batch_size x dim] (row-major)
 * @param weight Per-dimension scaling [dim]
 * @param epsilon Small constant (e.g., 1e-6)
 * @param batch_size Number of vectors
 * @param dim Dimension per vector
 * 
 * @return 0 on success, -1 on error
 */
int rmsnorm_batch(float *out, const float *in, const float *weight,
                  float epsilon, int batch_size, int dim);

/**
 * @brief Result struct for QK-Norm weight loading.
 */
typedef struct {
    float* q_scale;
    float* k_scale;
} qk_norm_result_t;

/**
 * @brief Apply QK-Norm with weight extraction from layer.
 * 
 * Loads Q and K scale weights from the layer, handles broadcasting,
 * and applies QK-Norm in-place to query and key projections.
 * 
 * @param buf Layer buffers containing q_proj, k_proj, and scratch space
 * @param layer Model layer weights containing q_norm_weight, k_norm_weight
 * @param config Model configuration with head counts and dimensions
 * @param head_dim Dimension per attention head
 * @param layer_idx Layer index for logging
 * @return Struct containing pointers to extracted q_scale and k_scale
 */
qk_norm_result_t qk_norm_from_layer(layer_buffers_t buf,
                                           model_layer_weights_t* layer,
                                           gemma3_270m_config_t* config,
                                           int head_dim,
                                           int layer_idx);

#endif // NORMALIZATION_H

#ifndef ATTENTION_H
#define ATTENTION_H

#include "attention_strategy.h"

/**
 * @brief Computes attention scores using standard scaled dot-product attention.
 * 
 * This is a convenience wrapper around the modular attention system.
 * Uses the scaled_dot_product_strategy: Softmax(Q·K^T / sqrt(d_k))
 * 
 * @param q The Query vector (d_k dimension).
 * @param kv_cache_k The entire Key cache (context_length x d_k matrix, row-major).
 * @param context_length The number of tokens currently in the cache.
 * @param d_k The dimension of the head (used for scaling: 1/sqrt(d_k)).
 * @param output_scores A float array to store the resulting Softmax-normalized 
 *                      attention weights (size: context_length). Modified in-place.
 */
void compute_attention_scores(
    const float *q,
    const float *kv_cache_k,
    int context_length,
    int d_k,
    float *output_scores
);

/**
 * @brief Computes attention scores with temperature scaling.
 * 
 * Uses the temperature_scaled_strategy: Softmax(Q·K^T / temperature)
 * 
 * Lower temperature concentrates attention on the highest-scoring token.
 * Higher temperature spreads attention more evenly.
 * 
 * @param q The Query vector (d_k dimension).
 * @param kv_cache_k The entire Key cache (context_length x d_k matrix, row-major).
 * @param context_length The number of tokens currently in the cache.
 * @param d_k The dimension of the head.
 * @param temperature The temperature scaling factor (>0). 
 *                    0.5 = sharper, 1.0 = normal, 2.0 = softer.
 * @param output_scores Output attention weights (size: context_length).
 */
void compute_attention_scores_with_temperature(
    const float *q,
    const float *kv_cache_k,
    int context_length,
    int d_k,
    float temperature,
    float *output_scores
);

/**
 * @brief Computes attention scores using ALiBi (Attention with Linear Biases).
 * 
 * Uses the alibi_attention_strategy: Applies position-dependent biases.
 * 
 * @param q The Query vector (d_k dimension).
 * @param kv_cache_k The entire Key cache (context_length x d_k matrix, row-major).
 * @param context_length The number of tokens currently in the cache.
 * @param d_k The dimension of the head.
 * @param output_scores Output attention weights (size: context_length).
 */
void compute_attention_scores_with_alibi(
    const float *q,
    const float *kv_cache_k,
    int context_length,
    int d_k,
    float *output_scores
);

#endif // ATTENTION_H

#include "attention_strategy.h"
#include "utils.h"
#include <math.h>
#include <stddef.h>

/**
 * @brief Standard scaled dot-product attention strategy.
 * 
 * Scales scores by 1/sqrt(d_k), then applies softmax.
 * This is the core mechanism used in the Transformer architecture.
 * 
 * The scaling factor 1/sqrt(d_k) prevents the dot products from growing too large,
 * which would cause softmax to concentrate probability on a single token.
 */
void scaled_dot_product_strategy(
    float *scores,
    int context_length,
    int d_k,
    void *user_data
) {
    (void)user_data;  // Unused for this strategy

    if (scores == NULL || context_length <= 0 || d_k <= 0) {
        return;
    }

    // Scale by 1/sqrt(d_k)
    float scale_factor = 1.0f / sqrtf((float)d_k);
    for (int i = 0; i < context_length; i++) {
        scores[i] *= scale_factor;
    }

    // Apply softmax normalization
    softmax(scores, context_length);
}

/**
 * @brief Temperature-scaled attention strategy.
 * 
 * Scales scores by 1/temperature, then applies softmax.
 * - Lower temperature (e.g., 0.5): Sharpens the distribution, concentrates attention.
 * - Higher temperature (e.g., 2.0): Flattens the distribution, spreads attention.
 * - Temperature = 1.0: No temperature effect (equivalent to scaled dot-product).
 * 
 * This is useful for controlling generation diversity in language models.
 * Lower temperature = more deterministic / focused. Higher temperature = more diverse / exploratory.
 */
void temperature_scaled_strategy(
    float *scores,
    int context_length,
    int d_k,
    void *user_data
) {
    (void)d_k;  // Temperature doesn't depend on d_k

    if (scores == NULL || user_data == NULL || context_length <= 0) {
        return;
    }

    float temperature = *(float *)user_data;
    if (temperature <= 0.0f) {
        temperature = 1.0f;  // Fallback to no temperature effect
    }

    // Scale by 1/temperature
    float scale_factor = 1.0f / temperature;
    for (int i = 0; i < context_length; i++) {
        scores[i] *= scale_factor;
    }

    // Apply softmax normalization
    softmax(scores, context_length);
}

/**
 * @brief ALiBi (Attention with Linear Biases) strategy.
 * 
 * ALiBi is an alternative to scaling that adds position-dependent biases to attention scores.
 * Instead of using rotary embeddings or scaling, ALiBi directly biases the attention matrix.
 * 
 * Formula: attention_bias[i, j] = -|i - j| * slope_h
 * where i is query position, j is key position, h is the head index.
 * 
 * Benefits:
 * - Doesn't require position embeddings in the input.
 * - Naturally generalizes to longer sequences.
 * - Simpler than RoPE.
 * 
 * In this placeholder implementation, we just apply softmax without scaling.
 * In production, you would:
 * 1. Track the head index.
 * 2. Compute head-specific slope (e.g., slope = 2^(-8i/num_heads) for head i).
 * 3. Add the bias before softmax.
 */
void alibi_attention_strategy(
    float *scores,
    int context_length,
    int d_k,
    void *user_data
) {
    (void)d_k;
    (void)user_data;

    if (scores == NULL || context_length <= 0) {
        return;
    }

    // Placeholder: Just apply softmax without scaling.
    // In a full implementation, ALiBi biases would be added to scores before softmax.
    softmax(scores, context_length);
}

/**
 * @brief Generic attention score computation with pluggable strategy.
 * 
 * Core algorithm:
 * 1. Compute dot product Q · K^T for all cached key vectors.
 * 2. Apply the chosen scaling/normalization strategy (handles scaling and softmax).
 * 
 * This function separates the dot product computation from the normalization logic,
 * allowing different attention mechanisms to be swapped in without changing the
 * core dot product computation.
 */
void compute_attention_scores_with_strategy(
    const float *q,
    const float *kv_cache_k,
    int context_length,
    int d_k,
    AttentionScalingStrategy strategy,
    void *user_data,
    float *output_scores
) {
    if (q == NULL || kv_cache_k == NULL || output_scores == NULL || strategy == NULL) {
        return;
    }

    if (context_length <= 0 || d_k <= 0) {
        return;
    }

    // Step 1: Compute dot products between Q and all K vectors in the cache.
    // Each row of kv_cache_k is a key vector of dimension d_k (row-major layout).
    for (int i = 0; i < context_length; i++) {
        float dot_product = 0.0f;

        // Compute Q · K_i (dot product with i-th key vector)
        for (int j = 0; j < d_k; j++) {
            dot_product += q[j] * kv_cache_k[i * d_k + j];
        }

        output_scores[i] = dot_product;
    }

    // Step 2: Apply the scaling/normalization strategy.
    // This handles all strategy-specific logic: scaling, bias application, softmax, etc.
    strategy(output_scores, context_length, d_k, user_data);
}

#include "attention.h"
#include "kernels.h"

#include <stdbool.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "log.h"

#define GEMMA3_QK_HEAD_DIM 256
/* NOTE: Use model->config.query_pre_attn_scalar at runtime (e.g., 256).
 * The legacy GEMMA3_QK_SCALE used 1/sqrt(256) which is misleading; prefer
 * applying 1.0 / query_pre_attn_scalar to scale unit-normalized Q·K.
 */
#define GEMMA3_QK_SCALE 0.00390625f  // 1 / 256 (legacy informational constant)

static bool attention_debug_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* env = getenv("SAPPHIRE_DEBUG_ATTENTION");
        cached = (env && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached == 1;
}

/**
 * @brief Standard scaled dot-product attention strategy.
 *
 * For Gemma 3: NO scaling applied (scale_factor = 1.0).
 * QK-norm already normalizes Q and K to unit vectors, preventing score explosion.
 *
 * For other models: Scales scores by 1/sqrt(d_k), then applies softmax.
 * The scaling factor prevents dot products from growing too large.
 */
void scaled_dot_product_strategy(
    float* scores,
    int context_length,
    int d_k,
    const void* user_data) {
    if (scores == NULL || context_length <= 0 || d_k <= 0) {
        return;
    }

    const bool debug_enabled = attention_debug_enabled();
    const gemma3_attention_params_t* params = (const gemma3_attention_params_t*)user_data;

    // Gemma 3: QK-norm replaces 1/sqrt(d_k) scaling
    // After QK-norm, Q and K are unit vectors, so no additional scaling needed
    float scale_factor = (d_k == GEMMA3_QK_HEAD_DIM) ? 1.0f : (1.0f / sqrtf((float)d_k));

    if (params && params->manual_scale > 0.0f) {
        scale_factor = params->manual_scale;
    }

    for (int i = 0; i < context_length; i++) {
        scores[i] *= scale_factor;
    }

    // Apply softcap if specified (Gemma 3)
    // REMOVED: Soft-capping is replaced by QK-Norm in Gemma 3
    /*
    if (params && params->softcap > 0.0f) {
        float inv_cap = 1.0f / params->softcap;
        for (int i = 0; i < context_length; i++) {
            scores[i] = params->softcap * tanhf(scores[i] * inv_cap);
        }
    }
    */

    if (debug_enabled) {
        if (context_length <= 20) {
            LOG_DEBUG("Attention scores (after scaling by %.4f, before softmax):", scale_factor);
            for (int i = 0; i < context_length; i++) {
                LOG_DEBUG("  scores[%d] = %.6f", i, scores[i]);
            }
        } else {
            LOG_DEBUG("Scaled scores (first 10 of %d): %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f",
                      context_length, scores[0], scores[1], scores[2], scores[3], scores[4],
                      scores[5], scores[6], scores[7], scores[8], scores[9]);
        }
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
    float* scores,
    int context_length,
    int d_k,
    void* user_data) {
    (void)d_k;  // Temperature doesn't depend on d_k

    if (scores == NULL || user_data == NULL || context_length <= 0) {
        return;
    }

    float temperature = *(float*)user_data;
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
    float* scores,
    int context_length,
    int d_k,
    void* user_data) {
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
    const float* q,
    const float* kv_cache_k,
    float* output_scores,
    const attention_scoring_config_t* config) {
    if (q == NULL || kv_cache_k == NULL || output_scores == NULL || config == NULL || config->strategy == NULL) {
        return;
    }

    int context_length = config->context_length;
    int d_k = config->d_k;

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
    config->strategy(output_scores, context_length, d_k, config->user_data);
}

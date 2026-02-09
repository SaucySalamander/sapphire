#ifndef ATTENTION_STRATEGY_H
#define ATTENTION_STRATEGY_H

/**
 * @brief Attention scaling/normalization strategy function type.
 * 
 * Computes and applies attention scaling logic to raw dot-product scores.
 * Different strategies may:
 * - Scale by 1/sqrt(d_k) (standard scaled dot-product)
 * - Apply temperature scaling
 * - Apply ALiBi bias
 * - Use custom masking or normalization
 * 
 * @param scores Raw attention scores (length: context_length). Modified in-place.
 * @param context_length Number of tokens in the cache.
 * @param d_k Dimension of the attention head.
 * @param user_data Pointer to strategy-specific parameters (e.g., temperature).
 */
typedef void (*AttentionScalingStrategy)(
    float *scores,
    int context_length,
    int d_k,
    void *user_data
);

/**
 * @brief Parameters for Gemma 3 style attention (includes softcapping).
 */
typedef struct {
    float softcap;      /**< Logit softcap value (e.g., 50.0). 0.0 to disable. */
    float manual_scale; /**< Override scale factor. If > 0, replaces 1/sqrt(d_k). */
} gemma3_attention_params_t;

/**
 * @brief Configuration for attention scoring.
 */
typedef struct {
    AttentionScalingStrategy strategy; /**< Strategy function to apply. */
    int context_length;                /**< Length of the context (number of tokens). */
    int d_k;                           /**< Head dimension. */
    void *user_data;                   /**< Strategy-specific data. */
} attention_scoring_config_t;

/**
 * @brief Standard scaled dot-product attention strategy.
 * 
 * Scales scores by 1/sqrt(d_k), then applies softmax.
 * Formula: Softmax(Q·K^T / sqrt(d_k))
 * 
 * For Gemma 3, user_data can point to a gemma3_attention_params_t struct.
 * 
 * @param scores Raw dot-product scores. Modified in-place to contain softmax output.
 * @param context_length Number of tokens in the cache.
 * @param d_k Dimension of the attention head.
 * @param user_data Optional pointer to gemma3_attention_params_t.
 */
void scaled_dot_product_strategy(
    float *scores,
    int context_length,
    int d_k,
    void *user_data
);

/**
 * @brief Temperature-scaled attention strategy.
 * 
 * Scales scores by 1/temperature, then applies softmax.
 * Lower temperature → sharper distribution (concentrates attention).
 * Higher temperature → softer distribution (spreads attention).
 * Formula: Softmax(Q·K^T / temperature)
 * 
 * user_data should point to a float containing the temperature value (>0).
 * 
 * @param scores Raw dot-product scores. Modified in-place to contain softmax output.
 * @param context_length Number of tokens in the cache.
 * @param d_k Dimension of the attention head (unused for temperature).
 * @param user_data Pointer to float containing temperature value.
 */
void temperature_scaled_strategy(
    float *scores,
    int context_length,
    int d_k,
    void *user_data
);

/**
 * @brief ALiBi (Attention with Linear Biases) strategy.
 * 
 * Instead of scaling, applies position-dependent bias to attention scores.
 * ALiBi bias: bias[i, j] = -|i - j| * slope for position pair (i, j).
 * 
 * In this implementation, bias is not explicitly added (placeholder).
 * In production, you would track head index and apply proper ALiBi slopes.
 * Currently, this strategy applies softmax without scaling.
 * 
 * user_data can be NULL or point to ALiBi parameters.
 * 
 * @param scores Raw dot-product scores. Modified in-place to contain softmax output.
 * @param context_length Number of tokens in the cache.
 * @param d_k Dimension of the attention head.
 * @param user_data Pointer to ALiBi parameters (unused in placeholder version).
 */
void alibi_attention_strategy(
    float *scores,
    int context_length,
    int d_k,
    void *user_data
);

/**
 * @brief Generic attention score computation with pluggable scaling strategy.
 * 
 * Performs:
 * 1. Dot product: Q · K^T for all cached keys.
 * 2. Apply the chosen scaling/normalization strategy.
 * 
 * @param q The Query vector (d_k dimension).
 * @param kv_cache_k The Key cache (context_length x d_k, row-major).
 * @param output_scores Output attention weights (size: context_length). Modified in-place.
 * @param config Scoring configuration (strategy, dimensions, user data).
 */
void compute_attention_scores_with_strategy(
    const float *q,
    const float *kv_cache_k,
    float *output_scores,
    const attention_scoring_config_t *config
);

#endif // ATTENTION_STRATEGY_H

#include "attention.h"
#include "attention_strategy.h"
#include <stddef.h>

/**
 * @brief Convenience wrapper: Compute attention with standard scaled dot-product.
 */
void compute_attention_scores(
    const float *q,
    const float *kv_cache_k,
    int context_length,
    int d_k,
    float *output_scores
) {
    compute_attention_scores_with_strategy(
        q,
        kv_cache_k,
        context_length,
        d_k,
        scaled_dot_product_strategy,
        NULL,
        output_scores
    );
}

/**
 * @brief Convenience wrapper: Compute attention with temperature scaling.
 */
void compute_attention_scores_with_temperature(
    const float *q,
    const float *kv_cache_k,
    int context_length,
    int d_k,
    float temperature,
    float *output_scores
) {
    compute_attention_scores_with_strategy(
        q,
        kv_cache_k,
        context_length,
        d_k,
        temperature_scaled_strategy,
        (void *)&temperature,
        output_scores
    );
}

/**
 * @brief Convenience wrapper: Compute attention using ALiBi strategy.
 */
void compute_attention_scores_with_alibi(
    const float *q,
    const float *kv_cache_k,
    int context_length,
    int d_k,
    float *output_scores
) {
    compute_attention_scores_with_strategy(
        q,
        kv_cache_k,
        context_length,
        d_k,
        alibi_attention_strategy,
        NULL,
        output_scores
    );
}

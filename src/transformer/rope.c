#include "rope.h"
#include "positional_encoding.h"
#include <stddef.h>

#define ROPE_DEFAULT_BASE 10000.0f

/**
 * @brief Convenience wrapper: Apply RoPE with a custom base.
 * 
 * @param x The input vector. Modified in-place.
 * @param pos The position index of the token.
 * @param head_dim The dimension of a single attention head.
 * @param base The base for frequency calculation.
 */
void apply_rope(float *x, int pos, int head_dim, float base) {
    if (x == NULL) {
        return;
    }

    // Create a mutable copy of base to pass to the strategy
    float base_copy = base;
    apply_positional_encoding(x, pos, head_dim, rope_encoding_strategy, (void *)&base_copy);
}

/**
 * @brief Convenience wrapper: Apply RoPE with default base (10000.0f).
 * 
 * @param x The input vector. Modified in-place.
 * @param pos The position index of the token.
 * @param head_dim The dimension of a single attention head.
 */
void apply_rope_default(float *x, int pos, int head_dim) {
    apply_rope(x, pos, head_dim, ROPE_DEFAULT_BASE);
}

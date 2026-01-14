#ifndef ROPE_H
#define ROPE_H

#include "positional_encoding.h"

/**
 * @brief Convenience wrapper: Apply RoPE (Rotary Positional Embedding) to a vector.
 * 
 * This is a high-level API that uses the modular positional encoding system.
 * It applies RoPE using the standard base value (10000).
 * 
 * @param x The input vector (e.g., Query or Key). Modified in-place.
 * @param pos The position index (m) of the token being processed.
 * @param head_dim The dimension of a single attention head.
 * @param base The base for frequency calculation (typically 10000.0f).
 */
void apply_rope(float *x, int pos, int head_dim, float base);

/**
 * @brief Apply RoPE with default base (10000.0f).
 * 
 * @param x The input vector. Modified in-place.
 * @param pos The position index of the token.
 * @param head_dim The dimension of a single attention head.
 */
void apply_rope_default(float *x, int pos, int head_dim);

/**
 * @brief Precompute RoPE frequencies (cos and sin for all positions and dimensions).
 * 
 * @param freqs_cos Pointer to pre-allocated buffer for cosine frequencies.
 * @param freqs_sin Pointer to pre-allocated buffer for sine frequencies.
 * @param d_k The hidden dimension of a single head.
 * @param max_context_len Maximum context length to precompute for.
 * @param rope_base The base for frequency calculation.
 * @return 0 on success, non-zero on error.
 */
int rope_precompute_freqs(float* freqs_cos, float* freqs_sin,
                         int d_k, int max_context_len, float rope_base);

/**
 * @brief Fast RoPE application using precomputed frequencies.
 * 
 * @param x The input vector to modify.
 * @param pos The position index.
 * @param head_dim The head dimension.
 * @param freqs_cos Pointer to the precomputed cosine frequencies buffer.
 * @param freqs_sin Pointer to the precomputed sine frequencies buffer.
 */
void rope_apply_fast(float* x, int pos, int head_dim, 
                    const float* freqs_cos, const float* freqs_sin);

#endif // ROPE_H

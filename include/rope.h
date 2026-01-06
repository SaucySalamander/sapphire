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

#endif // ROPE_H

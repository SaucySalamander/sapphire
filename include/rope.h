/**
 * @file rope.h
 * @brief Rotary Positional Embedding (RoPE) and general positional encoding.
 *
 * This module is the exclusive owner of RoPE math, including dual-base (10k/1M)
 * support required for Gemma 3.
 */

#ifndef ROPE_H
#define ROPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// STRATEGY INTERFACE (Pluggable positional encodings)
// ============================================================================

/**
 * @brief Positional encoding strategy function type.
 */
typedef void (*PositionalEncodingStrategy)(
    int head_dim,
    int dim_pair,
    int pos,
    void *user_data,
    float *out_cos,
    float *out_sin
);

/**
 * @brief Generic positional encoding application using a pluggable strategy.
 */
void apply_positional_encoding(
    float *x,
    int pos,
    int head_dim,
    PositionalEncodingStrategy strategy,
    void *user_data
);

// ============================================================================
// ROPE IMPLEMENTATIONS
// ============================================================================

/**
 * @brief RoPE (Rotary Positional Embedding) encoding strategy.
 * Implements: theta_i = m * (1/base)^(2i/d)
 */
void rope_encoding_strategy(
    int head_dim,
    int dim_pair,
    int pos,
    void *user_data,
    float *out_cos,
    float *out_sin
);

/**
 * @brief Apply RoPE (Rotary Positional Embedding) to a vector.
 * @param x The input vector (modified in-place)
 * @param pos Position index
 * @param head_dim Hidden dimension of a single head
 * @param base Base for frequency calculation (typically 10000.0f or 1000000.0f)
 */
void apply_rope(float *x, int pos, int head_dim, float base);

/**
 * @brief Apply RoPE with default base (10000.0f).
 */
void apply_rope_default(float *x, int pos, int head_dim);

// ============================================================================
// PERFORMANCE OPTIMIZATIONS (Precomputation)
// ============================================================================

/**
 * @brief Precompute RoPE frequencies for all positions and dimensions.
 */
int rope_precompute_freqs(float* freqs_cos, float* freqs_sin,
                         int d_k, int max_context_len, float rope_base);

/**
 * @brief Fast RoPE application using precomputed frequencies.
 */
void rope_apply_fast(float* x, int pos, int head_dim, 
                    const float* freqs_cos, const float* freqs_sin);

// ============================================================================
// FALLBACKS / OTHER STRATEGIES
// ============================================================================

/**
 * @brief ALiBi (Attention with Linear Biases) placeholder strategy.
 */
void alibi_encoding_strategy(
    int head_dim,
    int dim_pair,
    int pos,
    void *user_data,
    float *out_cos,
    float *out_sin
);

#ifdef __cplusplus
}
#endif

#endif // ROPE_H

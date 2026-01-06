#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <stdint.h>

/**
 * @brief Positional encoding strategy function type.
 * 
 * Computes cosine and sine frequency values for a given dimension pair and position.
 * This allows swapping between different positional encoding schemes (RoPE, ALiBi, etc.)
 * without changing the application logic.
 * 
 * @param head_dim The dimension of a single attention head.
 * @param dim_pair The dimension pair index (0, 1, 2, ...) for pair-based encodings.
 * @param pos The token position.
 * @param user_data Pointer to strategy-specific parameters (e.g., base for RoPE, slope for ALiBi).
 * @param out_cos Output: cosine value for rotation (or 1.0 for identity).
 * @param out_sin Output: sine value for rotation (or 0.0 for identity).
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
 * @brief RoPE (Rotary Positional Embedding) encoding strategy.
 * 
 * Implements the formula: theta_i = m * (1/base)^(2i/d)
 * where m is the token position, i is the dimension index, d is the dimension size.
 * Returns cos(theta_i) and sin(theta_i) for vector rotation.
 * 
 * @param head_dim The dimension of a single attention head.
 * @param dim_pair The dimension pair index.
 * @param pos The token position.
 * @param user_data Should point to a float containing the base value (typically 10000.0f).
 * @param out_cos Output: cosine frequency.
 * @param out_sin Output: sine frequency.
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
 * @brief ALiBi (Attention with Linear Biases) encoding strategy.
 * 
 * ALiBi does not use vector rotation; instead, it applies positional bias directly
 * to attention scores. For compatibility with the rotation-based interface,
 * this strategy returns identity rotation (cos=1.0, sin=0.0).
 * 
 * Note: ALiBi bias should be applied in the attention computation, not here.
 * This is a placeholder for interface compatibility.
 * 
 * @param head_dim The dimension of a single attention head.
 * @param dim_pair The dimension pair index.
 * @param pos The token position.
 * @param user_data Pointer to ALiBi slope parameters (optional).
 * @param out_cos Output: always 1.0 (identity).
 * @param out_sin Output: always 0.0 (identity).
 */
void alibi_encoding_strategy(
    int head_dim,
    int dim_pair,
    int pos,
    void *user_data,
    float *out_cos,
    float *out_sin
);

/**
 * @brief Generic positional encoding application using a pluggable strategy.
 * 
 * Applies rotation to vector element pairs based on the chosen encoding strategy.
 * The rotation is applied to pairs (x[0], x[1]), (x[2], x[3]), etc.
 * 
 * Rotation matrix for each pair:
 *   [x'_j]     [cos -sin] [x_j]
 *   [x'_{j+1}] = [sin  cos] [x_{j+1}]
 * 
 * @param x The input vector. Modified in-place.
 * @param pos The token position.
 * @param head_dim The dimension of a single attention head.
 * @param strategy The encoding strategy function pointer.
 * @param user_data Strategy-specific parameters (passed to the strategy function).
 */
void apply_positional_encoding(
    float *x,
    int pos,
    int head_dim,
    PositionalEncodingStrategy strategy,
    void *user_data
);

#endif // POSITIONAL_ENCODING_H

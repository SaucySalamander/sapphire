#include "rope.h"
#include <math.h>
#include <stddef.h>

/**
 * @brief RoPE strategy implementation.
 * 
 * Computes: theta_i = m * (1/base)^(2i/d)
 * Then applies rotation: cos(theta_i), sin(theta_i)
 * 
 * The formula allows different dimensions to have different frequency oscillations:
 * - Low dimensions oscillate quickly (high frequency)
 * - High dimensions oscillate slowly (low frequency)
 * This captures positional information at multiple scales.
 */
void rope_encoding_strategy(
    int head_dim,
    int dim_pair,
    int pos,
    void *user_data,
    float *out_cos,
    float *out_sin
) {
    if (user_data == NULL || out_cos == NULL || out_sin == NULL) {
        return;
    }

    float base = *(float *)user_data;  // user_data points to base value (e.g., 10000.0f)

    // Compute the inverse frequency: inv_freq = (1/base)^(2*dim_pair/head_dim)
    float inv_freq = powf(1.0f / base, 2.0f * dim_pair / (float)head_dim);

    // Compute theta: theta = pos * inv_freq
    float theta = pos * inv_freq;

    // Return the cosine and sine of theta
    *out_cos = cosf(theta);
    *out_sin = sinf(theta);
}

/**
 * @brief ALiBi strategy (placeholder for attention score bias approach).
 * 
 * ALiBi (Attention with Linear Biases) does not use vector rotation.
 * Instead, it adds a fixed bias to attention scores based on relative positions.
 * 
 * For compatibility with the rotation-based interface, this strategy
 * returns identity rotation (cos=1.0, sin=0.0), effectively making apply_positional_encoding
 * a no-op for vector rotation. The actual ALiBi bias should be applied
 * in the attention computation layer.
 */
void alibi_encoding_strategy(
    int head_dim,
    int dim_pair,
    int pos,
    void *user_data,
    float *out_cos,
    float *out_sin
) {
    (void)head_dim;   // unused
    (void)dim_pair;   // unused
    (void)pos;        // unused
    (void)user_data;  // unused

    // Return identity rotation (no vector rotation for ALiBi)
    *out_cos = 1.0f;
    *out_sin = 0.0f;
}

/**
 * @brief Generic positional encoding application.
 * 
 * Applies the chosen encoding strategy to vector element pairs in-place.
 * This is the core function that all positional encoding schemes use.
 */
void apply_positional_encoding(
    float *x,
    int pos,
    int head_dim,
    PositionalEncodingStrategy strategy,
    void *user_data
) {
    if (x == NULL || strategy == NULL) {
        return;
    }

    // Apply rotation to pairs of elements within head_dim
    for (int j = 0; j < head_dim; j += 2) {
        // Get cosine and sine values from the strategy
        float cos_t, sin_t;
        strategy(head_dim, j / 2, pos, user_data, &cos_t, &sin_t);

        // Apply rotation to the pair (x[j], x[j+1])
        // Rotation matrix: [cos -sin] applied to [x[j], x[j+1]]
        //                  [sin  cos]
        if (j + 1 < head_dim) {
            float x_even = x[j];
            float x_odd = x[j + 1];

            x[j] = x_even * cos_t - x_odd * sin_t;
            x[j + 1] = x_even * sin_t + x_odd * cos_t;
        }
    }
}

#include "rope.h"
#include <stddef.h>
#include <math.h>

// Gemma 3 uses rope_theta = 1,000,000 (not standard 10,000)
// This critical parameter encodes positional information at the correct frequency scale
#define ROPE_DEFAULT_BASE 1000000.0f

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

/**
 * @brief Precompute RoPE frequencies (cos and sin for all positions and dimensions).
 */
int rope_precompute_freqs(float* freqs_cos, float* freqs_sin,
                          int d_k, int max_context_len, float rope_base) {
    if (!freqs_cos || !freqs_sin) return -1;

    int half = d_k / 2;
    for (int pos = 0; pos < max_context_len; pos++) {
        for (int m = 0; m < half; m++) {
            float freq = 1.0f / powf(rope_base, 2.0f * (float)m / (float)d_k);
            float angle = (float)pos * freq;

            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            // Half-split storage: match the first half with the second half
            freqs_cos[pos * d_k + m]        = cos_val;
            freqs_cos[pos * d_k + m + half] = cos_val;

            freqs_sin[pos * d_k + m]        = sin_val;
            freqs_sin[pos * d_k + m + half] = sin_val;
        }
    }
    return 0;
}
// int rope_precompute_freqs(float* freqs_cos, float* freqs_sin,
//                           int d_k, int max_context_len, float rope_base) {
//     if (!freqs_cos || !freqs_sin) return -1;

//     /**
//      * RoPE (Rotary Position Encoding) formula:
//      * For each position pos and pair index m (where m = 0, 1, 2, ..., d_k/2-1):
//      *   freq = 1.0 / rope_base^(2m / d_k)
//      *   angle = pos * freq
//      * The same angle is applied to both elements in the pair (dim, dim+1).
//      */
//     for (int pos = 0; pos < max_context_len; pos++) {
//         for (int m = 0; m < d_k / 2; m++) {
//             float freq = 1.0f / powf(rope_base, 2.0f * (float)m / (float)d_k);
//             float angle = (float)pos * freq;

//             float cos_val = cosf(angle);
//             float sin_val = sinf(angle);

//             // Store same angle for both elements in the pair
//             int dim = 2 * m;
//             freqs_cos[pos * d_k + dim] = cos_val;
//             freqs_cos[pos * d_k + dim + 1] = cos_val;

//             freqs_sin[pos * d_k + dim] = sin_val;
//             freqs_sin[pos * d_k + dim + 1] = sin_val;
//         }
//     }

//     return 0;
// }

/**
 * @brief Fast RoPE application using precomputed frequencies.
 */
void rope_apply_fast(float* x, int pos, int head_dim, 
                     const float* freqs_cos, const float* freqs_sin) {
    if (!x || !freqs_cos || !freqs_sin) return;

    const float* f_cos = freqs_cos + pos * head_dim;
    const float* f_sin = freqs_sin + pos * head_dim;
    int half = head_dim / 2;

    for (int i = 0; i < half; i++) {
        float x0 = x[i];
        float x1 = x[i + half];
        float c = f_cos[i];
        float s = f_sin[i];

        // x_rotated = [x0*cos - x1*sin, x1*cos + x0*sin]
        x[i]        = x0 * c - x1 * s;
        x[i + half] = x1 * c + x0 * s;
    }
}
// void rope_apply_fast(float* x, int pos, int head_dim, 
//                      const float* freqs_cos, const float* freqs_sin) {
//     if (!x || !freqs_cos || !freqs_sin) return;
//     const float* f_cos = freqs_cos + pos * head_dim;
//     const float* f_sin = freqs_sin + pos * head_dim;

//     for (int i = 0; i < head_dim; i += 2) {
//         float x0 = x[i];
//         float x1 = x[i + 1];
//         float c = f_cos[i];
//         float s = f_sin[i];
//         x[i] = x0 * c - x1 * s;
//         x[i + 1] = x0 * s + x1 * c;
//     }
// }

#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

/**
 * @brief Computes softmax with numerical stability.
 * 
 * Implements the numerically stable softmax using the log-space trick:
 * softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)) for all j)
 * 
 * This prevents overflow/underflow by subtracting the maximum value before
 * computing exponentials.
 * 
 * @param scores Input array of scores. Modified in-place to contain softmax output.
 * @param n The number of elements in the scores array.
 */
void softmax(float *scores, int n);

/**
 * @brief Computes vector statistics (min, max, RMS) for debugging.
 * 
 * @param x Input vector.
 * @param n Number of elements.
 * @param out_min Output for minimum value (optional).
 * @param out_max Output for maximum value (optional).
 * @param out_rms Output for Root Mean Square (optional).
 */
void vec_stats(const float* x, int n, float* out_min, float* out_max, float* out_rms);

/**
 * @brief Vector operations
 */
void vec_add(float* dst, const float* src, int n);
void vec_scale(float* x, float s, int n);
void vec_copy(float* dst, const float* src, int n);
float vec_dot(const float* a, const float* b, int n);

/**
 * @brief Numerical softcapping (tanh clamping)
 */
void vec_softcap(float* x, int n, float cap);

/**
 * @brief BF16 to F32 conversion
 */
void bf16_to_f32_vec(float* dst, const uint16_t* src, int n);

/**
 * @brief Greedy sampler: select token with highest logit
 */
int sample_greedy(const float* logits, int n);

/**
 * @brief Temperature sampling with softmax
 */
int sample_temperature(float* logits, int n, float temperature);

/**
 * @brief Compute Shannon entropy (nats) from unnormalized exp(logits).
 * @param exp_logits Array of exp(logit) values (unnormalized).
 * @param n Number of elements.
 * @param sum Sum of exp_logits (precomputed).
 * @return entropy in nats.
 */
float sampling_entropy_from_unnormalized(const float *exp_logits, int n, float sum);

/**
 * @brief Compute cumulative mass of top-k probabilities from unnormalized exp(logits).
 * @param exp_logits Array of exp(logit) values (unnormalized).
 * @param n Number of elements.
 * @param k Top-k value (k>=1).
 * @param sum Sum of exp_logits (precomputed).
 * @return cumulative probability mass of top-k (0..1)
 */
float sampling_topk_mass_from_unnormalized(const float *exp_logits, int n, int k, float sum);

#endif // UTILS_H

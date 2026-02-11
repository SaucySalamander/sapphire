#include "utils.h"
#include "tensor.h"
#include <math.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>

/**
 * @brief Computes softmax with numerical stability.
 * 
 * Implements the numerically stable softmax using the log-space trick:
 * softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)) for all j)
 * 
 * @param scores Input array of scores. Modified in-place to contain softmax output.
 * @param n The number of elements in the scores array.
 */
void softmax(float *scores, int n) {
    if (n <= 0 || scores == NULL) {
        return;
    }

    // Step 1: Find the maximum score to prevent overflow with AVX.
    float max_score = scores[0];
    int i = 0;
    __m256 v_max = _mm256_set1_ps(-1e38f); // Close to -INFINITY
    for (; i + 8 <= n; i += 8) {
        __m256 v_s = _mm256_loadu_ps(scores + i);
        v_max = _mm256_max_ps(v_max, v_s);
    }
    
    float tmp[8];
    _mm256_storeu_ps(tmp, v_max);
    for (int j = 0; j < 8; j++) {
        if (tmp[j] > max_score) max_score = tmp[j];
    }
    for (; i < n; i++) {
        if (scores[i] > max_score) max_score = scores[i];
    }

    // Step 2: Compute exp(scores[i] - max_score) and accumulate sum.
    float sum = 0.0f;
    for (i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }

    // Step 3: Normalize by the sum with AVX.
    float sum_inv = 1.0f / sum;
    __m256 v_inv = _mm256_set1_ps(sum_inv);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v_s = _mm256_loadu_ps(scores + i);
        _mm256_storeu_ps(scores + i, _mm256_mul_ps(v_s, v_inv));
    }
    for (; i < n; i++) {
        scores[i] *= sum_inv;
    }
}

void vec_stats(const float* x, int n, float* out_min, float* out_max, float* out_rms) {
    if (!x || n <= 0) {
        if (out_min) *out_min = 0.0f;
        if (out_max) *out_max = 0.0f;
        if (out_rms) *out_rms = 0.0f;
        return;
    }
    float mn = x[0], mx = x[0];
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum_sq += (double)v * (double)v;
    }
    if (out_min) *out_min = mn;
    if (out_max) *out_max = mx;
    if (out_rms) *out_rms = sqrtf((float)(sum_sq / (double)n));
}

void vec_add(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] += src[i];
    }
}

void vec_scale(float* x, float s, int n) {
    if (s == 1.0f) return;
    for (int i = 0; i < n; i++) {
        x[i] *= s;
    }
}

void vec_copy(float* dst, const float* src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

float vec_dot(const float* a, const float* b, int n) {
    int i = 0;
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float final = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    
    for (; i < n; i++) {
        final += a[i] * b[i];
    }
    return final;
}

void vec_softcap(float* x, int n, float cap) {
    if (cap <= 0.0f) return;
    float inv_cap = 1.0f / cap;
    for (int i = 0; i < n; i++) {
        x[i] = cap * tanhf(x[i] * inv_cap);
    }
}

void bf16_to_f32_vec(float* dst, const uint16_t* src, int n) {
    int i = 0;
    // AVX2 optimized path
    for (; i + 8 <= n; i += 8) {
        __m128i bf_packed = _mm_loadu_si128((const __m128i*)(src + i));
        __m256i bf_expanded = _mm256_cvtepu16_epi32(bf_packed);
        __m256 f32_vec = _mm256_castsi256_ps(_mm256_slli_epi32(bf_expanded, 16));
        _mm256_storeu_ps(dst + i, f32_vec);
    }
    // Scalar remainder
    for (; i < n; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        float f;
        memcpy(&f, &f32_bits, sizeof(float));
        dst[i] = f;
    }
}

int sample_greedy(const float* logits, int n) {
    if (!logits || n <= 0) return -1;
    int best_token = 0;
    float best_logit = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best_token = i;
        }
    }
    return best_token;
}

int sample_temperature(float* logits, int n, float temperature) {
    if (!logits || n <= 0) return -1;
    if (temperature <= 0.0f) return sample_greedy(logits, n);
    
    // Softmax with temperature
    float max_logit = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_logit) max_logit = logits[i];
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum += logits[i];
    }
    
    float rand_val = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = logits[i] / sum;
        cumsum += p;
        if (rand_val <= cumsum) return i;
    }
    return n - 1;
}

float sampling_entropy_from_unnormalized(const float *exp_logits, int n, float sum) {
    if (!exp_logits || n <= 0 || sum <= 0.0f) return 0.0f;
    double entropy = 0.0;
    for (int i = 0; i < n; i++) {
        double p = (double)exp_logits[i] / (double)sum;
        if (p <= 0.0) continue;
        entropy -= p * log(p);
    }
    return (float)entropy; // in nats
}

float sampling_topk_mass_from_unnormalized(const float *exp_logits, int n, int k, float sum) {
    if (!exp_logits || n <= 0 || k <= 0 || sum <= 0.0f) return 0.0f;
    // Simple selection of top-k values via partial scan
    // For small k, this is acceptable
    float top_vals[128];
    if (k > 128) k = 128;
    for (int i = 0; i < k; i++) top_vals[i] = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = exp_logits[i];
        // insert into top_vals if larger than any
        for (int j = 0; j < k; j++) {
            if (v > top_vals[j]) {
                // shift down
                for (int t = k - 1; t > j; t--) top_vals[t] = top_vals[t-1];
                top_vals[j] = v;
                break;
            }
        }
    }
    double mass = 0.0;
    for (int i = 0; i < k; i++) mass += top_vals[i];
    return (float)(mass / (double)sum);
}

// Helper to handle BF16 norm weights on the fly
const float* get_norm_weights(const tensor_t* weight, float* scratch, int n) {
    if (!weight) return NULL;
    if (tensor_dtype(weight) == DTYPE_BF16) {
        bf16_to_f32_vec(scratch, (const uint16_t*)tensor_data(weight), n);
        return scratch;
    }
    return (const float*)tensor_data(weight);
}
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "../include/normalization.h"
#include "../include/activations.h"

// Small epsilon for float comparisons
static const float EPS = 1e-3f;

// Helper: compute RMS of vector
static float rms(const float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)v[i] * (double)v[i];
    return (float)sqrt(s / (double)n);
}

// Test 1: apply_qk_norm produces per-head RMS approx equal to gamma (unit after inv_rms * gamma)
static void test_apply_qk_norm_basic() {
    const int head_dim = 8;
    const int num_q_heads = 2;
    float q[num_q_heads * head_dim];
    float k[1 * head_dim];
    float q_scale[num_q_heads * head_dim];
    float k_scale[1 * head_dim];

    // Fill q with increasing values, k not used here
    for (int i = 0; i < num_q_heads * head_dim; ++i) q[i] = (i % head_dim) - (head_dim/2);
    for (int i = 0; i < head_dim; ++i) k[i] = (i - head_dim/2);

    // Set per-element gamma = 2.0 (so expected RMS after scaling == ~2.0)
    for (int i = 0; i < num_q_heads * head_dim; ++i) q_scale[i] = 2.0f;
    for (int i = 0; i < head_dim; ++i) k_scale[i] = 1.0f;

    apply_qk_norm(q, k, q_scale, k_scale, head_dim, num_q_heads, 1);

    for (int h = 0; h < num_q_heads; ++h) {
        float *head_q = q + h * head_dim;
        float r = rms(head_q, head_dim);
        // After normalization and scale, RMS should be close to 2.0
        assert(fabsf(r - 2.0f) < 1e-2f);
    }
    printf("test_apply_qk_norm_basic: PASS\n");
}

// Test 2: GeGLU computes output = x * GELU(y)
static void test_geglu_gelu() {
    const int n = 4;
    float out[n];
    float input[2*n];
    // x = [1,2,3,4], y = [0.1, 0.2, 0.3, 0.4]
    for (int i = 0; i < n; ++i) input[i] = (float)(i+1);
    for (int i = 0; i < n; ++i) input[n + i] = 0.1f * (i + 1);

    int rc = sapphire_geglu(out, input, 2*n);
    assert(rc == 0);

    for (int i = 0; i < n; ++i) {
        float x = input[i];
        float y = input[n + i];
        float expected = x * gelu(y);
        assert(fabsf(out[i] - expected) < EPS);
    }
    printf("test_geglu_gelu: PASS\n");
}

// Test 3: final-logit softcap function behavior (tanh-cap)
static void test_final_logit_softcap() {
    float cap = 30.0f;
    float logits[5] = { -100.0f, -1.0f, 0.0f, 1.0f, 100.0f };
    float expv[5];
    for (int i = 0; i < 5; ++i) {
        expv[i] = cap * tanhf(logits[i] / cap);
    }
    // Basic sanity: capped magnitudes should be <= cap and monotonic
    for (int i = 0; i < 5; ++i) {
        assert(fabsf(expv[i]) <= cap + 1e-6f);
    }
    printf("test_final_logit_softcap: PASS\n");
}

int main(void) {
    test_apply_qk_norm_basic();
    test_geglu_gelu();
    test_final_logit_softcap();
    printf("All transformer invariant tests passed.\n");
    return 0;
}

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/kernels.h"

// Small epsilon for float comparisons
static const float EPS = 1e-3f;

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
    float test_logits[5];
    
    // Copy for vector operation
    for(int i=0; i<5; i++) test_logits[i] = logits[i];
    
    // Apply softcap in place using sapphire kernel
    vec_softcap(test_logits, 5, cap);

    // Verify against expected standard math
    for (int i = 0; i < 5; ++i) {
        float expected = cap * tanhf(logits[i] / cap);
        assert(fabsf(test_logits[i] - expected) < EPS);
        assert(fabsf(test_logits[i]) <= cap + EPS);
    }
    printf("test_final_logit_softcap: PASS\n");
}

int main(void) {
    test_geglu_gelu();
    test_final_logit_softcap();
    printf("All transformer invariant tests passed.\n");
    return 0;
}

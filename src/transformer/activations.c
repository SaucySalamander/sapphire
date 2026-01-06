#include <math.h>
#include "activations.h"

// ============================================================================
// SiLU (Sigmoid Linear Unit / Swish)
// ============================================================================

/**
 * SiLU: x * sigmoid(x) = x / (1 + exp(-x))
 * 
 * Implementation notes:
 * - For large positive x: sigmoid(x) ≈ 1, so output ≈ x
 * - For large negative x: sigmoid(x) ≈ 0, so output ≈ 0
 * - Smooth gradient everywhere (unlike ReLU)
 */
float silu(float x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    float sigmoid = 1.0f / (1.0f + expf(-x));
    return x * sigmoid;
}

void silu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = silu(x[i]);
    }
}

// ============================================================================
// ReLU (Rectified Linear Unit)
// ============================================================================

/**
 * ReLU: max(0, x)
 * 
 * Fast and simple. Breaks linearity at x=0 which helps model learn non-linear features.
 */
float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

void relu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = relu(x[i]);
    }
}

// ============================================================================
// GELU (Gaussian Error Linear Unit)
// ============================================================================

/**
 * GELU approximation:
 * 
 * Exact formula (requires erf):
 *   output = 0.5 * x * (1 + erf(x / sqrt(2)))
 * 
 * Approximation (used here for efficiency):
 *   output ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * 
 * This approximation is accurate to ~0.01 and avoids the erf call.
 * 
 * Reference: Hendrycks & Gimpel (2016) "Gaussian Error Linear Units"
 */
float gelu(float x) {
    // Coefficient: sqrt(2/π) ≈ 0.7978845608...
    static const float GELU_COEF_A = 0.7978845608f;
    // Coefficient for cubic term
    static const float GELU_COEF_B = 0.044715f;
    
    // tanh approximation: tanh(y) = (exp(2y) - 1) / (exp(2y) + 1)
    float y = GELU_COEF_A * (x + GELU_COEF_B * x * x * x);
    float tanh_y = tanhf(y);
    
    // output = 0.5 * x * (1 + tanh_y)
    return 0.5f * x * (1.0f + tanh_y);
}

void gelu_inplace(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = gelu(x[i]);
    }
}

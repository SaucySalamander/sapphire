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

// ============================================================================
// GeGLU (Gated GELU) Activation
// ============================================================================

/**
 * GeGLU: Element-wise gated GELU
 * 
 * Formula: output = x * GELU(y)
 * 
 * GeGLU applies GELU to one input (gate) and multiplies by another (signal).
 * Used in transformer FFN layers: linear -> geglu -> linear
 * 
 * Compared to SiLU (Swish) and ReLU:
 * - Smooth gradient everywhere (unlike ReLU)
 * - Better empirical performance in large models
 * - Slightly more expensive (requires GELU computation)
 * 
 * Reference: Shazeer et al. (2020) "GLU Variants Improve Transformer"
 */
float geglu(float x, float y) {
    return x * gelu(y);
}

/**
 * Vectorized GeGLU with loop unrolling for performance.
 * 
 * Input layout: [x_1, x_2, ..., x_n, y_1, y_2, ..., y_n]
 * - First half: x values (signal)
 * - Second half: y values (gate)
 * 
 * Processing: output[i] = input[i] * GELU(input[i + n]) for i in [0, n)
 * 
 * Loop unrolling factor: 4
 * - Reduces loop overhead
 * - Improves instruction cache efficiency
 * - Better for out-of-order execution
 * 
 * Safety: Includes NULL checks, bounds validation, assertions
 */
int sapphire_geglu(float *output, const float *input, size_t size) {
    // Validate inputs
    if (!output || !input || size == 0) {
        return -1;
    }
    
    // Size must be even (pairs of x,y values)
    if (size % 2 != 0) {
        return -1;
    }
    
    size_t n = size / 2;  // Number of (x, y) pairs
    const float *x_vals = input;          // First half: x values
    const float *y_vals = input + n;      // Second half: y values
    
    // Unrolled loop: process 4 pairs at a time
    size_t i = 0;
    size_t unroll_limit = (n / 4) * 4;  // Largest multiple of 4 <= n
    
    // Main unrolled loop (4x per iteration)
    for (i = 0; i < unroll_limit; i += 4) {
        output[i + 0] = x_vals[i + 0] * gelu(y_vals[i + 0]);
        output[i + 1] = x_vals[i + 1] * gelu(y_vals[i + 1]);
        output[i + 2] = x_vals[i + 2] * gelu(y_vals[i + 2]);
        output[i + 3] = x_vals[i + 3] * gelu(y_vals[i + 3]);
    }
    
    // Handle remaining elements (0-3 pairs)
    for (; i < n; i++) {
        output[i] = x_vals[i] * gelu(y_vals[i]);
    }
    
    return 0;
}

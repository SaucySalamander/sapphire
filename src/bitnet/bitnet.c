#include <transformer.h>
#include <string.h> // for memset
#include <math.h>
#include <stdint.h>

static inline int8_t signf(float x) {
    if (x > 0.0f) return 1;
    if (x < 0.0f) return -1;
    return 0;
}

// Forward Pass (per-row scale, ternary quantization)
void bitlinear_forward(float* out, const float* input, const float* weights_fp, int n, int d) {
    const float eps = 1e-8f;

    for (int i = 0; i < n; i++) {
        // per-row gamma (mean absolute value)
        float gamma = 0.0f;
        for (int j = 0; j < d; j++) {
            gamma += fabsf(weights_fp[i * d + j]);
        }
        gamma /= (float)d;
        gamma = fmaxf(gamma, eps);

        // accumulate
        float acc = 0.0f;
        for (int j = 0; j < d; j++) {
            float scaled_w = weights_fp[i * d + j] / gamma;
            int8_t w_quant = 0;
            if (scaled_w > 0.5f) w_quant = 1;
            else if (scaled_w < -0.5f) w_quant = -1;

            if (w_quant == 1) acc += input[j];
            else if (w_quant == -1) acc -= input[j];
        }
        out[i] = acc * gamma;
    }
}

// Backward Pass (STE for weight gradients)
void bitlinear_backward(float* d_input, float* d_weights_fp, const float* d_out,
                        const float* input, const float* weights_fp, int n, int d) {
    const float eps = 1e-8f;

    // zero gradients before accumulation
    memset(d_input, 0, d * sizeof(float));
    // caller may zero d_weights_fp; we'll zero here to be safe
    memset(d_weights_fp, 0, n * d * sizeof(float));

    for (int i = 0; i < n; i++) {
        float grad_incoming = d_out[i];

        // per-row gamma
        float gamma = 0.0f;
        for (int j = 0; j < d; j++) {
            gamma += fabsf(weights_fp[i * d + j]);
        }
        gamma /= (float)d;
        gamma = fmaxf(gamma, eps);

        for (int j = 0; j < d; j++) {
            float w = weights_fp[i * d + j];
            float scaled_w = w / gamma;
            int8_t w_quant = 0;
            if (scaled_w > 0.5f) w_quant = 1;
            else if (scaled_w < -0.5f) w_quant = -1;

            // gradient w.r.t. input via STE
            if (w_quant == 1) {
                d_input[j] += grad_incoming * gamma;
            } else if (w_quant == -1) {
                d_input[j] -= grad_incoming * gamma;
            }

            // straight-through estimator for weight gradients
            // approximate d(out)/d(w) ~= grad_incoming * input[j]
            float d_w = grad_incoming * input[j];
            d_weights_fp[i * d + j] += d_w;
        }
    }
}
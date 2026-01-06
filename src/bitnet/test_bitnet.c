#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// forward/backward declarations from bitnet.c
void bitlinear_forward(float* out, const float* input, const float* weights_fp, int n, int d);
void bitlinear_backward(float* d_input, float* d_weights_fp, const float* d_out,
                        const float* input, const float* weights_fp, int n, int d);

int main(void) {
    int n = 4; // output dim
    int d = 8; // input dim

    float *input = malloc(d * sizeof(float));
    float *weights = malloc(n * d * sizeof(float));
    float *out = malloc(n * sizeof(float));

    // init input and weights
    for (int i = 0; i < d; ++i) input[i] = (i % 2) ? 1.0f : -1.0f;
    for (int i = 0; i < n * d; ++i) weights[i] = ((i % 3) - 1) * 0.3f + 0.1f; // small variety

    bitlinear_forward(out, input, weights, n, d);

    printf("forward output:\n");
    for (int i = 0; i < n; ++i) printf(" out[%d] = %f\n", i, out[i]);

    // backward
    float *d_input = malloc(d * sizeof(float));
    float *d_weights = malloc(n * d * sizeof(float));
    float *d_out = malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) d_out[i] = 1.0f; // simple gradient

    bitlinear_backward(d_input, d_weights, d_out, input, weights, n, d);

    printf("\nbackward d_input:\n");
    for (int i = 0; i < d; ++i) printf(" d_input[%d] = %f\n", i, d_input[i]);

    printf("\nbackward d_weights (first row):\n");
    for (int j = 0; j < d; ++j) printf(" d_w[0,%d] = %f\n", j, d_weights[j]);

    free(input);
    free(weights);
    free(out);
    free(d_input);
    free(d_weights);
    free(d_out);

    return 0;
}

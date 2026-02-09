#include "../../include/kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    const int blocks = 4;
    const int block_size = 32;

    ggml_block_q4_0 *row = (ggml_block_q4_0*)calloc(blocks, sizeof(ggml_block_q4_0));
    if (!row) return 2;

    for (int b = 0; b < blocks; ++b) {
        row[b].scale = 0.05f * (1.0f + b);
        for (int i = 0; i < 16; ++i) {
            uint8_t lo = (uint8_t)(i & 0x0F);
            uint8_t hi = (uint8_t)((i + 1) & 0x0F);
            row[b].q_data[i] = (hi << 4) | (lo & 0x0F);
        }
    }

    float *x = (float*)aligned_alloc(32, sizeof(float) * blocks * block_size);
    if (!x) return 3;
    for (int i = 0; i < blocks * block_size; ++i) x[i] = 0.001f * (float)(i + 1);

    float s = quantized_gemv_q4_0_unaligned(row, x, blocks, block_size);
    float a = quantized_gemv_q4_0_unaligned(row, x, blocks, block_size);
    float aa = quantized_gemv_q4_0_aligned(row, x, blocks, block_size);

    printf("unaligned1=%.8f unaligned2=%.8f aligned=%.8f\n", s, a, aa);
    float d1 = fabsf(s - a);
    float d2 = fabsf(s - aa);

    free(row);
    free(x);

    if (d1 > 1e-4f || d2 > 1e-4f) {
        printf("TEST FAILED diffs: %f %f\n", d1, d2);
        return 1;
    }
    printf("TEST PASSED\n");
    return 0;
}

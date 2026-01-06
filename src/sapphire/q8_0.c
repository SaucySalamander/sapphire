// q8_0.c - AVX2-optimized Q8_0 kernels (aligned + unaligned)

#define _POSIX_C_SOURCE 200809L
#include "../../include/sapphire.h"

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Process one row consisting of block_count blocks, each block contains 32 int8 weights.
// Unaligned version uses _mm256_loadu_ps for x loads; aligned version uses _mm256_load_ps.

static inline __m256 load_x_unaligned(const float *x, int idx) {
    return _mm256_loadu_ps(x + idx);
}

static inline __m256 load_x_aligned(const float *x, int idx) {
    return _mm256_load_ps(x + idx);
}

float quantized_gemv_q8_0_unaligned(const ggml_block_q8_0 *W_row, const float *x, int block_count, int block_size) {
    if (block_size != 32) return 0.0f;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int b = 0; b < block_count; ++b) {
        const ggml_block_q8_0 *blk = &W_row[b];
        // each block has 32 int8 weights; process in 4 groups of 8
        const int base = b * 32;

        // load x (unaligned)
        __m256 xv0 = load_x_unaligned(x, base + 0);
        __m256 xv1 = load_x_unaligned(x, base + 8);
        __m256 xv2 = load_x_unaligned(x, base + 16);
        __m256 xv3 = load_x_unaligned(x, base + 24);

        // load 8 int8s into 128-bit, then sign-extend to 32-bit ints in 256-bit lanes
    (void)0; // keep consistent style; reading bytes handled via memcpy

    // process first 8 (bytes 0..7)
    uint64_t tmp0;
    memcpy(&tmp0, blk->q_data + 0, sizeof(uint64_t));
    __m128i eight0 = _mm_cvtsi64_si128((long long)tmp0);
        __m256i wi0 = _mm256_cvtepi8_epi32(eight0);
        __m256 wf0 = _mm256_cvtepi32_ps(wi0);
        __m256 scale_v = _mm256_set1_ps(blk->scale);
        __m256 w0 = _mm256_mul_ps(wf0, scale_v);
        acc0 = _mm256_fmadd_ps(w0, xv0, acc0);

        // bytes 8..15
        uint64_t tmp1;
        memcpy(&tmp1, blk->q_data + 8, sizeof(uint64_t));
        __m128i eight1 = _mm_cvtsi64_si128((long long)tmp1);
        __m256i wi1 = _mm256_cvtepi8_epi32(eight1);
        __m256 wf1 = _mm256_cvtepi32_ps(wi1);
        __m256 w1 = _mm256_mul_ps(wf1, scale_v);
        acc1 = _mm256_fmadd_ps(w1, xv1, acc1);

        // bytes 16..23
        uint64_t tmp2;
        memcpy(&tmp2, blk->q_data + 16, sizeof(uint64_t));
        __m128i eight2 = _mm_cvtsi64_si128((long long)tmp2);
        __m256i wi2 = _mm256_cvtepi8_epi32(eight2);
        __m256 wf2 = _mm256_cvtepi32_ps(wi2);
        __m256 w2 = _mm256_mul_ps(wf2, scale_v);
        acc2 = _mm256_fmadd_ps(w2, xv2, acc2);

        // bytes 24..31
        uint64_t tmp3;
        memcpy(&tmp3, blk->q_data + 24, sizeof(uint64_t));
        __m128i eight3 = _mm_cvtsi64_si128((long long)tmp3);
        __m256i wi3 = _mm256_cvtepi8_epi32(eight3);
        __m256 wf3 = _mm256_cvtepi32_ps(wi3);
        __m256 w3 = _mm256_mul_ps(wf3, scale_v);
        acc3 = _mm256_fmadd_ps(w3, xv3, acc3);
    }

    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum = _mm256_add_ps(sum01, sum23);

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float final = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    return final;
}

float quantized_gemv_q8_0_aligned(const ggml_block_q8_0 *W_row, const float *x, int block_count, int block_size) {
    if (block_size != 32) return 0.0f;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int b = 0; b < block_count; ++b) {
        const ggml_block_q8_0 *blk = &W_row[b];
        const int base = b * 32;

        __m256 xv0 = load_x_aligned(x, base + 0);
        __m256 xv1 = load_x_aligned(x, base + 8);
        __m256 xv2 = load_x_aligned(x, base + 16);
        __m256 xv3 = load_x_aligned(x, base + 24);

        uint64_t tmp0; memcpy(&tmp0, blk->q_data + 0, sizeof(uint64_t));
        uint64_t tmp1; memcpy(&tmp1, blk->q_data + 8, sizeof(uint64_t));
        uint64_t tmp2; memcpy(&tmp2, blk->q_data + 16, sizeof(uint64_t));
        uint64_t tmp3; memcpy(&tmp3, blk->q_data + 24, sizeof(uint64_t));

        __m128i eight0 = _mm_cvtsi64_si128((long long)tmp0);
        __m128i eight1 = _mm_cvtsi64_si128((long long)tmp1);
        __m128i eight2 = _mm_cvtsi64_si128((long long)tmp2);
        __m128i eight3 = _mm_cvtsi64_si128((long long)tmp3);

        __m256i wi0 = _mm256_cvtepi8_epi32(eight0);
        __m256i wi1 = _mm256_cvtepi8_epi32(eight1);
        __m256i wi2 = _mm256_cvtepi8_epi32(eight2);
        __m256i wi3 = _mm256_cvtepi8_epi32(eight3);

        __m256 wf0 = _mm256_cvtepi32_ps(wi0);
        __m256 wf1 = _mm256_cvtepi32_ps(wi1);
        __m256 wf2 = _mm256_cvtepi32_ps(wi2);
        __m256 wf3 = _mm256_cvtepi32_ps(wi3);

        __m256 scale_v = _mm256_set1_ps(blk->scale);
        __m256 w0 = _mm256_mul_ps(wf0, scale_v);
        __m256 w1 = _mm256_mul_ps(wf1, scale_v);
        __m256 w2 = _mm256_mul_ps(wf2, scale_v);
        __m256 w3 = _mm256_mul_ps(wf3, scale_v);

        acc0 = _mm256_fmadd_ps(w0, xv0, acc0);
        acc1 = _mm256_fmadd_ps(w1, xv1, acc1);
        acc2 = _mm256_fmadd_ps(w2, xv2, acc2);
        acc3 = _mm256_fmadd_ps(w3, xv3, acc3);
    }

    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum = _mm256_add_ps(sum01, sum23);

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float final = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    return final;
}

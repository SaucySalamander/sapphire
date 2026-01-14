// q4_0.c - Q4_0 kernel implementations (scalar + AVX2 aligned/unaligned)
// Model-agnostic: takes void *W_row (opaque weight block data)

#define _POSIX_C_SOURCE 200809L
#include "../../include/sapphire.h"

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/**
 * Q4_0 block structure (local definition for decoupling from ggml)
 * Each block contains:
 *   - scale: float (4 bytes)
 *   - q_data: 16 bytes of quantized 4-bit values (32 x 4-bit = 16 bytes)
 * Total: 20 bytes per block
 */
typedef struct {
    float scale;
    uint8_t q_data[16];
} q4_0_block_t;

// Note: AVX2 + FMA unaligned implementation
// W_row: opaque void pointer to Q4_0 block array
float quantized_gemv_q4_0_unaligned(const void *W_row, const float *x, int block_count, int block_size) {
    if (block_size != 32 || !W_row || !x) return 0.0f;
    
    const q4_0_block_t *blocks = (const q4_0_block_t *)W_row;
    const int offset = 8;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    int b = 0;
    const __m128i mask0f = _mm_set1_epi8((char)0x0F);
    for (; b + 1 < block_count; b += 2) {
        const q4_0_block_t *blk0 = &blocks[b];
        __m128i packed0 = _mm_loadu_si128((const __m128i*)blk0->q_data);
        __m128i low0 = _mm_and_si128(packed0, mask0f);
        __m128i high0 = _mm_and_si128(_mm_srli_epi16(packed0, 4), mask0f);
        __m128i ilo0 = _mm_unpacklo_epi8(low0, high0);
        __m128i ihi0 = _mm_unpackhi_epi8(low0, high0);
        const float *xptr0 = x + b*32;
        __m256 offset_v0 = _mm256_set1_ps((float)offset);
        __m256 scale_v0 = _mm256_set1_ps(blk0->scale);

        _mm_prefetch((const char*)(xptr0 + 16), _MM_HINT_T0);
        int aligned0 = (((uintptr_t)(const void*)xptr0) & 31) == 0;
        __m256 xv0 = aligned0 ? _mm256_load_ps(xptr0 + 0)  : _mm256_loadu_ps(xptr0 + 0);
        __m256 xv1 = aligned0 ? _mm256_load_ps(xptr0 + 8)  : _mm256_loadu_ps(xptr0 + 8);
        __m256 xv2 = aligned0 ? _mm256_load_ps(xptr0 + 16) : _mm256_loadu_ps(xptr0 + 16);
        __m256 xv3 = aligned0 ? _mm256_load_ps(xptr0 + 24) : _mm256_loadu_ps(xptr0 + 24);

        __m128i g0_0 = ilo0;
        __m256i wi0 = _mm256_cvtepu8_epi32(g0_0);
        __m256 wf0 = _mm256_cvtepi32_ps(wi0);
        __m256 w0 = _mm256_mul_ps(_mm256_sub_ps(wf0, offset_v0), scale_v0);
        acc0 = _mm256_fmadd_ps(w0, xv0, acc0);

        __m128i g1_0 = _mm_srli_si128(ilo0, 8);
        __m256i wi1 = _mm256_cvtepu8_epi32(g1_0);
        __m256 wf1 = _mm256_cvtepi32_ps(wi1);
        __m256 w1 = _mm256_mul_ps(_mm256_sub_ps(wf1, offset_v0), scale_v0);
        acc1 = _mm256_fmadd_ps(w1, xv1, acc1);

        __m128i g2_0 = ihi0;
        __m256i wi2 = _mm256_cvtepu8_epi32(g2_0);
        __m256 wf2 = _mm256_cvtepi32_ps(wi2);
        __m256 w2 = _mm256_mul_ps(_mm256_sub_ps(wf2, offset_v0), scale_v0);
        acc2 = _mm256_fmadd_ps(w2, xv2, acc2);

        __m128i g3_0 = _mm_srli_si128(ihi0, 8);
        __m256i wi3 = _mm256_cvtepu8_epi32(g3_0);
        __m256 wf3 = _mm256_cvtepi32_ps(wi3);
        __m256 w3 = _mm256_mul_ps(_mm256_sub_ps(wf3, offset_v0), scale_v0);
        acc3 = _mm256_fmadd_ps(w3, xv3, acc3);

        const q4_0_block_t *blk1 = &blocks[b+1];
        __m128i packed1 = _mm_loadu_si128((const __m128i*)blk1->q_data);
        __m128i low1 = _mm_and_si128(packed1, mask0f);
        __m128i high1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), mask0f);
        __m128i ilo1 = _mm_unpacklo_epi8(low1, high1);
        __m128i ihi1 = _mm_unpackhi_epi8(low1, high1);
        const float *xptr1 = x + (b+1)*32;
        __m256 offset_v1 = _mm256_set1_ps((float)offset);
        __m256 scale_v1 = _mm256_set1_ps(blk1->scale);

        _mm_prefetch((const char*)(xptr1 + 16), _MM_HINT_T0);
        int aligned1 = (((uintptr_t)(const void*)xptr1) & 31) == 0;
        __m256 xv4 = aligned1 ? _mm256_load_ps(xptr1 + 0)  : _mm256_loadu_ps(xptr1 + 0);
        __m256 xv5 = aligned1 ? _mm256_load_ps(xptr1 + 8)  : _mm256_loadu_ps(xptr1 + 8);
        __m256 xv6 = aligned1 ? _mm256_load_ps(xptr1 + 16) : _mm256_loadu_ps(xptr1 + 16);
        __m256 xv7 = aligned1 ? _mm256_load_ps(xptr1 + 24) : _mm256_loadu_ps(xptr1 + 24);

        __m128i g0_1 = ilo1;
        __m256i wi4 = _mm256_cvtepu8_epi32(g0_1);
        __m256 wf4 = _mm256_cvtepi32_ps(wi4);
        __m256 w4 = _mm256_mul_ps(_mm256_sub_ps(wf4, offset_v1), scale_v1);
        acc4 = _mm256_fmadd_ps(w4, xv4, acc4);

        __m128i g1_1 = _mm_srli_si128(ilo1, 8);
        __m256i wi5 = _mm256_cvtepu8_epi32(g1_1);
        __m256 wf5 = _mm256_cvtepi32_ps(wi5);
        __m256 w5 = _mm256_mul_ps(_mm256_sub_ps(wf5, offset_v1), scale_v1);
        acc5 = _mm256_fmadd_ps(w5, xv5, acc5);

        __m128i g2_1 = ihi1;
        __m256i wi6 = _mm256_cvtepu8_epi32(g2_1);
        __m256 wf6 = _mm256_cvtepi32_ps(wi6);
        __m256 w6 = _mm256_mul_ps(_mm256_sub_ps(wf6, offset_v1), scale_v1);
        acc6 = _mm256_fmadd_ps(w6, xv6, acc6);

        __m128i g3_1 = _mm_srli_si128(ihi1, 8);
        __m256i wi7 = _mm256_cvtepu8_epi32(g3_1);
        __m256 wf7 = _mm256_cvtepi32_ps(wi7);
        __m256 w7 = _mm256_mul_ps(_mm256_sub_ps(wf7, offset_v1), scale_v1);
        acc7 = _mm256_fmadd_ps(w7, xv7, acc7);
    }

    for (; b < block_count; ++b) {
        const q4_0_block_t *blk = &blocks[b];
        __m128i packed = _mm_loadu_si128((const __m128i*)blk->q_data);
        __m128i low = _mm_and_si128(packed, mask0f);
        __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask0f);
        __m128i ilo = _mm_unpacklo_epi8(low, high);
        __m128i ihi = _mm_unpackhi_epi8(low, high);
        const float *xptr = x + b*32;
        __m256 offset_v = _mm256_set1_ps((float)offset);
        __m256 scale_v = _mm256_set1_ps(blk->scale);

        _mm_prefetch((const char*)(xptr + 16), _MM_HINT_T0);
        int aligned = (((uintptr_t)(const void*)xptr) & 31) == 0;
        __m256 xv0 = aligned ? _mm256_load_ps(xptr + 0)  : _mm256_loadu_ps(xptr + 0);
        __m256 xv1 = aligned ? _mm256_load_ps(xptr + 8)  : _mm256_loadu_ps(xptr + 8);
        __m256 xv2 = aligned ? _mm256_load_ps(xptr + 16) : _mm256_loadu_ps(xptr + 16);
        __m256 xv3 = aligned ? _mm256_load_ps(xptr + 24) : _mm256_loadu_ps(xptr + 24);

        __m128i g0 = ilo;
        __m256i wi0 = _mm256_cvtepu8_epi32(g0);
        __m256 wf0 = _mm256_cvtepi32_ps(wi0);
        __m256 w0 = _mm256_mul_ps(_mm256_sub_ps(wf0, offset_v), scale_v);
        acc0 = _mm256_fmadd_ps(w0, xv0, acc0);

        __m128i g1 = _mm_srli_si128(ilo, 8);
        __m256i wi1 = _mm256_cvtepu8_epi32(g1);
        __m256 wf1 = _mm256_cvtepi32_ps(wi1);
        __m256 w1 = _mm256_mul_ps(_mm256_sub_ps(wf1, offset_v), scale_v);
        acc1 = _mm256_fmadd_ps(w1, xv1, acc1);

        __m128i g2 = ihi;
        __m256i wi2 = _mm256_cvtepu8_epi32(g2);
        __m256 wf2 = _mm256_cvtepi32_ps(wi2);
        __m256 w2 = _mm256_mul_ps(_mm256_sub_ps(wf2, offset_v), scale_v);
        acc2 = _mm256_fmadd_ps(w2, xv2, acc2);

        __m128i g3 = _mm_srli_si128(ihi, 8);
        __m256i wi3 = _mm256_cvtepu8_epi32(g3);
        __m256 wf3 = _mm256_cvtepi32_ps(wi3);
        __m256 w3 = _mm256_mul_ps(_mm256_sub_ps(wf3, offset_v), scale_v);
        acc3 = _mm256_fmadd_ps(w3, xv3, acc3);
    }

    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum45 = _mm256_add_ps(acc4, acc5);
    __m256 sum67 = _mm256_add_ps(acc6, acc7);
    __m256 sum0123 = _mm256_add_ps(sum01, sum23);
    __m256 sum4567 = _mm256_add_ps(sum45, sum67);
    __m256 sum = _mm256_add_ps(sum0123, sum4567);

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float final = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    return final;
}

// Aligned fast-path
// W_row: opaque void pointer to Q4_0 block array
float quantized_gemv_q4_0_aligned(const void *W_row, const float *x, int block_count, int block_size) {
    if (block_size != 32 || !W_row || !x) return 0.0f;
    
    const q4_0_block_t *blocks = (const q4_0_block_t *)W_row;
    const int offset = 8;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    int b = 0;
    const __m128i mask0f = _mm_set1_epi8((char)0x0F);
    for (; b + 1 < block_count; b += 2) {
        const q4_0_block_t *blk0 = &blocks[b];
        __m128i packed0 = _mm_loadu_si128((const __m128i*)blk0->q_data);
        __m128i low0 = _mm_and_si128(packed0, mask0f);
        __m128i high0 = _mm_and_si128(_mm_srli_epi16(packed0, 4), mask0f);
        __m128i ilo0 = _mm_unpacklo_epi8(low0, high0);
        __m128i ihi0 = _mm_unpackhi_epi8(low0, high0);
        const float *xptr0 = x + b*32;
        __m256 offset_v0 = _mm256_set1_ps((float)offset);
        __m256 scale_v0 = _mm256_set1_ps(blk0->scale);

        __m256 xv0 = _mm256_load_ps(xptr0 + 0);
        __m256 xv1 = _mm256_load_ps(xptr0 + 8);
        __m256 xv2 = _mm256_load_ps(xptr0 + 16);
        __m256 xv3 = _mm256_load_ps(xptr0 + 24);

        __m128i g0_0 = ilo0;
        __m256i wi0 = _mm256_cvtepu8_epi32(g0_0);
        __m256 wf0 = _mm256_cvtepi32_ps(wi0);
        __m256 w0 = _mm256_mul_ps(_mm256_sub_ps(wf0, offset_v0), scale_v0);
        acc0 = _mm256_fmadd_ps(w0, xv0, acc0);

        __m128i g1_0 = _mm_srli_si128(ilo0, 8);
        __m256i wi1 = _mm256_cvtepu8_epi32(g1_0);
        __m256 wf1 = _mm256_cvtepi32_ps(wi1);
        __m256 w1 = _mm256_mul_ps(_mm256_sub_ps(wf1, offset_v0), scale_v0);
        acc1 = _mm256_fmadd_ps(w1, xv1, acc1);

        __m128i g2_0 = ihi0;
        __m256i wi2 = _mm256_cvtepu8_epi32(g2_0);
        __m256 wf2 = _mm256_cvtepi32_ps(wi2);
        __m256 w2 = _mm256_mul_ps(_mm256_sub_ps(wf2, offset_v0), scale_v0);
        acc2 = _mm256_fmadd_ps(w2, xv2, acc2);

        __m128i g3_0 = _mm_srli_si128(ihi0, 8);
        __m256i wi3 = _mm256_cvtepu8_epi32(g3_0);
        __m256 wf3 = _mm256_cvtepi32_ps(wi3);
        __m256 w3 = _mm256_mul_ps(_mm256_sub_ps(wf3, offset_v0), scale_v0);
        acc3 = _mm256_fmadd_ps(w3, xv3, acc3);

        const q4_0_block_t *blk1 = &blocks[b+1];
        __m128i packed1 = _mm_loadu_si128((const __m128i*)blk1->q_data);
        __m128i low1 = _mm_and_si128(packed1, mask0f);
        __m128i high1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), mask0f);
        __m128i ilo1 = _mm_unpacklo_epi8(low1, high1);
        __m128i ihi1 = _mm_unpackhi_epi8(low1, high1);
        const float *xptr1 = x + (b+1)*32;
        __m256 offset_v1 = _mm256_set1_ps((float)offset);
        __m256 scale_v1 = _mm256_set1_ps(blk1->scale);

        __m256 xv4 = _mm256_load_ps(xptr1 + 0);
        __m256 xv5 = _mm256_load_ps(xptr1 + 8);
        __m256 xv6 = _mm256_load_ps(xptr1 + 16);
        __m256 xv7 = _mm256_load_ps(xptr1 + 24);

        __m128i g0_1 = ilo1;
        __m256i wi4 = _mm256_cvtepu8_epi32(g0_1);
        __m256 wf4 = _mm256_cvtepi32_ps(wi4);
        __m256 w4 = _mm256_mul_ps(_mm256_sub_ps(wf4, offset_v1), scale_v1);
        acc4 = _mm256_fmadd_ps(w4, xv4, acc4);

        __m128i g1_1 = _mm_srli_si128(ilo1, 8);
        __m256i wi5 = _mm256_cvtepu8_epi32(g1_1);
        __m256 wf5 = _mm256_cvtepi32_ps(wi5);
        __m256 w5 = _mm256_mul_ps(_mm256_sub_ps(wf5, offset_v1), scale_v1);
        acc5 = _mm256_fmadd_ps(w5, xv5, acc5);

        __m128i g2_1 = ihi1;
        __m256i wi6 = _mm256_cvtepu8_epi32(g2_1);
        __m256 wf6 = _mm256_cvtepi32_ps(wi6);
        __m256 w6 = _mm256_mul_ps(_mm256_sub_ps(wf6, offset_v1), scale_v1);
        acc6 = _mm256_fmadd_ps(w6, xv6, acc6);

        __m128i g3_1 = _mm_srli_si128(ihi1, 8);
        __m256i wi7 = _mm256_cvtepu8_epi32(g3_1);
        __m256 wf7 = _mm256_cvtepi32_ps(wi7);
        __m256 w7 = _mm256_mul_ps(_mm256_sub_ps(wf7, offset_v1), scale_v1);
        acc7 = _mm256_fmadd_ps(w7, xv7, acc7);
    }

    for (; b < block_count; ++b) {
        const q4_0_block_t *blk = &blocks[b];
        __m128i packed = _mm_loadu_si128((const __m128i*)blk->q_data);
        __m128i low = _mm_and_si128(packed, mask0f);
        __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask0f);
        __m128i ilo = _mm_unpacklo_epi8(low, high);
        __m128i ihi = _mm_unpackhi_epi8(low, high);
        const float *xptr = x + b*32;
        __m256 offset_v = _mm256_set1_ps((float)offset);
        __m256 scale_v = _mm256_set1_ps(blk->scale);

        __m256 xv0 = _mm256_load_ps(xptr + 0);
        __m256 xv1 = _mm256_load_ps(xptr + 8);
        __m256 xv2 = _mm256_load_ps(xptr + 16);
        __m256 xv3 = _mm256_load_ps(xptr + 24);

        __m128i g0 = ilo;
        __m256i wi0 = _mm256_cvtepu8_epi32(g0);
        __m256 wf0 = _mm256_cvtepi32_ps(wi0);
        __m256 w0 = _mm256_mul_ps(_mm256_sub_ps(wf0, offset_v), scale_v);
        acc0 = _mm256_fmadd_ps(w0, xv0, acc0);

        __m128i g1 = _mm_srli_si128(ilo, 8);
        __m256i wi1 = _mm256_cvtepu8_epi32(g1);
        __m256 wf1 = _mm256_cvtepi32_ps(wi1);
        __m256 w1 = _mm256_mul_ps(_mm256_sub_ps(wf1, offset_v), scale_v);
        acc1 = _mm256_fmadd_ps(w1, xv1, acc1);

        __m128i g2 = ihi;
        __m256i wi2 = _mm256_cvtepu8_epi32(g2);
        __m256 wf2 = _mm256_cvtepi32_ps(wi2);
        __m256 w2 = _mm256_mul_ps(_mm256_sub_ps(wf2, offset_v), scale_v);
        acc2 = _mm256_fmadd_ps(w2, xv2, acc2);

        __m128i g3 = _mm_srli_si128(ihi, 8);
        __m256i wi3 = _mm256_cvtepu8_epi32(g3);
        __m256 wf3 = _mm256_cvtepi32_ps(wi3);
        __m256 w3 = _mm256_mul_ps(_mm256_sub_ps(wf3, offset_v), scale_v);
        acc3 = _mm256_fmadd_ps(w3, xv3, acc3);
    }

    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum45 = _mm256_add_ps(acc4, acc5);
    __m256 sum67 = _mm256_add_ps(acc6, acc7);
    __m256 sum0123 = _mm256_add_ps(sum01, sum23);
    __m256 sum4567 = _mm256_add_ps(sum45, sum67);
    __m256 sum = _mm256_add_ps(sum0123, sum4567);

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float final = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    return final;
}


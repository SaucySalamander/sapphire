// q4_0.c - Q4_0 kernel implementations (scalar + AVX2 aligned/unaligned)
// Model-agnostic: takes void *W_row (opaque weight block data)

#define _POSIX_C_SOURCE 200809L
#include "../../include/kernels.h"

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

/**
 * Dequantize one 8-element section and FMA into accumulator.
 * group: 8 bytes of 4-bit quantized values (unpacked to 8 bytes)
 * offset_v: broadcast offset (8)
 * scale_v: broadcast scale factor
 * xv: 8 floats to multiply with dequantized weights
 * acc: pointer to accumulator to update in-place
 */
static inline void dequant_and_fma(
    __m256 *acc,
    __m128i group,
    __m256 offset_v,
    __m256 scale_v,
    __m256 xv)
{
    __m256i wi = _mm256_cvtepu8_epi32(group);
    __m256 wf = _mm256_cvtepi32_ps(wi);
    __m256 w = _mm256_mul_ps(_mm256_sub_ps(wf, offset_v), scale_v);
    *acc = _mm256_fmadd_ps(w, xv, *acc);
}

/**
 * Processes a single Q4_0 block (32 weights).
 * blk: Pointer to the Q4_0 block
 * xptr: Pointer to the corresponding section of x
 * acc: Array of 4 __m256 accumulators to update
 * mask0f: Constant mask for nibble extraction
 * offset_v: Constant broadcasted offset (8.0f)
 * force_aligned: If true, uses aligned loads for xptr
 */
static inline void compute_block_q4_0(
    const q4_0_block_t *blk,
    const float *xptr,
    __m256 *acc,
    __m128i mask0f,
    __m256 offset_v,
    int force_aligned)
{
    __m128i packed = _mm_loadu_si128((const __m128i*)blk->q_data);
    __m128i low = _mm_and_si128(packed, mask0f);
    __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask0f);
    __m128i ilo = _mm_unpacklo_epi8(low, high);
    __m128i ihi = _mm_unpackhi_epi8(low, high);

    __m256 scale_v = _mm256_set1_ps(blk->scale);

    _mm_prefetch((const char*)(const void*)(xptr + 16), _MM_HINT_T0);
    int is_aligned = force_aligned || (((uintptr_t)(const void*)xptr) & 31) == 0;

    __m256 xv0 = is_aligned ? _mm256_load_ps(xptr + 0)  : _mm256_loadu_ps(xptr + 0);
    __m256 xv1 = is_aligned ? _mm256_load_ps(xptr + 8)  : _mm256_loadu_ps(xptr + 8);
    __m256 xv2 = is_aligned ? _mm256_load_ps(xptr + 16) : _mm256_loadu_ps(xptr + 16);
    __m256 xv3 = is_aligned ? _mm256_load_ps(xptr + 24) : _mm256_loadu_ps(xptr + 24);

    dequant_and_fma(&acc[0], ilo, offset_v, scale_v, xv0);
    dequant_and_fma(&acc[1], _mm_srli_si128(ilo, 8), offset_v, scale_v, xv1);
    dequant_and_fma(&acc[2], ihi, offset_v, scale_v, xv2);
    dequant_and_fma(&acc[3], _mm_srli_si128(ihi, 8), offset_v, scale_v, xv3);
}

/**
 * Reduces 8 __m256 accumulators to a single float sum.
 */
static inline float reduce_accumulators_q4_0(const __m256 *acc) {
    __m256 sum01 = _mm256_add_ps(acc[0], acc[1]);
    __m256 sum23 = _mm256_add_ps(acc[2], acc[3]);
    __m256 sum45 = _mm256_add_ps(acc[4], acc[5]);
    __m256 sum67 = _mm256_add_ps(acc[6], acc[7]);
    __m256 sum0123 = _mm256_add_ps(sum01, sum23);
    __m256 sum4567 = _mm256_add_ps(sum45, sum67);
    __m256 sum = _mm256_add_ps(sum0123, sum4567);

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

// Note: AVX2 + FMA unaligned implementation
// W_row: opaque void pointer to Q4_0 block array
float quantized_gemv_q4_0_unaligned(const void *W_row, const float *x, int block_count, int block_size) {
    if (block_size != 32 || !W_row || !x) return 0.0f;
    
    const q4_0_block_t *blocks = (const q4_0_block_t *)W_row;
    __m256 acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = _mm256_setzero_ps();

    int b = 0;
    const __m128i mask0f = _mm_set1_epi8((char)0x0F);
    const __m256 offset_v = _mm256_set1_ps(8.0f);

    for (; b + 1 < block_count; b += 2) {
        compute_block_q4_0(&blocks[b], x + b*32, &acc[0], mask0f, offset_v, 0);
        compute_block_q4_0(&blocks[b+1], x + (b+1)*32, &acc[4], mask0f, offset_v, 0);
    }

    for (; b < block_count; ++b) {
        compute_block_q4_0(&blocks[b], x + b*32, &acc[0], mask0f, offset_v, 0);
    }

    return reduce_accumulators_q4_0(acc);
}

// Aligned fast-path
// W_row: opaque void pointer to Q4_0 block array
float quantized_gemv_q4_0_aligned(const void *W_row, const float *x, int block_count, int block_size) {
    if (block_size != 32 || !W_row || !x) return 0.0f;
    
    const q4_0_block_t *blocks = (const q4_0_block_t *)W_row;
    __m256 acc[8];
    for (int i = 0; i < 8; ++i) acc[i] = _mm256_setzero_ps();

    int b = 0;
    const __m128i mask0f = _mm_set1_epi8((char)0x0F);
    const __m256 offset_v = _mm256_set1_ps(8.0f);

    for (; b + 1 < block_count; b += 2) {
        compute_block_q4_0(&blocks[b], x + b*32, &acc[0], mask0f, offset_v, 1);
        compute_block_q4_0(&blocks[b+1], x + (b+1)*32, &acc[4], mask0f, offset_v, 1);
    }

    for (; b < block_count; ++b) {
        compute_block_q4_0(&blocks[b], x + b*32, &acc[0], mask0f, offset_v, 1);
    }

    return reduce_accumulators_q4_0(acc);
}



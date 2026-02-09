// bf16.c - Brain Float 16-bit kernel implementations
// Model-agnostic: takes void *W_row (opaque BF16 weight data)

#define _POSIX_C_SOURCE 200809L
#include "../../include/kernels.h"
#include "../../include/tensor.h"

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/**
 * BF16 scalar implementation.
 * Converts BF16 (16-bit) to F32 on-the-fly during accumulation.
 * 
 * @param W_row Opaque pointer to BF16 weight data (array of uint16_t)
 * @param x Input vector (F32)
 * @param block_count Number of blocks (not used for BF16, kept for API compatibility)
 * @param block_size Elements per block (typically 32)
 * @return Dot product result (float)
 */
float quantized_gemv_bf16_scalar(const void *W_row, const float *x, int block_count, int block_size) {
    if (!W_row || !x) return 0.0f;
    
    const uint16_t *bf16_data = (const uint16_t *)W_row;
    int cols = (block_size == 1) ? block_count : block_count * block_size;
    float acc = 0.0f;
    
    for (int j = 0; j < cols; ++j) {
        // Convert BF16 to F32: shift left 16 bits (padding in lower 16 bits)
        uint32_t f32_bits = ((uint32_t)bf16_data[j]) << 16;
        float w;
        memcpy(&w, &f32_bits, sizeof(float));
        acc += w * x[j];
    }
    return acc;
}

/**
 * BF16 AVX2 implementation with FMA.
 * Uses alignment detection for fast-path (_mm256_load_ps vs _mm256_loadu_ps).
 * 
 * CRITICAL: Input vector 'x' MUST be padded to a multiple of 8 floats (32 bytes)
 * to allow safe SIMD over-reads. The AVX2 loads (_mm256_loadu_ps) read 256 bits
 * (8 floats) at a time, and kernels do not bounds-check within these loads.
 * The caller (inference.c) ensures all input vectors are allocated with proper
 * padding via posix_memalign(32, ...).
 * 
 * @param W_row Opaque pointer to BF16 weight data
 * @param x Input vector (F32), must be padded to multiple of 8 floats
 * @param block_count Number of blocks
 * @param block_size Elements per block (typically 32)
 * @return Dot product result (float)
 */
float quantized_gemv_bf16_avx2(const void *W_row, const float *x, int block_count, int block_size) {
    if (!W_row || !x) return 0.0f;
    
    const uint16_t *bf16_data = (const uint16_t *)W_row;
    int cols = (block_size == 1) ? block_count : block_count * block_size;
    
    // Check alignment of x for fast-path selection
    int x_aligned = (((uintptr_t)(const void*)x) & 31) == 0;
    
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    
    // Process in chunks of 32 elements
    // Ensure buffers are padded to allow safe SIMD loads beyond the actual data.
    int j;
    for (j = 0; j + 32 <= cols; j += 32) {
        // Process 8 elements at a time (4 iterations of 8)
        for (int k = 0; k < 4; ++k) {
            // Load 8 BF16 values (128 bits = 8 x uint16) and convert to F32
            // _mm_loadu_si128 loads 128 bits (8 uint16 values)
            __m128i bf_packed = _mm_loadu_si128((const __m128i*)(bf16_data + j + k*8));  // 8 x uint16
            __m256i bf_expanded = _mm256_cvtepu16_epi32(bf_packed);  // Expand to 8 x uint32
            __m256 wv = _mm256_castsi256_ps(_mm256_slli_epi32(bf_expanded, 16));  // Shift left 16 bits
            
            // Load 8 F32 input values with alignment check
            __m256 xv = x_aligned ? _mm256_load_ps(x + j + k*8) : _mm256_loadu_ps(x + j + k*8);
            
            if (k == 0) acc0 = _mm256_fmadd_ps(wv, xv, acc0);
            else if (k == 1) acc1 = _mm256_fmadd_ps(wv, xv, acc1);
            else if (k == 2) acc2 = _mm256_fmadd_ps(wv, xv, acc2);
            else acc3 = _mm256_fmadd_ps(wv, xv, acc3);
        }
    }
    
    // Handle remainder with scalar
    float remainder = 0.0f;
    for (; j < cols; ++j) {
        uint32_t f32_bits = ((uint32_t)bf16_data[j]) << 16;
        float w;
        memcpy(&w, &f32_bits, sizeof(float));
        remainder += w * x[j];
    }
    
    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum = _mm256_add_ps(sum01, sum23);
    
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float final = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] + remainder;
    return final;
}

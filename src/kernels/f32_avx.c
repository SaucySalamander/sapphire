// f32.c - Standard 32-bit float kernel implementations
// Model-agnostic: takes void *W_row (opaque F32 weight data)

#define _POSIX_C_SOURCE 200809L
#include "../../include/kernels.h"
#include "../../include/tensor.h"

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

/**
 * F32 scalar implementation (trivial dot product).
 * 
 * @param W_row Opaque pointer to F32 weight data (array of float)
 * @param x Input vector (F32)
 * @param block_count Number of blocks (not used for F32, kept for API compatibility)
 * @param block_size Elements per block (typically 32)
 * @return Dot product result (float)
 */
float quantized_gemv_f32_scalar(const void *W_row, const float *x, int block_count, int block_size) {
    if (!W_row || !x) return 0.0f;
    
    const float *f32_data = (const float *)W_row;
    int cols = (block_size == 1) ? block_count : block_count * block_size;
    float acc = 0.0f;
    
    for (int j = 0; j < cols; ++j) {
        acc += f32_data[j] * x[j];
    }
    return acc;
}

/**
 * F32 AVX2 implementation with FMA.
 * Uses alignment detection for fast-path (_mm256_load_ps vs _mm256_loadu_ps).
 * 
 * @param W_row Opaque pointer to F32 weight data
 * @param x Input vector (F32)
 * @param block_count Number of blocks
 * @param block_size Elements per block (typically 32)
 * @return Dot product result (float)
 */
float quantized_gemv_f32_avx2(const void *W_row, const float *x, int block_count, int block_size) {
    if (!W_row || !x) return 0.0f;
    
    const float *f32_data = (const float *)W_row;
    int cols = (block_size == 1) ? block_count : block_count * block_size;
    int x_aligned = (((uintptr_t)(const void*)x) & 31) == 0;
    
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    
    int j;
    for (j = 0; j + 32 <= cols; j += 32) {
        for (int k = 0; k < 4; ++k) {
            __m256 wv = _mm256_loadu_ps(f32_data + j + k*8);  // Weights may not be aligned
            __m256 xv = x_aligned ? _mm256_load_ps(x + j + k*8) : _mm256_loadu_ps(x + j + k*8);
            
            if (k == 0) acc0 = _mm256_fmadd_ps(wv, xv, acc0);
            else if (k == 1) acc1 = _mm256_fmadd_ps(wv, xv, acc1);
            else if (k == 2) acc2 = _mm256_fmadd_ps(wv, xv, acc2);
            else acc3 = _mm256_fmadd_ps(wv, xv, acc3);
        }
    }
    
    // Remainder
    float remainder = 0.0f;
    for (; j < cols; ++j) {
        remainder += f32_data[j] * x[j];
    }
    
    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum = _mm256_add_ps(sum01, sum23);
    
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float final = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] + remainder;
    return final;
}

// sapphire.c - dispatcher + small wiring for sapphire kernels

#define _POSIX_C_SOURCE 200809L
#include "../../include/sapphire.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>

// Dispatcher thin wrappers: call q4_0 implementations
float quantized_gemv_row_dot_product(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size) {
	// choose unaligned q4 kernel
	return quantized_gemv_q4_0_unaligned(W_row, x, block_count, block_size);
}

float quantized_gemv_row_dot_product_aligned(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size) {
	return quantized_gemv_q4_0_aligned(W_row, x, block_count, block_size);
}

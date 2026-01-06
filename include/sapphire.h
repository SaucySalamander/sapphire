// sapphire.h - Q4_0 quantized GEMM API for sapphire core
#ifndef SAPPHIRE_H
#define SAPPHIRE_H

#include <stdint.h>
#include <stddef.h>

// Define the structure for a block of 32 4-bit quantized weights.
// Each block stores a shared scale and 16 bytes packing 32 4-bit weights.
typedef struct {
    float scale;         // Scaling factor for the entire block (f32)
    uint8_t q_data[16];  // 16 bytes * 2 weights/byte = 32 packed 4-bit weights
} ggml_block_q4_0;

// Performs a dot-product between one quantized row (W_row) and a float vector (x).
// W_row: Array of ggml_block_q4_0 blocks representing one row of the weight matrix.
// x: The input vector (float array).
// block_count: The total number of blocks in the row.
// block_size: The number of weights per block (fixed at 32 for Q4_0).
// Returns the resulting dot product as a float.
float quantized_gemv_row_dot_product(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size);
// Fast path: caller promises x is 32-byte aligned. Uses _mm256_load_ps for speed.
float quantized_gemv_row_dot_product_aligned(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size);
// Scalar reference implementation (useful for tests and validation)
float quantized_gemv_row_dot_product_scalar(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size);

// Persistent thread-pool context for batched GEMV
typedef struct sapphire_context sapphire_context;

// Create/destroy thread-pool context. num_threads=0 => autodetect hardware_concurrency.
sapphire_context *sapphire_context_create(int num_threads, int chunk_size);
void sapphire_context_destroy(sapphire_context *ctx);

// Q8 block format (one byte per weight)
typedef struct {
    float scale;
    uint8_t q_data[32];
} ggml_block_q8_0;

float quantized_gemv_q8_0_unaligned(const ggml_block_q8_0 *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_q8_0_aligned(const ggml_block_q8_0 *W_row, const float *x, int block_count, int block_size);

// Minimal GGML reader API
typedef enum { GGML_TYPE_F32 = 0, GGML_TYPE_Q8_0 = 1, GGML_TYPE_Q4_0 = 2 } ggml_type_t;
typedef struct {
    ggml_type_t type;
    int rows;
    int cols;
    void *data; /* pointer to tensor payload (heap-owned) */
    size_t data_size;
} ggml_tensor_t;

// Run batched GEMV using the context's thread pool. tensors points to an array of ggml_tensor_t describing the
// weight tensors to use (one or more). rows is the number of output rows to compute. Returns 0 on success.
int sapphire_batched_gemv(sapphire_context *ctx, const ggml_tensor_t *tensors, size_t tensor_count, int rows, int blocks_per_row, const float *x, float *y);

// Additional kernel wrappers
float quantized_gemv_q4_0_unaligned(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size);
float quantized_gemv_q4_0_aligned(const ggml_block_q4_0 *W_row, const float *x, int block_count, int block_size);
int ggml_model_load(const char *path, ggml_tensor_t **out_tensors, size_t *out_count);
void ggml_model_free(ggml_tensor_t *tensors, size_t count);

#endif // SAPPHIRE_H

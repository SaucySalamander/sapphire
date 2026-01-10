#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Supported data types for tensors.
 * 
 * This enum allows tensors to store data in different formats:
 * - DTYPE_F32: Standard 32-bit floats (unquantized)
 * - DTYPE_BF16: 16-bit brain float (bfloat16)
 * - DTYPE_F16: 16-bit float (half precision)
 * - DTYPE_Q4_0: 4-bit quantized (from Phase 1 quantization)
 * - DTYPE_Q8_0: 8-bit quantized (from Phase 1 quantization)
 */
typedef enum {
    DTYPE_F32,     // 32-bit float
    DTYPE_BF16,    // 16-bit brain float (bfloat16)
    DTYPE_F16,     // 16-bit float (half precision)
    DTYPE_Q4_0,    // 4-bit quantized (Phase 1)
    DTYPE_Q8_0,    // 8-bit quantized (Phase 1)
} tensor_dtype_t;

/**
 * @brief Memory layout strategy for tensor storage.
 * 
 * Determines how multi-dimensional tensor data is arranged in linear memory.
 * Row-major (C-style) is default and best for LLM inference.
 */
typedef enum {
    LAYOUT_ROW_MAJOR,      // C-style: element[i][j] at index i*cols + j
    LAYOUT_COLUMN_MAJOR,   // Fortran-style: element[i][j] at index j*rows + i
} memory_layout_t;

/*
 * Opaque tensor type. Implementation details are private to src/tensor.c.
 * Use accessor functions to query shape, dtype, data, etc.
 */
typedef struct tensor_t tensor_t;

/**
 * @brief Create a tensor with given shape and dtype.
 * 
 * Allocates memory for the tensor and initializes metadata.
 * Memory is zero-initialized.
 * 
 * @param ndim Number of dimensions (1-8).
 * @param shape Array of shape values (length: ndim). Example: shape=[4, 8] for 4x8 matrix.
 * @param dtype Data type (F32, Q4_0, Q8_0).
 * 
 * @return Pointer to allocated tensor_t, or NULL if allocation fails or arguments invalid.
 * 
 * @note Caller is responsible for calling tensor_release() when done.
 * 
 * Example:
 *   int shape[] = {3, 4};
 *   tensor_t *t = tensor_create(2, shape, DTYPE_F32);  // 3x4 float matrix
 */
tensor_t* tensor_create(int ndim, const int *shape, tensor_dtype_t dtype);

// Create a new F32 tensor that is the transpose of a 2D F32 tensor 'src' and
// store pointer to newly allocated tensor in *dst. Returns 0 on success.
int tensor_transpose(const tensor_t *src, tensor_t **dst);

/**
 * @brief Create a tensor sharing existing data (zero-copy).
 * @param ndim Number of dimensions.
 * @param shape Shape array.
 * @param dtype Data type.
 * @param data Pointer to external memory (e.g. mmap).
 * @return Tensor object.
 */
tensor_t* tensor_create_external(int ndim, const int *shape, tensor_dtype_t dtype, void *data);

/**
 * @brief Clone a tensor (deep copy).
 * 
 * Creates a new tensor with identical shape, dtype, layout, and data.
 * The new tensor is independent (separate data allocation).
 * 
 * @param src Source tensor to clone.
 * @return New tensor_t with copied data, or NULL on failure.
 * 
 * @note Caller is responsible for calling tensor_release() on the cloned tensor.
 */
tensor_t* tensor_clone(const tensor_t *src);

/**
 * @brief Get total number of elements in tensor.
 * 
 * @param t Tensor.
 * @return Product of all shape dimensions. Example: tensor_numel([3, 4]) = 12.
 */
size_t tensor_numel(const tensor_t *t);

/**
 * @brief Get element size in bytes for a dtype.
 * 
 * @param dtype Data type.
 * @return Bytes per element: F32=4, Q4_0=0.5, Q8_0=1, etc.
 */
size_t dtype_element_size(tensor_dtype_t dtype);

/**
 * @brief Get data type name (for printing/debugging).
 */
const char* dtype_name(tensor_dtype_t dtype);

/**
 * @brief Access element by linear index.
 * 
 * Linear indexing assumes row-major layout.
 * For 2D tensor shape=[3,4]: element[i][j] is at linear index i*4 + j.
 * 
 * @param t Tensor.
 * @param idx Linear index (0 to tensor_numel(t)-1).
 * @return F32 value. If dtype is quantized, value is dequantized.
 * 
 * @note This is slow for quantized tensors (requires dequantization per-element).
 *       Use for validation/testing only, not performance-critical code.
 */
float tensor_get_f32(const tensor_t *t, size_t idx);

/**
 * @brief Set element by linear index (F32 tensors only).
 * 
 * @param t Tensor (must be DTYPE_F32).
 * @param idx Linear index.
 * @param val F32 value.
 * 
 * @return 0 on success, -1 if dtype is not F32.
 */
int tensor_set_f32(tensor_t *t, size_t idx, float val);

/**
 * @brief Increment reference count.
 * 
 * @param t Tensor (must not be NULL).
 */
void tensor_ref_inc(tensor_t *t);

/**
 * @brief Decrement reference count; free if zero.
 * 
 * When reference count reaches 0, tensor data is freed.
 * 
 * @param t Tensor to release (can be NULL, safe no-op).
 */
void tensor_release(tensor_t *t);

/**
 * @brief Pretty-print tensor metadata (shape, dtype, layout, nbytes).
 * 
 * Example output: "Tensor [3, 4] F32 row-major 48 bytes"
 * 
 * @param t Tensor.
 */
void tensor_print_info(const tensor_t *t);

/**
 * @brief Accessor: get number of dimensions.
 * @return ndim or 0 for NULL
 */
int tensor_ndim(const tensor_t *t);

/**
 * @brief Accessor: get shape pointer (length = ndim). Do not modify the returned array.
 * @return pointer to internal shape array or NULL for NULL tensor
 */
const int* tensor_shape(const tensor_t *t);

/**
 * @brief Accessor: get pointer to raw data buffer (read-only).
 * @return pointer to internal data buffer or NULL
 */
const void* tensor_data(const tensor_t *t);

/**
 * @brief Accessor: get pointer to raw f32 data (writable) if dtype == DTYPE_F32.
 * @return pointer to internal f32 buffer or NULL (and prints a warning) if dtype != F32
 */
float* tensor_data_f32(tensor_t *t);

/**
 * @brief Accessor: get pointer to raw data buffer (writable) for any dtype.
 * 
 * Use this when you need mutable access to the underlying tensor data,
 * particularly for loading quantized weights from disk or other initialization.
 * 
 * @param t Tensor.
 * @return pointer to writable internal data buffer or NULL
 */
void* tensor_data_mutable(tensor_t *t);

/**
 * @brief Accessor: get tensor dtype.
 * @return dtype enum
 */
tensor_dtype_t tensor_dtype(const tensor_t *t);

/**
 * @brief Accessor: total bytes allocated for the tensor data buffer.
 * @return number of bytes or 0 for NULL
 */
size_t tensor_nbytes(const tensor_t *t);

/**
 * @brief Accessor: reference count (for debugging/tests).
 * @return reference count, or 0 for NULL
 */
int tensor_ref_count(const tensor_t *t);

/**
 * @brief Create a tensor that points to existing data (no allocation).
 * Used for memory-mapped weights.
 */
tensor_t* tensor_create_view(tensor_dtype_t dtype, int ndim, const int *shape, void *data);

#endif // TENSOR_H

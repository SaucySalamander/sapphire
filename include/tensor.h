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
 * - DTYPE_TERNARY_2BIT: packed ternary weights with grouped FP32 scales
 */
typedef enum {
    DTYPE_F32,     // 32-bit float
    DTYPE_BF16,    // 16-bit brain float (bfloat16)
    DTYPE_F16,     // 16-bit float (half precision)
    DTYPE_Q4_0,    // 4-bit quantized (Phase 1)
    DTYPE_Q8_0,    // 8-bit quantized (Phase 1)
    DTYPE_TERNARY_2BIT,    // 2-bit packed ternary symbols + grouped scales
    DTYPE_TERNARY_HYBRID,  // 2-bit ternary bulk + BF16 anchor patch
} tensor_dtype_t;

typedef struct {
    const uint8_t *packed_weights;
    const float *scales;
    uint32_t rows;
    uint32_t cols;
    uint32_t packed_cols;
    uint32_t scale_group_size;
    uint32_t groups_per_row;
    size_t packed_weight_bytes;
    size_t scale_count;
    size_t scale_bytes;
} tensor_ternary_view_t;

typedef struct {
    const uint8_t *packed_weights;
    size_t packed_weight_bytes;
    const float *scales;
    size_t scale_count;
    uint32_t scale_group_size;
} tensor_ternary_payload_t;

/**
 * @brief Hybrid ternary + BF16 anchor view for runtime.
 *
 * The anchor_entries array is sorted by (row, col).
 * Row i's anchors span indices [anchor_row_offsets[i], anchor_row_offsets[i+1]).
 */
typedef struct {
    /* Ternary bulk (same as tensor_ternary_view_t) */
    const uint8_t *packed_weights;
    const float *scales;
    uint32_t rows;
    uint32_t cols;
    uint32_t packed_cols;
    uint32_t scale_group_size;
    uint32_t groups_per_row;
    size_t packed_weight_bytes;
    size_t scale_count;
    size_t scale_bytes;
    /* Anchor patch */
    const void *anchor_entries;        /* ternary_anchor_entry_t[] or NULL */
    const uint32_t *anchor_row_offsets; /* CSR-like row offsets (rows+1) or NULL */
    uint32_t anchor_count;
    int owns_anchor_memory;
} tensor_hybrid_view_t;

typedef struct {
    tensor_ternary_payload_t bulk;
    const void *anchor_entries;
    const uint32_t *anchor_row_offsets;
    uint32_t anchor_count;
    int owns_anchor_memory;
} tensor_hybrid_payload_t;

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
float* tensor_data_f32(const tensor_t *t);

/**
 * @brief Accessor: get pointer to raw data buffer (writable) for any dtype.
 * 
 * Use this when you need mutable access to the underlying tensor data,
 * particularly for loading quantized weights from disk or other initialization.
 * 
 * @param t Tensor.
 * @return pointer to writable internal data buffer or NULL
 */
void* tensor_data_mutable(const tensor_t *t);

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
 * @brief Returns non-zero if the tensor's data is externally owned (e.g. mmap).
 *
 * Only externally-owned tensors are safe to advise with MADV_DONTNEED because
 * file-backed pages are re-faulted from the file on next access. Malloc-backed
 * (non-external) tensors will read back as zeros after MADV_DONTNEED.
 *
 * @return 1 if external (mmap-backed), 0 if malloc-owned.
 */
int tensor_is_external(const tensor_t *t);

/**
 * @brief Create a tensor that points to existing data (no allocation).
 * Used for memory-mapped weights.
 */
tensor_t* tensor_create_view(tensor_dtype_t dtype, int ndim, const int *shape, void *data);

/**
 * @brief Create an owned tensor wrapper for packed ternary weights.
 *
 * The returned tensor stores the logical matrix shape [rows, cols], while its
 * backing payload points at a packed 2-bit symbol stream plus grouped F32
 * scales.
 */
tensor_t* tensor_create_ternary_view(uint32_t rows,
                                     uint32_t cols,
                                     const tensor_ternary_payload_t *payload,
                                     int is_external);

/**
 * @brief Return the packed ternary payload for a tensor, or NULL if not ternary.
 */
const tensor_ternary_view_t* tensor_data_ternary(const tensor_t *t);

/**
 * @brief Create a hybrid ternary+anchor tensor view.
 *
 * @param rows         Number of rows
 * @param cols         Number of columns
 * @param payload      Hybrid payload bundle
 * @param is_external  If 1, data is externally owned and not freed
 * @return Tensor or NULL on error
 */
tensor_t* tensor_create_hybrid_view(uint32_t rows,
                                    uint32_t cols,
                                    const tensor_hybrid_payload_t *payload,
                                    int is_external);

/**
 * @brief Return the hybrid ternary+anchor view for a tensor, or NULL if not hybrid.
 */
const tensor_hybrid_view_t* tensor_data_hybrid(const tensor_t *t);

#endif // TENSOR_H

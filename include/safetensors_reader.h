/**
 * @file safetensors_reader.h
 * @brief Safetensors format (.safetensors) reader with zero-copy mmap loading.
 *
 * Implements efficient loading of Hugging Face Safetensors format files using
 * memory-mapped I/O for zero-copy tensor access. Supports F32 and BF16 tensors.
 *
 * The Safetensors format:
 * - Byte 0-7: uint64_t (little-endian) = length of JSON header
 * - Byte 8 to (8 + length - 1): JSON metadata string
 * - Rest: Binary tensor data (offset calculated from JSON)
 *
 * This implementation uses mmap for efficient zero-copy loading on 8GB VRAM
 * hardware without buffering the entire file in memory.
 */

#ifndef SAFETENSORS_READER_H
#define SAFETENSORS_READER_H

#include <stdint.h>
#include <stddef.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Data type for Safetensors format.
 *
 * Maps to standard PyTorch/Hugging Face dtypes:
 * - "F32" or "FLOAT": 32-bit float
 * - "BF16": 16-bit bfloat (Brain Float)
 * - "F16": 16-bit float (half precision)
 * - "I32": 32-bit signed integer
 * - "I64": 64-bit signed integer
 */
typedef enum {
    SAFETENSORS_F32 = 0,     // float32
    SAFETENSORS_BF16 = 1,    // bfloat16
    SAFETENSORS_F16 = 2,     // float16
    SAFETENSORS_I32 = 3,     // int32
    SAFETENSORS_I64 = 4,     // int64
    SAFETENSORS_UNKNOWN = -1
} safetensors_dtype_t;

/**
 * @brief Single tensor metadata from Safetensors header.
 *
 * Stores shape, dtype, and byte offset for a tensor within the file.
 */
typedef struct {
    char name[256];              // Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
    safetensors_dtype_t dtype;   // Data type (F32, BF16, etc.)
    int ndim;                    // Number of dimensions (1-8)
    uint32_t shape[8];           // Shape array
    uint64_t offset;             // Byte offset in file where tensor data starts
    uint64_t size_bytes;         // Size in bytes of this tensor's data
} safetensors_tensor_meta_t;

/**
 * @brief Opaque handle to an open Safetensors file.
 *
 * Manages mmapped file pointer, file descriptor, and metadata array.
 * Implementation is in safetensors_reader.c.
 */
typedef struct safetensors_file_t safetensors_file_t;

/**
 * @brief Open a Safetensors file and parse its header.
 *
 * Memory-maps the file for zero-copy access. Parses the JSON header to extract
 * tensor metadata (shapes, dtypes, offsets). Must be paired with safetensors_close().
 *
 * @param path File system path to .safetensors file (required).
 *
 * @return Pointer to safetensors_file_t on success, NULL on error.
 *         Errors: file not found, mmap failure, invalid Safetensors format,
 *         JSON parse error, malloc failure.
 *
 * @note File remains open (mmapped) until safetensors_close() is called.
 * @note On error, NULL is returned and errno/stderr may contain details.
 */
safetensors_file_t* safetensors_open(const char *path);

/**
 * @brief Get the number of tensors in the Safetensors file.
 *
 * @param st Safetensors file handle (opened via safetensors_open).
 *
 * @return Number of tensors, or 0 if st is NULL.
 */
int safetensors_tensor_count(const safetensors_file_t *st);

/**
 * @brief Get metadata for a tensor by index.
 *
 * @param st Safetensors file handle.
 * @param index Zero-based tensor index (0 to count-1).
 *
 * @return Pointer to metadata (valid until safetensors_close), or NULL if index out of bounds.
 */
const safetensors_tensor_meta_t* safetensors_get_tensor_by_index(
    const safetensors_file_t *st, int index);

/**
 * @brief Get metadata for a tensor by name.
 *
 * Linear search through metadata array to find the named tensor.
 *
 * @param st Safetensors file handle.
 * @param name Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight").
 *
 * @return Pointer to metadata (valid until safetensors_close), or NULL if not found.
 */
const safetensors_tensor_meta_t* safetensors_get_tensor_by_name(
    const safetensors_file_t *st, const char *name);

/**
 * @brief Create a tensor_t from Safetensors metadata.
 *
 * Constructs a tensor_t object whose data pointer points directly into the
 * mmapped file memory (zero-copy). The returned tensor is a reference to
 * mmapped memory and must NOT be freed with tensor_release().
 *
 * @param st Safetensors file handle (must remain open).
 * @param meta Tensor metadata (from safetensors_get_tensor_by_* functions).
 *
 * @return tensor_t* pointing to mmapped data, or NULL if dtype unsupported
 *         or meta is NULL.
 *
 * @note IMPORTANT: The returned tensor points into mmapped memory managed
 *       by the safetensors_file_t. Do NOT call tensor_release() on it.
 *       The file must remain open for the lifetime of all created tensors.
 * @note The tensor dtype will be DTYPE_F32 for SAFETENSORS_F32,
 *       and DTYPE_Q8_0 for SAFETENSORS_BF16 (will be improved in future phases).
 */
tensor_t* safetensors_create_tensor_ref(safetensors_file_t *st,
                                        const safetensors_tensor_meta_t *meta);

/**
 * @brief Load a Safetensors tensor into a freshly allocated tensor_t.
 *
 * Unlike safetensors_create_tensor_ref(), this copies the data from the
 * mmapped region into a newly allocated buffer. The returned tensor owns
 * its memory and must be freed with tensor_release().
 *
 * Use this if you need independent ownership of the tensor data, or if you
 * want to close the Safetensors file but keep the tensor.
 *
 * @param st Safetensors file handle.
 * @param meta Tensor metadata.
 *
 * @return Newly allocated tensor_t with copied data, or NULL on failure.
 *
 * @note Caller must call tensor_release() on the returned tensor.
 */
tensor_t* safetensors_load_tensor_copy(const safetensors_file_t *st,
                                       const safetensors_tensor_meta_t *meta);

/**
 * @brief Close a Safetensors file and free all resources.
 *
 * Unmaps memory, closes file descriptor, and frees metadata array.
 * All tensor_t references created via safetensors_create_tensor_ref()
 * become invalid after this call.
 *
 * @param st Safetensors file handle (may be NULL; safe noop).
 *
 * @note After calling this, st is invalid and must not be used.
 */
void safetensors_close(safetensors_file_t *st);

/**
 * @brief Print Safetensors file metadata for debugging.
 *
 * Prints all tensor names, shapes, dtypes, and offsets to stdout.
 *
 * @param st Safetensors file handle.
 */
void safetensors_print_info(const safetensors_file_t *st);

#ifdef __cplusplus
}
#endif

#endif // SAFETENSORS_READER_H

/**
 * @file ggml_model.h
 * @brief Model structure and loading interface for GGML-format model files.
 */

#ifndef GGML_MODEL_H
#define GGML_MODEL_H

/*
 * NOTE: This header is kept for backward compatibility. The canonical model
 * definitions have been moved to `llm_model.h`.  Include that file instead
 * for new code and migrations.
 */

#include "llm_model.h"

#include <stdint.h>
#include <stdio.h>

/**
 * GGML-specific tensor metadata.
 */
typedef struct {
    char name[256];              /**< Tensor name (e.g., "layers.0.attention.q_proj.weight"). */
    uint32_t ndim;               /**< Number of dimensions. */
    uint32_t shape[8];           /**< Shape array. */
    tensor_dtype_t dtype;        /**< Data type (F32, Q4_0, Q8_0, etc.). */
    size_t file_offset;          /**< Byte offset in file where data starts. */
    size_t data_size;            /**< Size in bytes of tensor data. */
} ggml_tensor_meta_t;

/**
 * GGML file header metadata.
 */
typedef struct {
    uint32_t magic;              /**< 0x67676d6c ("ggml"). */
    uint32_t version;            /**< File format version. */
    uint32_t tensor_count;       /**< Number of tensors. */
    ggml_tensor_meta_t *tensors; /**< Metadata for all tensors. */
} ggml_file_header_t;

/**
 * GGML reader API (used by GGML loader implementation and tests)
 */
FILE* ggml_file_open_and_parse_header(const char *filename, ggml_file_header_t *header);

tensor_t* ggml_load_tensor(FILE *fp, const ggml_tensor_meta_t *meta);

const ggml_tensor_meta_t* ggml_find_tensor_meta(const ggml_file_header_t *header, const char *name);

void ggml_header_destroy(ggml_file_header_t *header);

#endif // GGML_MODEL_H

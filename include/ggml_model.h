/**
 * @file ggml_model.h
 * @brief Model structure and loading interface for GGML-format model files.
 */

#ifndef GGML_MODEL_H
#define GGML_MODEL_H

#include <stdint.h>
#include <stdio.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Model configuration.
 */
typedef struct {
    int vocab_size;              /**< Size of vocabulary (for embedding). */
    int d_model;                 /**< Embedding/hidden dimension. */
    int num_heads;               /**< Number of attention heads. */
    int d_k;                     /**< Dimension per head (d_model / num_heads). */
    int num_layers;              /**< Number of transformer layers. */
    int max_context_len;         /**< Maximum sequence length. */
    float rope_base;             /**< Base for RoPE frequency (default 10000.0). */
} model_config_t;

/**
 * Tensor metadata from GGML file.
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
 * Single transformer layer's weights.
 */
typedef struct {
    tensor_t *norm_attn_weight;  /**< Layer norm (attention): [d_model]. */
    tensor_t *q_proj_weight;     /**< Query projection: [d_model, d_model]. */
    tensor_t *k_proj_weight;     /**< Key projection: [d_model, d_model]. */
    tensor_t *v_proj_weight;     /**< Value projection: [d_model, d_model]. */
    tensor_t *out_proj_weight;   /**< Output projection: [d_model, d_model]. */
    
    tensor_t *norm_ffn_weight;   /**< Layer norm (FFN): [d_model]. */
    tensor_t *up_proj_weight;    /**< Up projection: [d_model, d_ff]. */
    tensor_t *gate_proj_weight;  /**< Gate projection: [d_model, d_ff]. */
    tensor_t *down_proj_weight;  /**< Down projection: [d_ff, d_model]. */
} model_layer_weights_t;

/**
 * Complete loaded LLM model.
 */
typedef struct {
    model_config_t config;
    tensor_t *embedding_weight;              /**< Token embeddings: [vocab_size, d_model]. */
    tensor_t *norm_final_weight;             /**< Final layer norm: [d_model]. */
    tensor_t *lm_head_weight;                /**< Logit projection: [vocab_size, d_model]. */
    
    model_layer_weights_t *layers;           /**< Array of [num_layers]. */
    
    FILE *weight_file;                       /**< Open file handle (for lazy loading if needed). */
    ggml_file_header_t file_header;          /**< Parsed file header metadata. */
} llm_model_t;

/**
 * Open and parse a GGML file header.
 *
 * @param filename Path to the .ggml file.
 * @param header Output: parsed header metadata.
 * @return FILE* pointer (caller must fclose), or NULL on error.
 */
FILE* ggml_file_open_and_parse_header(const char *filename, ggml_file_header_t *header);

/**
 * Load a single tensor from file into memory.
 *
 * @param fp File pointer (from ggml_file_open_and_parse_header).
 * @param meta Tensor metadata (shape, dtype, file offset).
 * @return Allocated tensor_t with data loaded, or NULL on failure.
 */
tensor_t* ggml_load_tensor(FILE *fp, const ggml_tensor_meta_t *meta);

/**
 * Find a tensor by name in the file header.
 *
 * @param header Parsed GGML header.
 * @param name Tensor name (e.g., "layers.0.attention.q_proj.weight").
 * @return Pointer to tensor metadata, or NULL if not found.
 */
const ggml_tensor_meta_t* ggml_find_tensor_meta(const ggml_file_header_t *header, const char *name);

/**
 * Free parsed header and metadata.
 *
 * @param header Header to free.
 */
void ggml_header_destroy(ggml_file_header_t *header);

/**
 * Load entire GGML model from file.
 *
 * @param filename Path to .ggml model file.
 * @param config Model configuration (required; must not be NULL).
 * @return Allocated llm_model_t with all weights loaded, or NULL on failure.
 */
llm_model_t* llm_model_load(const char *filename, const model_config_t *config);

/**
 * Free all model memory.
 *
 * @param model Model to free.
 */
void llm_model_destroy(llm_model_t *model);

/**
 * Print model configuration and layer summary.
 *
 * @param model Model to print.
 */
void llm_model_print_info(const llm_model_t *model);

#ifdef __cplusplus
}
#endif

#endif // GGML_MODEL_H

/*
 * @file llm_model.h
 * @brief Core model types (format-agnostic) and lifecycle APIs.
 *
 * This header contains the canonical model structures used by the tensor
 * mapper loaders (Safetensors, GGML/GGUF) and the runtime. It replaces the
 * legacy ggml_model.h as the primary place for model definitions.
 */

#ifndef LLM_MODEL_H
#define LLM_MODEL_H

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
    int d_model;                 /**< Embedding/input dimension. */
    int d_inner;                 /**< Transformer inner/hidden dimension (for attention and FFN). May differ from d_model in hybrid architectures. */
    int d_kv;                    /**< Key/Value projection dimension (may differ from d_inner in GQA). Set during model load. */
    int num_heads;               /**< Number of attention heads (query heads). */
    int d_k;                     /**< Dimension per head (typically d_inner / num_heads). */
    int num_kv_heads;            /**< Number of KV heads (for GQA, typically num_heads / reduction). */
    int num_layers;              /**< Number of transformer layers. */
    int max_context_len;         /**< Maximum sequence length. */
    float rope_base;             /**< Base for RoPE frequency (default 10000.0). */
} model_config_t;

/**
 * Single transformer layer's weights.
 * 
 * In hybrid architectures (e.g., Gemma 3), projections use d_inner for the attention space,
 * while the FFN projections follow the actual tensor dimensions from the model.
 */
typedef struct {
    tensor_t *norm_attn_weight;  /**< Layer norm (attention): [d_model]. */
    tensor_t *q_proj_weight;     /**< Query projection: [d_model, d_inner] for hybrid, or [d_model, d_model]. */
    tensor_t *k_proj_weight;     /**< Key projection: [d_model, d_inner] for hybrid, or [d_model, d_model]. */
    tensor_t *v_proj_weight;     /**< Value projection: [d_model, d_inner] for hybrid, or [d_model, d_model]. */
    tensor_t *out_proj_weight;   /**< Output projection: [d_inner, d_model] for hybrid, or [d_model, d_model]. */
    
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
    
    /* Implementation detail: format loaders may store file handles here */
    FILE *weight_file;                       /**< Open file handle (for lazy loading if needed). */

    /* Format-specific header metadata (used by e.g., GGML loader). */
    void *file_header;                       /**< Opaque file header pointer (format-specific). */
} llm_model_t;

/**
 * Free all model memory / resources.
 *
 * Frees allocated layer arrays and closes associated file handles.
 */
void llm_model_destroy(llm_model_t *model);

/**
 * Print model configuration and layer summary.
 */
void llm_model_print_info(const llm_model_t *model);

#ifdef __cplusplus
}
#endif

#endif // LLM_MODEL_H

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

/* Forward declare model_spec_t so header does not need to include model_spec.h */
typedef struct model_spec model_spec_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Model configuration.
 */
#define SAPPHIRE_MAX_LAYERS 256


/**
 * Single transformer layer's weights.
 * 
 * In hybrid architectures (e.g., Gemma 3), projections use d_inner for the attention space,
 * while the FFN projections follow the actual tensor dimensions from the model.
 * 
 * QK-Norm: Gemma 3 includes per-head normalization of Q and K before attention.
 * This stabilizes attention scores and prevents magnitude explosion.
 */
typedef struct {
    tensor_t *norm_attn_weight;  /**< Layer norm (attention input): [d_model]. */
    tensor_t *norm_attn_post_weight; /**< Layer norm (attention output): [d_model] (Gemma 3). */
    tensor_t *q_proj_weight;     /**< Query projection: [d_model, d_inner] for hybrid, or [d_model, d_model]. */
    tensor_t *k_proj_weight;     /**< Key projection: [d_model, d_inner] for hybrid, or [d_model, d_model]. */
    tensor_t *v_proj_weight;     /**< Value projection: [d_model, d_inner] for hybrid, or [d_model, d_model]. */
    tensor_t *q_norm_weight;     /**< Q normalization (QK-Norm): [d_inner]. Applied per-head after Q projection. */
    tensor_t *k_norm_weight;     /**< K normalization (QK-Norm): [d_kv]. Applied per-head after K projection. */
    tensor_t *out_proj_weight;   /**< Output projection: [d_inner, d_model] for hybrid, or [d_model, d_model]. */
    
    tensor_t *norm_ffn_weight;   /**< Layer norm (FFN input): [d_model]. */
    tensor_t *norm_ffn_post_weight; /**< Layer norm (FFN output): [d_model] (Gemma 3). */
    tensor_t *up_proj_weight;    /**< Up projection: [d_model, d_ff]. */
    tensor_t *gate_proj_weight;  /**< Gate projection: [d_model, d_ff]. */
    tensor_t *down_proj_weight;  /**< Down projection: [d_ff, d_model]. */
} model_layer_weights_t;

/**
 * Complete loaded LLM model.
 */
typedef struct {
    tensor_t *embedding_weight;              /**< Token embeddings: [vocab_size, d_model]. */
    tensor_t *norm_final_weight;             /**< Final layer norm: [d_model]. */
    tensor_t *lm_head_weight;                /**< Logit projection: [vocab_size, d_model]. */
    model_layer_weights_t *layers;           /**< Array of [num_layers]. */
    void *safetensors_handle;                /**< Opaque handle to safetensors_file_t for cleanup. */
} llm_model_t;

/**
 * Free all model memory / resources.
 *
 * Frees allocated layer arrays and closes associated file handles.
 */
void llm_model_destroy(llm_model_t *model);

/**
 * Destroy model referenced by the provided `spec`.
 *
 * The model pointer is owned by `spec->llm_model`. This function will free
 * model-owned tensors and close any underlying safetensors handle. The
 * implementation will consult `spec->variant_config` (when present) to
 * determine the actual `num_hidden_layers` to free rather than assuming a
 * hard-coded maximum.
 */
void llm_model_destroy_ex(const model_spec_t *spec);

/**
 * Print model configuration and layer summary.
 */
void llm_model_print_info(const llm_model_t *model);

#ifdef __cplusplus
}
#endif

#endif // LLM_MODEL_H

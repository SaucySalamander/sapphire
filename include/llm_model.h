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
 * Per-layer memory page tracking for sliding-window eviction.
 *
 * When low_memory_mode is enabled, the inference loop uses these ranges
 * to call madvise(MADV_DONTNEED) on layers outside the window and
 * madvise(MADV_WILLNEED) on layers about to be processed.
 */
typedef struct {
    int shard_idx;       /**< Index into safetensors_shard_handles[]. */
    size_t offset;       /**< Byte offset within shard mmap. */
    size_t size;         /**< Total bytes for this layer's tensors. */
} model_layer_page_info_t;


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
    int    num_layers;                       /**< Actual number of entries in layers[]. */
    void  *safetensors_handle;               /**< Single-shard handle (NULL when using shard array). */
    void **safetensors_shard_handles;        /**< Array of per-shard handles (multi-part safetensors). */
    int    safetensors_shard_count;          /**< Number of entries in safetensors_shard_handles. */
    int    low_memory_mode;                  /**< If set, inference uses sliding-window layer eviction. */
    model_layer_page_info_t *layer_page_info; /**< Per-layer mmap ranges for eviction [num_layers]. */
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
 * Evict layer pages from resident memory (MADV_DONTNEED).
 *
 * Use this in a sliding-window inference loop to bound RSS. The mmap
 * stays valid; pages fault back in on next access.
 *
 * @param model Loaded model with layer_page_info populated.
 * @param layer_idx Layer index to evict.
 */
void llm_model_evict_layer(llm_model_t *model, int layer_idx);

/**
 * Prefetch layer pages for upcoming access (MADV_WILLNEED).
 *
 * Starts async readahead for the layer's mmapped pages.
 *
 * @param model Loaded model with layer_page_info populated.
 * @param layer_idx Layer index to prefetch.
 */
void llm_model_prefetch_layer(llm_model_t *model, int layer_idx);

#ifdef __cplusplus
}
#endif

#endif // LLM_MODEL_H

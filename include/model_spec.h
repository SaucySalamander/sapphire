/*
 * @file model_spec.h
 * @brief Generic model specification structures (tensor mapping, tokenizer spec).
 */

#ifndef MODEL_SPEC_H
#define MODEL_SPEC_H

#include "llm_model.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Single tensor map entry (format-agnostic) */
typedef struct {
    const char* hf_name;      /**< Hugging Face tensor name (e.g., "model.embed_tokens.weight") */
    const char* internal_key; /**< Logical grouping key (e.g., "blk.0" or "embedding") */
    const char* field_name;   /**< Field name in llm_model_t/model_layer_weights_t (e.g., "q_proj_weight") */
} tensor_map_entry_t;

/** Tokenizer-related static specification (filenames/ids). */
typedef struct {
    const char* tokenizer_json;     /**< Expected tokenizer.json filename (relative to model dir) */
    const char* tokenizer_model;    /**< Expected tokenizer.model / tokenizer.model filename */
    const char* special_tokens_map; /**< Optional special_tokens_map.json filename */
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
} tokenizer_spec_t;

/** Important files to reference for a model (relative to model directory) */
typedef struct {
    const char* config_json;        /**< e.g., "config.json" */
    const char* tokenizer_json;     /**< e.g., "tokenizer.json" */
    const char* tokenizer_model;    /**< e.g., "tokenizer.model" */
    const char* added_tokens;       /**< e.g., "added_tokens.json" */
    const char* special_tokens_map; /**< e.g., "special_tokens_map.json" */
    const char* generation_config;  /**< e.g., "generation_config.json" */
    const char* chat_template;      /**< e.g., "chat_template.jinja" */
    const char* readme;             /**< e.g., "README.md" */
} model_files_t;

/**
 * Generic model specification combining a default config, tensor map and tokenizer info.
 */
/* Forward declaration so loader hooks can accept a spec pointer */
typedef struct model_spec model_spec_t;

typedef struct model_loader_hooks {
    /**
     * Populate model fields from files referenced in the spec (config.json, tokenizer files, etc.).
     * `spec` points to the owning `model_spec_t` and may be NULL if unknown.
     * Logs detailed errors directly via LOG_ERROR(); caller checks return code only.
     * Returns 0 on success, non-zero on error.
     */
    int (*populate_from_files)(const char* model_dir, const model_spec_t* spec);

    /**
     * Optional postprocessing step invoked after populate_from_files. May compute derived fields.
     */
    void (*postprocess_model)(const model_spec_t* spec);
} model_loader_hooks_t;

struct model_spec {
    const char* model_id;                   /**< Unique model identifier (e.g., "gemma3-270m-it") */
    const tensor_map_entry_t* tensor_map;   /**< Pointer to static tensor mapping table */
    int tensor_map_size;                    /**< Number of entries in tensor_map (excluding sentinel) */
    const tokenizer_spec_t* tokenizer_spec; /**< Tokenizer filenames and id overrides */
    sapphire_tokenizer_t *tokenizer_handle; /**< Optional loaded tokenizer instance (owned by spec) */
    void* llm_model;
    const model_files_t* files;               /**< Optional pointer to important model file paths (relative to model dir) */
    void* variant_config;                     /**< Optional pointer to a model-specific configuration object (loader may populate/cast to the appropriate type, e.g., `gemma3_config_t`) */
    const model_loader_hooks_t* loader_hooks; /**< Optional loader hooks for model-specific parsing/validation */
};

#ifdef __cplusplus
}
#endif

#endif /* MODEL_SPEC_H */

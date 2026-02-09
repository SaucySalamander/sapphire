/*
 * @file gemma3_270m_config.h
 * @brief Gemma 3 (270M IT) model configuration helper (header-only)
 *
 * This header provides canonical configuration values extracted from the
 * model's config.json. It is intended to be included by loaders or tests to
 * initialize missing fields in runtime `model_config_t` structures when the
 * on-disk model metadata is incomplete.
 */

#ifndef GEMMA3_270M_CONFIG_H
#define GEMMA3_270M_CONFIG_H

#include "llm_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* No default config defined here; parsing must populate a `gemma3_config_t` instance explicitly. */

/**
 * Gemma3-specific configuration structure (maps to config.json)
 */
typedef struct {
    /* Per-config fields (directly map to keys in config.json) */
    int sliding_window_pattern;      /* _sliding_window_pattern */
    const char *architectures_first; /* first architecture string (helpers may parse full array if needed) */

    int attention_bias;              /* boolean */
    float attention_dropout;
    float attn_logit_softcapping;    /* nullable (NAN if null) */

    int bos_token_id;
    int eos_token_id;
    float final_logit_softcapping;   /* nullable */
    int head_dim;
    const char *hidden_activation;
    int hidden_size;
    float initializer_range;
    int intermediate_size;

    /* Layer types represented as a bitmask (bit i set => full_attention) */
    unsigned long long layer_types_mask;
    int layer_types_count;

    int max_position_embeddings;
    const char *model_type;

    int num_attention_heads;
    int num_hidden_layers;
    int num_key_value_heads;
    int pad_token_id;
    float query_pre_attn_scalar;
    float rms_norm_eps;
    float rope_local_base_freq;
    float rope_scaling;               /* nullable */
    float rope_theta;
    int sliding_window;
    const char *torch_dtype;
    const char *transformers_version;
    int use_bidirectional_attention;
    int use_cache;
    int vocab_size;
} gemma3_270m_config_t;

/* No default config defined here; parsing must populate a `gemma3_config_t` instance explicitly. */

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_270M_CONFIG_H */

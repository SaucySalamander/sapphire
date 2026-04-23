/*
 * @file gemma3_config.h
 * @brief Generic Gemma 3 runtime configuration shared across model sizes.
 *
 * This header defines the common config.json-backed runtime configuration
 * structure used by Sapphire's Gemma 3 loaders and runtime.
 */

#ifndef GEMMA3_CONFIG_H
#define GEMMA3_CONFIG_H

#include "llm_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gemma3-specific configuration structure (maps to config.json)
 */
typedef struct gemma3_config {
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
    int sapphire_ffn_down_proj_input_rmsnorm;
    int sapphire_mixed_precision_anchors;  /* Model uses ternary+BF16 anchor hybrid */
    uint32_t sapphire_anchor_budget_ppm;   /* Anchor budget in parts-per-million */
    int sliding_window;
    const char *torch_dtype;
    const char *transformers_version;
    int use_bidirectional_attention;
    int use_cache;
    int vocab_size;
} gemma3_config_t;

/* Backward-compatible alias for existing code paths. */
typedef gemma3_config_t gemma3_270m_config_t;

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_CONFIG_H */

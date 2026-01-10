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

/* Canonical fields from models/gemma/270m-it/config.json */
static const int GEMMA3_270M_MAX_CONTEXT = 32768;           /* max_position_embeddings */
static const int GEMMA3_270M_NUM_LAYERS = 18;              /* num_hidden_layers */
static const int GEMMA3_270M_VOCAB_SIZE = 262144;          /* vocab_size */
static const int GEMMA3_270M_D_MODEL = 640;                /* hidden_size */
static const int GEMMA3_270M_D_INNER = 1024;               /* intermediate_size */
static const int GEMMA3_270M_NUM_HEADS = 4;                /* num_attention_heads */
static const int GEMMA3_270M_D_K = 256;                    /* head_dim */
static const int GEMMA3_270M_NUM_KV_HEADS = 1;             /* num_key_value_heads */
static const int GEMMA3_270M_D_FF = 2048;                  /* FFN/intermediate size */
static const float GEMMA3_270M_ROPE_BASE = 1000000.0f;     /* rope_theta */
static const float GEMMA3_270M_ROPE_LOCAL_BASE = 10000.0f; /* rope_local_base_freq */
static const int GEMMA3_270M_SLIDING_WINDOW = 512;         /* sliding_window */
static const int GEMMA3_270M_SLIDING_PATTERN = 6;          /* _sliding_window_pattern */
static const float GEMMA3_270M_QUERY_PRE_ATTN = 256.0f;    /* query_pre_attn_scalar */

/* Additional fields present in config.json */
static const int GEMMA3_270M_BOS_TOKEN_ID = 2;
static const int GEMMA3_270M_EOS_TOKEN_ID = 1;
static const int GEMMA3_270M_PAD_TOKEN_ID = 0;
static const int GEMMA3_270M_HEAD_DIM = 256;               /* same as GEMMA3_270M_D_K */
static const float GEMMA3_270M_INIT_RANGE = 0.02f;         /* initializer_range */
static const char GEMMA3_270M_HIDDEN_ACTIVATION[] = "gelu_pytorch_tanh";
static const char GEMMA3_270M_MODEL_TYPE[] = "gemma3_text";
static const char GEMMA3_270M_TORCH_DTYPE[] = "bfloat16";
static const char GEMMA3_270M_TRANSFORMERS_VERSION[] = "4.55.0.dev0";
static const float GEMMA3_270M_RMS_NORM_EPS = 1e-06f;
static const int GEMMA3_270M_USE_BIDIRECTIONAL_ATTENTION = 0; /* use_bidirectional_attention */
static const int GEMMA3_270M_USE_CACHE = 1;                 /* use_cache */
static const float GEMMA3_270M_ATTENTION_DROPOUT = 0.0f;     /* attention_dropout */
static const int GEMMA3_270M_ATTENTION_BIAS = 0;            /* attention_bias (false) */

/* Nullable fields in config.json (represented as NAN if missing) */
#include <math.h>
static const float GEMMA3_270M_ATTN_LOGIT_SOFTCAPPING = NAN; /* attn_logit_softcapping (null) */
static const float GEMMA3_270M_FINAL_LOGIT_SOFTCAPPING = NAN; /* final_logit_softcapping (null) */
static const float GEMMA3_270M_ROPE_SCALING = NAN;          /* rope_scaling (null) */

/* Layer types - represented as a bitmask for compactness. Bit i is 1 if layer i uses full attention. */
/* From models/gemma/270m-it/config.json: full_attention at indices 5, 11, 17 (0-based) */
static const unsigned int GEMMA3_270M_LAYER_TYPE_FULL_MASK = 0x00020820u; /* bits: 17,11,5 */


/**
 * Apply recommended Gemma3 defaults into an existing model_config_t.
 * Only sets fields that are zero or non-positive, preserving explicit values
 * parsed from the model files.
 */
/* Gemma3-specific configuration structure (maps to config.json) */
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

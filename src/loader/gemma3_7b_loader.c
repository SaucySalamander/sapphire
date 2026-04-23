/*
 * @file gemma3_7b_loader.c
 * @brief Loader hooks and static spec objects for Gemma 3 7B IT.
 *
 * This loader keeps the Gemma 3 HF tensor naming convention used by the
 * existing 4B/27B assets and loads runtime configuration from the local
 * model directory's config.json.
 */

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "file_reader.h"
#include "gemma3_config.h"
#include "gemma3_7b_spec.h"
#include "layer_config_loader.h"
#include "llm_model.h"
#include "log.h"
#include "model_reader.h"
#include "model_spec.h"
#include "simple_json.h"
#include "tokenizer.h"

const tokenizer_spec_t GEMMA3_7B_TOKENIZER_SPEC = {
    .tokenizer_json     = "tokenizer.json",
    .tokenizer_model    = "tokenizer.model",
    .special_tokens_map = "special_tokens_map.json",
    .bos_token_id       = 2,
    .eos_token_id       = 1,
    .pad_token_id       = 0
};

const model_files_t GEMMA3_7B_FILES = {
    .config_json        = "config.json",
    .tokenizer_json     = "tokenizer.json",
    .tokenizer_model    = "tokenizer.model",
    .added_tokens       = "added_tokens.json",
    .special_tokens_map = "special_tokens_map.json",
    .generation_config  = "generation_config.json",
    .chat_template      = "chat_template.jinja",
    .readme             = "README.md"
};

gemma3_270m_config_t GEMMA3_7B_RUNTIME_CONFIG = {0};

model_spec_t GEMMA3_7B_IT_SPEC = {
    .model_id        = "gemma-3-7b-it",
    .tensor_map      = GEMMA3_7B_TENSOR_MAP,
    .tensor_map_size = GEMMA3_7B_TENSOR_MAP_SIZE,
    .tokenizer_spec  = &GEMMA3_7B_TOKENIZER_SPEC,
    .files           = &GEMMA3_7B_FILES,
    .variant_config  = &GEMMA3_7B_RUNTIME_CONFIG,
    .loader_hooks    = &GEMMA3_7B_LOADER_HOOKS
};

typedef struct {
    unsigned long long mask;
    int index;
    int limit;
} layer_types_build_ctx_t;

static int find_key_in_object_or_root(const char *json,
                                      const sjson_token_t *tokens,
                                      int nt,
                                      int obj_idx,
                                      const char *key)
{
    int value_idx = -1;

    if (!json || !tokens || nt <= 0 || !key) {
        return -1;
    }

    if (obj_idx >= 0 && obj_idx < nt) {
        value_idx = sjson_find_key(json, tokens, nt, obj_idx, key);
        if (value_idx >= 0) {
            return value_idx;
        }
    }

    if (obj_idx != 0) {
        value_idx = sjson_find_key(json, tokens, nt, 0, key);
    }
    return value_idx;
}

static int parse_int_token_at(const char *json,
                              const sjson_token_t *tokens,
                              int nt,
                              int token_idx,
                              int *out_value)
{
    const sjson_token_t *token = NULL;
    int64_t value = 0;

    if (!json || !tokens || nt <= 0 || token_idx < 0 || token_idx >= nt || !out_value) {
        return -1;
    }

    token = &tokens[token_idx];
    if (token->type == SJSON_ARR) {
        if (token->size <= 0) {
            return -1;
        }
        return parse_int_token_at(json, tokens, nt, token_idx + 1, out_value);
    }

    if (sjson_token_to_int64(json, token, &value) != 0) {
        return -1;
    }

    *out_value = (int)value;
    return 0;
}

static int parse_float_token_at(const char *json,
                               const sjson_token_t *tokens,
                               int nt,
                               int token_idx,
                               float *out_value)
{
    const sjson_token_t *token = NULL;
    double value = 0.0;

    if (!json || !tokens || nt <= 0 || token_idx < 0 || token_idx >= nt || !out_value) {
        return -1;
    }

    token = &tokens[token_idx];
    if (token->type == SJSON_ARR) {
        if (token->size <= 0) {
            return -1;
        }
        return parse_float_token_at(json, tokens, nt, token_idx + 1, out_value);
    }

    if (sjson_token_to_double(json, token, &value) != 0) {
        return -1;
    }

    *out_value = (float)value;
    return 0;
}

static int parse_u64_token_at(const char *json,
                              const sjson_token_t *tokens,
                              int nt,
                              int token_idx,
                              unsigned long long *out_value)
{
    const sjson_token_t *token = NULL;
    char buffer[64];
    char *endptr = NULL;
    unsigned long long value = 0ULL;

    if (!json || !tokens || nt <= 0 || token_idx < 0 || token_idx >= nt || !out_value) {
        return -1;
    }

    token = &tokens[token_idx];
    if (token->type == SJSON_ARR) {
        if (token->size <= 0) {
            return -1;
        }
        return parse_u64_token_at(json, tokens, nt, token_idx + 1, out_value);
    }

    if (token->type == SJSON_STR) {
        if (sjson_token_to_str(json, token, buffer, (int)sizeof(buffer)) != 0) {
            return -1;
        }
    } else {
        int len = token->end - token->start;
        if (len <= 0 || len >= (int)sizeof(buffer)) {
            return -1;
        }
        memcpy(buffer, json + token->start, (size_t)len);
        buffer[len] = '\0';
    }

    errno = 0;
    value = strtoull(buffer, &endptr, 0);
    if (errno != 0 || endptr == buffer || *endptr != '\0') {
        return -1;
    }

    *out_value = value;
    return 0;
}

static int parse_int_field(const char *json,
                           const sjson_token_t *tokens,
                           int nt,
                           int obj_idx,
                           const char *key,
                           int *out_value)
{
    int value_idx = find_key_in_object_or_root(json, tokens, nt, obj_idx, key);
    if (value_idx < 0) {
        return 1;
    }
    if (parse_int_token_at(json, tokens, nt, value_idx, out_value) != 0) {
        return -1;
    }
    return 0;
}

static int parse_float_field(const char *json,
                             const sjson_token_t *tokens,
                             int nt,
                             int obj_idx,
                             const char *key,
                             float *out_value)
{
    int value_idx = find_key_in_object_or_root(json, tokens, nt, obj_idx, key);
    if (value_idx < 0) {
        return 1;
    }
    if (parse_float_token_at(json, tokens, nt, value_idx, out_value) != 0) {
        return -1;
    }
    return 0;
}

static void parse_7b_sapphire_feature_flags(const char *json,
                                            const sjson_token_t *tokens,
                                            int nt,
                                            int obj_idx,
                                            gemma3_270m_config_t *cfg)
{
    int value_idx = -1;
    int len = 0;
    int budget_ppm = 0;

    if (!json || !tokens || nt <= 0 || !cfg) {
        return;
    }

    value_idx = find_key_in_object_or_root(json,
                                           tokens,
                                           nt,
                                           obj_idx,
                                           "sapphire_ffn_down_proj_input_rmsnorm");
    if (value_idx < 0) {
        return;
    }

    len = tokens[value_idx].end - tokens[value_idx].start;
    cfg->sapphire_ffn_down_proj_input_rmsnorm =
        (len == 4 && strncmp(json + tokens[value_idx].start, "true", 4) == 0) ? 1 : 0;

    value_idx = find_key_in_object_or_root(json,
                                           tokens,
                                           nt,
                                           obj_idx,
                                           "sapphire_mixed_precision_anchors");
    if (value_idx >= 0) {
        len = tokens[value_idx].end - tokens[value_idx].start;
        cfg->sapphire_mixed_precision_anchors =
            (len == 4 && strncmp(json + tokens[value_idx].start, "true", 4) == 0) ? 1 : 0;
    }

    value_idx = find_key_in_object_or_root(json,
                                           tokens,
                                           nt,
                                           obj_idx,
                                           "sapphire_anchor_budget_ppm");
    if (value_idx >= 0 && parse_int_token_at(json, tokens, nt, value_idx, &budget_ppm) == 0 && budget_ppm >= 0) {
        cfg->sapphire_anchor_budget_ppm = (uint32_t)budget_ppm;
    }
}

static int is_global_layer_type_string(const char *value, int len)
{
    if (!value || len <= 0) {
        return -1;
    }

    if ((len == 1 && value[0] == '1') ||
        (len >= 4 && strncmp(value, "full", 4) == 0) ||
        (len >= 6 && strncmp(value, "global", 6) == 0) ||
        (len >= 4 && strncmp(value, "true", 4) == 0)) {
        return 1;
    }

    if ((len == 1 && value[0] == '0') ||
        (len >= 5 && strncmp(value, "local", 5) == 0) ||
        (len >= 7 && strncmp(value, "sliding", 7) == 0) ||
        (len >= 5 && strncmp(value, "false", 5) == 0)) {
        return 0;
    }

    return -1;
}

static int layer_types_numeric_cb(uint32_t value, void *user)
{
    layer_types_build_ctx_t *ctx = (layer_types_build_ctx_t *)user;

    if (!ctx || ctx->index >= ctx->limit) {
        return -1;
    }

    if (value != 0u) {
        ctx->mask |= (1ULL << ctx->index);
    }
    ctx->index++;
    return 0;
}

static int layer_types_string_cb(const char *value, int len, void *user)
{
    layer_types_build_ctx_t *ctx = (layer_types_build_ctx_t *)user;
    int is_global = 0;

    if (!ctx || ctx->index >= ctx->limit) {
        return -1;
    }

    is_global = is_global_layer_type_string(value, len);
    if (is_global < 0) {
        return -1;
    }
    if (is_global > 0) {
        ctx->mask |= (1ULL << ctx->index);
    }
    ctx->index++;
    return 0;
}

static int parse_layer_types_from_array(const char *json,
                                        const sjson_token_t *tokens,
                                        int nt,
                                        int obj_idx,
                                        gemma3_270m_config_t *cfg)
{
    int arr_idx = find_key_in_object_or_root(json, tokens, nt, obj_idx, "layer_types");
    layer_types_build_ctx_t ctx;
    int rc = -1;

    if (arr_idx < 0) {
        return 1;
    }
    if (tokens[arr_idx].type != SJSON_ARR || tokens[arr_idx].size <= 0) {
        LOG_ERROR("gemma3_7b_loader: layer_types must be a non-empty array");
        return -1;
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.limit = cfg->num_hidden_layers;

    if (tokens[arr_idx + 1].type == SJSON_STR) {
        rc = sjson_iterate_str_array(json, tokens, nt, arr_idx, layer_types_string_cb, &ctx);
    } else {
        rc = sjson_iterate_num_array(json, tokens, nt, arr_idx, layer_types_numeric_cb, &ctx);
    }
    if (rc < 0 || ctx.index != cfg->num_hidden_layers) {
        LOG_ERROR("gemma3_7b_loader: layer_types count mismatch (got=%d expected=%d)",
                  ctx.index, cfg->num_hidden_layers);
        return -1;
    }

    cfg->layer_types_mask = ctx.mask;
    cfg->layer_types_count = ctx.index;
    return 0;
}

static int parse_layer_types_mask_field(const char *json,
                                        const sjson_token_t *tokens,
                                        int nt,
                                        int obj_idx,
                                        gemma3_270m_config_t *cfg)
{
    int value_idx = find_key_in_object_or_root(json, tokens, nt, obj_idx, "layer_types_mask");
    unsigned long long mask = 0ULL;

    if (value_idx < 0) {
        return 1;
    }
    if (parse_u64_token_at(json, tokens, nt, value_idx, &mask) != 0) {
        LOG_ERROR("gemma3_7b_loader: failed to parse layer_types_mask");
        return -1;
    }
    if (cfg->num_hidden_layers < 64 && (mask >> (unsigned)cfg->num_hidden_layers) != 0ULL) {
        LOG_ERROR("gemma3_7b_loader: layer_types_mask has bits beyond num_hidden_layers=%d",
                  cfg->num_hidden_layers);
        return -1;
    }

    cfg->layer_types_mask = mask;
    cfg->layer_types_count = cfg->num_hidden_layers;
    return 0;
}

static void build_layer_types_from_pattern(gemma3_270m_config_t *cfg)
{
    int pattern = 6;
    unsigned long long mask = 0ULL;

    if (!cfg || cfg->num_hidden_layers <= 0 || cfg->layer_types_mask != 0ULL) {
        return;
    }

    if (cfg->sliding_window_pattern > 0) {
        pattern = cfg->sliding_window_pattern;
    }

    for (int i = 0; i < cfg->num_hidden_layers && i < 64; ++i) {
        if (pattern > 1 && ((i + 1) % pattern) == 0) {
            mask |= (1ULL << i);
        }
    }

    cfg->layer_types_mask = mask;
    cfg->layer_types_count = cfg->num_hidden_layers;
}

static int validate_7b_layer_configs(const model_spec_t *spec,
                                     const gemma3_270m_config_t *cfg)
{
    sapphire_layer_config_t *layer_configs = NULL;
    model_spec_t temp_spec;
    int rc = -1;

    if (!spec || !cfg || cfg->num_hidden_layers <= 0) {
        return -1;
    }

    layer_configs = (sapphire_layer_config_t *)calloc((size_t)cfg->num_hidden_layers,
                                                      sizeof(*layer_configs));
    if (!layer_configs) {
        LOG_ERROR("gemma3_7b_loader: OOM allocating layer config validation buffer");
        return -1;
    }

    temp_spec = *spec;
    temp_spec.variant_config = (void *)cfg;
    rc = layer_config_load_from_spec(&temp_spec, cfg->num_hidden_layers, layer_configs);
    free(layer_configs);
    return rc;
}

static void init_7b_runtime_config(gemma3_270m_config_t *cfg)
{
    if (!cfg) {
        return;
    }

    cfg->bos_token_id = 2;
    cfg->eos_token_id = 1;
    cfg->pad_token_id = 0;
    cfg->vocab_size = 262144;
    cfg->sliding_window = 1024;
    cfg->sliding_window_pattern = 6;
    cfg->rms_norm_eps = 1e-6f;
    cfg->rope_theta = 1000000.0f;
    cfg->rope_local_base_freq = 10000.0f;
    cfg->attn_logit_softcapping = NAN;
    cfg->final_logit_softcapping = NAN;
    cfg->rope_scaling = NAN;
}

static int load_7b_required_dimensions(const char *json,
                                       const sjson_token_t *tokens,
                                       int nt,
                                       int cfg_obj_idx,
                                       gemma3_270m_config_t *cfg)
{
    if (parse_int_field(json, tokens, nt, cfg_obj_idx, "hidden_size", &cfg->hidden_size) != 0) {
        return -1;
    }
    if (parse_int_field(json, tokens, nt, cfg_obj_idx, "intermediate_size", &cfg->intermediate_size) != 0) {
        return -1;
    }
    if (parse_int_field(json, tokens, nt, cfg_obj_idx, "num_hidden_layers", &cfg->num_hidden_layers) != 0) {
        return -1;
    }
    if (parse_int_field(json, tokens, nt, cfg_obj_idx, "num_attention_heads", &cfg->num_attention_heads) != 0) {
        return -1;
    }
    if (parse_int_field(json, tokens, nt, cfg_obj_idx, "num_key_value_heads", &cfg->num_key_value_heads) != 0) {
        return -1;
    }
    if (parse_int_field(json, tokens, nt, cfg_obj_idx, "head_dim", &cfg->head_dim) != 0) {
        return -1;
    }
    return 0;
}

static void load_7b_optional_fields(const char *json,
                                    const sjson_token_t *tokens,
                                    int nt,
                                    int cfg_obj_idx,
                                    gemma3_270m_config_t *cfg)
{
    int result;

    result = parse_int_field(json, tokens, nt, cfg_obj_idx, "vocab_size", &cfg->vocab_size);
    if (result == 1 || cfg->vocab_size <= 0) {
        LOG_WARN("gemma3_7b_loader: vocab_size missing; defaulting to 262144");
        cfg->vocab_size = 262144;
    }

    result = parse_int_field(json, tokens, nt, cfg_obj_idx, "sliding_window", &cfg->sliding_window);
    if (result == 1 || cfg->sliding_window <= 0) {
        LOG_WARN("gemma3_7b_loader: sliding_window missing; defaulting to 1024");
        cfg->sliding_window = 1024;
    }

    parse_int_field(json, tokens, nt, cfg_obj_idx, "sliding_window_pattern", &cfg->sliding_window_pattern);
    parse_float_field(json, tokens, nt, cfg_obj_idx, "query_pre_attn_scalar", &cfg->query_pre_attn_scalar);
    parse_float_field(json, tokens, nt, cfg_obj_idx, "rms_norm_eps", &cfg->rms_norm_eps);
    parse_float_field(json, tokens, nt, cfg_obj_idx, "attention_dropout", &cfg->attention_dropout);
    parse_float_field(json, tokens, nt, cfg_obj_idx, "attn_logit_softcapping", &cfg->attn_logit_softcapping);
    parse_float_field(json, tokens, nt, cfg_obj_idx, "final_logit_softcapping", &cfg->final_logit_softcapping);

    result = parse_float_field(json, tokens, nt, cfg_obj_idx, "rope_local_base_freq", &cfg->rope_local_base_freq);
    if (result == 1 || cfg->rope_local_base_freq <= 0.0f) {
        LOG_WARN("gemma3_7b_loader: rope_local_base_freq missing; defaulting to 10000");
        cfg->rope_local_base_freq = 10000.0f;
    }

    result = parse_float_field(json, tokens, nt, cfg_obj_idx, "rope_theta", &cfg->rope_theta);
    if (result == 1 &&
        parse_float_field(json, tokens, nt, cfg_obj_idx, "rope_theta_base", &cfg->rope_theta) == 1 &&
        parse_float_field(json, tokens, nt, cfg_obj_idx, "rope_base_freq", &cfg->rope_theta) == 1) {
        LOG_WARN("gemma3_7b_loader: rope_theta missing; defaulting to 1000000");
        cfg->rope_theta = 1000000.0f;
    }

    parse_int_field(json, tokens, nt, cfg_obj_idx, "bos_token_id", &cfg->bos_token_id);
    parse_int_field(json, tokens, nt, cfg_obj_idx, "pad_token_id", &cfg->pad_token_id);
}

static void load_7b_token_ids(const char *json,
                              const sjson_token_t *tokens,
                              int nt,
                              int cfg_obj_idx,
                              gemma3_270m_config_t *cfg)
{
    if (parse_int_field(json, tokens, nt, cfg_obj_idx, "eos_token_id", &cfg->eos_token_id) == 1) {
        int eos_idx = find_key_in_object_or_root(json, tokens, nt, cfg_obj_idx, "eos_token_id");
        if (eos_idx >= 0 && tokens[eos_idx].type == SJSON_ARR && tokens[eos_idx].size > 0) {
            if (parse_int_token_at(json, tokens, nt, eos_idx + 1, &cfg->eos_token_id) != 0) {
                LOG_WARN("gemma3_7b_loader: failed to parse eos_token_id array; keeping default");
            }
        }
    }
}

static int load_7b_layer_types(const char *json,
                               const sjson_token_t *tokens,
                               int nt,
                               int cfg_obj_idx,
                               gemma3_270m_config_t *cfg)
{
    int result = parse_layer_types_mask_field(json, tokens, nt, cfg_obj_idx, cfg);
    if (result < 0) {
        return -1;
    }
    if (cfg->layer_types_mask == 0ULL) {
        result = parse_layer_types_from_array(json, tokens, nt, cfg_obj_idx, cfg);
        if (result < 0) {
            return -1;
        }
        if (result == 1) {
            LOG_WARN("gemma3_7b_loader: layer_types metadata missing; using sliding_window_pattern fallback");
        }
    }

    build_layer_types_from_pattern(cfg);
    return 0;
}

static int validate_7b_runtime_config(const model_spec_t *spec,
                                      gemma3_270m_config_t *cfg)
{
    if (cfg->num_hidden_layers <= 0 || cfg->num_hidden_layers > GEMMA3_7B_MAX_LAYERS) {
        LOG_ERROR("gemma3_7b_loader: unsupported layer count=%d (max=%d)",
                  cfg->num_hidden_layers, GEMMA3_7B_MAX_LAYERS);
        return -1;
    }
    if (cfg->hidden_size <= 0 || cfg->intermediate_size <= 0 || cfg->head_dim <= 0) {
        LOG_ERROR("gemma3_7b_loader: incomplete dimensions (hidden=%d inter=%d head_dim=%d)",
                  cfg->hidden_size, cfg->intermediate_size, cfg->head_dim);
        return -1;
    }
    if (cfg->num_attention_heads <= 0 || cfg->num_key_value_heads <= 0) {
        LOG_ERROR("gemma3_7b_loader: missing attention head counts (heads=%d kv=%d)",
                  cfg->num_attention_heads, cfg->num_key_value_heads);
        return -1;
    }
    if ((cfg->num_attention_heads % cfg->num_key_value_heads) != 0) {
        LOG_ERROR("gemma3_7b_loader: invalid attention ratio heads=%d kv=%d",
                  cfg->num_attention_heads, cfg->num_key_value_heads);
        return -1;
    }

    if (cfg->rope_theta <= 0.0f) {
        cfg->rope_theta = 1000000.0f;
    }
    if (cfg->rope_local_base_freq <= 0.0f) {
        cfg->rope_local_base_freq = 10000.0f;
    }

    if (validate_7b_layer_configs(spec, cfg) != 0) {
        LOG_ERROR("gemma3_7b_loader: layer configuration validation failed");
        return -1;
    }
    return 0;
}

static void apply_7b_runtime_config(model_spec_t *spec,
                                    const gemma3_270m_config_t *cfg)
{
    spec->variant_config = (void *)cfg;
    spec->tensor_map_size = (int)((size_t)cfg->num_hidden_layers * 13u + 3u);
}

static int load_7b_config(const char *model_dir, const model_spec_t *spec)
{
    char *cfg_path = NULL;
    char *json = NULL;
    size_t json_len = 0;
    sjson_token_t tokens[4096];
    gemma3_270m_config_t *cfg = NULL;
    int nt = -1;
    int cfg_obj_idx = 0;
    int rc = -1;
    model_spec_t *mutable_spec = NULL;

    if (!model_dir || !spec) {
        return -1;
    }

    cfg_path = construct_safe_path(model_dir, "config.json", NULL);
    if (!cfg_path) {
        return -1;
    }

    if (file_read_json(cfg_path, &json, &json_len) != 0) {
        free(cfg_path);
        return -1;
    }
    free(cfg_path);
    (void)json_len;

    nt = sjson_tokenize(json, tokens, (int)(sizeof(tokens) / sizeof(tokens[0])));
    if (nt < 0) {
        LOG_ERROR("gemma3_7b_loader: failed to tokenize config.json");
        goto done;
    }

    cfg_obj_idx = sjson_find_key(json, tokens, nt, 0, "text_config");
    if (cfg_obj_idx >= 0 && tokens[cfg_obj_idx].type != SJSON_OBJ) {
        LOG_ERROR("gemma3_7b_loader: text_config must be an object");
        goto done;
    }
    if (cfg_obj_idx < 0) {
        cfg_obj_idx = 0;
    }

    cfg = (gemma3_270m_config_t *)calloc(1, sizeof(*cfg));
    if (!cfg) {
        LOG_ERROR("gemma3_7b_loader: OOM allocating config");
        goto done;
    }

    init_7b_runtime_config(cfg);

    if (load_7b_required_dimensions(json, tokens, nt, cfg_obj_idx, cfg) != 0) {
        goto missing_required;
    }

    load_7b_optional_fields(json, tokens, nt, cfg_obj_idx, cfg);
    load_7b_token_ids(json, tokens, nt, cfg_obj_idx, cfg);
    parse_7b_sapphire_feature_flags(json, tokens, nt, cfg_obj_idx, cfg);

    if (load_7b_layer_types(json, tokens, nt, cfg_obj_idx, cfg) != 0) {
        goto missing_required;
    }

    if (validate_7b_runtime_config(spec, cfg) != 0) {
        goto done;
    }

    mutable_spec = (model_spec_t *)(void *)spec;
    apply_7b_runtime_config(mutable_spec, cfg);

    LOG_INFO("gemma3_7b: hidden=%d inter=%d layers=%d heads=%d kv=%d head_dim=%d vocab=%d rope_theta=%.0f rope_local=%.0f",
             cfg->hidden_size,
             cfg->intermediate_size,
             cfg->num_hidden_layers,
             cfg->num_attention_heads,
             cfg->num_key_value_heads,
             cfg->head_dim,
             cfg->vocab_size,
             (double)cfg->rope_theta,
             (double)cfg->rope_local_base_freq);

    LOG_INFO("gemma3_7b: tensor map size set to %d entries", mutable_spec->tensor_map_size);

    rc = 0;
    goto done;

missing_required:
    LOG_ERROR("gemma3_7b_loader: config.json is missing required 7B fields");

    if (cfg) {
        free(cfg);
        cfg = NULL;
    }

done:
    free(json);
    return rc;
}

static int gemma3_7b_populate_from_files(const char *model_dir, const model_spec_t *spec)
{
    llm_model_t *loaded = NULL;
    llm_model_t *model = NULL;
    sapphire_tokenizer_t *tok = NULL;

    if (!model_dir || !spec || !spec->llm_model) {
        LOG_ERROR("gemma3_7b_loader: invalid arguments");
        return -1;
    }

    if (load_7b_config(model_dir, spec) != 0) {
        LOG_ERROR("gemma3_7b_loader: failed to load config.json from %s", model_dir);
        return -1;
    }

    loaded = load_model(model_dir, spec);
    if (!loaded) {
        LOG_ERROR("gemma3_7b_loader: failed to load weights from %s", model_dir);
        return -1;
    }

    model = (llm_model_t *)spec->llm_model;
    memcpy(model, loaded, sizeof(*loaded));
    free(loaded);

    tok = tokenizer_load(model_dir);
    if (!tok) {
        LOG_WARN("gemma3_7b_loader: tokenizer not found in %s", model_dir);
    } else {
        const gemma3_270m_config_t *cfg = (const gemma3_270m_config_t *)spec->variant_config;
        if (cfg) {
            if (cfg->bos_token_id >= 0) tok->bos_token_id = cfg->bos_token_id;
            if (cfg->eos_token_id >= 0) tok->eos_token_id = cfg->eos_token_id;
            if (cfg->pad_token_id >= 0) tok->pad_token_id = cfg->pad_token_id;
        }
        ((model_spec_t *)(void *)spec)->tokenizer_handle = tok;
    }

    return 0;
}

static void gemma3_7b_postprocess_model(const model_spec_t *spec)
{
    (void)spec;
}

const model_loader_hooks_t GEMMA3_7B_LOADER_HOOKS = {
    .populate_from_files = (int (*)(const char *, const model_spec_t *))gemma3_7b_populate_from_files,
    .postprocess_model   = (void (*)(const model_spec_t *))gemma3_7b_postprocess_model
};
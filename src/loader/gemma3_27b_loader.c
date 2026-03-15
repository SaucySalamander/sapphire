/*
 * @file gemma3_27b_loader.c
 * @brief Loader hooks and static spec objects for Gemma 3 27B IT.
 *
 * Key differences from the 270M loader:
 *  - config.json fields are nested under "text_config" (not at root)
 *  - layer_types_mask is derived from the 5:1 local/global pattern
 *    (global at layers where layer_idx % 6 == 5)
 *  - Tensor prefix: language_model.model.layers.N.*
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "file_reader.h"
#include "gemma3_270m_config.h"
#include "gemma3_27b_spec.h"
#include "layer_config_loader.h"
#include "llm_model.h"
#include "log.h"
#include "model_reader.h"
#include "model_spec.h"
#include "simple_json.h"
#include "tokenizer.h"

/* -------------------------------------------------------------------------
 * Static data objects
 * -------------------------------------------------------------------------*/

const tokenizer_spec_t GEMMA3_27B_TOKENIZER_SPEC = {
    .tokenizer_json     = "tokenizer.json",
    .tokenizer_model    = "tokenizer.model",
    .special_tokens_map = "special_tokens_map.json",
    .bos_token_id       = 2,
    .eos_token_id       = 1,
    .pad_token_id       = 0
};

const model_files_t GEMMA3_27B_FILES = {
    .config_json        = "config.json",
    .tokenizer_json     = "tokenizer.json",
    .tokenizer_model    = "tokenizer.model",
    .added_tokens       = "added_tokens.json",
    .special_tokens_map = "special_tokens_map.json",
    .generation_config  = "generation_config.json",
    .chat_template      = "chat_template.jinja",
    .readme             = "README.md"
};

gemma3_270m_config_t GEMMA3_27B_RUNTIME_CONFIG = {0};

model_spec_t GEMMA3_27B_IT_SPEC = {
    .model_id        = "gemma-3-27b-it",
    .tensor_map      = GEMMA3_27B_TENSOR_MAP,
    .tensor_map_size = GEMMA3_27B_TENSOR_MAP_SIZE,
    .tokenizer_spec  = &GEMMA3_27B_TOKENIZER_SPEC,
    .files           = &GEMMA3_27B_FILES,
    .variant_config  = &GEMMA3_27B_RUNTIME_CONFIG,
    .loader_hooks    = &GEMMA3_27B_LOADER_HOOKS
};

/* -------------------------------------------------------------------------
 * 27B-specific field table (int or float, offset into gemma3_270m_config_t)
 * -------------------------------------------------------------------------*/

typedef struct {
    const char *key;
    size_t      off;
    int         is_float; /* 0 = int, 1 = float */
} g27b_field_t;

static const g27b_field_t G27B_TC_FIELDS[] = {
    {"hidden_size",           offsetof(gemma3_270m_config_t, hidden_size),           0},
    {"intermediate_size",     offsetof(gemma3_270m_config_t, intermediate_size),     0},
    {"num_hidden_layers",     offsetof(gemma3_270m_config_t, num_hidden_layers),     0},
    {"num_attention_heads",   offsetof(gemma3_270m_config_t, num_attention_heads),   0},
    {"num_key_value_heads",   offsetof(gemma3_270m_config_t, num_key_value_heads),   0},
    {"head_dim",              offsetof(gemma3_270m_config_t, head_dim),              0},
    {"sliding_window",        offsetof(gemma3_270m_config_t, sliding_window),        0},
    {"vocab_size",            offsetof(gemma3_270m_config_t, vocab_size),            0},
    {"query_pre_attn_scalar", offsetof(gemma3_270m_config_t, query_pre_attn_scalar), 1},
    {"rope_theta",            offsetof(gemma3_270m_config_t, rope_theta),            1},
    {"rope_local_base_freq",  offsetof(gemma3_270m_config_t, rope_local_base_freq),  1},
    {"rms_norm_eps",          offsetof(gemma3_270m_config_t, rms_norm_eps),          1},
};

#define G27B_TC_FIELDS_COUNT \
    (sizeof(G27B_TC_FIELDS) / sizeof(G27B_TC_FIELDS[0]))

/* -------------------------------------------------------------------------
 * Parse all numeric fields from the text_config object
 * -------------------------------------------------------------------------*/

static void parse_27b_tc_fields(const char *json, const sjson_token_t *tokens, int nt,
                                 int tc_idx, gemma3_270m_config_t *cfg)
{
    for (size_t i = 0; i < G27B_TC_FIELDS_COUNT; ++i) {
        int vi = sjson_find_key(json, tokens, nt, tc_idx, G27B_TC_FIELDS[i].key);
        if (vi < 0) continue;

        double dv = 0.0;
        if (sjson_token_to_double(json, &tokens[vi], &dv) != 0) continue;

        if (G27B_TC_FIELDS[i].is_float) {
            float *fp = (float *)(void *)((char *)cfg + G27B_TC_FIELDS[i].off);
            *fp = (float)dv;
        } else {
            int *ip = (int *)(void *)((char *)cfg + G27B_TC_FIELDS[i].off);
            *ip = (int)(dv + 0.5);
        }
    }
}

/* -------------------------------------------------------------------------
 * Parse rope_scaling.factor from inside text_config
 * -------------------------------------------------------------------------*/

static void parse_27b_rope_scaling(const char *json, const sjson_token_t *tokens, int nt,
                                    int tc_idx, gemma3_270m_config_t *cfg)
{
    int rs_idx = sjson_find_key(json, tokens, nt, tc_idx, "rope_scaling");
    if (rs_idx < 0 || tokens[rs_idx].type != SJSON_OBJ) return;

    int factor_idx = sjson_find_key(json, tokens, nt, rs_idx, "factor");
    if (factor_idx < 0) return;

    double dv = 0.0;
    if (sjson_token_to_double(json, &tokens[factor_idx], &dv) == 0) {
        cfg->rope_scaling = (float)dv;
    }
}

/* -------------------------------------------------------------------------
 * Build layer_types_mask: global attention at (layer_idx % 6) == 5
 * -------------------------------------------------------------------------*/

static void build_27b_layer_types_mask(gemma3_270m_config_t *cfg)
{
    int n = cfg->num_hidden_layers > 0 ? cfg->num_hidden_layers : 62;
    unsigned long long mask = 0ULL;

    for (int i = 0; i < n && i < 64; ++i) {
        if ((i % 6) == 5) {
            mask |= (1ULL << i);
        }
    }
    cfg->layer_types_mask  = mask;
    cfg->layer_types_count = n;
}

/* -------------------------------------------------------------------------
 * Load config.json, navigate into text_config, populate cfg
 * -------------------------------------------------------------------------*/

static int load_27b_config(const char *model_dir, model_spec_t *spec)
{
    char *cfg_path = construct_safe_path(model_dir, "config.json", NULL);
    if (!cfg_path) return -1;

    char  *json     = NULL;
    size_t json_len = 0;
    int    rc       = -1;

    if (file_read_json(cfg_path, &json, &json_len) != 0) {
        free(cfg_path);
        return -1;
    }
    free(cfg_path);

    sjson_token_t tokens[4096];
    int nt = sjson_tokenize(json, tokens, (int)(sizeof(tokens) / sizeof(tokens[0])));
    if (nt < 0) {
        LOG_ERROR("gemma3_27b_loader: failed to tokenize config.json");
        goto done;
    }

    /* Locate the text_config object */
    int tc_idx = sjson_find_key(json, tokens, nt, 0, "text_config");
    if (tc_idx < 0 || tokens[tc_idx].type != SJSON_OBJ) {
        LOG_ERROR("gemma3_27b_loader: text_config not found in config.json");
        goto done;
    }

    gemma3_270m_config_t *cfg = (gemma3_270m_config_t *)calloc(1, sizeof(*cfg));
    if (!cfg) {
        LOG_ERROR("gemma3_27b_loader: OOM allocating config");
        goto done;
    }

    /* Hardcoded 27B defaults (overridden by whatever is in JSON) */
    cfg->vocab_size          = 262144;
    cfg->rope_theta          = 1000000.0f;
    cfg->rope_local_base_freq = 10000.0f;
    cfg->rms_norm_eps        = 1e-6f;
    cfg->bos_token_id        = 2;
    cfg->eos_token_id        = 1;
    cfg->pad_token_id        = 0;
    cfg->attn_logit_softcapping  = NAN;
    cfg->final_logit_softcapping = NAN;
    cfg->rope_scaling            = NAN;

    /* Parse all JSON fields from text_config */
    parse_27b_tc_fields(json, tokens, nt, tc_idx, cfg);
    parse_27b_rope_scaling(json, tokens, nt, tc_idx, cfg);
    build_27b_layer_types_mask(cfg);

    LOG_INFO("gemma3_27b: hidden=%d inter=%d layers=%d heads=%d kv=%d dim=%d rope_scale=%.1f",
             cfg->hidden_size, cfg->intermediate_size, cfg->num_hidden_layers,
             cfg->num_attention_heads, cfg->num_key_value_heads, cfg->head_dim,
             (double)cfg->rope_scaling);

    if (cfg->num_attention_heads == 0 || cfg->head_dim == 0 || cfg->num_hidden_layers == 0) {
        LOG_ERROR("gemma3_27b_loader: incomplete config (heads=%d dim=%d layers=%d)",
                  cfg->num_attention_heads, cfg->head_dim, cfg->num_hidden_layers);
        free(cfg);
        goto done;
    }

    spec->variant_config = cfg;
    rc = 0;

done:
    free(json);
    return rc;
}

/* -------------------------------------------------------------------------
 * Loader hooks
 * -------------------------------------------------------------------------*/

static int gemma3_27b_populate_from_files(const char *model_dir, model_spec_t *spec)
{
    if (!model_dir || !spec) return -1;

    if (load_27b_config(model_dir, spec) != 0) {
        LOG_WARN("gemma3_27b_loader: failed to load config.json from %s", model_dir);
        /* Not fatal — leave GEMMA3_27B_RUNTIME_CONFIG defaults in place */
    }

    llm_model_t *loaded = load_model(model_dir, (const model_spec_t *)spec);
    if (!loaded) {
        LOG_ERROR("gemma3_27b_loader: failed to load weights from %s", model_dir);
        return -1;
    }

    llm_model_t *model = (llm_model_t *)spec->llm_model;
    model->embedding_weight  = loaded->embedding_weight;
    model->norm_final_weight = loaded->norm_final_weight;
    model->lm_head_weight    = loaded->lm_head_weight;
    model->layers            = loaded->layers;
    model->safetensors_handle = loaded->safetensors_handle;
    free(loaded);

    sapphire_tokenizer_t *tok = tokenizer_load(model_dir);
    if (!tok) {
        LOG_WARN("gemma3_27b_loader: tokenizer not found in %s", model_dir);
    } else {
        const gemma3_270m_config_t *cfg =
            (const gemma3_270m_config_t *)spec->variant_config;
        if (cfg) {
            if (cfg->bos_token_id >= 0) tok->bos_token_id = cfg->bos_token_id;
            if (cfg->eos_token_id >= 0) tok->eos_token_id = cfg->eos_token_id;
            if (cfg->pad_token_id >= 0) tok->pad_token_id = cfg->pad_token_id;
        }
        spec->tokenizer_handle = tok;
    }

    return 0;
}

static void gemma3_27b_postprocess_model(const model_spec_t *spec)
{
    (void)spec;
}

const model_loader_hooks_t GEMMA3_27B_LOADER_HOOKS = {
    .populate_from_files = (int (*)(const char *, const model_spec_t *))gemma3_27b_populate_from_files,
    .postprocess_model   = (void (*)(const model_spec_t *))gemma3_27b_postprocess_model
};

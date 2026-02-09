/*
 * @file gemma3_loader.c
 * @brief Gemma3 specific loader hooks
 */

#include <ctype.h>
#include <log.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "file_reader.h"
#include "gemma3_270m_map.h"
#include "gemma3_270m_config.h"
#include "gemma3_270m_spec.h"  /* Now provides GEMMA3_270M_TENSOR_MAP and getter */
#include "llm_model.h"
#include "model_spec.h"
#include "safetensors_reader.h"
#include "simple_json.h"
#include "tensor_mapper.h"
#include "tokenizer.h"
#include "utils.h"
#include "model_reader.h"
#include "gemma3_270m_spec.h"

/* Tokenizer spec (filenames expected under model dir) */
const tokenizer_spec_t GEMMA3_270M_TOKENIZER_SPEC = {
    .tokenizer_json = "tokenizer.json",
    .tokenizer_model = "tokenizer.model",
    .special_tokens_map = "special_tokens_map.json",
    .bos_token_id = 2,
    .eos_token_id = 1,
    .pad_token_id = 0
};

/* Important model files (relative to model directory) */
const model_files_t GEMMA3_270M_FILES = {
    .config_json = "config.json",
    .tokenizer_json = "tokenizer.json",
    .tokenizer_model = "tokenizer.model",
    .added_tokens = "added_tokens.json",
    .special_tokens_map = "special_tokens_map.json",
    .generation_config = "generation_config.json",
    .chat_template = "chat_template.jinja",
    .readme = "README.md"
};

/* Runtime configuration instance (populated by the Gemma3 loader hook) */
gemma3_270m_config_t GEMMA3_270M_RUNTIME_CONFIG = {0};

/* The public model specification objects */
model_spec_t GEMMA3_270M_IT_SPEC = {
    .model_id = "gemma-3-270m-it",
    .tensor_map = GEMMA3_270M_TENSOR_MAP,
    .tensor_map_size = GEMMA3_270M_TENSOR_MAP_SIZE,
    .tokenizer_spec = &GEMMA3_270M_TOKENIZER_SPEC,
    .files = &GEMMA3_270M_FILES,
    .variant_config = &GEMMA3_270M_RUNTIME_CONFIG,
    .loader_hooks = &GEMMA3_LOADER_HOOKS
};

model_spec_t GEMMA3_270M_SPEC = {
    .model_id = "gemma-3-270m",
    .tensor_map = GEMMA3_270M_TENSOR_MAP,
    .tensor_map_size = GEMMA3_270M_TENSOR_MAP_SIZE,
    .tokenizer_spec = &GEMMA3_270M_TOKENIZER_SPEC,
    .files = &GEMMA3_270M_FILES,
    .variant_config = &GEMMA3_270M_RUNTIME_CONFIG,
    .loader_hooks = &GEMMA3_LOADER_HOOKS
};

/* Forward declarations for helpers used below */

/* Field type enum for table-driven parsing */
typedef enum {
    FIELD_INT,
    FIELD_FLOAT,
    FIELD_STRING,
    FIELD_BOOL
} gemma3_field_type_t;

/* Field specification for table-driven parsing */
typedef struct {
    const char *json_key;
    size_t cfg_offset;
    gemma3_field_type_t type;
} gemma3_field_spec_t;

/* Parse a single numeric field (int or float) from JSON tokens */
static void parse_numeric_field(const char* json, const sjson_token_t* tokens, int nt,
                                 gemma3_270m_config_t* cfg, const gemma3_field_spec_t* spec) {
    int v_idx = sjson_find_key(json, tokens, nt, 0, spec->json_key);
    if (v_idx < 0) return;
    
    double dv = 0.0;
    if (sjson_token_to_double(json, &tokens[v_idx], &dv) != 0) return;
    
    if (spec->type == FIELD_INT) {
        int* field_ptr = (int*)((char*)cfg + spec->cfg_offset);
        *field_ptr = (int)(dv + 0.5);
    } else if (spec->type == FIELD_FLOAT) {
        float* field_ptr = (float*)((char*)cfg + spec->cfg_offset);
        *field_ptr = (float)dv;
    }
}

/* Parse a nullable softcap field (checks for "null" literal) */
static void parse_nullable_softcap(const char* json, const sjson_token_t* tokens, int nt,
                                    float* target, const char* key_primary, 
                                    const char* key_alt) {
    int v_idx = sjson_find_key(json, tokens, nt, 0, key_primary);
    if (v_idx < 0 && key_alt) {
        v_idx = sjson_find_key(json, tokens, nt, 0, key_alt);
    }
    if (v_idx < 0) return;
    
    int len = tokens[v_idx].end - tokens[v_idx].start;
    if (len == 4 && strncmp(json + tokens[v_idx].start, "null", 4) == 0) {
        *target = NAN;
    } else {
        double dv = 0.0;
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) {
            *target = (float)dv;
        }
    }
}

/* Parse a string field (allocates new string via strdup) */
static void parse_string_field(const char* json, const sjson_token_t* tokens, int nt,
                               gemma3_270m_config_t* cfg, const gemma3_field_spec_t* spec) {
    int v_idx = sjson_find_key(json, tokens, nt, 0, spec->json_key);
    if (v_idx < 0) return;
    
    char buf[512];
    if (sjson_token_to_str(json, &tokens[v_idx], buf, sizeof(buf)) != 0) return;
    
    char** field_ptr = (char**)((char*)cfg + spec->cfg_offset);
    *field_ptr = strdup(buf);
}

/* Parse a boolean field ("true" or "false") */
static void parse_bool_field(const char* json, const sjson_token_t* tokens, int nt,
                             gemma3_270m_config_t* cfg, const gemma3_field_spec_t* spec) {
    int v_idx = sjson_find_key(json, tokens, nt, 0, spec->json_key);
    if (v_idx < 0) return;
    
    int len = tokens[v_idx].end - tokens[v_idx].start;
    int value = (len == 4 && strncmp(json + tokens[v_idx].start, "true", 4) == 0) ? 1 : 0;
    
    int* field_ptr = (int*)((char*)cfg + spec->cfg_offset);
    *field_ptr = value;
}

/* Helper to populate all standard numeric, string, and boolean fields via tables */
static void populate_config_standard_fields(const char* json, const sjson_token_t* tokens, int nt,
                                           gemma3_270m_config_t* cfg) {
    /* Table-driven parsing for standard numeric fields */
    static const gemma3_field_spec_t NUMERIC_FIELDS[] = {
        {"rope_theta", offsetof(gemma3_270m_config_t, rope_theta), FIELD_FLOAT},
        {"rope_local_base_freq", offsetof(gemma3_270m_config_t, rope_local_base_freq), FIELD_FLOAT},
        {"num_hidden_layers", offsetof(gemma3_270m_config_t, num_hidden_layers), FIELD_INT},
        {"hidden_size", offsetof(gemma3_270m_config_t, hidden_size), FIELD_INT},
        {"intermediate_size", offsetof(gemma3_270m_config_t, intermediate_size), FIELD_INT},
        {"num_attention_heads", offsetof(gemma3_270m_config_t, num_attention_heads), FIELD_INT},
        {"num_key_value_heads", offsetof(gemma3_270m_config_t, num_key_value_heads), FIELD_INT},
        {"vocab_size", offsetof(gemma3_270m_config_t, vocab_size), FIELD_INT},
        {"sliding_window", offsetof(gemma3_270m_config_t, sliding_window), FIELD_INT},
        {"_sliding_window_pattern", offsetof(gemma3_270m_config_t, sliding_window_pattern), FIELD_INT},
        {"query_pre_attn_scalar", offsetof(gemma3_270m_config_t, query_pre_attn_scalar), FIELD_FLOAT},
        {"attention_dropout", offsetof(gemma3_270m_config_t, attention_dropout), FIELD_FLOAT},
        {"max_position_embeddings", offsetof(gemma3_270m_config_t, max_position_embeddings), FIELD_INT},
        {"head_dim", offsetof(gemma3_270m_config_t, head_dim), FIELD_INT},
    };

    /* Table-driven parsing for string fields */
    static const gemma3_field_spec_t STRING_FIELDS[] = {
        {"hidden_activation", offsetof(gemma3_270m_config_t, hidden_activation), FIELD_STRING},
        {"model_type", offsetof(gemma3_270m_config_t, model_type), FIELD_STRING},
        {"torch_dtype", offsetof(gemma3_270m_config_t, torch_dtype), FIELD_STRING},
        {"transformers_version", offsetof(gemma3_270m_config_t, transformers_version), FIELD_STRING},
    };

    /* Table-driven parsing for boolean fields */
    static const gemma3_field_spec_t BOOL_FIELDS[] = {
        {"attention_bias", offsetof(gemma3_270m_config_t, attention_bias), FIELD_BOOL},
        {"use_bidirectional_attention", offsetof(gemma3_270m_config_t, use_bidirectional_attention), FIELD_BOOL},
        {"use_cache", offsetof(gemma3_270m_config_t, use_cache), FIELD_BOOL},
    };

    for (size_t i = 0; i < sizeof(NUMERIC_FIELDS) / sizeof(NUMERIC_FIELDS[0]); ++i) {
        parse_numeric_field(json, tokens, nt, cfg, &NUMERIC_FIELDS[i]);
    }
    for (size_t i = 0; i < sizeof(STRING_FIELDS) / sizeof(STRING_FIELDS[0]); ++i) {
        parse_string_field(json, tokens, nt, cfg, &STRING_FIELDS[i]);
    }
    for (size_t i = 0; i < sizeof(BOOL_FIELDS) / sizeof(BOOL_FIELDS[0]); ++i) {
        parse_bool_field(json, tokens, nt, cfg, &BOOL_FIELDS[i]);
    }

    /* Handle nullable softcaps separately */
    parse_nullable_softcap(json, tokens, nt, &cfg->final_logit_softcapping, "final_logit_softcap", "final_logit_softcapping");
    parse_nullable_softcap(json, tokens, nt, &cfg->attn_logit_softcapping, "attn_logit_softcap", "attn_logit_softcapping");
}

static void populate_config_layer_types(const char* json, const sjson_token_t* tokens, int nt,
                                       gemma3_270m_config_t* cfg) {
    int v_idx = sjson_find_key(json, tokens, nt, 0, "layer_types");
    if (v_idx < 0 || tokens[v_idx].type != SJSON_ARR) return;

    unsigned long long mask = 0ULL;
    int count = tokens[v_idx].size;
    for (int i = 0; i < count && i < 64; i++) {
        const sjson_token_t* t = &tokens[v_idx + 1 + i];
        if (t->type != SJSON_STR) continue;
        int len = t->end - t->start;
        if (len >= 4 && strncmp(json + t->start, "full", 4) == 0) {
            mask |= (1ULL << i);
        }
    }
    cfg->layer_types_mask = mask;
    cfg->layer_types_count = count;
}

static int load_gemma3_config_json_from_dir(const char* dir, model_spec_t* spec) {
    if (!dir || !spec) return -1;

    char cfg_path[1200];
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", dir);

    char* json = NULL;
    size_t json_len = 0;
    gemma3_270m_config_t* cfg = NULL;
    int result = -1;

    if (file_read_json(cfg_path, &json, &json_len) != 0) {
        return -1;
    }

    sjson_token_t tokens[2048];
    int nt = sjson_tokenize(json, tokens, (int)(sizeof(tokens) / sizeof(tokens[0])));
    if (nt < 0) goto cleanup;

    cfg = (gemma3_270m_config_t*)calloc(1, sizeof(*cfg));
    if (!cfg) goto cleanup;

    /* Initialize nullable floats to NAN */
    cfg->attn_logit_softcapping = NAN;
    cfg->final_logit_softcapping = NAN;
    cfg->rope_scaling = NAN;

    /* Populate fields using helpers */
    populate_config_standard_fields(json, tokens, nt, cfg);
    populate_config_layer_types(json, tokens, nt, cfg);

    /* Validate critical fields */
    if (cfg->num_attention_heads == 0 || cfg->head_dim == 0 || 
        cfg->num_hidden_layers == 0 || cfg->vocab_size == 0) {
        LOG_WARN("config.json incomplete: heads=%d dim=%d layers=%d vocab=%d",
                 cfg->num_attention_heads, cfg->head_dim, cfg->num_hidden_layers, cfg->vocab_size);
        goto cleanup;
    }

    spec->variant_config = cfg;
    result = 0;

cleanup:
    if (result != 0 && cfg) free(cfg);
    if (json) free(json);
    return result;
}

/* We now rely on the shared `load_model(model_dir, spec)` helper from
 * `src/io/model_reader.c` to open and map model files. This keeps Gemma3
 * loader logic small and consistent with other loaders.
 */

/* Gemma3-specific populate: use generic parser then apply defaults and any postprocessing */
static int gemma3_populate_from_files(const char* model_dir, model_spec_t* spec) {
    if (!model_dir || !spec || !spec->llm_model) return -1;

    /* Load config first */
    if (load_gemma3_config_json_from_dir(model_dir, spec) != 0) {
        LOG_WARN("Failed to load Gemma3 config.json from %s", model_dir);
        /* Not fatal: leave spec->variant_config as-is (may have defaults from static instance) */
    }

    /* Load weights using the shared model reader helper */
    llm_model_t* loaded_model = load_model(model_dir, (const model_spec_t*)spec);
    if (!loaded_model) {
        LOG_ERROR("Failed to load Gemma3 weights from safetensors");
        return -1;
    }

    llm_model_t* model = (llm_model_t*)spec->llm_model;
    /* Copy loaded model data into the provided model structure */
    model->embedding_weight = loaded_model->embedding_weight;
    model->norm_final_weight = loaded_model->norm_final_weight;
    model->lm_head_weight = loaded_model->lm_head_weight;
    model->layers = loaded_model->layers;
    model->safetensors_handle = loaded_model->safetensors_handle;

    /* Compute derived fields on the Gemma runtime config */
    gemma3_270m_config_t* cfg = (gemma3_270m_config_t*)spec->variant_config;

    /* Free the shell structure (loaded_model) but keep its contents */
    free(loaded_model);

    if (!cfg) return 0;

    sapphire_tokenizer_t *tokenizer = tokenizer_load(model_dir);
    if (!tokenizer) {
        LOG_WARN("Failed to load tokenizer for Gemma3 from %s", model_dir);
        /* Not fatal */
    } else {
        /* Set tokenizer special token IDs if not already set from config */
        if (cfg->bos_token_id >= 0) tokenizer->bos_token_id = cfg->bos_token_id;
        if (cfg->eos_token_id >= 0) tokenizer->eos_token_id = cfg->eos_token_id;
        if (cfg->pad_token_id >= 0) tokenizer->pad_token_id = cfg->pad_token_id;

        /* Store tokenizer handle in spec for later use by runtime */
        spec->tokenizer_handle = tokenizer;
    }

    return 0;
}

static float gemma3_bf16_to_f32(uint16_t bf16_val) {
    uint32_t f32_bits = ((uint32_t)bf16_val) << 16;
    float f;
    memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

static tensor_t* gemma3_scale_norm_tensor(const tensor_t* src, float scale, const char* label) {
    if (!src) return NULL;

    int ndim = tensor_ndim(src);
    const int* shape = tensor_shape(src);
    size_t numel = tensor_numel(src);
    if (ndim <= 0 || !shape || numel == 0) return NULL;

    tensor_dtype_t dtype = tensor_dtype(src);
    if (dtype != DTYPE_F32 && dtype != DTYPE_BF16) {
        LOG_WARN("Gemma3 loader: skipping norm scale for %s (dtype=%s unsupported)",
                 label ? label : "(unnamed)", dtype_name(dtype));
        return NULL;
    }

    tensor_t* dst = tensor_create(ndim, shape, DTYPE_F32);
    if (!dst) return NULL;

    float* out = tensor_data_f32(dst);
    if (!out) {
        tensor_release(dst);
        return NULL;
    }

    if (dtype == DTYPE_F32) {
        const float* in = (const float*)tensor_data(src);
        if (!in) {
            tensor_release(dst);
            return NULL;
        }
        for (size_t i = 0; i < numel; ++i) {
            out[i] = in[i] * scale;
        }
    } else {
        const uint16_t* in = (const uint16_t*)tensor_data(src);
        if (!in) {
            tensor_release(dst);
            return NULL;
        }
        for (size_t i = 0; i < numel; ++i) {
            out[i] = gemma3_bf16_to_f32(in[i]) * scale;
        }
    }

    LOG_DEBUG("Gemma3 loader: scaled %s by %.6f into F32", label ? label : "(unnamed)", scale);
    return dst;
}

static void gemma3_scale_norm_inplace(tensor_t** slot, float scale, const char* label) {
    if (!slot || !*slot) return;
    tensor_t* scaled = gemma3_scale_norm_tensor(*slot, scale, label);
    if (!scaled) return;
    tensor_release(*slot);
    *slot = scaled;
}

static void gemma3_postprocess_model(const model_spec_t* model_spec) {
    if (!model_spec || !model_spec->llm_model || !model_spec->variant_config) return;

    /* No postprocessing for now - investigate core bug */
    (void)model_spec;
    return;
}

const model_loader_hooks_t GEMMA3_LOADER_HOOKS = {
    .populate_from_files = (int (*)(const char*, const model_spec_t*))gemma3_populate_from_files,
    .postprocess_model = (void (*)(const model_spec_t*))gemma3_postprocess_model};


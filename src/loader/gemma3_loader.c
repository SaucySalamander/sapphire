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

#include "gemma3_270m_config.h"
#include "gemma3_270m_map.h"
#include "llm_model.h"
#include "model_spec.h"
#include "safetensors_reader.h"
#include "simple_json.h"
#include "tensor_mapper.h"
#include "tokenizer.h"

/* Forward declarations for helpers used below */
static int load_gemma3_config_json_from_dir(const char* dir, model_spec_t* spec, char* error_msg, int max_err_len);

/* Use the generic model reader to load safetensors/gguf models from a directory */
#include "model_reader.h"

/* We now rely on the shared `load_model(model_dir, spec)` helper from
 * `src/io/model_reader.c` to open and map model files. This keeps Gemma3
 * loader logic small and consistent with other loaders.
 */

/* Gemma3-specific populate: use generic parser then apply defaults and any postprocessing */
static int gemma3_populate_from_files(const char* model_dir, model_spec_t* spec, char* error_msg, int max_err_len) {
    if (!model_dir || !spec || !spec->llm_model) return -1;

    /* Load config first */
    if (load_gemma3_config_json_from_dir(model_dir, spec, error_msg, max_err_len) != 0) {
        LOG_WARN("Failed to load Gemma3 config.json from %s", model_dir);
        /* Not fatal: leave spec->variant_config as-is (may have defaults from static instance) */
    }

    /* Load weights using the shared model reader helper */
    llm_model_t* loaded_model = load_model(model_dir, (const model_spec_t*)spec);
    if (!loaded_model) {
        if (error_msg && max_err_len > 0) {
            snprintf(error_msg, max_err_len, "Failed to load Gemma3 weights from safetensors");
        }
        return -1;
    }

    llm_model_t* model = (llm_model_t*)spec->llm_model;
    /* Copy loaded model data into the provided model structure */
    model->embedding_weight = loaded_model->embedding_weight;
    model->norm_final_weight = loaded_model->norm_final_weight;
    model->lm_head_weight = loaded_model->lm_head_weight;
    model->layers = loaded_model->layers;
    model->safetensors_handle = loaded_model->safetensors_handle;

    LOG_DEBUG("LOADER: After copy, layer 0 norm_attn_post_weight = %p", model->layers[0].norm_attn_post_weight);
    LOG_DEBUG("LOADER: After copy, layer 1 norm_attn_post_weight = %p", model->layers[1].norm_attn_post_weight);

    /* Compute derived fields on the Gemma runtime config */
    gemma3_270m_config_t* cfg = (gemma3_270m_config_t*)spec->variant_config;

    /* Optional per-layer norm presence diagnostic for debugging mapping issues.
     * Enable by setting SAPPHIRE_DEBUG_LAYER_NORMS=1 or running with LOG_LEVEL=DEBUG.
     */
    if (getenv("SAPPHIRE_DEBUG_LAYER_NORMS") || log_get_level() == LOG_LEVEL_DEBUG) {
        int dbg_layers = cfg ? (cfg->num_hidden_layers > 0 ? cfg->num_hidden_layers : 0) : 0;
        if (dbg_layers <= 0) dbg_layers = 18;
        LOG_DEBUG("Per-layer norm presence (showing norms if present, MISSING otherwise):");
        for (int li = 0; li < dbg_layers && li < SAPPHIRE_MAX_LAYERS; ++li) {
            model_layer_weights_t *lw = &model->layers[li];
            LOG_DEBUG("Layer %d:", li);
            if (lw->norm_attn_weight) { LOG_DEBUG("  norm_attn_weight:"); tensor_print_info(lw->norm_attn_weight); } else LOG_DEBUG("  norm_attn_weight: MISSING");
            if (lw->norm_attn_post_weight) { LOG_DEBUG("  norm_attn_post_weight:"); tensor_print_info(lw->norm_attn_post_weight); } else LOG_DEBUG("  norm_attn_post_weight: MISSING");
            if (lw->q_norm_weight) { LOG_DEBUG("  q_norm_weight:"); tensor_print_info(lw->q_norm_weight); } else LOG_DEBUG("  q_norm_weight: MISSING");
            if (lw->k_norm_weight) { LOG_DEBUG("  k_norm_weight:"); tensor_print_info(lw->k_norm_weight); } else LOG_DEBUG("  k_norm_weight: MISSING");
            if (lw->norm_ffn_weight) { LOG_DEBUG("  norm_ffn_weight:"); tensor_print_info(lw->norm_ffn_weight); } else LOG_DEBUG("  norm_ffn_weight: MISSING");
            if (lw->norm_ffn_post_weight) { LOG_DEBUG("  norm_ffn_post_weight:"); tensor_print_info(lw->norm_ffn_post_weight); } else LOG_DEBUG("  norm_ffn_post_weight: MISSING");
        }
    }
    /* Free the shell structure (loaded_model) but keep its contents */
    free(loaded_model);

    /* Debug: print shapes/info for each loaded layer to aid diagnosis */
    LOG_DEBUG("Model loaded; printing per-layer tensor info (first layers may be long)...");
    if (cfg) {
        int num_layers_dbg = cfg->num_hidden_layers > 0 ? cfg->num_hidden_layers : 0;
        for (int li = 0; li < num_layers_dbg && li < SAPPHIRE_MAX_LAYERS; ++li) {
            model_layer_weights_t *lw = &model->layers[li];
            LOG_DEBUG("Layer %d:", li);
            if (lw->q_proj_weight) { LOG_DEBUG(" q_proj_weight:"); tensor_print_info(lw->q_proj_weight); }
            if (lw->k_proj_weight) { LOG_DEBUG(" k_proj_weight:"); tensor_print_info(lw->k_proj_weight); }
            if (lw->v_proj_weight) { LOG_DEBUG(" v_proj_weight:"); tensor_print_info(lw->v_proj_weight); }
            if (lw->out_proj_weight) { LOG_DEBUG(" out_proj_weight:"); tensor_print_info(lw->out_proj_weight); }
            if (lw->up_proj_weight) { LOG_DEBUG(" up_proj_weight:"); tensor_print_info(lw->up_proj_weight); }
            if (lw->gate_proj_weight) { LOG_DEBUG(" gate_proj_weight:"); tensor_print_info(lw->gate_proj_weight); }
            if (lw->down_proj_weight) { LOG_DEBUG(" down_proj_weight:"); tensor_print_info(lw->down_proj_weight); }
        }
    } else {
        LOG_DEBUG("No variant config present; skipping per-layer debug prints");
    }

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

static void gemma3_postprocess_model(const model_spec_t* model_spec) {
    /* currently none; placeholder for future per-model transformations */
    (void)model_spec;
}

const model_loader_hooks_t GEMMA3_LOADER_HOOKS = {
    .populate_from_files = (int (*)(const char*, const model_spec_t*, char*, int))gemma3_populate_from_files,
    .postprocess_model = (void (*)(const model_spec_t*))gemma3_postprocess_model};

static int load_gemma3_config_json_from_dir(const char* dir, model_spec_t* spec, char* error_msg, int max_err_len) {
    if (!dir || !spec) return -1;

    /* read file */
    char cfg_path[1200];
    snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", dir);

    char* json = NULL;
    size_t json_len = 0;
    if (json_read_file(cfg_path, &json, &json_len) != 0) {
        return -1; /* no config.json — not fatal to callers but indicate we did not populate */
    }

    sjson_token_t tokens[2048];
    int nt = sjson_tokenize(json, tokens, (int)(sizeof(tokens) / sizeof(tokens[0])));
    if (nt < 0) {
        free(json);
        return -1;
    }

    LOG_DEBUG("sjson_tokenize returned nt=%d", nt);
    for (int ti = 0; ti < nt && ti < 200; ti++) {
        const sjson_token_t *tk = &tokens[ti];
        int len = tk->end - tk->start;
        char buf[128] = {0};
        if (len > 0 && len < (int)sizeof(buf)) memcpy(buf, json + tk->start, len);
        LOG_DEBUG("token %d: type=%d parent=%d size=%d text='%s'", ti, tk->type, tk->parent, tk->size, buf);
    }

    /* allocate and zero/initialize config instance */
    gemma3_270m_config_t* cfg = (gemma3_270m_config_t*)calloc(1, sizeof(*cfg));
    if (!cfg) {
        free(json);
        return -1;
    }

    /* set nullable floats to NAN to mark missing fields explicitly */
    cfg->attn_logit_softcapping = NAN;
    cfg->final_logit_softcapping = NAN;
    cfg->rope_scaling = NAN;

/* helpers */
auto_find_num:; /* label for gcc-style local goto-less coding, keep scope compact */

    /* numeric fields */
    double dv;
    int v_idx;
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "rope_theta")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->rope_theta = (float)dv;
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "rope_local_base_freq")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->rope_local_base_freq = (float)dv;
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "num_hidden_layers")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->num_hidden_layers = (int)(dv + 0.5);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "hidden_size")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->hidden_size = (int)(dv + 0.5);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "intermediate_size")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->intermediate_size = (int)(dv + 0.5);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "num_attention_heads")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->num_attention_heads = (int)(dv + 0.5);
    }
    LOG_DEBUG("parsed num_attention_heads = %d (v_idx=%d)", cfg->num_attention_heads, v_idx);
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "num_key_value_heads")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->num_key_value_heads = (int)(dv + 0.5);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "vocab_size")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->vocab_size = (int)(dv + 0.5);
    }
    LOG_DEBUG("parsed vocab_size = %d (v_idx=%d)", cfg->vocab_size, v_idx);
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "sliding_window")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->sliding_window = (int)(dv + 0.5);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "_sliding_window_pattern")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->sliding_window_pattern = (int)(dv + 0.5);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "query_pre_attn_scalar")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->query_pre_attn_scalar = (float)dv;
    }
    

    /* nullable numeric softcaps */
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "final_logit_softcap")) >= 0 ||
        (v_idx = sjson_find_key(json, tokens, nt, 0, "final_logit_softcapping")) >= 0) {
        /* check for null */
        int len = tokens[v_idx].end - tokens[v_idx].start;
        if (len == 4 && strncmp(json + tokens[v_idx].start, "null", 4) == 0) {
            cfg->final_logit_softcapping = NAN;
        } else {
            if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->final_logit_softcapping = (float)dv;
        }
    }

    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "attn_logit_softcap")) >= 0 ||
        (v_idx = sjson_find_key(json, tokens, nt, 0, "attn_logit_softcapping")) >= 0) {
        int len = tokens[v_idx].end - tokens[v_idx].start;
        if (len == 4 && strncmp(json + tokens[v_idx].start, "null", 4) == 0) {
            cfg->attn_logit_softcapping = NAN;
        } else {
            if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->attn_logit_softcapping = (float)dv;
        }
    }

    /* strings */
    char buf[512];
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "hidden_activation")) >= 0) {
        if (sjson_token_to_str(json, &tokens[v_idx], buf, sizeof(buf)) == 0) cfg->hidden_activation = strdup(buf);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "model_type")) >= 0) {
        if (sjson_token_to_str(json, &tokens[v_idx], buf, sizeof(buf)) == 0) cfg->model_type = strdup(buf);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "torch_dtype")) >= 0) {
        if (sjson_token_to_str(json, &tokens[v_idx], buf, sizeof(buf)) == 0) cfg->torch_dtype = strdup(buf);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "transformers_version")) >= 0) {
        if (sjson_token_to_str(json, &tokens[v_idx], buf, sizeof(buf)) == 0) cfg->transformers_version = strdup(buf);
    }

    /* booleans/ints */
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "attention_bias")) >= 0) {
        int len = tokens[v_idx].end - tokens[v_idx].start;
        if (len == 4 && strncmp(json + tokens[v_idx].start, "true", 4) == 0)
            cfg->attention_bias = 1;
        else
            cfg->attention_bias = 0;
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "attention_dropout")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->attention_dropout = (float)dv;
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "use_bidirectional_attention")) >= 0) {
        int len = tokens[v_idx].end - tokens[v_idx].start;
        if (len == 4 && strncmp(json + tokens[v_idx].start, "true", 4) == 0)
            cfg->use_bidirectional_attention = 1;
        else
            cfg->use_bidirectional_attention = 0;
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "use_cache")) >= 0) {
        int len = tokens[v_idx].end - tokens[v_idx].start;
        if (len == 4 && strncmp(json + tokens[v_idx].start, "true", 4) == 0)
            cfg->use_cache = 1;
        else
            cfg->use_cache = 0;
    }

    /* layer_types array -> bitmask */
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "layer_types")) >= 0 && tokens[v_idx].type == SJSON_ARR) {
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

    /* some additional fields */
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "max_position_embeddings")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->max_position_embeddings = (int)(dv + 0.5);
    }
    if ((v_idx = sjson_find_key(json, tokens, nt, 0, "head_dim")) >= 0) {
        if (sjson_token_to_double(json, &tokens[v_idx], &dv) == 0) cfg->head_dim = (int)(dv + 0.5);
    }
    LOG_DEBUG("parsed head_dim = %d (v_idx=%d)", cfg->head_dim, v_idx);

    /* Validate critical fields were populated */
    if (cfg->num_attention_heads == 0 || cfg->head_dim == 0 || cfg->num_hidden_layers == 0 || cfg->vocab_size == 0) {
        LOG_WARN("config.json appears incomplete or unparsable: num_attention_heads=%d head_dim=%d num_hidden_layers=%d vocab_size=%d",
                 cfg->num_attention_heads, cfg->head_dim, cfg->num_hidden_layers, cfg->vocab_size);
        free(cfg);
        free(json);
        return -1;
    }

    spec->variant_config = cfg;

    free(json);
    return 0;
}

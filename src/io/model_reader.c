#include "llm_model.h"
#include "model_spec.h"
#include "safetensors_reader.h"
#include "gemma3_270m_config.h"
#include "tensor_mapper.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <log.h>
#include <file_reader.h>

typedef enum {
    MODEL_FORMAT_UNKNOWN = 0,
    MODEL_FORMAT_GGML = 1,
    MODEL_FORMAT_SAFETENSORS = 2
} model_format_t;


/**
 * @brief Load a model from Safetensors file
 * 
 * Uses safetensors_reader to:
 * 1. Open the safetensors file (mmapped zero-copy)
 * 2. Parse the JSON header and extract tensor metadata
 * 3. Create an llm_model_t structure
 * 4. Map all tensors using the tensor_map from model_spec
 * 5. Store the model pointer in the model_spec_t
 *
 * @param model_spec The model specification containing the tensor mapping table
 * @param safetensors_path Path to the safetensors file
 *
 * @return Pointer to populated llm_model_t, or NULL on error
 */
static llm_model_t* load_model_safetensors(const model_spec_t *model_spec, 
                                           const char *safetensors_path) {
    if (!model_spec || !safetensors_path) {
        LOG_ERROR("model_spec or safetensors_path is NULL");
        return NULL;
    }
    
    // Open the safetensors file
    safetensors_file_t *st = safetensors_open(safetensors_path);
    if (!st) {
        LOG_ERROR("Failed to open safetensors file: %s", safetensors_path);
        return NULL;
    }
    
    // Allocate and initialize the model structure
    llm_model_t *model = (llm_model_t *)malloc(sizeof(llm_model_t));
    if (!model) {
        LOG_ERROR("Failed to allocate model structure");
        safetensors_close(st);
        return NULL;
    }
    memset(model, 0, sizeof(llm_model_t));
    
    // Derive layer count from spec's variant_config when available
    int num_layers = 18;  /* safe default for Gemma3 270M */
    if (model_spec->variant_config) {
        const gemma3_270m_config_t *cfg =
            (const gemma3_270m_config_t *)model_spec->variant_config;
        if (cfg->num_hidden_layers > 0 &&
            cfg->num_hidden_layers <= SAPPHIRE_MAX_LAYERS) {
            num_layers = cfg->num_hidden_layers;
        }
    }
    model->layers = (model_layer_weights_t *)malloc(num_layers * sizeof(model_layer_weights_t));
    if (!model->layers) {
        LOG_ERROR("Failed to allocate layer array");
        free(model);
        safetensors_close(st);
        return NULL;
    }
    memset(model->layers, 0, num_layers * sizeof(model_layer_weights_t));
    model->num_layers = num_layers;
    int rc = safetensors_map_all_tensors_with_table(st, 
                                                     model_spec->tensor_map,
                                                     model_spec->tensor_map_size,
                                                     NULL,  // No dynamic handler
                                                     model);
    
    if (rc != 0) {
        LOG_ERROR("Failed to map tensors");
        free(model->layers);
        free(model);
        safetensors_close(st);
        return NULL;
    }
    
    LOG_INFO("✓ Successfully loaded all tensors from %s", safetensors_path);
    
    // Store the safetensors file handle for cleanup in llm_model_destroy()
    // The file must remain open since tensors are zero-copy references into mmapped memory
    model->safetensors_handle = st;
    
    return model;
}

/* -------------------------------------------------------------------------
 * Sharded safetensors loader
 * -------------------------------------------------------------------------*/

/** qsort comparator for shard filename strings */
static int shard_name_cmp(const void *a, const void *b) {
    return strcmp(*(const char * const *)a, *(const char * const *)b);
}

/**
 * @brief Collect sorted shard filenames from model_dir.
 *
 * Scans for files matching "model-NNNNN-of-NNNNN.safetensors" and returns
 * a heap-allocated, sorted array of heap-allocated path strings.
 * Caller must free each element and the array itself.
 *
 * @param model_dir Directory to scan.
 * @param out_count Number of shards found.
 * @return Sorted array of paths, or NULL on error / none found.
 */
static char **collect_shard_paths(const char *model_dir, int *out_count) {
    *out_count = 0;
    DIR *d = opendir(model_dir);
    if (!d) return NULL;

    char **names = NULL;
    int   cap    = 0;
    int   cnt    = 0;

    const struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *n = ent->d_name;
        /* Match "model-NNNNN-of-NNNNN.safetensors" */
        if (strncmp(n, "model-", 6) != 0) continue;
        const char *ext = strstr(n, ".safetensors");
        if (!ext || ext[sizeof(".safetensors") - 1] != '\0') continue;

        if (cnt >= cap) {
            int new_cap = cap ? cap * 2 : 8;
            char **tmp = (char **)realloc(names, new_cap * sizeof(char *));
            if (!tmp) { closedir(d); goto oom; }
            names = tmp;
            cap   = new_cap;
        }
        names[cnt] = construct_safe_path(model_dir, n, NULL);
        if (!names[cnt]) { closedir(d); goto oom; }
        cnt++;
    }
    closedir(d);

    if (cnt == 0) { free(names); return NULL; }

    qsort(names, cnt, sizeof(char *), shard_name_cmp);
    *out_count = cnt;
    return names;

oom:
    for (int i = 0; i < cnt; i++) free(names[i]);
    free(names);
    return NULL;
}

/**
 * @brief Dynamic handler used during sharded loading.
 *
 * Each shard only contains a subset of the full tensor map.  Tensors absent
 * from the current shard will be present in another shard, so we silently
 * skip them here.  The lm_head weight-tie and any truly fatal misses are
 * resolved in a post-pass after all shards have been loaded.
 */
static int shard_skip_missing(const safetensors_file_t *st,
                              const safetensors_tensor_meta_t *meta,
                              llm_model_t *model) {
    (void)st; (void)meta; (void)model;
    return 0; /* silently skip — tensor lives in another shard */
}

/**
 * @brief Post-pass validation after all shards have been mapped.
 *
 * Applies lm_head weight-tying when the tensor was absent from every shard
 * (tied-embedding models), and verifies that mandatory top-level tensors are
 * present.  Returns 0 on success, -1 on fatal missing tensor.
 */
static int resolve_post_shard(llm_model_t *model) {
    /* Weight-tied lm_head: share embedding tensor */
    if (!model->lm_head_weight && model->embedding_weight) {
        model->lm_head_weight = model->embedding_weight;
        tensor_ref_inc(model->lm_head_weight);
        LOG_INFO("lm_head weight tied to embedding");
    }

    if (!model->embedding_weight) {
        LOG_ERROR("resolve_post_shard: embedding_weight missing after all shards");
        return -1;
    }
    if (!model->norm_final_weight) {
        LOG_ERROR("resolve_post_shard: norm_final_weight missing after all shards");
        return -1;
    }
    if (!model->lm_head_weight) {
        LOG_ERROR("resolve_post_shard: lm_head_weight missing after all shards");
        return -1;
    }
    return 0;
}

/**
 * @brief Load a sharded safetensors model by mapping all shards in order.
 *
 * Opens each shard file (sorted lexicographically), mmaps it, and maps
 * its tensors into the model.  All shard handles are kept open and stored
 * in model->safetensors_shard_handles so that the zero-copy tensor views
 * remain valid for the model's lifetime.
 */
static llm_model_t *load_model_sharded(const model_spec_t *model_spec,
                                        const char *model_dir) {
    int    shard_count = 0;
    char **shard_paths = collect_shard_paths(model_dir, &shard_count);
    if (!shard_paths || shard_count == 0) {
        LOG_ERROR("load_model_sharded: no shards found in %s", model_dir);
        return NULL;
    }
    LOG_INFO("Sharded safetensors: found %d shard(s) in %s", shard_count, model_dir);

    llm_model_t *model = (llm_model_t *)malloc(sizeof(llm_model_t));
    if (!model) { goto fail_paths; }
    memset(model, 0, sizeof(llm_model_t));

    int num_layers = 18;
    if (model_spec->variant_config) {
        const gemma3_270m_config_t *cfg =
            (const gemma3_270m_config_t *)model_spec->variant_config;
        if (cfg->num_hidden_layers > 0 &&
            cfg->num_hidden_layers <= SAPPHIRE_MAX_LAYERS) {
            num_layers = cfg->num_hidden_layers;
        }
    }

    model->layers = (model_layer_weights_t *)malloc(
        num_layers * sizeof(model_layer_weights_t));
    if (!model->layers) { goto fail_model; }
    memset(model->layers, 0, num_layers * sizeof(model_layer_weights_t));
    model->num_layers = num_layers;

    model->safetensors_shard_handles =
        (void **)malloc(shard_count * sizeof(void *));
    if (!model->safetensors_shard_handles) { goto fail_layers; }
    memset(model->safetensors_shard_handles, 0, shard_count * sizeof(void *));

    for (int s = 0; s < shard_count; s++) {
        LOG_INFO("  Loading shard %d/%d: %s", s + 1, shard_count, shard_paths[s]);
        safetensors_file_t *st = safetensors_open(shard_paths[s]);
        if (!st) {
            LOG_ERROR("Failed to open shard: %s", shard_paths[s]);
            goto fail_shards;
        }
        model->safetensors_shard_handles[s] = st;
        model->safetensors_shard_count      = s + 1;

        int rc = safetensors_map_all_tensors_with_table(
            st,
            model_spec->tensor_map,
            model_spec->tensor_map_size,
            shard_skip_missing,
            model);
        if (rc != 0) {
            LOG_ERROR("Failed to map tensors from shard: %s", shard_paths[s]);
            goto fail_shards;
        }
    }

    for (int i = 0; i < shard_count; i++) free(shard_paths[i]);
    free(shard_paths);
    shard_paths  = NULL;
    shard_count  = 0;

    /* Post-pass: weight-tie lm_head and verify mandatory tensors */
    if (resolve_post_shard(model) != 0) {
        goto fail_shards;
    }

    LOG_INFO("\u2713 All %d shards loaded successfully", shard_count);
    return model;

fail_shards:
    for (int i = 0; i < model->safetensors_shard_count; i++) {
        if (model->safetensors_shard_handles[i])
            safetensors_close((safetensors_file_t *)model->safetensors_shard_handles[i]);
    }
    free(model->safetensors_shard_handles);
fail_layers:
    free(model->layers);
fail_model:
    free(model);
fail_paths:
    for (int i = 0; i < shard_count; i++) free(shard_paths[i]);
    free(shard_paths);
    return NULL;
}

/**
 * @brief Load a model from directory (looks for model.safetensors or model.gguf)
 */
llm_model_t* load_model(const char *model_dir, const model_spec_t *model_spec) {
    if (!model_dir) {
        LOG_ERROR("Model directory path is NULL");
        return NULL;
    }
    
    if (!model_spec) {
        LOG_ERROR("Model specification is NULL");
        return NULL;
    }
    
    LOG_INFO("Loading model from directory: %s", model_dir);
    
    // Check if directory exists
    if (access(model_dir, F_OK) == -1) {
        LOG_ERROR("Model directory not found: %s", model_dir);
        return NULL;
    }
    
    // Try to find model.safetensors first
    char *model_path = construct_safe_path(model_dir, "model.safetensors", NULL);
    if (!model_path) {
        return NULL;
    }
    
    // Check if model.safetensors exists
    if (access(model_path, F_OK) != -1) {
        LOG_INFO("✓ Found model.safetensors, loading...");
        
        LOG_INFO("Loading Safetensors format model...");
        llm_model_t *model = load_model_safetensors(model_spec, model_path);
        free(model_path);
        
        if (model) {
            LOG_INFO("Model loaded successfully from Safetensors");
            return model;
        } else {
            LOG_ERROR("Failed to load model from Safetensors");
            return NULL;
        }
    }
    
    free(model_path);

    /* Try sharded safetensors (model-00001-of-NNNNN.safetensors, ...) */
    {
        llm_model_t *model = load_model_sharded(model_spec, model_dir);
        if (model) {
            LOG_INFO("Model loaded successfully from sharded Safetensors");
            return model;
        }
    }

    LOG_ERROR("No supported model file found in directory: %s", model_dir);
    return NULL;
}

/**
 * @brief Free all model memory and close associated files.
 *
 * Releases all tensors and closes the safetensors file handle if present.
 * After calling this, the model pointer becomes invalid.
 *
 * @param model Model to destroy (may be NULL; safe noop)
 */
/* Legacy destroy that accepts a direct model pointer. This preserves
 * compatibility with unit tests and older call-sites. When possible prefer
 * `llm_model_destroy_ex(spec)` which can consult the owning `spec` for
 * accurate layer counts. */
void llm_model_destroy(llm_model_t *model) {
    if (!model) return;

    if (model->embedding_weight) tensor_release(model->embedding_weight);
    if (model->norm_final_weight) tensor_release(model->norm_final_weight);
    if (model->lm_head_weight) tensor_release(model->lm_head_weight);

    if (model->layers) {
        int n = (model->num_layers > 0) ? model->num_layers : SAPPHIRE_MAX_LAYERS;
        for (int i = 0; i < n; i++) {
            model_layer_weights_t *layer = &model->layers[i];
            if (layer->norm_attn_weight) tensor_release(layer->norm_attn_weight);
            if (layer->norm_attn_post_weight) tensor_release(layer->norm_attn_post_weight);
            if (layer->q_proj_weight) tensor_release(layer->q_proj_weight);
            if (layer->k_proj_weight) tensor_release(layer->k_proj_weight);
            if (layer->v_proj_weight) tensor_release(layer->v_proj_weight);
            if (layer->q_norm_weight) tensor_release(layer->q_norm_weight);
            if (layer->k_norm_weight) tensor_release(layer->k_norm_weight);
            if (layer->out_proj_weight) tensor_release(layer->out_proj_weight);
            if (layer->norm_ffn_weight) tensor_release(layer->norm_ffn_weight);
            if (layer->norm_ffn_post_weight) tensor_release(layer->norm_ffn_post_weight);
            if (layer->up_proj_weight) tensor_release(layer->up_proj_weight);
            if (layer->gate_proj_weight) tensor_release(layer->gate_proj_weight);
            if (layer->down_proj_weight) tensor_release(layer->down_proj_weight);
        }
        free(model->layers);
    }

    if (model->safetensors_handle)
        safetensors_close((safetensors_file_t *)model->safetensors_handle);
    if (model->safetensors_shard_handles) {
        for (int i = 0; i < model->safetensors_shard_count; i++) {
            if (model->safetensors_shard_handles[i])
                safetensors_close((safetensors_file_t *)model->safetensors_shard_handles[i]);
        }
        free(model->safetensors_shard_handles);
    }
    free(model);
}

/* Spec-aware destroy: free the model referenced by `spec->llm_model`. This
 * allows using `spec->variant_config` to determine the exact number of
 * layers (e.g., `num_hidden_layers`) to free. */
void llm_model_destroy_ex(const struct model_spec *spec) {
    if (!spec) return;
    llm_model_t *model = (llm_model_t *)spec->llm_model;
    if (!model) return;

    if (model->embedding_weight) tensor_release(model->embedding_weight);
    if (model->norm_final_weight) tensor_release(model->norm_final_weight);
    if (model->lm_head_weight) tensor_release(model->lm_head_weight);

    int n_layers = SAPPHIRE_MAX_LAYERS;
    if (spec->variant_config) {
        /* Best-effort: many variant configs expose `num_hidden_layers` as an int. */
        /* We attempt to read that field from the common Gemma3 config if available. */
        const gemma3_270m_config_t *cfg = (const gemma3_270m_config_t *)spec->variant_config;
        if (cfg->num_hidden_layers > 0 && cfg->num_hidden_layers <= SAPPHIRE_MAX_LAYERS) {
            n_layers = cfg->num_hidden_layers;
        }
    }

    if (model->layers) {
        for (int i = 0; i < n_layers; i++) {
            model_layer_weights_t *layer = &model->layers[i];
            if (layer->norm_attn_weight) tensor_release(layer->norm_attn_weight);
            if (layer->norm_attn_post_weight) tensor_release(layer->norm_attn_post_weight);
            if (layer->q_proj_weight) tensor_release(layer->q_proj_weight);
            if (layer->k_proj_weight) tensor_release(layer->k_proj_weight);
            if (layer->v_proj_weight) tensor_release(layer->v_proj_weight);
            if (layer->q_norm_weight) tensor_release(layer->q_norm_weight);
            if (layer->k_norm_weight) tensor_release(layer->k_norm_weight);
            if (layer->out_proj_weight) tensor_release(layer->out_proj_weight);
            if (layer->norm_ffn_weight) tensor_release(layer->norm_ffn_weight);
            if (layer->norm_ffn_post_weight) tensor_release(layer->norm_ffn_post_weight);
            if (layer->up_proj_weight) tensor_release(layer->up_proj_weight);
            if (layer->gate_proj_weight) tensor_release(layer->gate_proj_weight);
            if (layer->down_proj_weight) tensor_release(layer->down_proj_weight);
        }
        free(model->layers);
    }

    if (model->safetensors_handle)
        safetensors_close((safetensors_file_t *)model->safetensors_handle);
    if (model->safetensors_shard_handles) {
        for (int i = 0; i < model->safetensors_shard_count; i++) {
            if (model->safetensors_shard_handles[i])
                safetensors_close((safetensors_file_t *)model->safetensors_shard_handles[i]);
        }
        free(model->safetensors_shard_handles);
    }

    /* Null out spec->llm_model to avoid dangling pointer in the spec */
    ((struct model_spec *)spec)->llm_model = NULL;

    free(model);
}
#include "llm_model.h"
#include "model_spec.h"
#include "safetensors_reader.h"
#include "gemma3_270m_config.h"
#include "tensor_mapper.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
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
    
    // Allocate layer array (assume 18 layers for Gemma3, could be made dynamic)
    // For now, hard-code to common sizes; could be derived from tensor metadata
    int num_layers = 18;  // Gemma3 270M has 18 layers
    model->layers = (model_layer_weights_t *)malloc(num_layers * sizeof(model_layer_weights_t));
    if (!model->layers) {
        LOG_ERROR("Failed to allocate layer array");
        free(model);
        safetensors_close(st);
        return NULL;
    }
    memset(model->layers, 0, num_layers * sizeof(model_layer_weights_t));
    
    // Map all tensors from the safetensors file using the spec's tensor map
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
    
    LOG_ERROR("No supported model file found in directory: %s", model_dir);
    free(model_path);
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
        for (int i = 0; i < SAPPHIRE_MAX_LAYERS; i++) {
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

    if (model->safetensors_handle) safetensors_close((safetensors_file_t *)model->safetensors_handle);
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

    if (model->safetensors_handle) safetensors_close((safetensors_file_t *)model->safetensors_handle);

    /* Null out spec->llm_model to avoid dangling pointer in the spec */
    ((struct model_spec *)spec)->llm_model = NULL;

    free(model);
}
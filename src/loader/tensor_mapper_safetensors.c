/**
 * @file tensor_mapper_safetensors.c
 * @brief Safetensors format implementation of the tensor mapper plugin.
 *
 * Maps Hugging Face Safetensors format to Sapphire llm_model_t using the
 * Gemma 3 270M tensor name mapping. Implements the tensor_mapper_plugin_t interface.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/tensor_mapper.h"
#include "../include/safetensors_reader.h"
#include "../include/gemma3_270m_map.h"

/**
 * Extract model configuration from Safetensors tensor metadata.
 *
 * Inspects shapes of known tensors (e.g., embedding, first layer) to infer
 * vocab_size, d_model, num_heads, num_layers, etc.
 */
static int extract_model_config(const safetensors_file_t *st, model_config_t *out_config) {
    if (!st || !out_config) return -1;

    // Find embedding tensor to extract vocab_size and d_model
    const safetensors_tensor_meta_t *embed = safetensors_get_tensor_by_name(st, "model.embed_tokens.weight");
    if (!embed) {
        fprintf(stderr, "ERROR: Missing model.embed_tokens.weight\n");
        return -1;
    }
    if (embed->ndim != 2) {
        fprintf(stderr, "ERROR: Embedding tensor must be 2D, got %d dimensions\n", embed->ndim);
        return -1;
    }
    out_config->vocab_size = embed->shape[0];
    out_config->d_model = embed->shape[1];

    // Find first layer Q projection to extract head info
    const safetensors_tensor_meta_t *q_proj = safetensors_get_tensor_by_name(st, "model.layers.0.self_attn.q_proj.weight");
    if (!q_proj) {
        fprintf(stderr, "ERROR: Missing model.layers.0.self_attn.q_proj.weight\n");
        return -1;
    }

    // For Gemma 3 with GQA: head_dim = 64 standard
    int head_dim = 64;
    int num_q_heads = q_proj->shape[0] / head_dim;
    out_config->num_heads = num_q_heads;
    out_config->d_k = head_dim;

    // Determine number of KV heads from the k_proj tensor (GQA may have fewer KV heads)
    const safetensors_tensor_meta_t *k_proj = safetensors_get_tensor_by_name(st, "model.layers.0.self_attn.k_proj.weight");
    if (k_proj && k_proj->ndim == 2 && k_proj->shape[0] > 0) {
        out_config->num_kv_heads = k_proj->shape[0] / head_dim;
    } else {
        out_config->num_kv_heads = num_q_heads; // fallback to same as query heads
    }

    // Count transformer layers
    int max_layer = -1;
    for (int idx = 0; idx < safetensors_tensor_count(st); idx++) {
        const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_index(st, idx);
        if (!meta) continue;

        int layer_idx = -1;
        if (sscanf(meta->name, "model.layers.%d.", &layer_idx) == 1) {
            if (layer_idx > max_layer) {
                max_layer = layer_idx;
            }
        }
    }

    if (max_layer < 0) {
        fprintf(stderr, "ERROR: Could not determine number of layers\n");
        return -1;
    }
    out_config->num_layers = max_layer + 1;

    // Set standard Gemma 3 parameters
    out_config->max_context_len = 8192;
    out_config->rope_base = 10000.0f;

    if (out_config->d_model <= 0 || out_config->num_heads <= 0 ||
        out_config->num_layers <= 0 || out_config->vocab_size <= 0) {
        fprintf(stderr, "ERROR: Invalid config derived from tensors\n");
        return -1;
    }

    return 0;
}

/**
 * Map a single Safetensors tensor to an llm_model_t field.
 */
static int map_tensor_to_model(const safetensors_file_t *st,
                               const safetensors_tensor_meta_t *meta,
                               llm_model_t *model,
                               char *error_msg,
                               int max_error_len) {
    if (!st || !meta || !model) return -1;

    // Look up this tensor in the mapping table
    const gemma3_tensor_map_entry_t *entry = NULL;
    for (int i = 0; GEMMA3_270M_TENSOR_MAP[i].hf_name != NULL; i++) {
        if (strcmp(GEMMA3_270M_TENSOR_MAP[i].hf_name, meta->name) == 0) {
            entry = &GEMMA3_270M_TENSOR_MAP[i];
            break;
        }
    }

    if (!entry) {
        // Skip metadata and unmapped tensors
        if (meta->name[0] == '_') {
            return 0;
        }
        snprintf(error_msg, max_error_len, "Unknown tensor: %s", meta->name);
        return -1;
    }

    // Parse internal_key and route to appropriate model field
    tensor_t **target_pointer = NULL;
    int expected_ndim = 0;

    if (strcmp(entry->internal_key, "embedding") == 0) {
        if (strcmp(entry->field_name, "embedding_weight") == 0) {
            target_pointer = &model->embedding_weight;
            expected_ndim = 2;
        }
    } else if (strncmp(entry->internal_key, "blk.", 4) == 0) {
        int layer_idx = atoi(&entry->internal_key[4]);
        if (layer_idx < 0 || layer_idx >= model->config.num_layers) {
            snprintf(error_msg, max_error_len, "Invalid layer index %d", layer_idx);
            return -1;
        }

        model_layer_weights_t *layer = &model->layers[layer_idx];

        if (strcmp(entry->field_name, "norm_attn_weight") == 0) {
            target_pointer = &layer->norm_attn_weight;
            expected_ndim = 1;
        } else if (strcmp(entry->field_name, "q_proj_weight") == 0) {
            target_pointer = &layer->q_proj_weight;
            expected_ndim = 2;
        } else if (strcmp(entry->field_name, "k_proj_weight") == 0) {
            target_pointer = &layer->k_proj_weight;
            expected_ndim = 2;
        } else if (strcmp(entry->field_name, "v_proj_weight") == 0) {
            target_pointer = &layer->v_proj_weight;
            expected_ndim = 2;
        } else if (strcmp(entry->field_name, "out_proj_weight") == 0) {
            target_pointer = &layer->out_proj_weight;
            expected_ndim = 2;
        } else if (strcmp(entry->field_name, "norm_ffn_weight") == 0) {
            target_pointer = &layer->norm_ffn_weight;
            expected_ndim = 1;
        } else if (strcmp(entry->field_name, "gate_proj_weight") == 0) {
            target_pointer = &layer->gate_proj_weight;
            expected_ndim = 2;
        } else if (strcmp(entry->field_name, "up_proj_weight") == 0) {
            target_pointer = &layer->up_proj_weight;
            expected_ndim = 2;
        } else if (strcmp(entry->field_name, "down_proj_weight") == 0) {
            target_pointer = &layer->down_proj_weight;
            expected_ndim = 2;
        } else {
            snprintf(error_msg, max_error_len, "Unknown field: %s", entry->field_name);
            return -1;
        }
    } else if (strcmp(entry->internal_key, "final") == 0) {
        if (strcmp(entry->field_name, "norm_final_weight") == 0) {
            target_pointer = &model->norm_final_weight;
            expected_ndim = 1;
        } else if (strcmp(entry->field_name, "lm_head_weight") == 0) {
            target_pointer = &model->lm_head_weight;
            expected_ndim = 2;
        } else {
            snprintf(error_msg, max_error_len, "Unknown final field: %s", entry->field_name);
            return -1;
        }
    } else {
        snprintf(error_msg, max_error_len, "Unknown internal key: %s", entry->internal_key);
        return -1;
    }

    if (!target_pointer) {
        snprintf(error_msg, max_error_len, "Could not determine target for %s", meta->name);
        return -1;
    }

    // Validate ndim
    if (meta->ndim != expected_ndim) {
        snprintf(error_msg, max_error_len,
                 "Shape mismatch for %s: expected %dD, got %dD",
                 meta->name, expected_ndim, meta->ndim);
        return -1;
    }

    // Create tensor reference pointing to mmapped data
    tensor_t *tensor = safetensors_create_tensor_ref((safetensors_file_t *)st, meta);
    if (!tensor) {
        snprintf(error_msg, max_error_len, "Failed to create tensor ref for %s", meta->name);
        return -1;
    }

    *target_pointer = tensor;
    return 0;
}

/**
 * Validate that all required tensors are present.
 */
static int validate_tensors(const safetensors_file_t *st, llm_model_t *model,
                            char *error_msg, int max_error_len) {
    if (!model->embedding_weight || !model->norm_final_weight || !model->lm_head_weight) {
        snprintf(error_msg, max_error_len,
                 "Missing critical tensors (embedding, norm_final, or lm_head)");
        return -1;
    }

    for (int i = 0; i < model->config.num_layers; i++) {
        model_layer_weights_t *layer = &model->layers[i];
        if (!layer->norm_attn_weight || !layer->q_proj_weight ||
            !layer->k_proj_weight || !layer->v_proj_weight ||
            !layer->out_proj_weight || !layer->norm_ffn_weight ||
            !layer->gate_proj_weight || !layer->up_proj_weight ||
            !layer->down_proj_weight) {
            snprintf(error_msg, max_error_len, "Layer %d is missing required tensors", i);
            return -1;
        }
    }

    return 0;
}

/**
 * Safetensors plugin implementation.
 */
int sapphire_load_safetensors(const char *safetensors_path, llm_model_t *out_model,
                              char *error_message, int max_error_len) {
    if (!safetensors_path || !out_model) {
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Invalid arguments (path or model NULL)");
        }
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    memset(out_model, 0, sizeof(llm_model_t));

    // Open Safetensors file
    safetensors_file_t *st = safetensors_open(safetensors_path);
    if (!st) {
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len,
                     "Failed to open Safetensors file: %s", safetensors_path);
        }
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    // Extract config
    if (extract_model_config(st, &out_model->config) != 0) {
        safetensors_close(st);
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len,
                     "Failed to extract model configuration from tensors");
        }
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    // Allocate layers
    out_model->layers = (model_layer_weights_t *)calloc(out_model->config.num_layers,
                                                         sizeof(model_layer_weights_t));
    if (!out_model->layers) {
        safetensors_close(st);
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len,
                     "Failed to allocate %d layers", out_model->config.num_layers);
        }
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    printf("Loaded config: vocab=%d, d_model=%d, heads=%d, layers=%d\n",
           out_model->config.vocab_size,
           out_model->config.d_model,
           out_model->config.num_heads,
           out_model->config.num_layers);

    // Map all tensors
    int tensor_count = safetensors_tensor_count(st);
    int mapped_count = 0;
    int skipped_count = 0;

    for (int i = 0; i < tensor_count; i++) {
        const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_index(st, i);
        if (!meta) continue;

        if (meta->name[0] == '_') {
            skipped_count++;
            continue;
        }

        // Check if in mapping
        const gemma3_tensor_map_entry_t *entry = NULL;
        for (int j = 0; GEMMA3_270M_TENSOR_MAP[j].hf_name != NULL; j++) {
            if (strcmp(GEMMA3_270M_TENSOR_MAP[j].hf_name, meta->name) == 0) {
                entry = &GEMMA3_270M_TENSOR_MAP[j];
                break;
            }
        }

        if (!entry) {
            skipped_count++;
            continue;
        }

        char err_buf[256] = {0};
        if (map_tensor_to_model(st, meta, out_model, err_buf, sizeof(err_buf)) != 0) {
            fprintf(stderr, "ERROR: %s\n", err_buf);
            safetensors_close(st);
            if (error_message && max_error_len > 0) {
                snprintf(error_message, max_error_len, "Tensor mapping failed: %s", err_buf);
            }
            free(out_model->layers);
            memset(out_model, 0, sizeof(llm_model_t));
            return SAPPHIRE_ERR_LOAD_MISSING_TENSOR;
        }

        mapped_count++;
    }

    printf("✓ Mapped %d tensors successfully (%d skipped/extra)\n", mapped_count, skipped_count);

    // Handle weight tying
    if (!out_model->lm_head_weight && out_model->embedding_weight) {
        printf("INFO: lm_head.weight not found, using weight tying with embedding\n");
        out_model->lm_head_weight = out_model->embedding_weight;
    }

    // Validate tensors
    char err_buf[256] = {0};
    if (validate_tensors(st, out_model, err_buf, sizeof(err_buf)) != 0) {
        fprintf(stderr, "ERROR: %s\n", err_buf);
        safetensors_close(st);
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Validation failed: %s", err_buf);
        }
        free(out_model->layers);
        memset(out_model, 0, sizeof(llm_model_t));
        return SAPPHIRE_ERR_LOAD_MISSING_TENSOR;
    }

    printf("✓ All required tensors present\n");

    safetensors_close(st);
    return SAPPHIRE_OK;
}

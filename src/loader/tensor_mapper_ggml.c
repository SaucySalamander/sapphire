/**
 * @file tensor_mapper_ggml.c
 * @brief GGML format implementation of the tensor mapper plugin.
 *
 * Loads GGML quantized models and maps them to Sapphire llm_model_t.
 * Implements the tensor_mapper_plugin_t interface.
 * 
 * Uses the GGML file header reading API from ggml_model.c to parse
 * GGML format files and map tensors to the unified model structure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/tensor_mapper.h"

/**
 * Extract model configuration from GGML header.
 * 
 * Since GGML files don't typically encode vocab_size and num_heads,
 * this function attempts to infer them from tensor shapes or uses defaults.
 */
static int ggml_extract_config(const ggml_file_header_t *header, model_config_t *out_config) {
    if (!header || !out_config) return -1;

    // Find embedding tensor to infer vocab_size and d_model
    const ggml_tensor_meta_t *embed = ggml_find_tensor_meta((ggml_file_header_t *)header, "embedding.weight");
    if (!embed || embed->ndim != 2) {
        fprintf(stderr, "ERROR: Could not find valid embedding tensor\n");
        return -1;
    }

    out_config->vocab_size = embed->shape[0];
    out_config->d_model = embed->shape[1];

    // Count layers by finding highest layer index
    int max_layer = -1;
    char tensor_name[256];
    for (uint32_t i = 0; i < header->tensor_count; i++) {
        const ggml_tensor_meta_t *meta = &header->tensors[i];
        if (!meta || !meta->name) continue;

        int layer_idx = -1;
        if (sscanf(meta->name, "layers.%d.", &layer_idx) == 1) {
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

    // Default head configuration (can be refined if found in metadata)
    out_config->num_heads = 16;  // Default for Gemma 3
    out_config->d_k = 64;
    out_config->max_context_len = 8192;
    out_config->rope_base = 10000.0f;

    if (out_config->d_model <= 0 || out_config->num_heads <= 0 ||
        out_config->num_layers <= 0 || out_config->vocab_size <= 0) {
        fprintf(stderr, "ERROR: Invalid config derived from GGML tensors\n");
        return -1;
    }

    return 0;
}

/**
 * Map a single GGML tensor to model field.
 */
static int ggml_map_tensor_to_model(FILE *fp,
                                    const ggml_file_header_t *header,
                                    const ggml_tensor_meta_t *meta,
                                    llm_model_t *model,
                                    char *error_msg,
                                    int max_error_len) {
    if (!fp || !meta || !model) return -1;

    tensor_t **target_pointer = NULL;

    // Map embedding
    if (strcmp(meta->name, "embedding.weight") == 0) {
        target_pointer = &model->embedding_weight;
    }
    // Map final norm
    else if (strcmp(meta->name, "norm_final.weight") == 0) {
        target_pointer = &model->norm_final_weight;
    }
    // Map LM head
    else if (strcmp(meta->name, "lm_head.weight") == 0) {
        target_pointer = &model->lm_head_weight;
    }
    // Map layer tensors
    else if (strncmp(meta->name, "layers.", 7) == 0) {
        int layer_idx = atoi(&meta->name[7]);
        if (layer_idx < 0 || layer_idx >= model->config.num_layers) {
            snprintf(error_msg, max_error_len, "Invalid layer index %d", layer_idx);
            return -1;
        }

        model_layer_weights_t *layer = &model->layers[layer_idx];
        const char *field = strchr(&meta->name[7], '.') + 1;
        field = strchr(field, '.') + 1;  // Skip "layers.N."

        if (strcmp(field, "attention_norm.weight") == 0) {
            target_pointer = &layer->norm_attn_weight;
        } else if (strcmp(field, "attention.q_proj.weight") == 0) {
            target_pointer = &layer->q_proj_weight;
        } else if (strcmp(field, "attention.k_proj.weight") == 0) {
            target_pointer = &layer->k_proj_weight;
        } else if (strcmp(field, "attention.v_proj.weight") == 0) {
            target_pointer = &layer->v_proj_weight;
        } else if (strcmp(field, "attention.out_proj.weight") == 0) {
            target_pointer = &layer->out_proj_weight;
        } else if (strcmp(field, "ffn_norm.weight") == 0) {
            target_pointer = &layer->norm_ffn_weight;
        } else if (strcmp(field, "ffn.up_proj.weight") == 0) {
            target_pointer = &layer->up_proj_weight;
        } else if (strcmp(field, "ffn.gate_proj.weight") == 0) {
            target_pointer = &layer->gate_proj_weight;
        } else if (strcmp(field, "ffn.down_proj.weight") == 0) {
            target_pointer = &layer->down_proj_weight;
        } else {
            // Skip unknown layer fields
            return 0;
        }
    } else {
        // Skip unknown tensors
        return 0;
    }

    if (!target_pointer) {
        return 0;  // Optional tensor, skip
    }

    // Load tensor from file
    tensor_t *tensor = ggml_load_tensor(fp, meta);
    if (!tensor) {
        snprintf(error_msg, max_error_len, "Failed to load tensor: %s", meta->name);
        return -1;
    }

    *target_pointer = tensor;
    return 0;
}

/**
 * Validate all required tensors are present.
 */
static int ggml_validate_tensors(llm_model_t *model,
                                  char *error_msg, int max_error_len) {
    if (!model->embedding_weight || !model->norm_final_weight) {
        snprintf(error_msg, max_error_len,
                 "Missing critical tensors (embedding or norm_final)");
        return -1;
    }

    for (int i = 0; i < model->config.num_layers; i++) {
        model_layer_weights_t *layer = &model->layers[i];
        if (!layer->norm_attn_weight || !layer->q_proj_weight ||
            !layer->k_proj_weight || !layer->v_proj_weight ||
            !layer->out_proj_weight || !layer->norm_ffn_weight ||
            !layer->gate_proj_weight || !layer->up_proj_weight ||
            !layer->down_proj_weight) {
            snprintf(error_msg, max_error_len, "Layer %d missing required tensors", i);
            return -1;
        }
    }

    return 0;
}

/**
 * Load model from GGML format file.
 */
int sapphire_load_ggml(const char *ggml_path, llm_model_t *out_model,
                       char *error_message, int max_error_len) {
    if (!ggml_path || !out_model) {
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Invalid arguments (path or model NULL)");
        }
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    memset(out_model, 0, sizeof(llm_model_t));

    // Open and parse GGML file header
    ggml_file_header_t file_header = {0};
    FILE *fp = ggml_file_open_and_parse_header(ggml_path, &file_header);
    if (!fp) {
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Failed to open GGML file: %s", ggml_path);
        }
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    // Extract model config
    if (ggml_extract_config(&file_header, &out_model->config) != 0) {
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Failed to extract config from GGML file");
        }
        fclose(fp);
        ggml_header_destroy(&file_header);
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    // Allocate layers
    out_model->layers = (model_layer_weights_t *)calloc(out_model->config.num_layers,
                                                         sizeof(model_layer_weights_t));
    if (!out_model->layers) {
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Failed to allocate layers");
        }
        fclose(fp);
        ggml_header_destroy(&file_header);
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    printf("Loaded GGML config: vocab=%d, d_model=%d, heads=%d, layers=%d\n",
           out_model->config.vocab_size,
           out_model->config.d_model,
           out_model->config.num_heads,
           out_model->config.num_layers);

    // Map all tensors
    int mapped_count = 0;
    int skipped_count = 0;

    for (uint32_t i = 0; i < file_header.tensor_count; i++) {
        const ggml_tensor_meta_t *meta = &file_header.tensors[i];
        if (!meta || !meta->name) continue;

        char err_buf[256] = {0};
        int result = ggml_map_tensor_to_model(fp, &file_header, meta, out_model, err_buf, sizeof(err_buf));

        if (result < 0) {
            fprintf(stderr, "ERROR: %s\n", err_buf);
            fclose(fp);
            ggml_header_destroy(&file_header);
            if (error_message && max_error_len > 0) {
                snprintf(error_message, max_error_len, "Tensor mapping failed: %s", err_buf);
            }
            free(out_model->layers);
            memset(out_model, 0, sizeof(llm_model_t));
            return SAPPHIRE_ERR_LOAD_MISSING_TENSOR;
        }

        if (result == 0) {
            skipped_count++;
        } else {
            mapped_count++;
        }
    }

    printf("✓ Mapped %d tensors from GGML file (%d skipped)\n", mapped_count, skipped_count);

    // Handle weight tying for lm_head
    if (!out_model->lm_head_weight && out_model->embedding_weight) {
        printf("INFO: lm_head.weight not found, using weight tying with embedding\n");
        out_model->lm_head_weight = out_model->embedding_weight;
    }

    // Validate tensors
    char err_buf[256] = {0};
    if (ggml_validate_tensors(out_model, err_buf, sizeof(err_buf)) != 0) {
        fprintf(stderr, "ERROR: %s\n", err_buf);
        fclose(fp);
        ggml_header_destroy(&file_header);
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Validation failed: %s", err_buf);
        }
        free(out_model->layers);
        memset(out_model, 0, sizeof(llm_model_t));
        return SAPPHIRE_ERR_LOAD_MISSING_TENSOR;
    }

    printf("✓ All required GGML tensors present\n");

    // Store file info (caller is responsible for cleanup)
    out_model->weight_file = fp;
    /* Move header to heap so it survives after this function returns. */
    ggml_file_header_t *hdr = (ggml_file_header_t *)malloc(sizeof(ggml_file_header_t));
    if (!hdr) {
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Failed to allocate GGML header copy");
        }
        fclose(fp);
        ggml_header_destroy(&file_header);
        free(out_model->layers);
        memset(out_model, 0, sizeof(llm_model_t));
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }
    *hdr = file_header; /* shallow copy is OK: tensors pointer is heap-allocated */
    out_model->file_header = (void *)hdr;

    return SAPPHIRE_OK;
}

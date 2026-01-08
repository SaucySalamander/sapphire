/**
 * @file ggml_model.c
 * @brief Model loading and management.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/ggml_model.h"
#include "../include/tensor.h"

/**
 * Load entire GGML model from file.
 */
llm_model_t* llm_model_load(const char *filename, const model_config_t *config) {
    if (!filename || !config) {
        fprintf(stderr, "ERROR: llm_model_load requires filename and config\n");
        return NULL;
    }
    
    // Parse GGML file header
    ggml_file_header_t file_header = {0};
    FILE *fp = ggml_file_open_and_parse_header(filename, &file_header);
    if (!fp) {
        fprintf(stderr, "ERROR: Failed to open and parse GGML file: %s\n", filename);
        return NULL;
    }
    
    // Allocate model structure
    llm_model_t *model = (llm_model_t *)malloc(sizeof(llm_model_t));
    if (!model) {
        fprintf(stderr, "ERROR: Failed to allocate model structure\n");
        ggml_header_destroy(&file_header);
        fclose(fp);
        return NULL;
    }
    
    model->config = *config;
    model->weight_file = fp;
    model->file_header = file_header;
    
    // Transfer ownership to model structure
    fp = NULL;
    file_header.tensors = NULL;

    model->layers = NULL;
    model->embedding_weight = NULL;
    model->norm_final_weight = NULL;
    model->lm_head_weight = NULL;
    
    // Load embedding weight
    const ggml_tensor_meta_t *emb_meta = ggml_find_tensor_meta(&file_header, "embedding.weight");
    if (emb_meta) {
        model->embedding_weight = ggml_load_tensor(fp, emb_meta);
        if (!model->embedding_weight) {
            fprintf(stderr, "ERROR: Failed to load embedding weights\n");
            llm_model_destroy(model);
            return NULL;
        }
    } else {
        fprintf(stderr, "WARNING: embedding.weight not found in model file\n");
    }
    
    // Load final layer norm
    const ggml_tensor_meta_t *norm_meta = ggml_find_tensor_meta(&file_header, "norm_final.weight");
    if (norm_meta) {
        model->norm_final_weight = ggml_load_tensor(fp, norm_meta);
        if (!model->norm_final_weight) {
            fprintf(stderr, "ERROR: Failed to load final norm weights\n");
            llm_model_destroy(model);
            return NULL;
        }
    } else {
        fprintf(stderr, "WARNING: norm_final.weight not found in model file\n");
    }
    
    // Load LM head
    const ggml_tensor_meta_t *lm_head_meta = ggml_find_tensor_meta(&file_header, "lm_head.weight");
    if (lm_head_meta) {
        model->lm_head_weight = ggml_load_tensor(fp, lm_head_meta);
        if (!model->lm_head_weight) {
            fprintf(stderr, "ERROR: Failed to load LM head weights\n");
            llm_model_destroy(model);
            return NULL;
        }
    } else {
        fprintf(stderr, "WARNING: lm_head.weight not found in model file\n");
    }
    
    // Load layer weights
    model->layers = (model_layer_weights_t *)malloc(config->num_layers * sizeof(model_layer_weights_t));
    if (!model->layers) {
        fprintf(stderr, "ERROR: Failed to allocate layer weights\n");
        llm_model_destroy(model);
        return NULL;
    }
    
    // Initialize all layer weights to NULL
    for (int i = 0; i < config->num_layers; i++) {
        model->layers[i].norm_attn_weight = NULL;
        model->layers[i].q_proj_weight = NULL;
        model->layers[i].k_proj_weight = NULL;
        model->layers[i].v_proj_weight = NULL;
        model->layers[i].out_proj_weight = NULL;
        model->layers[i].norm_ffn_weight = NULL;
        model->layers[i].up_proj_weight = NULL;
        model->layers[i].gate_proj_weight = NULL;
        model->layers[i].down_proj_weight = NULL;
    }
    
    // Load each layer
    char tensor_name[256];
    for (int layer = 0; layer < config->num_layers; layer++) {
        // Attention norm
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.attention_norm.weight", layer);
        const ggml_tensor_meta_t *meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].norm_attn_weight = ggml_load_tensor(fp, meta);
        }
        
        // Q, K, V projections
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.attention.q_proj.weight", layer);
        meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].q_proj_weight = ggml_load_tensor(fp, meta);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.attention.k_proj.weight", layer);
        meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].k_proj_weight = ggml_load_tensor(fp, meta);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.attention.v_proj.weight", layer);
        meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].v_proj_weight = ggml_load_tensor(fp, meta);
        }
        
        // Output projection
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.attention.out_proj.weight", layer);
        meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].out_proj_weight = ggml_load_tensor(fp, meta);
        }
        
        // FFN norm
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.ffn_norm.weight", layer);
        meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].norm_ffn_weight = ggml_load_tensor(fp, meta);
        }
        
        // FFN projections
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.ffn.up_proj.weight", layer);
        meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].up_proj_weight = ggml_load_tensor(fp, meta);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.ffn.gate_proj.weight", layer);
        meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].gate_proj_weight = ggml_load_tensor(fp, meta);
        }
        
        snprintf(tensor_name, sizeof(tensor_name), "layers.%d.ffn.down_proj.weight", layer);
        meta = ggml_find_tensor_meta(&file_header, tensor_name);
        if (meta) {
            model->layers[layer].down_proj_weight = ggml_load_tensor(fp, meta);
        }
    }
    
    fprintf(stderr, "INFO: Model loaded successfully: %d layers, vocab_size=%d, d_model=%d\n",
            config->num_layers, config->vocab_size, config->d_model);
    
    return model;
}

/**
 * Free all model memory.
 */
void llm_model_destroy(llm_model_t *model) {
    if (!model) return;
    
    if (model->embedding_weight) {
        tensor_release(model->embedding_weight);
    }
    if (model->norm_final_weight) {
        tensor_release(model->norm_final_weight);
    }
    if (model->lm_head_weight) {
        tensor_release(model->lm_head_weight);
    }
    
    if (model->layers) {
        for (int i = 0; i < model->config.num_layers; i++) {
            model_layer_weights_t *layer = &model->layers[i];
            if (layer->norm_attn_weight) tensor_release(layer->norm_attn_weight);
            if (layer->q_proj_weight) tensor_release(layer->q_proj_weight);
            if (layer->k_proj_weight) tensor_release(layer->k_proj_weight);
            if (layer->v_proj_weight) tensor_release(layer->v_proj_weight);
            if (layer->out_proj_weight) tensor_release(layer->out_proj_weight);
            if (layer->norm_ffn_weight) tensor_release(layer->norm_ffn_weight);
            if (layer->up_proj_weight) tensor_release(layer->up_proj_weight);
            if (layer->gate_proj_weight) tensor_release(layer->gate_proj_weight);
            if (layer->down_proj_weight) tensor_release(layer->down_proj_weight);
        }
        free(model->layers);
    }
    
    if (model->weight_file) {
        fclose(model->weight_file);
    }
    
    ggml_header_destroy(&model->file_header);
    
    free(model);
}

/**
 * Print model configuration and layer summary.
 */
void llm_model_print_info(const llm_model_t *model) {
    if (!model) return;
    
    printf("\n========== Model Configuration ==========\n");
    printf("Vocabulary Size: %d\n", model->config.vocab_size);
    printf("Hidden Dimension (d_model): %d\n", model->config.d_model);
    printf("Num Heads: %d\n", model->config.num_heads);
    printf("Dim per Head (d_k): %d\n", model->config.d_k);
    printf("Num Layers: %d\n", model->config.num_layers);
    printf("Max Context Length: %d\n", model->config.max_context_len);
    printf("RoPE Base: %.1f\n", model->config.rope_base);
    printf("Total Tensors in File: %u\n", model->file_header.tensor_count);
    printf("========================================\n\n");
}

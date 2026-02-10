/**
 * @file layer_config_loader.c
 * @brief Implementation of layer configuration loading.
 *
 * Populates layer configurations from model specification, deriving layer types
 * and per-layer settings from config.json metadata.
 */

#include "layer_config_loader.h"

#include <string.h>

#include "gemma3_270m_config.h"
#include "log.h"

/**
 * @brief Helper: Determine if a given layer is a global attention layer.
 *
 * Uses the layer_types_mask if available (bit i set = global attention),
 * otherwise defaults to pattern: every 6th layer (indices 5, 11, 17...).
 *
 * @param config Gemma3 configuration.
 * @param layer_idx Layer index (0-based).
 * @return 1 if layer should use global attention, 0 for local/sliding-window.
 */
static int is_global_attention_layer(const gemma3_270m_config_t* config, int layer_idx) {
    if (!config) return 0;
    
    /* Use loaded bitmask if available: bit i set => full (global) attention for layer i */
    if (config->layer_types_mask) {
        return ((config->layer_types_mask >> layer_idx) & 1ULL) != 0ULL;
    }
    
    /* Default pattern: 5 Sliding Window (Local) : 1 Global Attention */
    /* Global Layers (5, 11, 17, ...) use Base 1,000,000; Local Layers use Base 10,000 */
    return ((layer_idx + 1) % 6 == 0);
}

int layer_config_load_from_spec(const model_spec_t* spec, int num_layers, 
                                 sapphire_layer_config_t* out_configs) {
    if (!spec || !spec->variant_config || !out_configs) {
        LOG_ERROR("layer_config_load_from_spec: invalid arguments (spec=%p variant_config=%p out_configs=%p)",
                  spec, spec ? spec->variant_config : NULL, out_configs);
        return -1;
    }
    
    const gemma3_270m_config_t* cfg = (const gemma3_270m_config_t*)spec->variant_config;
    
    if (num_layers <= 0) {
        LOG_ERROR("layer_config_load_from_spec: invalid num_layers=%d", num_layers);
        return -1;
    }
    
    /* Populate each layer's configuration */
    for (int i = 0; i < num_layers; i++) {
        sapphire_layer_config_t* lc = &out_configs[i];
        
        /* Initialize to zero */
        memset(lc, 0, sizeof(sapphire_layer_config_t));
        
        /* Set layer index */
        lc->layer_idx = i;
        
        /* For now, all layers are softmax attention (extensible for SSM/linear later) */
        lc->type = LAYER_TYPE_ATTENTION_SOFTMAX;
        
        /* Configure attention settings */
        lc->config.attention.is_global = is_global_attention_layer(cfg, i);
        lc->config.attention.window_size = cfg->sliding_window;
        
        LOG_DEBUG("Layer %d: type=%d is_global=%d window_size=%d",
                  i, lc->type, lc->config.attention.is_global, lc->config.attention.window_size);
    }
    
    LOG_INFO("Loaded layer configs for %d layers", num_layers);
    return 0;
}

/**
 * @file layer_dispatch.h
 * @brief Layer-type dispatch system for hybrid transformer architectures.
 *
 * Supports runtime dispatch of different layer types:
 * - LAYER_TYPE_ATTENTION_SOFTMAX: Standard softmax attention (current Gemma 3)
 * - LAYER_TYPE_ATTENTION_LINEAR: LoLCATs linearized attention
 * - LAYER_TYPE_SSM_RECURRENT: Mamba-style state space model layers
 *
 * Each layer's forward pass is dispatched based on its configured type,
 * allowing mixed architectures within a single model.
 */

#ifndef LAYER_DISPATCH_H
#define LAYER_DISPATCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Enumeration of supported layer types in Sapphire.
 */
typedef enum {
    LAYER_TYPE_ATTENTION_SOFTMAX,   /**< Standard softmax attention (Gemma 3 default) */
    LAYER_TYPE_ATTENTION_LINEAR,    /**< LoLCATs linearized attention */
    LAYER_TYPE_SSM_RECURRENT,       /**< Mamba-style state space model */
} sapphire_layer_type_t;

/**
 * @brief Per-layer configuration for attention-based layers.
 */
typedef struct {
    int is_global;                  /**< 1 = full global attention, 0 = local sliding-window */
    int window_size;                /**< Sliding window size (ignored if is_global=1) */
} sapphire_attention_config_t;

/**
 * @brief Per-layer configuration for SSM (state space model) layers.
 */
typedef struct {
    int state_dim;                  /**< SSM state dimension */
    int conv_width;                 /**< Convolutional kernel width */
    int expand_ratio;               /**< Expansion ratio for internal state */
} sapphire_ssm_config_t;

/**
 * @brief Complete layer configuration (type + per-type settings).
 *
 * Represents all configuration needed to dispatch and execute a single
 * transformer layer. Populated from model metadata (config.json) and stored
 * in the inference session.
 */
typedef struct {
    sapphire_layer_type_t type;     /**< Layer type determining dispatch */
    int layer_idx;                  /**< Layer index in model */
    
    /* Per-type configuration */
    union {
        sapphire_attention_config_t attention;  /**< Config for softmax/linear attention */
        sapphire_ssm_config_t ssm;              /**< Config for SSM layers */
    } config;
} sapphire_layer_config_t;

#ifdef __cplusplus
}
#endif

#endif /* LAYER_DISPATCH_H */

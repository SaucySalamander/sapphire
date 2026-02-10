/**
 * @file layer_config_loader.h
 * @brief Layer configuration loading and initialization.
 *
 * Provides utilities to populate layer configurations from model metadata
 * and initialize them into an inference session.
 */

#ifndef LAYER_CONFIG_LOADER_H
#define LAYER_CONFIG_LOADER_H

#include "layer_dispatch.h"
#include "model_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load and populate layer configurations from model specification.
 *
 * Initializes an array of layer configs based on the model's layer_types
 * metadata and configuration. Currently defaults to LAYER_TYPE_ATTENTION_SOFTMAX
 * for all layers, with per-layer attention settings (global vs. local) derived
 * from the layer_types_mask.
 *
 * @param spec Model specification containing variant_config with layer metadata.
 * @param num_layers Number of layers to configure (usually config->num_hidden_layers).
 * @param out_configs Output array of layer configs (caller-allocated, size >= num_layers).
 *
 * @return 0 on success, -1 on error (logged via LOG_ERROR).
 */
int layer_config_load_from_spec(const model_spec_t* spec, int num_layers, 
                                 sapphire_layer_config_t* out_configs);

#ifdef __cplusplus
}
#endif

#endif /* LAYER_CONFIG_LOADER_H */

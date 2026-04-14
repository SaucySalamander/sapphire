/*
 * @file gemma3_7b_spec.h
 * @brief Static model & tokenizer specification for Gemma 3 7B IT.
 */

#ifndef GEMMA3_7B_SPEC_H
#define GEMMA3_7B_SPEC_H

#include "gemma3_config.h"
#include "model_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const model_loader_hooks_t GEMMA3_7B_LOADER_HOOKS;
extern const tensor_map_entry_t GEMMA3_7B_TENSOR_MAP[];
size_t get_gemma3_7b_tensor_map_size(void);

/*
 * 64-layer capacity template: 64 layers * 13 tensors/layer + 3 top-level tensors.
 * The loader trims spec->tensor_map_size at runtime to cfg->num_hidden_layers.
 */
#define GEMMA3_7B_MAX_LAYERS 64
#define GEMMA3_7B_TENSOR_MAP_SIZE ((GEMMA3_7B_MAX_LAYERS * 13) + 3)

extern const tokenizer_spec_t GEMMA3_7B_TOKENIZER_SPEC;
extern const model_files_t    GEMMA3_7B_FILES;
extern gemma3_270m_config_t   GEMMA3_7B_RUNTIME_CONFIG;
extern model_spec_t           GEMMA3_7B_IT_SPEC;

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_7B_SPEC_H */
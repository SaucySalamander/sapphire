/*
 * @file gemma3_270m_spec.h
 * @brief Static model & tokenizer specification for Gemma 3 270M IT
 */

#ifndef GEMMA3_270M_SPEC_H
#define GEMMA3_270M_SPEC_H

#include "model_spec.h"
#include "gemma3_270m_config.h" /* Provides `gemma3_config_t` and config constants */

/* Forward-declare Gemma3 loader hooks (defined in src/loader/gemma3_loader.c) */
extern const model_loader_hooks_t GEMMA3_LOADER_HOOKS;

#ifdef __cplusplus
extern "C" {
#endif

/* Exported tensor map info from src/loader/gemma3_map.c */
extern const tensor_map_entry_t GEMMA3_270M_TENSOR_MAP[];
size_t get_gemma3_270m_tensor_map_size(void);

/* Exported model specifications and supporting structs */
extern const tokenizer_spec_t GEMMA3_270M_TOKENIZER_SPEC;
extern const model_files_t GEMMA3_270M_FILES;
extern gemma3_270m_config_t GEMMA3_270M_RUNTIME_CONFIG;
extern model_spec_t GEMMA3_270M_IT_SPEC;
extern model_spec_t GEMMA3_270M_SPEC;

/* Convenience accessor */
#define GEMMA3_SPEC_RUNTIME_CONFIG(spec) ((gemma3_270m_config_t *)((spec)->variant_config))

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_270M_SPEC_H */

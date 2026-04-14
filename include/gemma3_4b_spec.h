/*
 * @file gemma3_4b_spec.h
 * @brief Static model & tokenizer specification for Gemma 3 4B IT
 */

#ifndef GEMMA3_4B_SPEC_H
#define GEMMA3_4B_SPEC_H

#include "model_spec.h"
#include "gemma3_config.h" /* Reuse gemma3_270m_config_t for 4B */

#ifdef __cplusplus
extern "C" {
#endif

/* Loader hooks defined in src/loader/gemma3_4b_loader.c */
extern const model_loader_hooks_t GEMMA3_4B_LOADER_HOOKS;

/* Tensor map defined in src/loader/gemma3_4b_map.c */
extern const tensor_map_entry_t GEMMA3_4B_TENSOR_MAP[];
size_t get_gemma3_4b_tensor_map_size(void);

/* 34 layers * 13 tensors/layer + 3 (embed, norm, lm_head) = 445 */
#define GEMMA3_4B_TENSOR_MAP_SIZE 445

/* Supporting structs and runtime state */
extern const tokenizer_spec_t  GEMMA3_4B_TOKENIZER_SPEC;
extern const model_files_t     GEMMA3_4B_FILES;
extern gemma3_270m_config_t    GEMMA3_4B_RUNTIME_CONFIG;
extern model_spec_t            GEMMA3_4B_IT_SPEC;

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_4B_SPEC_H */

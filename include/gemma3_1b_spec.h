/*
 * @file gemma3_1b_spec.h
 * @brief Static model & tokenizer specification for Gemma 3 1B IT
 */

#ifndef GEMMA3_1B_SPEC_H
#define GEMMA3_1B_SPEC_H

#include "gemma3_config.h"
#include "model_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const tensor_map_entry_t GEMMA3_1B_TENSOR_MAP[];
size_t get_gemma3_1b_tensor_map_size(void);

/* 26 layers * 13 tensors/layer + 3 (embed, norm, lm_head) = 341.
 * The on-disk 1B checkpoint ties lm_head to embeddings; missing lm_head is
 * resolved by the shared safetensors/model reader path. */
#define GEMMA3_1B_TENSOR_MAP_SIZE 341

extern const tokenizer_spec_t GEMMA3_1B_TOKENIZER_SPEC;
extern const model_files_t GEMMA3_1B_FILES;
extern gemma3_270m_config_t GEMMA3_1B_RUNTIME_CONFIG;
extern model_spec_t GEMMA3_1B_IT_SPEC;

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_1B_SPEC_H */
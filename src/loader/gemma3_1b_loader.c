/*
 * @file gemma3_1b_loader.c
 * @brief Static spec objects for Gemma 3 1B IT.
 */

#include "gemma3_1b_spec.h"
#include "gemma3_270m_spec.h"

const tokenizer_spec_t GEMMA3_1B_TOKENIZER_SPEC = {
    .tokenizer_json     = "tokenizer.json",
    .tokenizer_model    = "tokenizer.model",
    .special_tokens_map = "special_tokens_map.json",
    .bos_token_id       = 2,
    .eos_token_id       = 1,
    .pad_token_id       = 0
};

const model_files_t GEMMA3_1B_FILES = {
    .config_json        = "config.json",
    .tokenizer_json     = "tokenizer.json",
    .tokenizer_model    = "tokenizer.model",
    .added_tokens       = "added_tokens.json",
    .special_tokens_map = "special_tokens_map.json",
    .generation_config  = "generation_config.json",
    .chat_template      = "chat_template.jinja",
    .readme             = "README.md"
};

gemma3_270m_config_t GEMMA3_1B_RUNTIME_CONFIG = {0};

model_spec_t GEMMA3_1B_IT_SPEC = {
    .model_id        = "gemma-3-1b-it",
    .tensor_map      = GEMMA3_1B_TENSOR_MAP,
    .tensor_map_size = GEMMA3_1B_TENSOR_MAP_SIZE,
    .tokenizer_spec  = &GEMMA3_1B_TOKENIZER_SPEC,
    .files           = &GEMMA3_1B_FILES,
    .variant_config  = &GEMMA3_1B_RUNTIME_CONFIG,
    .loader_hooks    = &GEMMA3_LOADER_HOOKS
};
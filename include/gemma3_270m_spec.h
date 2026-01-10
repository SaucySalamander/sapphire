/*
 * @file gemma3_270m_spec.h
 * @brief Static model & tokenizer specification for Gemma 3 270M IT
 */

#ifndef GEMMA3_270M_SPEC_H
#define GEMMA3_270M_SPEC_H

#include "model_spec.h"
#include "gemma3_270m_map.h"  /* Provides GEMMA3_270M_TENSOR_MAP and size macro */
#include "gemma3_270m_config.h" /* Provides `gemma3_config_t` and config constants */

/* Forward-declare Gemma3 loader hooks (defined in src/loader/gemma3_loader.c) */
extern const model_loader_hooks_t GEMMA3_LOADER_HOOKS;

#ifdef __cplusplus
extern "C" {
#endif


/* Tokenizer spec (filenames expected under model dir) */
static const tokenizer_spec_t GEMMA3_270M_TOKENIZER_SPEC = {
    .tokenizer_json = "tokenizer.json",
    .tokenizer_model = "tokenizer.model",
    .special_tokens_map = "special_tokens_map.json",
    .bos_token_id = 2,
    .eos_token_id = 1,
    .pad_token_id = 0
};

/* Important model files (relative to model directory) */
static const model_files_t GEMMA3_270M_FILES = {
    .config_json = "config.json",
    .tokenizer_json = "tokenizer.json",
    .tokenizer_model = "tokenizer.model",
    .added_tokens = "added_tokens.json",
    .special_tokens_map = "special_tokens_map.json",
    .generation_config = "generation_config.json",
    .chat_template = "chat_template.jinja",
    .readme = "README.md"
};


/* Runtime configuration instance (populated by the Gemma3 loader hook) */
static gemma3_270m_config_t GEMMA3_270M_RUNTIME_CONFIG = {0};

/* Convenience accessor */
#define GEMMA3_SPEC_RUNTIME_CONFIG(spec) ((gemma3_270m_config_t *)((spec)->variant_config))

/* The public model specification object */
static model_spec_t GEMMA3_270M_SPEC = {
    .model_id = "gemma3-270m-it",
    .tensor_map = GEMMA3_270M_TENSOR_MAP,
    .tensor_map_size = GEMMA3_270M_TENSOR_MAP_SIZE,
    .tokenizer_spec = &GEMMA3_270M_TOKENIZER_SPEC,
    .files = &GEMMA3_270M_FILES,
    .variant_config = &GEMMA3_270M_RUNTIME_CONFIG,
    .loader_hooks = &GEMMA3_LOADER_HOOKS
};

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_270M_SPEC_H */

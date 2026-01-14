/*
 * @file gemma3_270m_tokenizer.h
 * @brief Tokenizer static references for Gemma 3 270M IT (header-only)
 *
 * Provides minimal data extracted from tokenizer files (added tokens and
 * special token IDs) to be used at loader initialization time. The real
 * tokenizer implementation will be redone later; this provides a lightweight
 * bridge for now.
 */

#ifndef GEMMA3_270M_TOKENIZER_H
#define GEMMA3_270M_TOKENIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Basic special token IDs (from tokenizer_config.json / config.json) */
#define GEMMA3_270M_BOS_ID 2
#define GEMMA3_270M_EOS_ID 1
#define GEMMA3_270M_PAD_ID 0

/* Added tokens from added_tokens.json */
static const struct {
    const char *token;
    int id;
} GEMMA3_270M_ADDED_TOKENS[] = {
    {"<image_soft_token>", 262144},
    {NULL, 0}
};

/* Convenience accessor: number of added tokens (excluding sentinel) */
static inline size_t gemma3_270m_added_tokens_count(void) {
    size_t n = 0;
    while (GEMMA3_270M_ADDED_TOKENS[n].token) n++;
    return n;
}

#ifdef __cplusplus
}
#endif

#endif /* GEMMA3_270M_TOKENIZER_H */

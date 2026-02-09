/*
 * @file simple_json.h
 * @brief Minimal, safe JSON tokenizer and helpers (jsmn-like)
 *
 * This is small and dependency-free. It tokenizes JSON into an array of
 * tokens (type/start/end/size) without allocating during tokenization.
 * Helpers are provided for reading top-level numbers, strings, and iterating
 * over string arrays.
 */

#ifndef SIMPLE_JSON_H
#define SIMPLE_JSON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SJSON_UNDEF = 0,
    SJSON_OBJ = 1,
    SJSON_ARR = 2,
    SJSON_STR = 3,
    SJSON_PRIM = 4
} sjson_type_t;

typedef struct {
    sjson_type_t type;
    int start; /* inclusive */
    int end;   /* exclusive */
    int size;  /* children count for objects/arrays */
    int parent; /* index of parent token or -1 */
} sjson_token_t;

/**
 * @brief Low-level Cursor API for streaming JSON parsing.
 *
 * Provides primitives for character-by-character JSON parsing without
 * allocating tokens. Useful for high-performance use cases like Safetensors.
 */
typedef struct {
    const char *json;  /* JSON buffer */
    size_t len;        /* Total length */
    size_t pos;        /* Current position */
} sjson_cursor_t;

/** Initialize cursor from JSON string. */
sjson_cursor_t sjson_cursor_init(const char *json, size_t len);

/** Skip whitespace and return next significant character without advancing. Returns '\0' if EOF. */
char sjson_cursor_peek(sjson_cursor_t *c);

/** If next significant character matches expected, advance and return 1; otherwise return 0. */
int sjson_cursor_consume(sjson_cursor_t *c, char expected);

/** Parse a JSON string (quotes included) into dst buffer. Returns 0 on success, -1 on error. */
int sjson_cursor_parse_string(sjson_cursor_t *c, char *dst, int dst_len);

/** Parse a JSON number into uint64_t using strtoull. Returns 0 on success, -1 on error. */
int sjson_cursor_parse_u64(sjson_cursor_t *c, uint64_t *out);

/** Skip the current value (primitive, object, or array) and advance cursor to next sibling. Returns 0 on success, -1 on error. */
int sjson_cursor_skip_value(sjson_cursor_t *c);

/** Tokenize `json` into `tokens` buffer (max tokens = max_tokens). Uses cursor internally. */
int sjson_tokenize(const char *json, sjson_token_t *tokens, int max_tokens);

/** Compare token (string) with C string. Returns 1 if equal. */
int sjson_tok_eq(const char *json, const sjson_token_t *t, const char *s);

/** Find a key inside an object token (tokens array, ntokens length). Returns index of value token or -1. */
int sjson_find_key(const char *json, const sjson_token_t *tokens, int ntokens, int obj_index, const char *key);

/** Convert a primitive token to double. Returns 0 on success, non-zero on error. */
int sjson_token_to_double(const char *json, const sjson_token_t *t, double *out);

/** Convert a primitive token to int64_t. Returns 0 on success, non-zero on error. */
int sjson_token_to_int64(const char *json, const sjson_token_t *t, int64_t *out);

/** Copy string token into dst (dst_len). Returns 0 on success. */
int sjson_token_to_str(const char *json, const sjson_token_t *t, char *dst, int dst_len);

/** Iterate over string array at token index arr_idx; calls cb for each string entry. Returns count or -1 on error. */
int sjson_iterate_str_array(const char *json, const sjson_token_t *tokens, int ntokens, int arr_idx, int (*cb)(const char*, int, void*), void *user);

/** Iterate over numeric array at token index arr_idx; calls cb for each uint32_t value. Returns count or -1 on error. */
int sjson_iterate_num_array(const char *json, const sjson_token_t *tokens, int ntokens, int arr_idx, int (*cb)(uint32_t, void*), void *user);

#ifdef __cplusplus
}
#endif

#endif /* SIMPLE_JSON_H */
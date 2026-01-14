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

/** Tokenize `json` into `tokens` buffer (max tokens = max_tokens). */
int sjson_tokenize(const char *json, sjson_token_t *tokens, int max_tokens);

/** Compare token (string) with C string. Returns 1 if equal. */
int sjson_tok_eq(const char *json, const sjson_token_t *t, const char *s);

/** Find a key inside an object token (tokens array, ntokens length). Returns index of value token or -1. */
int sjson_find_key(const char *json, const sjson_token_t *tokens, int ntokens, int obj_index, const char *key);

/** Convert a primitive token to double. Returns 0 on success, non-zero on error. */
int sjson_token_to_double(const char *json, const sjson_token_t *t, double *out);

/** Copy string token into dst (dst_len). Returns 0 on success. */
int sjson_token_to_str(const char *json, const sjson_token_t *t, char *dst, int dst_len);

/** Iterate over string array at token index arr_idx; calls cb for each string entry. Returns count or -1 on error. */
int sjson_iterate_str_array(const char *json, const sjson_token_t *tokens, int ntokens, int arr_idx, int (*cb)(const char*, int, void*), void *user);

/* Read entire file into a newly-allocated null-terminated buffer. Caller must free. */
int json_read_file(const char *path, char **out_buf, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* SIMPLE_JSON_H */
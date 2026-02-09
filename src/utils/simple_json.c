/*
 * Minimal JSON parser: Cursor API + Tokenizer
 * 
 * Architecture:
 * - Cursor API: Low-level streaming primitives (no allocations)
 * - Tokenizer: Uses Cursor internally, outputs sjson_token_t array
 * - Helpers: High-level token manipulation functions
 */

#include "../include/simple_json.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/**
 * @brief Initialize a cursor for streaming JSON parsing.
 */
sjson_cursor_t sjson_cursor_init(const char *json, size_t len) {
    sjson_cursor_t c = { .json = json, .len = len, .pos = 0 };
    return c;
}

/**
 * @brief Skip whitespace and return next significant character without advancing.
 * Returns '\0' if EOF.
 */
char sjson_cursor_peek(sjson_cursor_t *c) {
    if (!c || !c->json) return '\0';
    while (c->pos < c->len && isspace((unsigned char)c->json[c->pos])) {
        c->pos++;
    }
    if (c->pos >= c->len) return '\0';
    return c->json[c->pos];
}

/**
 * @brief If next significant character matches expected, advance and return 1.
 * Otherwise return 0. Skips whitespace before comparison.
 */
int sjson_cursor_consume(sjson_cursor_t *c, char expected) {
    if (!c || !c->json) return 0;
    char ch = sjson_cursor_peek(c);
    if (ch == expected) {
        c->pos++;
        return 1;
    }
    return 0;
}

/**
 * @brief Parse a JSON string (including quotes) into dst buffer.
 * Handles basic escape sequences (\", \\, \/, \b, \f, \n, \r, \t).
 * Returns 0 on success, -1 on parse error.
 */
int sjson_cursor_parse_string(sjson_cursor_t *c, char *dst, int dst_len) {
    if (!c || !c->json || !dst || dst_len <= 0) return -1;
    
    // Expect opening quote
    if (!sjson_cursor_consume(c, '"')) return -1;
    
    int di = 0;
    while (c->pos < c->len && di < dst_len - 1) {
        char ch = c->json[c->pos];
        
        if (ch == '"') {
            // End of string
            dst[di] = '\0';
            c->pos++;
            return 0;
        }
        
        if (ch == '\\' && c->pos + 1 < c->len) {
            c->pos++;
            char esc = c->json[c->pos];
            switch (esc) {
                case '"':  dst[di++] = '"'; break;
                case '\\': dst[di++] = '\\'; break;
                case '/':  dst[di++] = '/'; break;
                case 'b':  dst[di++] = '\b'; break;
                case 'f':  dst[di++] = '\f'; break;
                case 'n':  dst[di++] = '\n'; break;
                case 'r':  dst[di++] = '\r'; break;
                case 't':  dst[di++] = '\t'; break;
                default:   dst[di++] = esc; break;
            }
            c->pos++;
        } else {
            dst[di++] = ch;
            c->pos++;
        }
    }
    
    return -1; // Unterminated string
}

/**
 * @brief Parse a JSON number into uint64_t using strtoull(base 10).
 * Returns 0 on success, -1 on error.
 */
int sjson_cursor_parse_u64(sjson_cursor_t *c, uint64_t *out) {
    if (!c || !c->json || !out) return -1;
    
    char ch = sjson_cursor_peek(c);
    if (ch == '\0' || (!isdigit((unsigned char)ch) && ch != '-')) {
        return -1;
    }
    
    const char *start = c->json + c->pos;
    char *endptr = NULL;
    uint64_t val = strtoull(start, &endptr, 10);
    
    if (endptr == start) return -1; // No valid number parsed
    
    c->pos = (endptr - c->json);
    *out = val;
    return 0;
}

/* ============================================================================
 * TOKENIZE HELPER STRUCTURES AND FUNCTIONS (Low CCN)
 * ============================================================================ */

/**
 * @brief Parse state for tokenizer - consolidates all mutable state.
 * Passed to all helper functions to avoid excessive parameters.
 */
typedef struct {
    sjson_cursor_t *c;          // Pointer to cursor (shared, advanced by helpers)
    sjson_token_t *tokens;      // Token output array
    int max_tokens;             // Maximum tokens capacity
    int nt;                     // Current token count
    int parent;                 // Current parent token index
    int *parent_stack;          // Nesting depth stack
    int ps_cap;                 // Parent stack capacity
    int ps_top;                 // Parent stack top
} tokenize_state_t;

/**
 * @brief Ensure parent stack has capacity. Reallocates if needed.
 * @return 0 on success, -1 on realloc failure
 * Error Code: -1 (realloc failure)
 */
static int tokenize_expand_parent_stack(tokenize_state_t *state) {
    if (state->ps_top + 1 >= state->ps_cap) {
        int new_cap = state->ps_cap * 2;
        int *new_stack = (int*)realloc(state->parent_stack, sizeof(int) * new_cap);
        if (!new_stack) return -1;
        state->parent_stack = new_stack;
        state->ps_cap = new_cap;
    }
    return 0;
}

/**
 * @brief Emit a container token (object or array).
 * Advances cursor past opening brace/bracket.
 * If max_tokens exceeded, still tracks nesting but doesn't emit.
 * @return 0 on success, -1 on realloc failure
 */
static int tokenize_emit_container(tokenize_state_t *state, sjson_type_t type) {
    if (tokenize_expand_parent_stack(state) < 0) return -1;
    
    int start_pos = (int)state->c->pos;
    state->c->pos++;  // Consume opening char
    
    if (state->nt < state->max_tokens) {
        state->tokens[state->nt].type = type;
        state->tokens[state->nt].start = start_pos;
        state->tokens[state->nt].end = -1;
        state->tokens[state->nt].size = 0;
        state->tokens[state->nt].parent = state->parent;
        if (state->parent >= 0 && state->parent < state->max_tokens) {
            state->tokens[state->parent].size++;
        }
        state->parent = state->nt;
        state->parent_stack[state->ps_top++] = state->parent;
        state->nt++;
    } else {
        // Max tokens exceeded: still track nesting, just don't emit
        state->parent_stack[state->ps_top++] = -1;
        state->parent = -1;
    }
    
    return 0;
}

/**
 * @brief Close a container token (object or array).
 * Advances cursor past closing brace/bracket.
 * @return 0 on success, -2 on syntax error
 */
static int tokenize_close_container(tokenize_state_t *state) {
    if (state->ps_top <= 0) return -2;
    
    int close_pos = (int)state->c->pos;
    state->c->pos++;  // Consume closing char
    
    int popped = state->parent_stack[--state->ps_top];
    if (popped >= 0 && popped < state->max_tokens) {
        state->tokens[popped].end = close_pos + 1;
    }
    
    state->parent = (state->ps_top > 0) ? state->parent_stack[state->ps_top - 1] : -1;
    return 0;
}

/**
 * @brief Parse a string token. Consumes opening quote through closing quote.
 * Advances cursor to position after closing quote.
 * @return 0 on success, -2 on syntax error, -3 on max_tokens exceeded
 */
/**
 * @brief Parse a string token. Consumes opening quote through closing quote.
 * Advances cursor to position after closing quote.
 * If max_tokens exceeded, still consumes but doesn't emit.
 * @return 0 on success, -2 on syntax error
 */
static int tokenize_string_token(tokenize_state_t *state) {
    int str_start = (int)state->c->pos + 1;  // Position after opening quote
    
    // Manually scan string 
    state->c->pos++;  // Skip opening quote
    
    while (state->c->pos < state->c->len) {
        if (state->c->json[state->c->pos] == '\\' && state->c->pos + 1 < state->c->len) {
            state->c->pos += 2;  // Skip escaped character
        } else if (state->c->json[state->c->pos] == '"') {
            break;  // Found closing quote
        } else {
            state->c->pos++;
        }
    }
    
    if (state->c->pos >= state->c->len) return -2;  // Unterminated string
    
    int str_end = (int)state->c->pos;  // Position of closing quote
    state->c->pos++;  // Consume closing quote
    
    if (state->nt < state->max_tokens) {
        state->tokens[state->nt].type = SJSON_STR;
        state->tokens[state->nt].start = str_start;
        state->tokens[state->nt].end = str_end;
        state->tokens[state->nt].size = 0;
        state->tokens[state->nt].parent = state->parent;
        if (state->parent >= 0 && state->parent < state->max_tokens) {
            state->tokens[state->parent].size++;
        }
        state->nt++;
    }
    // If max_tokens exceeded, still consumed the string, just didn't emit
    
    return 0;
}

/**
 * @brief Parse a primitive token (number, boolean, null).
 * Uses 64-bit precision via strtoll/strtoull for large offsets.
 * Advances cursor to position after primitive.
 * If max_tokens exceeded, still consumes but doesn't emit.
 * @return 0 on success, -2 on syntax error
 */
static int tokenize_primitive_token(tokenize_state_t *state) {
    int prim_start = (int)state->c->pos;
    
    // Scan until we hit a structural delimiter
    while (state->c->pos < state->c->len) {
        char ch = state->c->json[state->c->pos];
        if (isspace((unsigned char)ch) || ch == ',' || ch == '}' || ch == ']' || ch == ':') {
            break;
        }
        state->c->pos++;
    }
    
    int prim_end = (int)state->c->pos;
    
    // Validate we actually consumed something
    if (prim_end == prim_start) return -2;
    
    if (state->nt < state->max_tokens) {
        state->tokens[state->nt].type = SJSON_PRIM;
        state->tokens[state->nt].start = prim_start;
        state->tokens[state->nt].end = prim_end;
        state->tokens[state->nt].size = 0;
        state->tokens[state->nt].parent = state->parent;
        if (state->parent >= 0 && state->parent < state->max_tokens) {
            state->tokens[state->parent].size++;
        }
        state->nt++;
    }
    // If max_tokens exceeded, still consumed the primitive, just didn't emit
    
    return 0;
}

/**
 * @brief Helper to skip a quoted string without copied data or unescaping.
 */
static int sjson_cursor_skip_string_raw(sjson_cursor_t *c) {
    if (c->pos >= c->len) return -1;
    c->pos++; // Consume opening quote
    while (c->pos < c->len) {
        if (c->json[c->pos] == '\\' && c->pos + 1 < c->len) {
            c->pos += 2; // Skip escaped character
        } else if (c->json[c->pos] == '"') {
            c->pos++; // Consume closing quote
            return 0;
        } else {
            c->pos++;
        }
    }
    return -1;
}

/**
 * @brief Helper to skip a primitive value (number, boolean, null).
 */
static int sjson_cursor_skip_primitive_raw(sjson_cursor_t *c) {
    while (c->pos < c->len) {
        char ch = c->json[c->pos];
        if (isspace((unsigned char)ch) || ch == ',' || ch == '}' || ch == ']' || ch == ':') {
            break;
        }
        c->pos++;
    }
    return 0;
}

/**
 * @brief Helper to skip a nested container (object or array).
 */
static int sjson_cursor_skip_container_raw(sjson_cursor_t *c) {
    char open = c->json[c->pos];
    char close = (open == '[') ? ']' : '}';
    int depth = 1;
    c->pos++; // Consume opening brace/bracket
    
    while (c->pos < c->len && depth > 0) {
        char ch = c->json[c->pos];
        if (ch == '"') {
            if (sjson_cursor_skip_string_raw(c) < 0) return -1;
        } else if (ch == open) {
            depth++;
            c->pos++;
        } else if (ch == close) {
            depth--;
            c->pos++;
        } else {
            c->pos++;
        }
    }
    return (depth == 0) ? 0 : -1;
}

/**
 * @brief Skip the current value (primitive, object, or array).
 * Advances cursor to the position after the value.
 * Returns 0 on success, -1 on parse error.
 */
int sjson_cursor_skip_value(sjson_cursor_t *c) {
    if (!c || !c->json) return -1;
    
    char ch = sjson_cursor_peek(c);
    if (ch == '\0') return -1;
    
    if (ch == '"') {
        return sjson_cursor_skip_string_raw(c);
    }
    
    if (ch == '[' || ch == '{') {
        return sjson_cursor_skip_container_raw(c);
    }
    
    // Fallback to primitive
    if (isdigit((unsigned char)ch) || ch == '-' || ch == 't' || ch == 'f' || ch == 'n') {
        return sjson_cursor_skip_primitive_raw(c);
    }
    
    return -1; // Unknown token type
}

/**
 * @brief Tokenize JSON string into token array.
 * 
 * Reverts to proven working implementation to ensure reliability.
 * Uses manual state management for robustness.
 * 
 * @return Number of tokens on success (>= 0), or negative error code
 */
int sjson_tokenize(const char *json, sjson_token_t *tokens, int max_tokens) {
    if (!json || !tokens || max_tokens <= 0) return -1;
    
    // Allocate parent stack for nesting tracking
    int ps_cap = 256;
    int *parent_stack = (int*)malloc(ps_cap * sizeof(int));
    if (!parent_stack) return -1;
    
    // Initialize cursor
    sjson_cursor_t cursor = sjson_cursor_init(json, strlen(json));
    
    // Initialize parse state
    tokenize_state_t state = {
        .c = &cursor,
        .tokens = tokens,
        .max_tokens = max_tokens,
        .nt = 0,
        .parent = -1,
        .parent_stack = parent_stack,
        .ps_cap = ps_cap,
        .ps_top = 0
    };
    
    // Main parsing loop: peek-only dispatcher
    while (state.c->pos < state.c->len) {
        char ch = sjson_cursor_peek(state.c);
        
        if (ch == '\0') break;  // EOF
        
        // Dispatch based on next character
        if (ch == '{') {
            int rc = tokenize_emit_container(&state, SJSON_OBJ);
            if (rc < 0) { free(parent_stack); return rc; }
        }
        else if (ch == '[') {
            int rc = tokenize_emit_container(&state, SJSON_ARR);
            if (rc < 0) { free(parent_stack); return rc; }
        }
        else if (ch == '}' || ch == ']') {
            int rc = tokenize_close_container(&state);
            if (rc < 0) { free(parent_stack); return rc; }
        }
        else if (ch == '"') {
            int rc = tokenize_string_token(&state);
            if (rc < 0) { free(parent_stack); return rc; }
        }
        else if (ch == '-' || (ch >= '0' && ch <= '9') || 
                 ch == 't' || ch == 'f' || ch == 'n') {
            int rc = tokenize_primitive_token(&state);
            if (rc < 0) { free(parent_stack); return rc; }
        }
        else if (ch == ',' || ch == ':') {
            // Skip structural separators
            state.c->pos++;
        }
        else {
            // Unexpected character
            free(parent_stack);
            return -2;
        }
    }
    
    // Validate all containers were closed
    if (state.ps_top != 0) {
        free(parent_stack);
        return -2;
    }
    
    free(parent_stack);
    return state.nt;
}

int sjson_tok_eq(const char *json, const sjson_token_t *t, const char *s) {
    int len = t->end - t->start;
    if ((int)strlen(s) != len) return 0;
    return (strncmp(json + t->start, s, len) == 0) ? 1 : 0;
}

int sjson_find_key(const char *json, const sjson_token_t *tokens, int ntokens, int obj_index, const char *key) {
    if (!json || !tokens || obj_index < 0 || obj_index >= ntokens) return -1;
    if (tokens[obj_index].type != SJSON_OBJ) return -1;

    /* Iterate tokens following the object token and look for top-level children whose parent == obj_index.
     * For each string key token, the value token should immediately follow. This approach is robust to
     * nested objects/arrays which also contribute to tokens[obj_index].size. */
    int idx = obj_index + 1;
    while (idx < ntokens) {
        const sjson_token_t *k = &tokens[idx];
        /* if this token is not a direct child of the object, skip it and continue scanning */
        if (k->parent != obj_index) { idx++; continue; }

        if (k->type == SJSON_STR) {
            const sjson_token_t *v = (idx + 1 < ntokens) ? &tokens[idx + 1] : NULL;
            if (sjson_tok_eq(json, k, key)) {
                return v ? (int)(v - tokens) : -1;
            }
            /* skip key and its value token */
            idx += 2;
        } else {
            /* skip unexpected token types */
            idx++;
        }
    }

    return -1;
}

int sjson_token_to_double(const char *json, const sjson_token_t *t, double *out) {
    if (!json || !t || !out) return -1;
    if (t->type != SJSON_PRIM) return -1;
    char buf[64];
    int len = t->end - t->start;
    if (len >= (int)sizeof(buf)) return -1;
    memcpy(buf, json + t->start, (size_t)len);
    buf[len] = '\0';
    char *endptr = NULL;
    double v = strtod(buf, &endptr);
    if (endptr == buf) return -1;
    *out = v;
    return 0;
}

int sjson_token_to_int64(const char *json, const sjson_token_t *t, int64_t *out) {
    if (!json || !t || !out) return -1;
    if (t->type != SJSON_PRIM) return -1;
    char buf[64];
    int len = t->end - t->start;
    if (len >= (int)sizeof(buf)) return -1;
    memcpy(buf, json + t->start, (size_t)len);
    buf[len] = '\0';
    char *endptr = NULL;
    int64_t v = strtoll(buf, &endptr, 10);
    if (endptr == buf) return -1;
    *out = v;
    return 0;
}

int sjson_token_to_str(const char *json, const sjson_token_t *t, char *dst, int dst_len) {
    if (!json || !t || !dst || dst_len <= 0) return -1;
    if (t->type != SJSON_STR) return -1;
    int len = t->end - t->start;
    if (len >= dst_len) return -1;
    /* unescape simple escapes (\" \\ \/ \b \f \n \r \t) */
    const char *src = json + t->start;
    int di = 0;
    for (int i = 0; i < len; i++) {
        char c = src[i];
        if (c == '\\' && i + 1 < len) {
            i++;
            char esc = src[i];
            switch (esc) {
                case '"': dst[di++] = '"'; break;
                case '\\': dst[di++] = '\\'; break;
                case '/': dst[di++] = '/'; break;
                case 'b': dst[di++] = '\b'; break;
                case 'f': dst[di++] = '\f'; break;
                case 'n': dst[di++] = '\n'; break;
                case 'r': dst[di++] = '\r'; break;
                case 't': dst[di++] = '\t'; break;
                default: dst[di++] = esc; break;
            }
        } else {
            dst[di++] = c;
        }
    }
    dst[di] = '\0';
    return 0;
}

int sjson_iterate_str_array(const char *json, const sjson_token_t *tokens, int ntokens, int arr_idx, int (*cb)(const char*, int, void*), void *user) {
    if (!json || !tokens || arr_idx < 0 || arr_idx >= ntokens || !cb) return -1;
    const sjson_token_t *arr = &tokens[arr_idx];
    if (arr->type != SJSON_ARR) return -1;
    /* iterate next `size` tokens */
    int idx = arr_idx + 1;
    int count = 0;
    for (int i = 0; i < arr->size && idx < ntokens; i++) {
        const sjson_token_t *e = &tokens[idx++];
        if (e->type != SJSON_STR) continue;
        int len = e->end - e->start;
        int rc = cb(json + e->start, len, user);
        if (rc != 0) return rc;
        count++;
    }
    return count;
}

int sjson_iterate_num_array(const char *json, const sjson_token_t *tokens, int ntokens, int arr_idx, int (*cb)(uint32_t, void*), void *user) {
    if (!json || !tokens || arr_idx < 0 || arr_idx >= ntokens || !cb) return -1;
    const sjson_token_t *arr = &tokens[arr_idx];
    if (arr->type != SJSON_ARR) return -1;
    /* iterate next `size` tokens (primitives only) */
    int idx = arr_idx + 1;
    int count = 0;
    for (int i = 0; i < arr->size && idx < ntokens; i++) {
        const sjson_token_t *e = &tokens[idx++];
        if (e->type != SJSON_PRIM) continue;
        /* parse as uint32_t */
        char buf[64];
        int len = e->end - e->start;
        if (len >= (int)sizeof(buf)) continue;
        memcpy(buf, json + e->start, (size_t)len);
        buf[len] = '\0';
        char *endptr = NULL;
        uint32_t v = (uint32_t)strtoul(buf, &endptr, 10);
        if (endptr == buf) continue;
        int rc = cb(v, user);
        if (rc != 0) return rc;
        count++;
    }
    return count;
}
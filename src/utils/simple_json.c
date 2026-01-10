/*
 * Minimal jsmn-like tokenizer and helpers
 */

#include "../include/simple_json.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/* Internal helper: return 1 if character is a JSON structural character */
static int is_struct_char(char c) {
    return (c == '{' || c == '}' || c == '[' || c == ']' || c == ':' || c == ',');
}

int sjson_tokenize(const char *json, sjson_token_t *tokens, int max_tokens) {
    if (!json || !tokens || max_tokens <= 0) return -1;
    int nt = 0;
    int pos = 0;
    int len = (int)strlen(json);
    int parent = -1;
    /* Parent stack to track nesting even if we stop emitting tokens due to max_tokens */
    int ps_cap = 256;
    int ps_top = 0;
    int *parent_stack = (int*)malloc(sizeof(int) * ps_cap);
    if (!parent_stack) return -1;

    while (pos < len) {
        char c = json[pos];
        if (isspace((unsigned char)c)) { pos++; continue; }
        if (c == '{' || c == '[') {
            /* push new nesting level; if we still have room emit a token, otherwise mark truncated */
            if (ps_top + 1 >= ps_cap) {
                int new_cap = ps_cap * 2;
                int *new_stack = (int*)realloc(parent_stack, sizeof(int) * new_cap);
                if (!new_stack) { free(parent_stack); return -1; }
                parent_stack = new_stack;
                ps_cap = new_cap;
            }

            if (nt < max_tokens) {
                tokens[nt].type = (c == '{') ? SJSON_OBJ : SJSON_ARR;
                tokens[nt].start = pos;
                tokens[nt].end = -1;
                tokens[nt].size = 0;
                tokens[nt].parent = parent;
                if (parent >= 0 && parent < max_tokens) tokens[parent].size++;
                parent = nt;
                parent_stack[ps_top++] = parent;
                nt++;
            } else {
                /* truncated level marker */
                parent_stack[ps_top++] = -1;
                parent = -1;
            }
            pos++;
            continue;
        }
        if (c == '}' || c == ']') {
            /* close current parent */
            if (ps_top <= 0) { free(parent_stack); return -2; }
            int popped = parent_stack[--ps_top];
            if (popped >= 0 && popped < max_tokens) {
                tokens[popped].end = pos + 1;
            }
            parent = (ps_top > 0) ? parent_stack[ps_top - 1] : -1;
            pos++;
            continue;
        }
        if (c == '"') {
            /* string token */
            int start = pos + 1;
            pos = start;
            while (pos < len) {
                if (json[pos] == '\\' && pos + 1 < len) { pos += 2; continue; }
                if (json[pos] == '"') break;
                pos++;
            }
            if (pos >= len) { free(parent_stack); return -2; } /* unterminated string */
            if (nt < max_tokens) {
                tokens[nt].type = SJSON_STR;
                tokens[nt].start = start;
                tokens[nt].end = pos; /* exclusive */
                tokens[nt].size = 0;
                tokens[nt].parent = parent;
                if (parent >= 0 && parent < max_tokens) tokens[parent].size++;
                nt++;
            }
            pos++; /* skip closing quote */
            continue;
        }
        /* primitives: number, true, false, null */
        if (c == '-' || (c >= '0' && c <= '9') || c == 't' || c == 'f' || c == 'n') {
            int start = pos;
            /* consume until comma or structural char */
            while (pos < len && !is_struct_char(json[pos]) && !isspace((unsigned char)json[pos])) pos++;
            if (nt < max_tokens) {
                tokens[nt].type = SJSON_PRIM;
                tokens[nt].start = start;
                tokens[nt].end = pos;
                tokens[nt].size = 0;
                tokens[nt].parent = parent;
                if (parent >= 0 && parent < max_tokens) tokens[parent].size++;
                nt++;
            }
            continue;
        }
        /* skip commas/colons if encountered at top-level between tokens */
        if (c == ',' || c == ':') { pos++; continue; }

        /* unexpected char */
        return -2;
    }

    if (ps_top != 0) { free(parent_stack); return -2; } /* unclosed object/array */
    free(parent_stack);
    return nt;
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

/* Simple file reader helper */
int json_read_file(const char *path, char **out_buf, size_t *out_len) {
    if (!path || !out_buf) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long l = ftell(f);
    if (l < 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    char *buf = (char*)malloc((size_t)l + 1);
    if (!buf) { fclose(f); return -1; }
    size_t got = fread(buf, 1, (size_t)l, f);
    fclose(f);
    if (got != (size_t)l) { free(buf); return -1; }
    buf[l] = '\0';
    *out_buf = buf;
    if (out_len) *out_len = (size_t)l;
    return 0;
}
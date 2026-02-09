#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <errno.h>
#include "../include/simple_json.h"
#include "../include/model_spec.h"
#include "../include/file_reader.h"
#include "log.h"

/* Simple DJB2 hash for strings */
static unsigned long hash_str(const char *s) {
    unsigned long h = 5381;
    unsigned char c;
    while ((c = (unsigned char)*s++) != '\0') {
        h = ((h << 5) + h) + c;
    }
    return h;
}

static int next_pow2(int v) {
    int n = 1;
    while (n < v) n <<= 1;
    return n;
}

/* Insert token id keyed by string into tokenizer hash table */
static int tok_hash_insert(sapphire_tokenizer_t *tok, const char *s, int id) {
    if (!tok || !tok->hash_table || !s) return -1;
    unsigned long h = hash_str(s);
    int mask = tok->hash_capacity - 1;
    int idx = (int)(h & mask);
    for (int probe = 0; probe < tok->hash_capacity; probe++) {
        int cur = tok->hash_table[idx];
        if (cur == -1) {
            tok->hash_table[idx] = id;
            return 0;
        }
        /* if same string, replace */
        if (tok->vocab[cur].str && strcmp(tok->vocab[cur].str, s) == 0) {
            tok->hash_table[idx] = id;
            return 0;
        }
        idx = (idx + 1) & mask;
    }
    return -1; /* table full */
}

/* Lookup token id by string; returns -1 if not found */
static int tok_hash_lookup(sapphire_tokenizer_t *tok, const char *s) {
    if (!tok || !tok->hash_table || !s) return -1;
    unsigned long h = hash_str(s);
    int mask = tok->hash_capacity - 1;
    int idx = (int)(h & mask);
    for (int probe = 0; probe < tok->hash_capacity; probe++) {
        int cur = tok->hash_table[idx];
        if (cur == -1) return -1;
        if (tok->vocab[cur].str && strcmp(tok->vocab[cur].str, s) == 0) return tok->vocab[cur].id;
        idx = (idx + 1) & mask;
    }
    return -1;
}

/**
 * Metadata from vocab JSON inspection phase
 * Used to pass information between inspect, allocate, and populate phases
 */
typedef struct {
    int vocab_idx_token;
    int ntokens;
    const sjson_token_t *tokens;
    const char *json_data;
} vocab_inspection_t;

/**
 * Safely construct and validate tokenizer path using centralized file_reader utilities
 * Returns allocated path on success, NULL on failure
 */
static char* construct_tokenizer_path(const char *model_dir) {
    if (!model_dir) {
        LOG_ERROR("model_dir is NULL");
        return NULL;
    }

    // construct_safe_path allocates and returns the path, or NULL on error
    return construct_safe_path(model_dir, "tokenizer.json", NULL);
}

/**
 * Helper: Inspect JSON to find and validate vocab object
 * Returns metadata struct containing token indices and pointers, or zeroed struct on error
 */
static vocab_inspection_t inspect_vocab_from_json(const char *json_data, const sjson_token_t *tokens, 
                                                   int ntokens) {
    vocab_inspection_t result = {0};
    
    // Find "model" key at root (token 0 is the root object)
    int model_val_idx = sjson_find_key(json_data, tokens, ntokens, 0, "model");
    if (model_val_idx < 0) {
        LOG_ERROR("No \"model\" found in tokenizer.json (expected model.vocab structure)");
        return result;
    }

    int vocab_idx_token = sjson_find_key(json_data, tokens, ntokens, model_val_idx, "vocab");
    if (vocab_idx_token < 0) {
        LOG_ERROR("No \"vocab\" found inside model in tokenizer.json");
        return result;
    }

    if (tokens[vocab_idx_token].type != SJSON_OBJ) {
        LOG_ERROR("vocab is not a JSON object");
        return result;
    }
    
    // Validation passed, populate result
    result.vocab_idx_token = vocab_idx_token;
    result.ntokens = ntokens;
    result.tokens = tokens;
    result.json_data = json_data;
    return result;
}

/**
 * Helper: Scan vocab object and allocate appropriately-sized vocabulary array
 * Performs Pass 1: count children, find max token ID, compute capacity
 * Returns zero-initialized vocab array ready for population, or NULL on error
 * Sets *out_vocab_size to the allocated capacity
 */
static token_t* allocate_vocab_from_json(const vocab_inspection_t *inspect, int *out_vocab_size) {
    // Count direct children (string keys = number of pairs)
    int child_count = 0;
    int probe_idx = inspect->vocab_idx_token + 1;
    while (probe_idx < inspect->ntokens && inspect->tokens[probe_idx].parent == inspect->vocab_idx_token) {
        if (inspect->tokens[probe_idx].type == SJSON_STR) child_count++;
        probe_idx++;
    }
    
    // Scan for maximum token ID
    int scan_idx = inspect->vocab_idx_token + 1;
    int max_token_id = -1;
    for (int i = 0; i < child_count && scan_idx + 1 < inspect->ntokens; i++) {
        scan_idx++;  // skip key
        const sjson_token_t *v = &inspect->tokens[scan_idx++];
        if (v->type != SJSON_PRIM) continue;
        
        int vlen = v->end - v->start;
        if (vlen <= 0 || vlen >= 64) continue;
        
        char tmp[64];
        memcpy(tmp, inspect->json_data + v->start, vlen);
        tmp[vlen] = '\0';
        int token_id = (int)strtol(tmp, NULL, 10);
        if (token_id > max_token_id) max_token_id = token_id;
    }
    
    // Calculate required capacity
    int capacity = 0;
    if (max_token_id >= 0) capacity = max_token_id + 1;
    if (capacity < child_count) capacity = child_count;
    if (capacity <= 0) capacity = 1;
    
    LOG_DEBUG("tokenizer.json: vocab_pairs=%d max_token_id=%d capacity=%d", child_count, max_token_id, capacity);
    
    // Allocate and zero-initialize
    token_t *vocab = calloc((size_t)capacity, sizeof(token_t));
    if (!vocab) {
        LOG_ERROR("Cannot allocate vocabulary array (capacity=%d)", capacity);
        return NULL;
    }
    
    if (out_vocab_size) *out_vocab_size = capacity;
    return vocab;
}

/**
 * Helper: Populate vocabulary array from JSON metadata
 * Performs Pass 2: extract token strings and IDs from tokens array
 * Returns: number of successfully parsed entries
 */
static int populate_vocab_from_json(sapphire_tokenizer_t *tok, const vocab_inspection_t *inspect) {
    if (!tok || !tok->vocab || !inspect) return 0;
    
    int idx = inspect->vocab_idx_token + 1;
    int parsed = 0;
    int skipped_out_of_range = 0;
    int skipped_malformed = 0;
    
    // Recount children for loop bounds
    int child_count = 0;
    int probe_idx = inspect->vocab_idx_token + 1;
    while (probe_idx < inspect->ntokens && inspect->tokens[probe_idx].parent == inspect->vocab_idx_token) {
        if (inspect->tokens[probe_idx].type == SJSON_STR) child_count++;
        probe_idx++;
    }
    
    printf("DEBUG: Parsing vocabulary entries from nested model.vocab...\n");
    for (int i = 0; i < child_count && idx + 1 < inspect->ntokens; i++) {
        const sjson_token_t *k = &inspect->tokens[idx++];
        const sjson_token_t *v = &inspect->tokens[idx++];

        if (k->type != SJSON_STR) continue;

        int key_len = k->end - k->start;
        char *token_text = malloc(key_len + 1);
        if (!token_text) continue;
        if (sjson_token_to_str(inspect->json_data, k, token_text, key_len + 1) != 0) {
            free(token_text);
            continue;
        }

        // Parse token id (primitive)
        int token_id = -1;
        int vlen = v->end - v->start;
        if (vlen > 0 && vlen < 64) {
            char tmp[64];
            memcpy(tmp, inspect->json_data + v->start, vlen);
            tmp[vlen] = '\0';
            token_id = (int)strtol(tmp, NULL, 10);
        } else {
            skipped_malformed++;
            free(token_text);
            continue;
        }

        if (token_id >= 0 && token_id < tok->vocab_size) {
            tok->vocab[token_id].id = token_id;
            tok->vocab[token_id].str = token_text;
            tok->vocab[token_id].len = key_len;
            parsed++;
        } else {
            skipped_out_of_range++;
            LOG_DEBUG("Skipping token id %d out-of-range (capacity=%d): %.*s", token_id, tok->vocab_size, key_len, token_text);
            free(token_text);
        }

        if ((parsed & 0xFFFF) == 0) {
            if (parsed % 50000 == 0) {
                printf("DEBUG: Parsed %d entries...\n", parsed);
                fflush(stdout);
            }
        }
    }

    printf("✓ Successfully parsed %d vocabulary entries (capacity=%d)\n", parsed, tok->vocab_size);
    LOG_DEBUG("Tokenizer parse summary: parsed=%d skipped_out_of_range=%d skipped_malformed=%d", parsed, skipped_out_of_range, skipped_malformed);
    return parsed;
}

/**
 * Helper: Build tokenizer hash table for fast token lookup
 * Sizes table to power-of-two >= 2 * parsed_count
 */
static void build_tokenizer_hash_table(sapphire_tokenizer_t *tok, int parsed) {
    if (!tok || parsed <= 0) return;
    
    int ht_size = next_pow2(parsed * 2 + 1);
    tok->hash_capacity = ht_size;
    tok->hash_table = malloc(sizeof(int) * (size_t)ht_size);
    if (tok->hash_table) {
        for (int i = 0; i < ht_size; i++) tok->hash_table[i] = -1;
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i].str) {
                if (tok_hash_insert(tok, tok->vocab[i].str, i) != 0) {
                    LOG_DEBUG("Hash insert failed for token id %d (%s)", i, tok->vocab[i].str);
                }
            }
        }
        LOG_DEBUG("Built tokenizer hash table: capacity=%d entries_indexed=%d", ht_size, parsed);
    } else {
        LOG_DEBUG("Failed to allocate tokenizer hash table (size=%d)", ht_size);
    }
}

/**
 * Helper: Load and parse the main vocabulary from tokenizer.json data
 * Consolidates tokenization, inspection, allocation, and population logic
 * Returns 0 on success, -1 on failure
 */
static int load_main_vocabulary(sapphire_tokenizer_t *tok, const char *json_data, size_t json_len) {
    if (!tok || !json_data) return -1;

    // Determine max tokens for JSON buffer
    size_t max_tokens = (json_len / 8) + 1024;
    if (max_tokens > 1500000) max_tokens = 1500000;
    
    sjson_token_t *tokens = malloc(sizeof(sjson_token_t) * max_tokens);
    if (!tokens) {
        LOG_ERROR("Cannot allocate JSON token buffer");
        return -1;
    }

    int ntokens = sjson_tokenize(json_data, tokens, (int)max_tokens);
    if (ntokens <= 0) {
        LOG_ERROR("Failed to tokenize tokenizer.json");
        free(tokens);
        return -1;
    }

    // Inspect vocab object structure
    vocab_inspection_t inspect = inspect_vocab_from_json(json_data, tokens, ntokens);
    if (inspect.vocab_idx_token == 0 && inspect.ntokens == 0) {
        // inspect_vocab_from_json logs the specific error
        free(tokens);
        return -1;
    }

    // Allocate vocab array with appropriate capacity
    tok->vocab = allocate_vocab_from_json(&inspect, &tok->vocab_size);
    if (!tok->vocab) {
        free(tokens);
        return -1;
    }
    LOG_DEBUG("Allocated tok->vocab size=%d", tok->vocab_size);

    // Populate vocabulary entries and build hash table
    int parsed = populate_vocab_from_json(tok, &inspect);
    build_tokenizer_hash_table(tok, parsed);

    free(tokens);
    return 0;
}

/**
 * Helper: Load and parse tokenizer_config.json for special tokens
 * Uses streaming sjson_cursor_t API for efficiency and safety
 */
static void load_tokenizer_config(sapphire_tokenizer_t *tok, const char *model_dir) {
    if (!tok || !model_dir) return;

    char *config_path = construct_safe_path(model_dir, "tokenizer_config.json", NULL);
    if (!config_path) {
        LOG_ERROR("Failed to construct tokenizer_config.json path");
        return;
    }

    size_t config_size = 0;
    char *config_data = NULL;
    if (file_read_json(config_path, &config_data, &config_size) != 0) {
        // Not a fatal error, but continue without custom config
        free(config_path);
        
        // Ensure defaults are set if config is missing
        tok->bos_token_id = 2;
        tok->eos_token_id = 1;
        tok->pad_token_id = 0;
        tok->unk_token_id = 3;
        tok->add_bos_token = 1; // CRITICAL FIX: Force BOS token for Gemma 3
        return;
    }

    // Default special token IDs (Gemma style)
    tok->bos_token_id = 2;
    tok->eos_token_id = 1;
    tok->pad_token_id = 0;
    tok->unk_token_id = 3;
    tok->add_bos_token = 1; // CRITICAL FIX: Force BOS token for Gemma 3

    // Parse config with Cursor API
    sjson_cursor_t c = sjson_cursor_init(config_data, config_size);
    if (sjson_cursor_consume(&c, '{')) {
        while (sjson_cursor_peek(&c) != '}' && sjson_cursor_peek(&c) != '\0') {
            char key[128];
            if (sjson_cursor_parse_string(&c, key, sizeof(key)) != 0) break;
            if (!sjson_cursor_consume(&c, ':')) break;

            if (strcmp(key, "bos_token_id") == 0) {
                uint64_t val;
                if (sjson_cursor_parse_u64(&c, &val) == 0) tok->bos_token_id = (int)val;
            } else if (strcmp(key, "eos_token_id") == 0) {
                uint64_t val;
                if (sjson_cursor_parse_u64(&c, &val) == 0) tok->eos_token_id = (int)val;
            } else if (strcmp(key, "pad_token_id") == 0) {
                uint64_t val;
                if (sjson_cursor_parse_u64(&c, &val) == 0) tok->pad_token_id = (int)val;
            } else if (strcmp(key, "add_bos_token") == 0) {
                // Peek at the next char for booleans (true/false)
                char peek = sjson_cursor_peek(&c);
                if (peek == 't') { // true
                    tok->add_bos_token = 1;
                } else if (peek == 'f') { // false
                    tok->add_bos_token = 0;
                }
                sjson_cursor_skip_value(&c); 
            } else if (strcmp(key, "add_eos_token") == 0) {
                char peek = sjson_cursor_peek(&c);
                if (peek == 't') {
                    tok->add_eos_token = 1;
                    sjson_cursor_skip_value(&c);
                } else if (peek == 'f') {
                    tok->add_eos_token = 0;
                    sjson_cursor_skip_value(&c);
                } else {
                    sjson_cursor_skip_value(&c);
                }
            } else {
                sjson_cursor_skip_value(&c);
            }
            sjson_cursor_consume(&c, ','); // consume separator if exists
        }
    }

    free(config_data);
    free(config_path);
    
    printf("Tokenizer special tokens: bos=%d, eos=%d, pad=%d, unk=%d, add_bos=%d, add_eos=%d\n",
           tok->bos_token_id, tok->eos_token_id, tok->pad_token_id, tok->unk_token_id,
           tok->add_bos_token, tok->add_eos_token);
}

/**
 * Load tokenizer from JSON files with streaming parser
 * Handles both root-level and nested model.vocab structures (Gemma 3 support)
 */
sapphire_tokenizer_t* tokenizer_load(const char *model_dir) {
    if (!model_dir) {
        LOG_ERROR("model_dir is NULL");
        return NULL;
    }

    sapphire_tokenizer_t *tok = NULL;
    char *tokenizer_path = NULL;
    char *json_data = NULL;
    size_t json_len = 0;

    // 1. Setup path and load JSON buffer
    tokenizer_path = construct_tokenizer_path(model_dir);
    if (!tokenizer_path) {
        goto cleanup;
    }

    if (file_read_json(tokenizer_path, &json_data, &json_len) != 0) {
        goto cleanup; // Error already logged by file_read_json()
    }
    LOG_DEBUG("Loading tokenizer.json (%zu bytes)...", json_len);

    // 2. Allocate tokenizer structure
    tok = calloc(1, sizeof(sapphire_tokenizer_t));
    if (!tok) {
        LOG_ERROR("Failed to allocate sapphire_tokenizer_t");
        goto cleanup;
    }

    // 3. Load Main Vocabulary
    if (load_main_vocabulary(tok, json_data, json_len) != 0) {
        goto cleanup;
    }

    // 4. Load Special Tokens Configuration
    load_tokenizer_config(tok, model_dir);

    // Success
    printf("Tokenizer loaded: vocab_size=%d\n", tok->vocab_size);
    goto final;

cleanup:
    if (tok) {
        tokenizer_free(tok);
        tok = NULL;
    }

final:
    if (tokenizer_path) free(tokenizer_path);
    if (json_data) free(json_data);
    return tok;
}

/**
 * Free tokenizer resources
 */
void tokenizer_free(sapphire_tokenizer_t *tok) {
    if (!tok) return;
    
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i].str) {
                free(tok->vocab[i].str);
            }
        }
        free(tok->vocab);
    }
    if (tok->hash_table) {
        free(tok->hash_table);
        tok->hash_table = NULL;
        tok->hash_capacity = 0;
    }
    
    if (tok->merges) {
        for (int i = 0; i < tok->num_merges; i++) {
            if (tok->merges[i].left) free(tok->merges[i].left);
            if (tok->merges[i].right) free(tok->merges[i].right);
        }
        free(tok->merges);
    }
    
    free(tok);
}

/**
 * Greedy BPE encoding with vocab lookup
 * 
 * Algorithm:
 * 1. Look up each word/character in vocabulary
 * 2. For subword encoding, use longest-match-first strategy
 * 3. If not found, return fallback (UNK or break into characters)
 */
int tokenize(sapphire_tokenizer_t *tok, const char *text, 
                      int *tokens, int max_tokens) {
    if (!tok || !text || !tokens || max_tokens <= 0) return -1;
    
    int token_count = 0;
    
    // Add BOS if configured
    if (tok->add_bos_token) {
        tokens[token_count++] = tok->bos_token_id;
    }
    
    int text_len = strlen(text);
    int i = 0;
    
    while (i < text_len && token_count < max_tokens) {
        int best_match_len = 0;
        int best_token_id = tok->unk_token_id;
        
        // 1. Check for SPECIAL TOKENS (critical for prompt structure)
        if (strncmp(&text[i], "<start_of_turn>", 15) == 0) {
            int sid = tok_hash_lookup(tok, "<start_of_turn>");
            if (sid >= 0) {
                tokens[token_count++] = sid;
                i += 15;
                goto next_token;
            }
        }
        if (strncmp(&text[i], "<end_of_turn>", 13) == 0) {
            int eid = tok_hash_lookup(tok, "<end_of_turn>");
            if (eid >= 0) {
                tokens[token_count++] = eid;
                i += 13;
                goto next_token;
            }
        }
        
        // 2. Greedy longest matching with SPIECE normalization
        for (int substr_len = 1; i + substr_len <= text_len && substr_len <= 100; substr_len++) {
            char raw_substr[256];
            strncpy(raw_substr, &text[i], substr_len);
            raw_substr[substr_len] = '\0';
            
            // Build SPIECE version: replace Space with \u2581 or Prepend \u2581
            char spiece_substr[256];
            spiece_substr[0] = '\0';
            
            if (raw_substr[0] == ' ') {
                 strcpy(spiece_substr, "\xe2\x96\x81"); // UTF-8 for U+2581
                 strcat(spiece_substr, raw_substr + 1);
            } else if (i > 0 && (text[i-1] == ' ')) {
                 // Start of sentence or after whitespace -> Prepend U+2581
                 // GEMMA 3 FIX: Do NOT treat '\n' as a trigger for prepending space.
                 // This fixes the issue where "\nThe" became "\n The" (ID 669) instead of "\nThe" (ID 818).
                 strcpy(spiece_substr, "\xe2\x96\x81");
                 strcat(spiece_substr, raw_substr);
            }

            // Reverse Lookup via hash table
            int hid = tok_hash_lookup(tok, raw_substr);
            if (hid >= 0) {
                best_match_len = substr_len;
                best_token_id = hid;
            } else if (spiece_substr[0] != '\0') {
                int sid = tok_hash_lookup(tok, spiece_substr);
                if (sid >= 0) {
                    best_match_len = substr_len;
                    best_token_id = sid;
                }
            }
        }
        
        if (best_match_len > 0) {
            tokens[token_count++] = best_token_id;
            i += best_match_len;
        } else {
            // Fallback: Skip space if not matched, or look for byte token
            if (text[i] == ' ') {
                i++;
                continue;
            }
            // Try single char match via hash
            const char single[2] = {text[i], '\0'};
            int sid = tok_hash_lookup(tok, single);
            if (sid >= 0) tokens[token_count++] = sid;
            else tokens[token_count++] = tok->unk_token_id;
            i++;
        }
        
        next_token:;
    }
    
    // Add EOS if configured
    if (tok->add_eos_token && token_count < max_tokens) {
        tokens[token_count++] = tok->eos_token_id;
    }
    
    return token_count;
}

int tokenize_fallback(const char *text, int *tokens, int max_tokens) {
    if (!text || !tokens) return -1;
    
    int len = strlen(text);
    int token_count = 0;
    
    // Add BOS token (Gemma uses 2)
    if (token_count < max_tokens) {
        tokens[token_count++] = 2;
    }
    
    // Map printable ASCII (32-126) to tokens [256, 382]
    for (int i = 0; i < len && token_count < max_tokens; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 32 && c <= 126) {
            tokens[token_count++] = 256 + (c - 32);
        }
    }
    
    return token_count;
}

/**
 * Decode single token ID to string
 * HYBRID OPTIMIZATION: Direct indexing for O(1) lookup (instant decode)
 */
const char* decode(sapphire_tokenizer_t *tok, int token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }
    
    // Direct access using token_id as index - no more slow linear search!
    return tok->vocab[token_id].str ? tok->vocab[token_id].str : "";
}

/**
 * Decode array of token IDs to text
 * Handles Gemma 3 SPIECE prefix (0xE2 0x96 0x81) by converting to spaces
 */
int detokenize(sapphire_tokenizer_t *tok, const int *tokens,
                        int num_tokens, char *output, int output_size) {
    if (!tok || !tokens || !output || output_size <= 0) return 0;
    
    int out_pos = 0;
    
    for (int i = 0; i < num_tokens && out_pos < output_size - 1; i++) {
        const char *token_str = decode(tok, tokens[i]);
        if (!token_str || !*token_str) continue;
        
        // Check for SPIECE prefix: 0xE2 0x96 0x81 (UTF-8 for U+2581)
        // If found, replace with space and skip the prefix bytes
        int src_idx = 0;
        if ((unsigned char)token_str[0] == 0xE2 && (unsigned char)token_str[1] == 0x96 && (unsigned char)token_str[2] == 0x81) {
            // Add space for SPIECE prefix
            output[out_pos++] = ' ';
            src_idx = 3;  // Skip the 3-byte SPIECE prefix
        }
        
        // Copy remaining bytes of token
        while (token_str[src_idx] != '\0' && out_pos < output_size - 1) {
            output[out_pos++] = token_str[src_idx++];
        }
    }
    
    output[out_pos] = '\0';
    return out_pos;
}

/**
 * Get vocabulary size
 */
int tokenizer_vocab_size(const sapphire_tokenizer_t *tok) {
    if (!tok) return 0;
    return tok->vocab_size;
}

/**
 * @brief Construct Gemma 3 IT instruction-tuned prompt with hardcoded token sequence
 *
 * This mirrors the previous helper in inference.c but belongs to the tokenizer
 * module because it uses tokenizer primitives (tokenize, tokenize_fallback,
 * decode) and the tokenizer's special-token configuration.
 */
int build_gemma3_prompt_it(sapphire_tokenizer_t* tok, const char* user_prompt,
                           int* tokens, int max_tokens) {
    if (!tok || !user_prompt || !tokens || max_tokens < 20) {
        return -1;
    }

    int idx = 0;

    /* CRITICAL: Hardcoded Gemma 3 IT turn markers (matches HuggingFace template) */
    tokens[idx++] = 2;     // <bos>
    tokens[idx++] = 2;     // <bos> (duplicate, as per HF template)
    tokens[idx++] = 105;   // <start_of_turn>
    tokens[idx++] = 2364;  // "user"
    tokens[idx++] = 107;   // "\n"

    /* Tokenize user's actual prompt message */
    int prompt_tokens[512];
    int prompt_len = tokenize(tok, user_prompt, prompt_tokens, 512);

    if (prompt_len <= 0) {
        LOG_WARN("Failed to tokenize user prompt, using fallback");
        prompt_len = tokenize_fallback(user_prompt, prompt_tokens, 512);
    }

    /* Append user prompt tokens (skip BOS and PAD tokens from tokenizer) */
    int skip_bos = (prompt_len > 0 && prompt_tokens[0] == 2) ? 1 : 0;
    for (int i = skip_bos; i < prompt_len && idx < max_tokens - 7; i++) {
        if (prompt_tokens[i] != 0) {  /* Skip pad tokens (token ID 0) */
            tokens[idx++] = prompt_tokens[i];
        }
    }

    /* End user turn and start model turn (hardcoded token sequence) */
    tokens[idx++] = 106;   // <end_of_turn>
    tokens[idx++] = 107;   // "\n" (added to match HF template)
    tokens[idx++] = 105;   // <start_of_turn>
    tokens[idx++] = 4368;  // "model"
    tokens[idx++] = 107;   // "\n"

    return idx;  /* Return actual prompt length */
}

/**
 * @brief Construct Gemma 3 base (non-IT) prompt
 *
 * For base Gemma 3 models (non-instruction-tuned), we use a minimal format:
 * BOS + tokenized_prompt + EOS
 *
 * This does NOT include chat-specific markers like <start_of_turn>, <end_of_turn>,
 * or role tokens ("user", "model").
 */
int build_gemma3_prompt_base(sapphire_tokenizer_t* tok, const char* user_prompt,
                             int* tokens, int max_tokens) {
    if (!tok || !user_prompt || !tokens || max_tokens < 5) {
        return -1;
    }

    int idx = 0;

    /* Add BOS token */
    tokens[idx++] = 2;  // <bos>

    /* Tokenize user's prompt message */
    int prompt_tokens[512];
    int prompt_len = tokenize(tok, user_prompt, prompt_tokens, 512);

    if (prompt_len <= 0) {
        LOG_WARN("Failed to tokenize user prompt, using fallback");
        prompt_len = tokenize_fallback(user_prompt, prompt_tokens, 512);
    }

    /* Append user prompt tokens (skip BOS added by tokenizer if present) */
    int skip_bos = (prompt_len > 0 && prompt_tokens[0] == 2) ? 1 : 0;
    for (int i = skip_bos; i < prompt_len && idx < max_tokens - 2; i++) {
        if (prompt_tokens[i] != 0) {  /* Skip pad tokens (token ID 0) */
            tokens[idx++] = prompt_tokens[i];
        }
    }

    /* Add EOS token */
    if (idx < max_tokens) {
        tokens[idx++] = 1;  // <eos>
    }

    return idx;  /* Return actual prompt length */
}

/**
 * @brief Smart prompt builder: detects model variant and calls appropriate builder
 *
 * Examines the model_spec_t's model_id field to determine if the model is IT
 * (instruction-tuned) or base, then calls the appropriate builder function.
 *
 * @param spec Pointer to model_spec_t (must have model_id and tokenizer_handle set)
 * @param user_prompt Raw user prompt string
 * @param tokens Output buffer for token ids
 * @param max_tokens Capacity of `tokens` buffer
 * @return Number of tokens written, or -1 on error
 */
int build_gemma3_prompt(const model_spec_t* spec, const char* user_prompt,
                        int* tokens, int max_tokens) {
    if (!spec || !spec->model_id || !user_prompt || !tokens) {
        LOG_ERROR("build_gemma3_prompt() received NULL arguments");
        return -1;
    }

    sapphire_tokenizer_t* tok = (sapphire_tokenizer_t*)spec->tokenizer_handle;
    if (!tok) {
        LOG_ERROR("model_spec has no tokenizer loaded");
        return -1;
    }

    /* Detect IT variant by checking model_id */
    int is_it = (strstr(spec->model_id, "-it") != NULL);

    LOG_DEBUG("build_gemma3_prompt: model_id='%s' is_it=%d", spec->model_id, is_it);

    if (is_it) {
        return build_gemma3_prompt_it(tok, user_prompt, tokens, max_tokens);
    } else {
        return build_gemma3_prompt_base(tok, user_prompt, tokens, max_tokens);
    }
}


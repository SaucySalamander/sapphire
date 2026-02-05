#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../include/simple_json.h"
#include "../include/model_spec.h"
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
 * Load tokenizer from JSON files with streaming parser
 * Handles both root-level and nested model.vocab structures (Gemma 3 support)
 */
sapphire_tokenizer_t* tokenizer_load(const char *model_dir) {
    if (!model_dir) return NULL;
    
    sapphire_tokenizer_t *tok = malloc(sizeof(sapphire_tokenizer_t));
    if (!tok) return NULL;
    memset(tok, 0, sizeof(sapphire_tokenizer_t));
    
    // Read tokenizer.json
    char tokenizer_path[512];
    snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json", model_dir);
    
    FILE *f = fopen(tokenizer_path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", tokenizer_path);
        free(tok);
        return NULL;
    }
    
    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    LOG_DEBUG("Loading tokenizer.json (%ld bytes)...", file_size);
    
    char *json_data = malloc(file_size + 1);
    if (!json_data) {
        fprintf(stderr, "ERROR: Cannot allocate memory for tokenizer.json (%ld bytes)\n", file_size);
        fclose(f);
        free(tok);
        return NULL;
    }
    
    if (fread(json_data, 1, file_size, f) != (size_t)file_size) {
        fprintf(stderr, "ERROR: Cannot read tokenizer.json\n");
        free(json_data);
        fclose(f);
        free(tok);
        return NULL;
    }
    fclose(f);
    json_data[file_size] = '\0';
    
    // Do not allocate vocab yet. We'll inspect the JSON `vocab` object
    // to determine the required capacity (max token id) and allocate
    // exactly what we need to avoid out-of-bounds writes.
    
    // Use simple_json tokenizer to parse nested model.vocab
    size_t max_tokens = (size_t)(file_size / 8) + 1024;
    if (max_tokens > 1500000) max_tokens = 1500000;
    sjson_token_t *tokens = malloc(sizeof(sjson_token_t) * max_tokens);
    if (!tokens) {
        fprintf(stderr, "ERROR: cannot allocate JSON token buffer\n");
        free(json_data);
        free(tok->vocab);
        free(tok);
        return NULL;
    }

    int ntokens = sjson_tokenize(json_data, tokens, (int)max_tokens);
    if (ntokens <= 0) {
        LOG_ERROR("Failed to tokenize tokenizer.json");
        free(tokens);
        free(json_data);
        free(tok->vocab);
        free(tok);
        return NULL;
    }

    // Find "model" key at root (token 0 is the root object)
    int model_val_idx = sjson_find_key(json_data, tokens, ntokens, 0, "model");
    if (model_val_idx < 0) {
        fprintf(stderr, "ERROR: No \"model\" found in tokenizer.json (expected model.vocab structure)\n");
        free(tokens);
        free(json_data);
        free(tok->vocab);
        free(tok);
        return NULL;
    }

    int vocab_idx_token = sjson_find_key(json_data, tokens, ntokens, model_val_idx, "vocab");
    if (vocab_idx_token < 0) {
        fprintf(stderr, "ERROR: No \"vocab\" found inside model in tokenizer.json\n");
        free(tokens);
        free(json_data);
        free(tok->vocab);
        free(tok);
        return NULL;
    }

    if (tokens[vocab_idx_token].type != SJSON_OBJ) {
        fprintf(stderr, "ERROR: vocab is not a JSON object\n");
        free(tokens);
        free(json_data);
        free(tok);
        return NULL;
    }

    // Compute required capacity (two-pass): first compute max token id
    // and child count, then allocate tok->vocab accordingly.
    int raw_child_count = tokens[vocab_idx_token].size;
    /* sjson records each child token (keys and values) in tokens[].size, so for
     * an object of N pairs this value will be ~2*N. Compute actual pair count by
     * scanning direct children and counting string keys. */
    int child_count = 0;
    int probe_idx = vocab_idx_token + 1;
    while (probe_idx < ntokens && tokens[probe_idx].parent == vocab_idx_token) {
        if (tokens[probe_idx].type == SJSON_STR) child_count++;
        probe_idx++;
    }
    LOG_DEBUG("tokenizer.json: ntokens=%d raw_child_tokens=%d vocab_pairs=%d", ntokens, raw_child_count, child_count);
    int scan_idx = vocab_idx_token + 1;
    int max_token_id = -1;
    for (int i = 0; i < child_count && scan_idx + 1 < ntokens; i++) {
        /* scan only the value token here to compute max id */
        scan_idx++; // skip key
        const sjson_token_t *v = &tokens[scan_idx++];
        if (v->type != SJSON_PRIM) continue;
        int vlen = v->end - v->start;
        if (vlen <= 0 || vlen >= 64) continue;
        char tmp[64];
        memcpy(tmp, json_data + v->start, vlen);
        tmp[vlen] = '\0';
        int token_id = (int)strtol(tmp, NULL, 10);
        if (token_id > max_token_id) max_token_id = token_id;
    }

    LOG_DEBUG("tokenizer.json: max_token_id=%d", max_token_id);

    int capacity = 0;
    if (max_token_id >= 0) capacity = max_token_id + 1;
    if (capacity < child_count) capacity = child_count;
    if (capacity <= 0) capacity = 1;

    tok->vocab = calloc((size_t)capacity, sizeof(token_t));
    if (!tok->vocab) {
        fprintf(stderr, "ERROR: Cannot allocate vocabulary array (capacity=%d)\n", capacity);
        free(tokens);
        free(json_data);
        free(tok);
        return NULL;
    }
    // Set vocab_size to capacity so decode() bounds checks are valid.
    tok->vocab_size = capacity;
    LOG_DEBUG("Allocated tok->vocab capacity=%d (vocab_size=%d)", capacity, tok->vocab_size);

    // Second pass: populate vocabulary entries
    int idx = vocab_idx_token + 1;
    int parsed = 0;
    int skipped_out_of_range = 0;
    int skipped_malformed = 0;
    printf("DEBUG: Parsing vocabulary entries from nested model.vocab...\n");
    for (int i = 0; i < child_count && idx + 1 < ntokens; i++) {
        const sjson_token_t *k = &tokens[idx++];
        const sjson_token_t *v = &tokens[idx++];

        if (k->type != SJSON_STR) continue;

        int key_len = k->end - k->start;
        char *token_text = malloc(key_len + 1);
        if (!token_text) continue;
        if (sjson_token_to_str(json_data, k, token_text, key_len + 1) != 0) {
            free(token_text);
            continue;
        }

        // Parse token id (primitive)
        int token_id = -1;
        int vlen = v->end - v->start;
        if (vlen > 0 && vlen < 64) {
            char tmp[64];
            memcpy(tmp, json_data + v->start, vlen);
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

    /* Build fast lookup hash table sized to power-of-two >= 2 * parsed */
    int ht_size = next_pow2(parsed * 2 + 1);
    tok->hash_capacity = ht_size;
    tok->hash_table = malloc(sizeof(int) * ht_size);
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

    free(tokens);
    free(json_data);
    
    // Read tokenizer_config.json for special tokens
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/tokenizer_config.json", model_dir);
    
    f = fopen(config_path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long config_size = ftell(f);
        fseek(f, 0, SEEK_SET);
        
        char *config_data = malloc(config_size + 1);
        if (config_data) {
            if (fread(config_data, 1, config_size, f) == (size_t)config_size) {
                config_data[config_size] = '\0';
                
                // Default special token IDs (from Gemma)
                tok->bos_token_id = 2;     // <bos>
                tok->eos_token_id = 1;     // <eos>
                tok->pad_token_id = 0;     // <pad>
                tok->unk_token_id = 3;     // <unk>
                
                // CRITICAL FIX: Force BOS token for Gemma 3
                // The model expects to start with <bos> (ID 2).
                tok->add_bos_token = 1;

                // Try to find actual values in config
                const char *bos_str = strstr(config_data, "\"bos_token_id\"");
                if (bos_str) {
                    const char *colon = strchr(bos_str, ':');
                    if (colon) {
                        tok->bos_token_id = strtol(colon + 1, NULL, 10);
                    }
                }
                
                const char *eos_str = strstr(config_data, "\"eos_token_id\"");
                if (eos_str) {
                    const char *colon = strchr(eos_str, ':');
                    if (colon) {
                        tok->eos_token_id = strtol(colon + 1, NULL, 10);
                    }
                }
                
                const char *pad_str = strstr(config_data, "\"pad_token_id\"");
                if (pad_str) {
                    const char *colon = strchr(pad_str, ':');
                    if (colon) {
                        tok->pad_token_id = strtol(colon + 1, NULL, 10);
                    }
                }
                
                // Parse add_bos_token and add_eos_token
                tok->add_bos_token = (strstr(config_data, "\"add_bos_token\": true") != NULL);
                tok->add_eos_token = (strstr(config_data, "\"add_eos_token\": true") != NULL);
            }
            free(config_data);
        }
        fclose(f);
    }
    
    printf("Tokenizer loaded: vocab_size=%d\n", tok->vocab_size);
    printf("Special tokens: bos=%d, eos=%d, pad=%d, unk=%d\n",
           tok->bos_token_id, tok->eos_token_id, tok->pad_token_id, tok->unk_token_id);
    
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
        fprintf(stderr, "WARN: Failed to tokenize user prompt, using fallback\n");
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
        fprintf(stderr, "WARN: Failed to tokenize user prompt, using fallback\n");
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
        fprintf(stderr, "ERROR: build_gemma3_prompt() received NULL arguments\n");
        return -1;
    }

    sapphire_tokenizer_t* tok = (sapphire_tokenizer_t*)spec->tokenizer_handle;
    if (!tok) {
        fprintf(stderr, "ERROR: model_spec has no tokenizer loaded\n");
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


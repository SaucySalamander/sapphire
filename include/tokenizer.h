#ifndef SAPPHIRE_TOKENIZER_H
#define SAPPHIRE_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Token-to-string mapping
 */
typedef struct {
    int id;
    char *str;
    int len;
} token_t;

/**
 * @brief Merge pair for BPE encoding
 */
typedef struct {
    char *left;   // First token string
    char *right;  // Second token string
    int left_len;
    int right_len;
    int priority; // Lower priority = applied first
} merge_rule_t;

/**
 * @brief Sapphire Tokenizer (S-Tok)
 * 
 * Handles:
 * - Vocabulary lookup (token ID <-> string)
 * - BPE encoding (text -> token IDs)
 * - BPE decoding (token IDs -> text)
 * - Special tokens (BOS, EOS, PAD, UNK)
 */
typedef struct sapphire_tokenizer_t {
    // Vocabulary: maps token ID to string
    token_t *vocab;
    int vocab_size;
    
    // BPE merge rules
    merge_rule_t *merges;
    int num_merges;
    
    // Special token IDs
    int bos_token_id;   // Beginning of sequence
    int eos_token_id;   // End of sequence
    int pad_token_id;   // Padding
    int unk_token_id;   // Unknown token
    
    // Configuration
    bool add_bos_token;
    bool add_eos_token;
    // Fast lookup: open-addressing hash table mapping token string -> id
    int *hash_table;      // length == hash_capacity, stores token id or -1
    int hash_capacity;    // power-of-two capacity of hash_table
} sapphire_tokenizer_t;

/**
 * @brief Load tokenizer from tokenizer.json and tokenizer_config.json
 * 
 * @param model_dir Directory containing tokenizer.json and tokenizer_config.json
 * @return Allocated tokenizer, or NULL on failure
 * 
 * The caller must free with tokenizer_free()
 */
sapphire_tokenizer_t* tokenizer_load(const char *model_dir);

/**
 * @brief Free tokenizer resources
 * 
 * @param tok Tokenizer to free
 */
void tokenizer_free(sapphire_tokenizer_t *tok);

/**
 * @brief Encode text to token IDs (BPE encoding)
 * 
 * @param tok Tokenizer
 * @param text Input text
 * @param tokens Output token array (must have capacity for at least result length)
 * @param max_tokens Maximum number of tokens to generate
 * @return Number of tokens generated, or -1 on error
 * 
 * Uses greedy BPE merging algorithm:
 * 1. Split text into individual UTF-8 characters
 * 2. Look for merge rules that can combine adjacent tokens
 * 3. Apply highest-priority merges iteratively
 * 4. Repeat until no more merges possible
 */
int tokenize(sapphire_tokenizer_t *tok, const char *text, 
                      int *tokens, int max_tokens);

/**
 * @brief Fallback character-level tokenization for Gemma 3
 */
int tokenize_fallback(const char *text, int *tokens, int max_tokens);

/**
 * @brief Decode token ID to string
 * 
 * @param tok Tokenizer
 * @param token_id Token ID
 * @return Pointer to token string, or "" if invalid ID
 * 
 * WARNING: Returned pointer is valid only while tokenizer is alive.
 * Do NOT modify or free the returned string.
 */
const char* decode(sapphire_tokenizer_t *tok, int token_id);

/**
 * @brief Decode array of token IDs to text
 * 
 * @param tok Tokenizer
 * @param tokens Array of token IDs
 * @param num_tokens Number of tokens
 * @param output Output buffer (caller must allocate)
 * @param output_size Size of output buffer
 * @return Number of characters written (not including null terminator)
 */
int detokenize(sapphire_tokenizer_t *tok, const int *tokens,
                        int num_tokens, char *output, int output_size);

/**
 * @brief Get vocabulary size
 * 
 * @param tok Tokenizer
 * @return Number of tokens in vocabulary
 */
int tokenizer_vocab_size(const sapphire_tokenizer_t *tok);

#ifdef __cplusplus
}
#endif

#endif /* SAPPHIRE_TOKENIZER_H */

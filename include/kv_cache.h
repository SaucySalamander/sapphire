#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stddef.h>

/* Forward declare types to keep header minimal */
typedef struct tensor_t tensor_t;
typedef struct kv_cache_t kv_cache_t;

// ============================================================================
// MULTI-LAYER, MULTI-HEAD KV CACHE API
// ============================================================================

/**
 * @brief Create a multi-layer, multi-head KV cache with GQA support.
 *
 * Creates a KV cache for a transformer with multiple layers and heads.
 * Supports Grouped Query Attention (GQA) where num_kv_heads < num_query_heads.
 * 
 * Structure:
 * - keys_per_layer[layer] is a tensor [num_kv_heads, max_seq_len, head_dim]
 * - values_per_layer[layer] is a tensor [num_kv_heads, max_seq_len, head_dim]
 * 
 * Example (Gemma 3 270M):
 *   - num_layers = 12
 *   - num_kv_heads = 1 (grouped from 8 query heads, 8:1 ratio)
 *   - max_seq_len = 8192
 *   - head_dim = 256
 *
 * @param num_layers Number of transformer layers
 * @param num_kv_heads Number of KV heads (after GQA reduction)
 * @param max_seq_len Maximum sequence length
 * @param head_dim Dimension per head
 * @return Initialized cache with current_pos=0, or NULL on failure
 */
kv_cache_t* kv_cache_create(int num_layers, int num_kv_heads, int max_seq_len, int head_dim);

/**
 * @brief Configure attention strategy for a specific layer.
 *
 * Used for models with interleaved attention patterns (e.g., Gemma 3).
 * Allows per-layer configuration of global vs. local (sliding window) attention.
 *
 * @param cache KV cache
 * @param layer Layer index (0 to num_layers-1)
 * @param is_local 1 for local/sliding-window attention, 0 for global
 * @param window_size Window size for local attention (ignored if is_local=0)
 * @return 0 on success, -1 on error
 */
int kv_cache_set_layer_config(kv_cache_t *cache, int layer, int is_local, int window_size);

/**
 * @brief Append a token's K and V vectors to the cache for all layers.
 *
 * Writes token vectors to all layers at the current position, then increments
 * the shared position counter.
 *
 * @param cache KV cache
 * @param k_token Key vectors [num_kv_heads, head_dim]
 * @param v_token Value vectors [num_kv_heads, head_dim]
 * @return 0 on success, -1 if cache is full
 */
int kv_cache_append_token(kv_cache_t *cache, const float *k_token, const float *v_token);

/**
 * @brief Get key tensor for a specific layer.
 *
 * @return Tensor pointer [num_kv_heads, max_seq_len, head_dim] or NULL
 */
tensor_t* kv_cache_get_keys(kv_cache_t *cache, int layer);

/**
 * @brief Get value tensor for a specific layer.
 *
 * @return Tensor pointer [num_kv_heads, max_seq_len, head_dim] or NULL
 */
tensor_t* kv_cache_get_values(kv_cache_t *cache, int layer);

/**
 * @brief Get current sequence length (shared across all layers).
 */
int kv_cache_get_seq_len(const kv_cache_t *cache);
int kv_cache_set_seq_len(kv_cache_t *cache, int seq_len);
int kv_cache_write_token(kv_cache_t *cache, int layer, int pos, const float *k_token, const float *v_token);

/**
 * @brief Check if cache is full.
 */
int kv_cache_is_full(const kv_cache_t *cache);

/**
 * @brief Reset cache for a new sequence (sets current_pos=0).
 */
void kv_cache_reset(kv_cache_t *cache);

/**
 * @brief Release all resources associated with the cache.
 */
void kv_cache_release(kv_cache_t *cache);

/**
 * @brief Print cache metadata and layer configuration.
 */
void kv_cache_print_info(const kv_cache_t *cache);

// ============================================================================
// ACCESSOR FUNCTIONS
// ============================================================================

int kv_cache_get_num_layers(const kv_cache_t *cache);
int kv_cache_get_num_kv_heads(const kv_cache_t *cache);
int kv_cache_get_max_seq_len(const kv_cache_t *cache);
int kv_cache_get_head_dim(const kv_cache_t *cache);
int kv_cache_get_layer_window_size(const kv_cache_t *cache, int layer);
int kv_cache_is_layer_local(const kv_cache_t *cache, int layer);

#endif // KV_CACHE_H

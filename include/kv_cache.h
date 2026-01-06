#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stddef.h>

/* Forward declare types to keep header minimal */
typedef struct tensor_t tensor_t;
typedef struct kv_cache_t kv_cache_t;

/**
 * @brief Create a KV cache for a single attention head.
 *
 * Allocates tensors for keys and values with shape [max_seq_len, d_k].
 * Initializes current_pos to 0 (empty cache).
 */
kv_cache_t* kv_cache_create(int max_seq_len, int d_k);

int kv_cache_append_token(kv_cache_t *cache, const float *k_token, const float *v_token);

tensor_t* kv_cache_get_keys(kv_cache_t *cache);
tensor_t* kv_cache_get_values(kv_cache_t *cache);
int kv_cache_get_seq_len(const kv_cache_t *cache);
int kv_cache_is_full(const kv_cache_t *cache);
void kv_cache_reset(kv_cache_t *cache);
void kv_cache_release(kv_cache_t *cache);
void kv_cache_print_info(const kv_cache_t *cache);

/* Accessors for previously-public fields */
int kv_cache_get_max_seq_len(const kv_cache_t *cache);
int kv_cache_get_d_k(const kv_cache_t *cache);

#endif // KV_CACHE_H

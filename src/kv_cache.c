#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kv_cache.h"
#include "tensor.h"

/* Concrete definition (private) */
struct kv_cache_t {
    tensor_t *keys;
    tensor_t *values;
    int current_pos;
    int max_seq_len;
    int d_k;
};

// ============================================================================
// Public API: kv_cache_create
// ============================================================================

/**
 * Create a new KV cache.
 * 
 * Allocates:
 * - keys tensor: [max_seq_len, d_k] for key vectors
 * - values tensor: [max_seq_len, d_k] for value vectors
 * - kv_cache_t struct with metadata
 * 
 * @return Initialized cache with current_pos=0, or NULL on failure.
 */
kv_cache_t* kv_cache_create(int max_seq_len, int d_k) {
    if (max_seq_len <= 0 || d_k <= 0) {
        fprintf(stderr, "ERROR: kv_cache_create invalid max_seq_len=%d or d_k=%d\n", 
                max_seq_len, d_k);
        return NULL;
    }
    
    // Allocate cache structure
    kv_cache_t *cache = (kv_cache_t *)malloc(sizeof(kv_cache_t));
    if (!cache) {
        fprintf(stderr, "ERROR: kv_cache_create malloc failed for kv_cache_t\n");
        return NULL;
    }
    
    // Allocate keys tensor [max_seq_len, d_k]
    int shape_kv[] = {max_seq_len, d_k};
    cache->keys = tensor_create(2, shape_kv, DTYPE_F32);
    if (!cache->keys) {
        fprintf(stderr, "ERROR: kv_cache_create failed to create keys tensor\n");
        free(cache);
        return NULL;
    }
    
    // Allocate values tensor [max_seq_len, d_k]
    cache->values = tensor_create(2, shape_kv, DTYPE_F32);
    if (!cache->values) {
        fprintf(stderr, "ERROR: kv_cache_create failed to create values tensor\n");
        tensor_release(cache->keys);
        free(cache);
        return NULL;
    }
    
    // Initialize metadata
    cache->max_seq_len = max_seq_len;
    cache->d_k = d_k;
    cache->current_pos = 0;  // Empty cache initially
    
    return cache;
}

// ============================================================================
// Public API: kv_cache_append_token
// ============================================================================

/**
 * Append a token's K and V vectors to the cache.
 * 
 * Writes:
 *   keys[current_pos, :] = k_token
 *   values[current_pos, :] = v_token
 * Then increments current_pos.
 * 
 * @return 0 on success, -1 if cache is full.
 */
int kv_cache_append_token(kv_cache_t *cache, const float *k_token, const float *v_token) {
    if (!cache || !k_token || !v_token) {
        fprintf(stderr, "ERROR: kv_cache_append_token null pointer\n");
        return -1;
    }
    
    // Check if cache is full
    if (cache->current_pos >= cache->max_seq_len) {
        fprintf(stderr, "ERROR: kv_cache_append_token cache full (pos=%d, max=%d)\n", 
                cache->current_pos, cache->max_seq_len);
        return -1;
    }
    
    // Write K vector at current_pos
    float *keys_data = tensor_data_f32(cache->keys);
    size_t k_offset = (size_t)cache->current_pos * cache->d_k;
    memcpy(keys_data + k_offset, k_token, cache->d_k * sizeof(float));
    
    // Write V vector at current_pos
    float *values_data = tensor_data_f32(cache->values);
    size_t v_offset = (size_t)cache->current_pos * cache->d_k;
    memcpy(values_data + v_offset, v_token, cache->d_k * sizeof(float));
    
    // Increment position
    cache->current_pos++;
    
    return 0;
}

// ============================================================================
// Public API: kv_cache_get_keys and kv_cache_get_values
// ============================================================================

tensor_t* kv_cache_get_keys(kv_cache_t *cache) {
    if (!cache) return NULL;
    return cache->keys;
}

tensor_t* kv_cache_get_values(kv_cache_t *cache) {
    if (!cache) return NULL;
    return cache->values;
}

// ============================================================================
// Public API: kv_cache_get_seq_len
// ============================================================================

int kv_cache_get_seq_len(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->current_pos;
}

// ============================================================================
// Public API: kv_cache_reset
// ============================================================================

/**
 * Reset the cache for a new sequence.
 * Sets current_pos=0 but keeps the allocated tensors.
 */
void kv_cache_reset(kv_cache_t *cache) {
    if (!cache) return;
    cache->current_pos = 0;
}

// ============================================================================
// Public API: kv_cache_is_full
// ============================================================================

int kv_cache_is_full(const kv_cache_t *cache) {
    if (!cache) return 1;
    return cache->current_pos >= cache->max_seq_len ? 1 : 0;
}

// ============================================================================
// Public API: kv_cache_release
// ============================================================================

/**
 * Release all resources associated with the cache.
 */
void kv_cache_release(kv_cache_t *cache) {
    if (!cache) return;
    
    if (cache->keys) {
        tensor_release(cache->keys);
    }
    if (cache->values) {
        tensor_release(cache->values);
    }
    
    free(cache);
}

// ============================================================================
// Public API: kv_cache_print_info
// ============================================================================

/**
 * Print metadata about the cache.
 * 
 * Example output:
 *   KV Cache: max_seq_len=2048 d_k=64 current_pos=42 (2.0% full)
 */
void kv_cache_print_info(const kv_cache_t *cache) {
    if (!cache) {
        printf("KV Cache: NULL\n");
        return;
    }
    
    double percent_full = 100.0 * cache->current_pos / cache->max_seq_len;
    printf("KV Cache: max_seq_len=%d d_k=%d current_pos=%d (%.1f%% full)\n",
           cache->max_seq_len, cache->d_k, cache->current_pos, percent_full);
}

int kv_cache_get_max_seq_len(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->max_seq_len;
}

int kv_cache_get_d_k(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->d_k;
}

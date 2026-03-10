/**
 * @file kv_cache.c
 * @brief Multi-layer, multi-head KV cache implementation with GQA support.
 *
 * This module implements a global KV cache for transformer models that:
 * - Supports multiple layers with independent K,V tensors per layer
 * - Handles Grouped Query Attention (GQA) with num_kv_heads < num_query_heads
 * - Tracks per-layer attention strategies (local vs global, window sizes)
 * - Maintains a shared sequence position across all layers
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/kv_cache.h"
#include "../include/kv_paged.h"
#include "../include/kv_cache_state.h"
#include "../include/tensor.h"
#include "../include/log.h"
#include "../include/utils.h"

#define KV_STATE_MAGIC_LOCAL "SPKVv1\0"
#define KV_STATE_VERSION_LOCAL 1u

/**
 * Multi-layer, multi-head KV cache structure.
 *
 * Shape per layer:
 *   keys[layer]:   [num_kv_heads, max_seq_len, head_dim]
 *   values[layer]: [num_kv_heads, max_seq_len, head_dim]
 */
struct kv_cache_t {
    tensor_t **keys_per_layer;      /**< [num_layers] K tensors */
    tensor_t **values_per_layer;    /**< [num_layers] V tensors */
    
    int num_layers;                 /**< Number of transformer layers */
    int num_kv_heads;               /**< Number of KV heads (GQA: < num_query_heads) */
    int max_seq_len;                /**< Maximum sequence length */
    int head_dim;                   /**< Dimension per head */
    
    int current_pos;                /**< Current sequence position (shared across all layers) */
    
    int *layer_window_size;         /**< [num_layers] Sliding window size per layer (0 = global) */
    int *layer_is_local;            /**< [num_layers] 1 = local/windowed, 0 = global attention */
    /* Debug snapshots: store last written K/V token per layer for readback verification */
    float **last_k_snapshot;        /**< [num_layers] pointer to last-written K token (num_kv_heads*head_dim) */
    float **last_v_snapshot;        /**< [num_layers] pointer to last-written V token (num_kv_heads*head_dim) */
    int *last_snapshot_pos;         /**< [num_layers] position associated with the snapshots, -1 if none */
    kv_pager_t *pager;
    kv_page_t **page_table;         /**< [num_layers * pages_per_layer] page mapping */
    float **cold_k_pages;           /**< [num_layers * pages_per_layer] RAM-backed cold K pages */
    float **cold_v_pages;           /**< [num_layers * pages_per_layer] RAM-backed cold V pages */
    uint8_t *cold_valid;            /**< [num_layers * pages_per_layer] cold page payload validity */
    size_t cold_page_bytes;
    size_t page_table_slots;
    uint32_t page_tokens;
    uint32_t pages_per_layer;
    uint64_t page_hits;
    uint64_t page_misses;
    uint64_t page_promotions;
    uint64_t page_evictions;
};

static uint32_t compute_default_max_pages(int num_layers, int max_seq_len, uint32_t page_tokens) {
    if (num_layers <= 0 || max_seq_len <= 0 || page_tokens == 0u) return 0u;
    uint32_t pages_per_layer = (uint32_t)(((uint32_t)max_seq_len + page_tokens - 1u) / page_tokens);
    return pages_per_layer * (uint32_t)num_layers;
}

static size_t kv_cache_page_float_count(const kv_cache_t *cache) {
    if (!cache || cache->page_tokens == 0u) return 0u;
    return (size_t)cache->num_kv_heads * (size_t)cache->page_tokens * (size_t)cache->head_dim;
}

static void kv_cache_copy_page_payload_out(const kv_cache_t *cache, int layer, uint32_t page_idx, float *dst_k, float *dst_v) {
    if (!cache || !dst_k || !dst_v) return;
    int token_start = (int)(page_idx * cache->page_tokens);
    if (token_start >= cache->max_seq_len) return;
    int token_count = (int)cache->page_tokens;
    if (token_start + token_count > cache->max_seq_len) {
        token_count = cache->max_seq_len - token_start;
    }

    const float *keys_data = (const float *)tensor_data(cache->keys_per_layer[layer]);
    const float *values_data = (const float *)tensor_data(cache->values_per_layer[layer]);
    size_t head_stride = (size_t)cache->max_seq_len * (size_t)cache->head_dim;
    size_t page_head_stride = (size_t)cache->page_tokens * (size_t)cache->head_dim;
    size_t token_floats = (size_t)token_count * (size_t)cache->head_dim;

    for (int head = 0; head < cache->num_kv_heads; ++head) {
        size_t src_base = (size_t)head * head_stride + (size_t)token_start * (size_t)cache->head_dim;
        size_t dst_base = (size_t)head * page_head_stride;
        memcpy(dst_k + dst_base, keys_data + src_base, token_floats * sizeof(float));
        memcpy(dst_v + dst_base, values_data + src_base, token_floats * sizeof(float));
    }
}

static void kv_cache_copy_page_payload_in(const kv_cache_t *cache, int layer, uint32_t page_idx, const float *src_k, const float *src_v) {
    if (!cache || !src_k || !src_v) return;
    int token_start = (int)(page_idx * cache->page_tokens);
    if (token_start >= cache->max_seq_len) return;
    int token_count = (int)cache->page_tokens;
    if (token_start + token_count > cache->max_seq_len) {
        token_count = cache->max_seq_len - token_start;
    }

    float *keys_data = (float *)tensor_data(cache->keys_per_layer[layer]);
    float *values_data = (float *)tensor_data(cache->values_per_layer[layer]);
    size_t head_stride = (size_t)cache->max_seq_len * (size_t)cache->head_dim;
    size_t page_head_stride = (size_t)cache->page_tokens * (size_t)cache->head_dim;
    size_t token_floats = (size_t)token_count * (size_t)cache->head_dim;

    for (int head = 0; head < cache->num_kv_heads; ++head) {
        size_t dst_base = (size_t)head * head_stride + (size_t)token_start * (size_t)cache->head_dim;
        size_t src_base = (size_t)head * page_head_stride;
        memcpy(keys_data + dst_base, src_k + src_base, token_floats * sizeof(float));
        memcpy(values_data + dst_base, src_v + src_base, token_floats * sizeof(float));
    }
}

static int kv_cache_stage_page_to_ram(kv_cache_t *cache, const kv_page_t *page) {
    if (!cache || !page || !cache->cold_k_pages || !cache->cold_v_pages || !cache->cold_valid) return -1;
    if (page->layer_id == UINT32_MAX || page->token_start == UINT64_MAX) return -1;

    uint32_t page_idx = (uint32_t)(page->token_start / cache->page_tokens);
    if (page->layer_id >= (uint32_t)cache->num_layers || page_idx >= cache->pages_per_layer) return -1;
    size_t slot = (size_t)page->layer_id * cache->pages_per_layer + page_idx;

    if (!cache->cold_k_pages[slot]) {
        cache->cold_k_pages[slot] = (float *)malloc(cache->cold_page_bytes);
    }
    if (!cache->cold_v_pages[slot]) {
        cache->cold_v_pages[slot] = (float *)malloc(cache->cold_page_bytes);
    }
    if (!cache->cold_k_pages[slot] || !cache->cold_v_pages[slot]) {
        LOG_WARN("KV page RAM spill allocation failed for slot=%zu", slot);
        return -1;
    }

    kv_cache_copy_page_payload_out(cache, (int)page->layer_id, page_idx, cache->cold_k_pages[slot], cache->cold_v_pages[slot]);
    cache->cold_valid[slot] = 1u;
    cache->page_evictions++;
    return 0;
}

static void kv_cache_maybe_promote_from_ram(kv_cache_t *cache, int layer, uint32_t page_idx, kv_page_t *page) {
    if (!cache || !page || !cache->cold_valid || !cache->cold_k_pages || !cache->cold_v_pages) return;
    size_t slot = (size_t)layer * cache->pages_per_layer + page_idx;
    if (slot >= cache->page_table_slots) return;
    if (!cache->cold_valid[slot]) return;
    if (!cache->cold_k_pages[slot] || !cache->cold_v_pages[slot]) return;

    kv_cache_copy_page_payload_in(cache, layer, page_idx, cache->cold_k_pages[slot], cache->cold_v_pages[slot]);
    cache->cold_valid[slot] = 0u;
    cache->page_promotions++;
    page->tier = KV_PAGE_TIER_RAM;
}

static int kv_cache_table_index(const kv_cache_t *cache, int layer, int pos, uint32_t *out_index) {
    if (!cache || !out_index || !cache->page_table || cache->pages_per_layer == 0u || cache->page_tokens == 0u) {
        return -1;
    }
    if (layer < 0 || layer >= cache->num_layers || pos < 0 || pos >= cache->max_seq_len) {
        return -1;
    }
    uint32_t page_idx = (uint32_t)pos / cache->page_tokens;
    if (page_idx >= cache->pages_per_layer) {
        return -1;
    }
    *out_index = (uint32_t)layer * cache->pages_per_layer + page_idx;
    return 0;
}

static void kv_cache_clear_page_owner(kv_cache_t *cache, const kv_page_t *page) {
    if (!cache || !page || !cache->page_table || cache->page_tokens == 0u || cache->pages_per_layer == 0u) return;
    if (page->layer_id == UINT32_MAX || page->token_start == UINT64_MAX) return;
    if (page->layer_id >= (uint32_t)cache->num_layers) return;
    uint32_t page_idx = (uint32_t)(page->token_start / cache->page_tokens);
    if (page_idx >= cache->pages_per_layer) return;
    uint32_t slot = page->layer_id * cache->pages_per_layer + page_idx;
    if (cache->page_table[slot] == page) {
        cache->page_table[slot] = NULL;
    }
}

static kv_page_t *kv_cache_bind_page_for_token(kv_cache_t *cache, int layer, int pos) {
    if (!cache || !cache->pager || !cache->page_table) return NULL;

    uint32_t slot = 0u;
    if (kv_cache_table_index(cache, layer, pos, &slot) != 0) return NULL;

    kv_page_t *page = cache->page_table[slot];
    uint32_t page_idx = (uint32_t)pos / cache->page_tokens;

    if (page) {
        cache->page_hits++;
        kv_pager_touch_page(cache->pager, page);
        return page;
    }

    cache->page_misses++;
    page = kv_pager_alloc_page(cache->pager);
    if (!page) {
        kv_page_t *victim = kv_pager_evict_candidate(cache->pager);
        if (victim) {
            (void)kv_cache_stage_page_to_ram(cache, victim);
            kv_cache_clear_page_owner(cache, victim);
            if (kv_pager_free_page(cache->pager, victim) == 0) {
                page = kv_pager_alloc_page(cache->pager);
            }
        }
    }

    if (!page) {
        return NULL;
    }

    page->layer_id = (uint32_t)layer;
    page->head_group_id = 0u;
    page->token_start = ((uint64_t)pos / cache->page_tokens) * cache->page_tokens;
    page->token_count = cache->page_tokens;
    if ((int)(page->token_start + page->token_count) > cache->max_seq_len) {
        page->token_count = (uint32_t)(cache->max_seq_len - (int)page->token_start);
    }
    page->tier = KV_PAGE_TIER_VRAM;

    kv_cache_maybe_promote_from_ram(cache, layer, page_idx, page);

    cache->page_table[slot] = page;
    return page;
}

static void kv_cache_init_core_fields(kv_cache_t *cache,
                                      int num_layers,
                                      int num_kv_heads,
                                      int max_seq_len,
                                      int head_dim) {
    cache->num_layers = num_layers;
    cache->num_kv_heads = num_kv_heads;
    cache->max_seq_len = max_seq_len;
    cache->head_dim = head_dim;
    cache->current_pos = 0;
    cache->pager = NULL;
    cache->page_table = NULL;
    cache->cold_k_pages = NULL;
    cache->cold_v_pages = NULL;
    cache->cold_valid = NULL;
    cache->cold_page_bytes = 0u;
    cache->page_table_slots = 0u;
    cache->page_tokens = 0u;
    cache->pages_per_layer = 0u;
    cache->page_hits = 0u;
    cache->page_misses = 0u;
    cache->page_promotions = 0u;
    cache->page_evictions = 0u;
}

static void kv_cache_free_layer_arrays(kv_cache_t *cache) {
    if (!cache) return;
    free(cache->keys_per_layer);
    free(cache->values_per_layer);
    free(cache->layer_window_size);
    free(cache->layer_is_local);
    free(cache->last_k_snapshot);
    free(cache->last_v_snapshot);
    free(cache->last_snapshot_pos);
    cache->keys_per_layer = NULL;
    cache->values_per_layer = NULL;
    cache->layer_window_size = NULL;
    cache->layer_is_local = NULL;
    cache->last_k_snapshot = NULL;
    cache->last_v_snapshot = NULL;
    cache->last_snapshot_pos = NULL;
}

static int kv_cache_alloc_layer_arrays(kv_cache_t *cache, int num_layers) {
    cache->keys_per_layer = (tensor_t **)malloc(num_layers * sizeof(tensor_t *));
    cache->values_per_layer = (tensor_t **)malloc(num_layers * sizeof(tensor_t *));
    cache->layer_window_size = (int *)malloc(num_layers * sizeof(int));
    cache->layer_is_local = (int *)malloc(num_layers * sizeof(int));
    cache->last_k_snapshot = (float **)malloc(num_layers * sizeof(float *));
    cache->last_v_snapshot = (float **)malloc(num_layers * sizeof(float *));
    cache->last_snapshot_pos = (int *)malloc(num_layers * sizeof(int));

    if (!cache->keys_per_layer || !cache->values_per_layer ||
        !cache->layer_window_size || !cache->layer_is_local ||
        !cache->last_k_snapshot || !cache->last_v_snapshot || !cache->last_snapshot_pos) {
        LOG_ERROR("Failed to allocate layer arrays");
        kv_cache_free_layer_arrays(cache);
        return -1;
    }

    memset(cache->keys_per_layer, 0, num_layers * sizeof(tensor_t *));
    memset(cache->values_per_layer, 0, num_layers * sizeof(tensor_t *));
    memset(cache->last_k_snapshot, 0, num_layers * sizeof(float *));
    memset(cache->last_v_snapshot, 0, num_layers * sizeof(float *));
    for (int i = 0; i < num_layers; ++i) {
        cache->last_snapshot_pos[i] = -1;
    }
    return 0;
}

static void kv_cache_disable_pager(kv_cache_t *cache) {
    if (!cache) return;
    free(cache->page_table);
    free(cache->cold_k_pages);
    free(cache->cold_v_pages);
    free(cache->cold_valid);
    cache->page_table = NULL;
    cache->cold_k_pages = NULL;
    cache->cold_v_pages = NULL;
    cache->cold_valid = NULL;
    cache->cold_page_bytes = 0u;
    cache->page_table_slots = 0u;
    if (cache->pager) {
        kv_pager_destroy(cache->pager);
        cache->pager = NULL;
    }
    cache->page_tokens = 0u;
    cache->pages_per_layer = 0u;
}

static void kv_cache_try_init_pager(kv_cache_t *cache, int num_layers, int max_seq_len) {
    kv_pager_config_t pager_cfg;
    if (kv_pager_config_from_env(&pager_cfg) != 0) return;

    if (pager_cfg.max_pages == 4096u) {
        uint32_t computed_pages = compute_default_max_pages(num_layers, max_seq_len, pager_cfg.page_tokens);
        if (computed_pages > 0u) pager_cfg.max_pages = computed_pages;
    }

    cache->pager = kv_pager_create(&pager_cfg);
    if (!cache->pager) {
        LOG_WARN("Paged KV allocator init failed, continuing with monolithic KV cache");
        return;
    }

    cache->page_tokens = kv_pager_page_tokens(cache->pager);
    cache->pages_per_layer = (uint32_t)(((uint32_t)cache->max_seq_len + cache->page_tokens - 1u) / cache->page_tokens);
    size_t table_count = (size_t)cache->num_layers * cache->pages_per_layer;
    cache->page_table_slots = table_count;
    cache->cold_page_bytes = kv_cache_page_float_count(cache) * sizeof(float);
    cache->page_table = (kv_page_t **)calloc(table_count, sizeof(kv_page_t *));
    cache->cold_k_pages = (float **)calloc(table_count, sizeof(float *));
    cache->cold_v_pages = (float **)calloc(table_count, sizeof(float *));
    cache->cold_valid = (uint8_t *)calloc(table_count, sizeof(uint8_t));
    if (!cache->page_table || !cache->cold_k_pages || !cache->cold_v_pages || !cache->cold_valid) {
        LOG_WARN("Paged KV page table allocation failed, disabling pager");
        kv_cache_disable_pager(cache);
    }
}

static void kv_cache_free_partial_layer_data(kv_cache_t *cache, int upto_layer) {
    for (int j = 0; j < upto_layer; j++) {
        tensor_release(cache->keys_per_layer[j]);
        tensor_release(cache->values_per_layer[j]);
        free(cache->last_k_snapshot[j]);
        free(cache->last_v_snapshot[j]);
    }
}

static int kv_cache_init_layer_tensors(kv_cache_t *cache,
                                       int num_layers,
                                       int num_kv_heads,
                                       int max_seq_len,
                                       int head_dim) {
    const int shape_kv[] = {num_kv_heads, max_seq_len, head_dim};
    int snapshot_size = cache->num_kv_heads * cache->head_dim;

    for (int i = 0; i < num_layers; i++) {
        cache->keys_per_layer[i] = tensor_create(3, shape_kv, DTYPE_F32);
        cache->values_per_layer[i] = tensor_create(3, shape_kv, DTYPE_F32);
        if (!cache->keys_per_layer[i] || !cache->values_per_layer[i]) {
            LOG_ERROR("Failed to create tensors for layer %d", i);
            kv_cache_free_partial_layer_data(cache, i);
            return -1;
        }

        cache->layer_is_local[i] = 0;
        cache->layer_window_size[i] = 0;
        cache->last_k_snapshot[i] = (float *)calloc(snapshot_size, sizeof(float));
        cache->last_v_snapshot[i] = (float *)calloc(snapshot_size, sizeof(float));
        cache->last_snapshot_pos[i] = -1;
    }
    return 0;
}

/**
 * Create a multi-layer KV cache with GQA support.
 *
 * @param num_layers Number of transformer layers
 * @param num_kv_heads Number of KV heads (after GQA reduction)
 * @param max_seq_len Maximum sequence length
 * @param head_dim Dimension per head
 * @return Initialized cache or NULL on failure
 */
kv_cache_t* kv_cache_create(int num_layers, int num_kv_heads, int max_seq_len, int head_dim) {
    if (num_layers <= 0 || num_kv_heads <= 0 || max_seq_len <= 0 || head_dim <= 0) {
        LOG_ERROR("Invalid KV cache parameters: "
                  "num_layers=%d, num_kv_heads=%d, max_seq_len=%d, head_dim=%d",
                  num_layers, num_kv_heads, max_seq_len, head_dim);
        return NULL;
    }
    
    kv_cache_t *cache = (kv_cache_t *)malloc(sizeof(kv_cache_t));
    if (!cache) {
        LOG_ERROR("Failed to allocate KV cache structure");
        return NULL;
    }
    
    kv_cache_init_core_fields(cache, num_layers, num_kv_heads, max_seq_len, head_dim);

    if (kv_cache_alloc_layer_arrays(cache, num_layers) != 0) {
        free(cache);
        return NULL;
    }

    kv_cache_try_init_pager(cache, num_layers, max_seq_len);

    if (kv_cache_init_layer_tensors(cache, num_layers, num_kv_heads, max_seq_len, head_dim) != 0) {
        kv_cache_disable_pager(cache);
        kv_cache_free_layer_arrays(cache);
        free(cache);
        return NULL;
    }
    
    return cache;
}

/**
 * Configure per-layer attention strategy (local vs global).
 *
 * @param cache KV cache
 * @param layer Layer index
 * @param is_local 1 for sliding window, 0 for global
 * @param window_size Window size (ignored if is_local=0)
 * @return 0 on success, -1 on error
 */
int kv_cache_set_layer_config(kv_cache_t *cache, int layer, int is_local, int window_size) {
    if (!cache || layer < 0 || layer >= cache->num_layers) {
        return -1;
    }
    
    if (is_local && window_size <= 0) {
        return -1;
    }
    
    cache->layer_is_local[layer] = is_local ? 1 : 0;
    cache->layer_window_size[layer] = is_local ? window_size : 0;
    
    return 0;
}

/**
 * Append a single token's K and V vectors to all layers.
 *
 * Writes token vectors to each layer's KV cache at current_pos, then increments
 * the shared position counter. All layers see the same token at the same position.
 *
 * @param cache KV cache
 * @param k_token Key vectors [num_kv_heads, head_dim]
 * @param v_token Value vectors [num_kv_heads, head_dim]
 * @return 0 on success, -1 if cache is full
 */
int kv_cache_append_token(kv_cache_t *cache, const float *k_token, const float *v_token) {
    if (!cache || !k_token || !v_token) {
        return -1;
    }
    
    if (cache->current_pos >= cache->max_seq_len) {
        return -1;  // Cache full
    }
    
    int pos = cache->current_pos;
    
    // Write token to all layers at the same position
    for (int layer = 0; layer < cache->num_layers; layer++) {
        if (kv_cache_write_token(cache, layer, pos, k_token, v_token) != 0) {
            return -1;
        }
    }
    
    cache->current_pos++;
    return 0;
}

/**
 * Get key tensor for a specific layer.
 *
 * @return Tensor [num_kv_heads, max_seq_len, head_dim] or NULL
 */
tensor_t* kv_cache_get_keys(kv_cache_t *cache, int layer) {
    if (!cache || layer < 0 || layer >= cache->num_layers) {
        return NULL;
    }
    return cache->keys_per_layer[layer];
}

/**
 * Get value tensor for a specific layer.
 *
 * @return Tensor [num_kv_heads, max_seq_len, head_dim] or NULL
 */
tensor_t* kv_cache_get_values(kv_cache_t *cache, int layer) {
    if (!cache || layer < 0 || layer >= cache->num_layers) {
        return NULL;
    }
    return cache->values_per_layer[layer];
}

/**
 * Get current sequence length (shared across all layers).
 */
int kv_cache_get_seq_len(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->current_pos;
}

int kv_cache_set_seq_len(kv_cache_t *cache, int seq_len) {
    if (!cache || seq_len < 0 || seq_len > cache->max_seq_len) {
        return -1;
    }
    cache->current_pos = seq_len;
    return 0;
}

int kv_cache_write_token(kv_cache_t *cache, int layer, int pos, const float *k_token, const float *v_token) {
    if (!cache || !k_token || !v_token) {
        return -1;
    }

    if (layer < 0 || layer >= cache->num_layers) {
        return -1;
    }

    if (pos < 0 || pos >= cache->max_seq_len) {
        return -1;
    }

    (void)kv_cache_bind_page_for_token(cache, layer, pos);

    const tensor_t *keys = cache->keys_per_layer[layer];
    const tensor_t *values = cache->values_per_layer[layer];

    float *keys_data = (float *)tensor_data(keys);
    float *values_data = (float *)tensor_data(values);

    for (int head = 0; head < cache->num_kv_heads; head++) {
        int base_offset = head * (cache->max_seq_len * cache->head_dim) + pos * cache->head_dim;

        for (int dim = 0; dim < cache->head_dim; dim++) {
            keys_data[base_offset + dim] = k_token[head * cache->head_dim + dim];
            values_data[base_offset + dim] = v_token[head * cache->head_dim + dim];
        }
    }

    /* Debug: snapshot of written K/V for quick integrity checks */
    if (log_get_level() == LOG_LEVEL_DEBUG && pos == 0) {
        float kmin = 0.0f, kmax = 0.0f, krms = 0.0f;
        float vmin = 0.0f, vmax = 0.0f, vrms = 0.0f;
        int total = cache->num_kv_heads * cache->head_dim;
        vec_stats(k_token, total, &kmin, &kmax, &krms);
        vec_stats(v_token, total, &vmin, &vmax, &vrms);
        LOG_DEBUG("KV Write L%d P%d: K min=%.6f max=%.6f rms=%.6f | V min=%.6f max=%.6f rms=%.6f",
                  layer, pos, kmin, kmax, krms, vmin, vmax, vrms);
    }

    /* Store a snapshot of the last written token for readback verification */
    if (cache->last_k_snapshot && cache->last_v_snapshot && cache->last_snapshot_pos) {
        int total = cache->num_kv_heads * cache->head_dim;
        float *k_snap = cache->last_k_snapshot[layer];
        float *v_snap = cache->last_v_snapshot[layer];
        if (k_snap && v_snap) {
            memcpy(k_snap, k_token, total * sizeof(float));
            memcpy(v_snap, v_token, total * sizeof(float));
            cache->last_snapshot_pos[layer] = pos;
        }
    }

    return 0;
}

int kv_cache_touch_range(kv_cache_t *cache, int layer, int start_pos, int end_pos) {
    if (!cache) return -1;
    if (!cache->pager || !cache->page_table || cache->page_tokens == 0u) return 0;
    if (layer < 0 || layer >= cache->num_layers) return -1;

    if (start_pos < 0) start_pos = 0;
    if (end_pos < 0) return 0;
    if (end_pos >= cache->max_seq_len) end_pos = cache->max_seq_len - 1;
    if (start_pos > end_pos) return 0;

    int pos = start_pos;
    while (pos <= end_pos) {
        (void)kv_cache_bind_page_for_token(cache, layer, pos);
        int next_pos = (int)(((uint32_t)pos / cache->page_tokens) + 1u) * (int)cache->page_tokens;
        if (next_pos <= pos) {
            next_pos = pos + 1;
        }
        pos = next_pos;
    }
    return 0;
}

int kv_cache_save_state(const kv_cache_t *cache, const char *path) {
    if (!cache || !path) return -1;
    if (cache->page_tokens == 0u) {
        LOG_ERROR("KV snapshot save requires paged KV configuration");
        return -1;
    }

    uint64_t pages_used_per_layer = (cache->current_pos > 0)
        ? (uint64_t)(((uint32_t)cache->current_pos + cache->page_tokens - 1u) / cache->page_tokens)
        : 0u;
    uint64_t page_count = pages_used_per_layer * (uint64_t)cache->num_layers;
    size_t page_float_count = kv_cache_page_float_count(cache);
    if (page_float_count == 0u) return -1;

    kv_state_file_header_t header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, KV_STATE_MAGIC_LOCAL, sizeof(header.magic));
    header.version = KV_STATE_VERSION_LOCAL;
    header.header_size = (uint32_t)sizeof(header);
    header.num_layers = (uint32_t)cache->num_layers;
    header.num_kv_heads = (uint32_t)cache->num_kv_heads;
    header.max_seq_len = (uint32_t)cache->max_seq_len;
    header.head_dim = (uint32_t)cache->head_dim;
    header.current_seq_len = (uint32_t)cache->current_pos;
    header.page_tokens = cache->page_tokens;
    header.page_float_count = (uint32_t)page_float_count;
    header.page_count = page_count;

    kv_state_writer_t *writer = NULL;
    if (kv_state_writer_open(path, &header, &writer) != 0) return -1;

    float *tmp_k = (float *)malloc(page_float_count * sizeof(float));
    float *tmp_v = (float *)malloc(page_float_count * sizeof(float));
    if (!tmp_k || !tmp_v) {
        if (tmp_k) free(tmp_k);
        if (tmp_v) free(tmp_v);
        kv_state_writer_close(writer);
        return -1;
    }

    for (int layer = 0; layer < cache->num_layers; ++layer) {
        for (uint32_t page_idx = 0; page_idx < (uint32_t)pages_used_per_layer; ++page_idx) {
            kv_state_page_record_t rec;
            memset(&rec, 0, sizeof(rec));
            rec.layer = (uint32_t)layer;
            rec.page_idx = page_idx;
            rec.token_start = page_idx * cache->page_tokens;
            rec.token_count = cache->page_tokens;
            if ((int)(rec.token_start + rec.token_count) > cache->max_seq_len) {
                rec.token_count = (uint32_t)(cache->max_seq_len - (int)rec.token_start);
            }

            size_t slot = (size_t)layer * cache->pages_per_layer + page_idx;
            if (cache->cold_valid && cache->cold_k_pages && cache->cold_v_pages &&
                slot < cache->page_table_slots && cache->cold_valid[slot] &&
                cache->cold_k_pages[slot] && cache->cold_v_pages[slot]) {
                memcpy(tmp_k, cache->cold_k_pages[slot], page_float_count * sizeof(float));
                memcpy(tmp_v, cache->cold_v_pages[slot], page_float_count * sizeof(float));
                rec.tier = (uint32_t)KV_PAGE_TIER_RAM;
                rec.flags = 1u;
            } else {
                kv_cache_copy_page_payload_out(cache, layer, page_idx, tmp_k, tmp_v);
                rec.tier = (uint32_t)KV_PAGE_TIER_VRAM;
                rec.flags = 0u;
            }

            if (kv_state_writer_write_page(writer, &rec, tmp_k, tmp_v, page_float_count) != 0) {
                free(tmp_k);
                free(tmp_v);
                kv_state_writer_close(writer);
                LOG_ERROR("KV snapshot write failed while writing layer=%d page=%u", layer, page_idx);
                return -1;
            }
        }
    }

    free(tmp_k);
    free(tmp_v);
    if (kv_state_writer_close(writer) != 0) {
        LOG_ERROR("KV snapshot close failed: %s", path);
        return -1;
    }

    LOG_INFO("KV snapshot saved: %s (seq_len=%d pages=%llu)",
             path,
             cache->current_pos,
             (unsigned long long)page_count);
    return 0;
}

int kv_cache_load_state(kv_cache_t *cache, const char *path) {
    if (!cache || !path) return -1;

    kv_state_file_header_t header;
    kv_state_reader_t *reader = NULL;
    if (kv_state_reader_open(path, &header, &reader) != 0) return -1;

    if ((int)header.num_layers != cache->num_layers ||
        (int)header.num_kv_heads != cache->num_kv_heads ||
        (int)header.max_seq_len != cache->max_seq_len ||
        (int)header.head_dim != cache->head_dim) {
        LOG_ERROR("KV snapshot incompatible with current model/cache dimensions");
        kv_state_reader_close(reader);
        return -1;
    }

    kv_cache_reset(cache);
    size_t page_float_count = kv_cache_page_float_count(cache);
    if (page_float_count == 0u || page_float_count != (size_t)header.page_float_count) {
        LOG_ERROR("KV snapshot page size mismatch (snapshot=%u local=%zu)",
                  header.page_float_count,
                  page_float_count);
        kv_state_reader_close(reader);
        return -1;
    }

    float *tmp_k = (float *)malloc(page_float_count * sizeof(float));
    float *tmp_v = (float *)malloc(page_float_count * sizeof(float));
    if (!tmp_k || !tmp_v) {
        if (tmp_k) free(tmp_k);
        if (tmp_v) free(tmp_v);
        kv_state_reader_close(reader);
        return -1;
    }

    for (uint64_t i = 0; i < header.page_count; ++i) {
        kv_state_page_record_t rec;
        int rc = kv_state_reader_read_page(reader, &rec, tmp_k, tmp_v, page_float_count);
        if (rc <= 0) {
            LOG_ERROR("KV snapshot truncated while loading page %llu", (unsigned long long)i);
            free(tmp_k);
            free(tmp_v);
            kv_state_reader_close(reader);
            return -1;
        }

        if ((int)rec.layer < 0 || rec.layer >= (uint32_t)cache->num_layers || rec.page_idx >= cache->pages_per_layer) {
            LOG_WARN("KV snapshot page record out of range, skipping layer=%u page=%u", rec.layer, rec.page_idx);
            continue;
        }

        kv_cache_copy_page_payload_in(cache, (int)rec.layer, rec.page_idx, tmp_k, tmp_v);

        size_t slot = (size_t)rec.layer * cache->pages_per_layer + rec.page_idx;
        if (cache->cold_k_pages && cache->cold_v_pages && cache->cold_valid && slot < cache->page_table_slots) {
            if (!cache->cold_k_pages[slot]) cache->cold_k_pages[slot] = (float *)malloc(cache->cold_page_bytes);
            if (!cache->cold_v_pages[slot]) cache->cold_v_pages[slot] = (float *)malloc(cache->cold_page_bytes);
            if (cache->cold_k_pages[slot] && cache->cold_v_pages[slot]) {
                memcpy(cache->cold_k_pages[slot], tmp_k, page_float_count * sizeof(float));
                memcpy(cache->cold_v_pages[slot], tmp_v, page_float_count * sizeof(float));
                cache->cold_valid[slot] = 1u;
            }
        }
    }

    cache->current_pos = (int)header.current_seq_len;

    free(tmp_k);
    free(tmp_v);
    if (kv_state_reader_close(reader) != 0) {
        LOG_WARN("KV snapshot reader close returned error: %s", path);
    }

    LOG_INFO("KV snapshot loaded: %s (seq_len=%d pages=%llu)",
             path,
             cache->current_pos,
             (unsigned long long)header.page_count);
    return 0;
}

/**
 * Check if cache is full.
 */
int kv_cache_is_full(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->current_pos >= cache->max_seq_len;
}

/**
 * Reset cache for a new sequence.
 */
void kv_cache_reset(kv_cache_t *cache) {
    if (!cache) return;
    cache->current_pos = 0;
    cache->page_hits = 0u;
    cache->page_misses = 0u;
    cache->page_promotions = 0u;
    cache->page_evictions = 0u;
    if (cache->pager) {
        kv_pager_reset(cache->pager);
    }
    if (cache->page_table && cache->page_table_slots > 0u) {
        memset(cache->page_table, 0, cache->page_table_slots * sizeof(kv_page_t *));
    }
    if (cache->cold_valid && cache->page_table_slots > 0u) {
        memset(cache->cold_valid, 0, cache->page_table_slots * sizeof(uint8_t));
    }
}

/**
 * Release all resources associated with the cache.
 */
void kv_cache_release(kv_cache_t *cache) {
    if (!cache) return;
    
    if (cache->keys_per_layer) {
        for (int i = 0; i < cache->num_layers; i++) {
            if (cache->keys_per_layer[i]) {
                tensor_release(cache->keys_per_layer[i]);
            }
        }
        free(cache->keys_per_layer);
    }
    
    if (cache->values_per_layer) {
        for (int i = 0; i < cache->num_layers; i++) {
            if (cache->values_per_layer[i]) {
                tensor_release(cache->values_per_layer[i]);
            }
        }
        free(cache->values_per_layer);
    }
    
    if (cache->last_k_snapshot) {
        for (int i = 0; i < cache->num_layers; i++) {
            if (cache->last_k_snapshot[i]) free(cache->last_k_snapshot[i]);
        }
        free(cache->last_k_snapshot);
    }

    if (cache->last_v_snapshot) {
        for (int i = 0; i < cache->num_layers; i++) {
            if (cache->last_v_snapshot[i]) free(cache->last_v_snapshot[i]);
        }
        free(cache->last_v_snapshot);
    }

    if (cache->last_snapshot_pos) {
        free(cache->last_snapshot_pos);
    }

    if (cache->pager) {
        kv_pager_destroy(cache->pager);
    }

    if (cache->page_table) {
        free(cache->page_table);
    }

    if (cache->cold_k_pages) {
        for (size_t i = 0; i < cache->page_table_slots; ++i) {
            if (cache->cold_k_pages[i]) free(cache->cold_k_pages[i]);
        }
        free(cache->cold_k_pages);
    }

    if (cache->cold_v_pages) {
        for (size_t i = 0; i < cache->page_table_slots; ++i) {
            if (cache->cold_v_pages[i]) free(cache->cold_v_pages[i]);
        }
        free(cache->cold_v_pages);
    }

    if (cache->cold_valid) {
        free(cache->cold_valid);
    }
    
    if (cache->layer_window_size) {
        free(cache->layer_window_size);
    }
    
    if (cache->layer_is_local) {
        free(cache->layer_is_local);
    }
    
    free(cache);
}

/**
 * Print cache metadata and configuration.
 */
void kv_cache_print_info(const kv_cache_t *cache) {
    if (!cache) return;
    
    LOG_INFO("KV Cache Info:");
    LOG_INFO("  Layers: %d, KV Heads: %d, Max Seq Len: %d, Head Dim: %d",
             cache->num_layers, cache->num_kv_heads, cache->max_seq_len, cache->head_dim);
    LOG_INFO("  Current Seq Len: %d / %d", cache->current_pos, cache->max_seq_len);
    LOG_INFO("  Per-Layer Config:");
    
    for (int i = 0; i < cache->num_layers; i++) {
        if (cache->layer_is_local[i]) {
            LOG_INFO("    Layer %d: LOCAL (window_size=%d)", i, cache->layer_window_size[i]);
        } else {
            LOG_INFO("    Layer %d: GLOBAL", i);
        }
    }

    if (cache->pager) {
        kv_pager_stats_t stats;
        if (kv_pager_get_stats(cache->pager, &stats) == 0) {
            LOG_INFO("  Paged KV: page_tokens=%u pages_per_layer=%u hits=%llu misses=%llu",
                     cache->page_tokens,
                     cache->pages_per_layer,
                     (unsigned long long)cache->page_hits,
                     (unsigned long long)cache->page_misses);
            LOG_INFO("  Paged KV Flow: promotions=%llu evictions=%llu",
                     (unsigned long long)cache->page_promotions,
                     (unsigned long long)cache->page_evictions);
            LOG_INFO("  Pager Stats: total=%u used=%u free=%u pinned=%u alloc=%llu free_ops=%llu evict=%llu touch=%llu",
                     stats.total_pages,
                     stats.used_pages,
                     stats.free_pages,
                     stats.pinned_pages,
                     (unsigned long long)stats.alloc_count,
                     (unsigned long long)stats.free_count,
                     (unsigned long long)stats.evict_count,
                     (unsigned long long)stats.touch_count);
        }
    }
}

// ============================================================================
// ACCESSOR FUNCTIONS
// ============================================================================

int kv_cache_get_num_layers(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->num_layers;
}

int kv_cache_get_num_kv_heads(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->num_kv_heads;
}

int kv_cache_get_max_seq_len(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->max_seq_len;
}

int kv_cache_get_head_dim(const kv_cache_t *cache) {
    if (!cache) return 0;
    return cache->head_dim;
}

int kv_cache_get_layer_window_size(const kv_cache_t *cache, int layer) {
    if (!cache || layer < 0 || layer >= cache->num_layers) {
        return 0;
    }
    return cache->layer_window_size[layer];
}

int kv_cache_is_layer_local(const kv_cache_t *cache, int layer) {
    if (!cache || layer < 0 || layer >= cache->num_layers) {
        return 0;
    }
    return cache->layer_is_local[layer];
}

int kv_cache_verify_entry(kv_cache_t *cache, int layer, int pos, const float *k_token, const float *v_token) {
    if (!cache || !k_token || !v_token) return -1;
    if (layer < 0 || layer >= cache->num_layers) return -1;

    if (!cache->last_k_snapshot || !cache->last_v_snapshot || !cache->last_snapshot_pos) return -1;

    if (cache->last_snapshot_pos[layer] != pos) {
        LOG_DEBUG("KV Verify L%d P%d: no matching snapshot (snapshot_pos=%d)", layer, pos, cache->last_snapshot_pos[layer]);
        return -1;
    }

    int total = cache->num_kv_heads * cache->head_dim;
    const float *k_snap = cache->last_k_snapshot[layer];
    const float *v_snap = cache->last_v_snapshot[layer];
    if (!k_snap || !v_snap) return -1;

    double k_sq = 0.0, v_sq = 0.0;
    double k_max_abs = 0.0, v_max_abs = 0.0;
    for (int i = 0; i < total; i++) {
        double kd = (double)k_snap[i] - (double)k_token[i];
        double vd = (double)v_snap[i] - (double)v_token[i];
        double k_abs = fabs(kd);
        double v_abs = fabs(vd);
        if (i == 0 || k_abs > k_max_abs) k_max_abs = k_abs;
        if (i == 0 || v_abs > v_max_abs) v_max_abs = v_abs;
        k_sq += kd * kd;
        v_sq += vd * vd;
    }
    double k_rms = sqrt(k_sq / (double)total);
    double v_rms = sqrt(v_sq / (double)total);

    LOG_DEBUG("KV Verify L%d P%d: K max_abs=%.6e rms=%.6e | V max_abs=%.6e rms=%.6e",
              layer, pos, k_max_abs, k_rms, v_max_abs, v_rms);

    return 0;
}

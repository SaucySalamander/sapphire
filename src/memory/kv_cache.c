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
#include "../include/tensor.h"
#include "../include/log.h"
#include "../include/utils.h"

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
};

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
        fprintf(stderr, "ERROR: Invalid KV cache parameters: "
                "num_layers=%d, num_kv_heads=%d, max_seq_len=%d, head_dim=%d\n",
                num_layers, num_kv_heads, max_seq_len, head_dim);
        return NULL;
    }
    
    kv_cache_t *cache = (kv_cache_t *)malloc(sizeof(kv_cache_t));
    if (!cache) {
        fprintf(stderr, "ERROR: Failed to allocate KV cache structure\n");
        return NULL;
    }
    
    cache->num_layers = num_layers;
    cache->num_kv_heads = num_kv_heads;
    cache->max_seq_len = max_seq_len;
    cache->head_dim = head_dim;
    cache->current_pos = 0;
    
    // Allocate layer arrays
    cache->keys_per_layer = (tensor_t **)malloc(num_layers * sizeof(tensor_t *));
    cache->values_per_layer = (tensor_t **)malloc(num_layers * sizeof(tensor_t *));
    cache->layer_window_size = (int *)malloc(num_layers * sizeof(int));
    cache->layer_is_local = (int *)malloc(num_layers * sizeof(int));
    cache->last_k_snapshot = (float **)malloc(num_layers * sizeof(float *));
    cache->last_v_snapshot = (float **)malloc(num_layers * sizeof(float *));
    cache->last_snapshot_pos = (int *)malloc(num_layers * sizeof(int));
    
    if (!cache->keys_per_layer || !cache->values_per_layer || 
        !cache->layer_window_size || !cache->layer_is_local) {
        fprintf(stderr, "ERROR: Failed to allocate layer arrays\n");
        if (cache->keys_per_layer) free(cache->keys_per_layer);
        if (cache->values_per_layer) free(cache->values_per_layer);
        if (cache->layer_window_size) free(cache->layer_window_size);
        if (cache->layer_is_local) free(cache->layer_is_local);
        if (cache->last_k_snapshot) free(cache->last_k_snapshot);
        if (cache->last_v_snapshot) free(cache->last_v_snapshot);
        if (cache->last_snapshot_pos) free(cache->last_snapshot_pos);
        free(cache);
        return NULL;
    }
    
    // Create tensors for each layer
    // Shape: [num_kv_heads, max_seq_len, head_dim]
    int shape_kv[] = {num_kv_heads, max_seq_len, head_dim};
    
    for (int i = 0; i < num_layers; i++) {
        cache->keys_per_layer[i] = tensor_create(3, shape_kv, DTYPE_F32);
        cache->values_per_layer[i] = tensor_create(3, shape_kv, DTYPE_F32);
        
        if (!cache->keys_per_layer[i] || !cache->values_per_layer[i]) {
            fprintf(stderr, "ERROR: Failed to create tensors for layer %d\n", i);
            for (int j = 0; j < i; j++) {
                tensor_release(cache->keys_per_layer[j]);
                tensor_release(cache->values_per_layer[j]);
            }
            for (int j = 0; j < i; j++) {
                if (cache->last_k_snapshot[j]) free(cache->last_k_snapshot[j]);
                if (cache->last_v_snapshot[j]) free(cache->last_v_snapshot[j]);
            }
            free(cache->last_k_snapshot);
            free(cache->last_v_snapshot);
            free(cache->last_snapshot_pos);
            free(cache->keys_per_layer);
            free(cache->values_per_layer);
            free(cache->layer_window_size);
            free(cache->layer_is_local);
            free(cache);
            return NULL;
        }
        
        // Default: all layers global attention with no window
        cache->layer_is_local[i] = 0;
        cache->layer_window_size[i] = 0;
            // allocate debug snapshots
            int snapshot_size = cache->num_kv_heads * cache->head_dim;
            cache->last_k_snapshot[i] = (float *)calloc(snapshot_size, sizeof(float));
            cache->last_v_snapshot[i] = (float *)calloc(snapshot_size, sizeof(float));
            cache->last_snapshot_pos[i] = -1;
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
    int token_size = cache->num_kv_heads * cache->head_dim;
    
    // Write token to all layers at the same position
    for (int layer = 0; layer < cache->num_layers; layer++) {
        tensor_t *keys = cache->keys_per_layer[layer];
        tensor_t *values = cache->values_per_layer[layer];
        
        // Offset into the 3D tensor [num_kv_heads, max_seq_len, head_dim]
        // Element at [head, pos, dim] has offset: head * (max_seq_len * head_dim) + pos * head_dim + dim
        
        float *keys_data = (float *)tensor_data(keys);
        float *values_data = (float *)tensor_data(values);
        
        for (int head = 0; head < cache->num_kv_heads; head++) {
            int base_offset = head * (cache->max_seq_len * cache->head_dim) + pos * cache->head_dim;
            
            for (int dim = 0; dim < cache->head_dim; dim++) {
                keys_data[base_offset + dim] = k_token[head * cache->head_dim + dim];
                values_data[base_offset + dim] = v_token[head * cache->head_dim + dim];
            }
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

    tensor_t *keys = cache->keys_per_layer[layer];
    tensor_t *values = cache->values_per_layer[layer];

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
    
    printf("KV Cache Info:\n");
    printf("  Layers: %d, KV Heads: %d, Max Seq Len: %d, Head Dim: %d\n",
           cache->num_layers, cache->num_kv_heads, cache->max_seq_len, cache->head_dim);
    printf("  Current Seq Len: %d / %d\n", cache->current_pos, cache->max_seq_len);
    printf("  Per-Layer Config:\n");
    
    for (int i = 0; i < cache->num_layers; i++) {
        if (cache->layer_is_local[i]) {
            printf("    Layer %d: LOCAL (window_size=%d)\n", i, cache->layer_window_size[i]);
        } else {
            printf("    Layer %d: GLOBAL\n", i);
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
    float *k_snap = cache->last_k_snapshot[layer];
    float *v_snap = cache->last_v_snapshot[layer];
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

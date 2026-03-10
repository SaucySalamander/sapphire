/**
 * @file vk_kv_cache.c
 * @brief KV cache buffer allocation and position tracking (Phase 5 ready).
 *
 * Implements:
 * - vk_kv_cache_create/destroy
 * - vk_kv_cache_reset/get_seq_pos/set_seq_pos (position tracking)
 */

#include "../../../../include/vk_buffers.h"
#include "../../../../include/log.h"

#include <stdlib.h>
#include <string.h>

/* Forward declare from vk_buffers.c */
extern int vk_buffer_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    size_t size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    vk_buffer_t *out_buffer
);

extern void vk_buffer_destroy(VkDevice device, vk_buffer_t *buffer);

/* ========================================================================
 * KV Cache
 * ======================================================================== */

int vk_kv_cache_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    const vk_kv_cache_cfg_t *cfg,
    vk_kv_cache_t *out_cache
) {
    if (!device || !phys_dev || !cfg || !out_cache) {
        LOG_ERROR("Invalid parameters (device, phys_dev, cfg, or out_cache is NULL)");
        return -1;
    }

    if (cfg->num_layers <= 0 || cfg->num_kv_heads <= 0 || cfg->max_seq_len <= 0 || cfg->head_dim <= 0) {
        LOG_ERROR("Invalid KV cache dimensions (layers=%d, heads=%d, max_seq=%d, head_dim=%d)",
                  cfg->num_layers, cfg->num_kv_heads, cfg->max_seq_len, cfg->head_dim);
        return -1;
    }

    /* Zero-initialize output */
    memset(out_cache, 0, sizeof(vk_kv_cache_t));

    /* Calculate buffer size: [num_layers × num_kv_heads × max_seq_len × head_dim] floats
     * Each KV pair stored separately, so double the size (K and V).
     */
    size_t kv_elements = (size_t)cfg->num_layers * cfg->num_kv_heads * cfg->max_seq_len * cfg->head_dim * 2;
    size_t kv_size = kv_elements * sizeof(float);

    LOG_DEBUG("Creating KV cache: layers=%d, heads=%d, max_seq=%d, head_dim=%d, size=%zu bytes",
              cfg->num_layers, cfg->num_kv_heads, cfg->max_seq_len, cfg->head_dim, kv_size);

    /* Allocate device-local buffer */
    int rc = vk_buffer_create(
        device, phys_dev, kv_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &out_cache->kv_data
    );
    if (rc != 0) {
        LOG_ERROR("Failed to create KV cache buffer");
        return -1;
    }

    /* Store metadata */
    out_cache->num_layers = cfg->num_layers;
    out_cache->num_kv_heads = cfg->num_kv_heads;
    out_cache->max_seq_len = cfg->max_seq_len;
    out_cache->head_dim = cfg->head_dim;
    out_cache->current_seq_pos = 0;

    LOG_INFO("Created KV cache: %zu bytes (%d layers, %d heads, max_seq=%d, head_dim=%d)",
             kv_size, cfg->num_layers, cfg->num_kv_heads, cfg->max_seq_len, cfg->head_dim);
    return 0;
}

void vk_kv_cache_reset(vk_kv_cache_t *cache) {
    if (!cache) {
        return;
    }
    cache->current_seq_pos = 0;
    LOG_DEBUG("Reset KV cache position to 0");
}

int vk_kv_cache_get_seq_pos(const vk_kv_cache_t *cache) {
    if (!cache) {
        return -1;
    }
    return cache->current_seq_pos;
}

int vk_kv_cache_set_seq_pos(vk_kv_cache_t *cache, int pos) {
    if (!cache) {
        LOG_ERROR("cache is NULL");
        return -1;
    }

    if (pos < 0 || pos > (int)cache->max_seq_len) {
        LOG_ERROR("Invalid sequence position (pos=%d, max_seq_len=%zu)", pos, cache->max_seq_len);
        return -1;
    }

    cache->current_seq_pos = pos;
    LOG_DEBUG("Set KV cache position to %d", pos);
    return 0;
}

void vk_kv_cache_destroy(VkDevice device, vk_kv_cache_t *cache) {
    if (!device || !cache) {
        return;
    }

    vk_buffer_destroy(device, &cache->kv_data);
    memset(cache, 0, sizeof(vk_kv_cache_t));

    LOG_DEBUG("Destroyed KV cache");
}


/**
 * @file vk_scratchpad.c
 * @brief GPU scratchpad buffer allocation for ephemeral activations.
 *
 * Implements:
 * - vk_scratchpad_create/destroy (batch-aware activation buffers)
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
 * GPU Scratchpad Buffers
 * ======================================================================== */

int vk_scratchpad_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    const vk_scratchpad_cfg_t *cfg,
    vk_gpu_scratchpad_t *out_scratchpad
) {
    if (!device || !phys_dev || !cfg || !out_scratchpad) {
        LOG_ERROR("Invalid parameters (device, phys_dev, cfg, or out_scratchpad is NULL)");
        return -1;
    }

    if (cfg->max_batch_size <= 0 || cfg->d_model <= 0 || cfg->d_ff <= 0 || cfg->num_kv_heads <= 0 || cfg->head_dim <= 0) {
        LOG_ERROR("Invalid scratchpad dimensions (batch=%d, d_model=%d, d_ff=%d, heads=%d, head_dim=%d)",
                  cfg->max_batch_size, cfg->d_model, cfg->d_ff, cfg->num_kv_heads, cfg->head_dim);
        return -1;
    }

    /* Zero-initialize output */
    memset(out_scratchpad, 0, sizeof(vk_gpu_scratchpad_t));

    /* Calculate sizes (all buffers sized for max_batch_size).
     *
     * CRITICAL: Q projection output size is num_attention_heads * head_dim per token,
     * NOT num_kv_heads * head_dim.  For GQA models (e.g. Gemma 3 270M: 4 Q heads, 1 KV head)
     * the Q buffer is 4× larger than K/V.  Using num_kv_heads here causes a GPU buffer
     * overflow that corrupts K/V data and produces NaN outputs.
     *
     * We use d_model as a safe upper bound for Q:  d_model >= num_q_heads * head_dim
     * in all standard transformer configurations.
     */
    size_t q_proj_size  = (size_t)cfg->max_batch_size * cfg->d_model * sizeof(float);
    size_t kv_proj_size = (size_t)cfg->max_batch_size * cfg->num_kv_heads * cfg->head_dim * sizeof(float);
    /* attn_output is reused across multiple pipeline stages:
     *   - attention output:   num_q_heads × head_dim  (≤ d_model in practice)
     *   - post_attn_norm out: d_model
     *   - up_proj output:     d_ff  ← LARGEST; must not undersize or GPU overflows into neighbours
     *   - down_proj output:   d_model
     * Allocate max(d_model, d_ff) to cover all usages safely. */
    size_t attn_out_size = (size_t)cfg->max_batch_size
                           * (cfg->d_ff > cfg->d_model ? cfg->d_ff : cfg->d_model)
                           * sizeof(float);
    size_t ffn_intermediate_size = (size_t)cfg->max_batch_size * cfg->d_ff * sizeof(float);
    size_t hidden_size = (size_t)cfg->max_batch_size * cfg->d_model * sizeof(float);
    /* norm_size is intentionally the same as hidden_size (per-layer norm outputs)
     * Use explicit assignment to avoid duplicate-expression warnings. */
    size_t norm_size = hidden_size;

    LOG_DEBUG("Creating GPU scratchpad: batch=%d, d_model=%d, d_ff=%d, kv_heads=%d, head_dim=%d",
              cfg->max_batch_size, cfg->d_model, cfg->d_ff, cfg->num_kv_heads, cfg->head_dim);
    LOG_DEBUG("  q_proj=%zu bytes (d_model bound), k/v_proj=%zu bytes (num_kv_heads×head_dim)",
              q_proj_size, kv_proj_size);

    /* Allocate all buffers (device-local, storage buffer usage)
     * CRITICAL: Add TRANSFER_SRC_BIT to allow vkCmdCopyBuffer from scratchpad to KV cache */
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    int rc = 0;

    /* Q projection buffer: sized for num_attention_heads (via d_model bound) */
    rc |= vk_buffer_create(device, phys_dev, q_proj_size,  usage, flags, &out_scratchpad->q_proj);
    /* K/V projection buffers: sized for num_kv_heads (correct for GQA) */
    rc |= vk_buffer_create(device, phys_dev, kv_proj_size, usage, flags, &out_scratchpad->k_proj);
    rc |= vk_buffer_create(device, phys_dev, kv_proj_size, usage, flags, &out_scratchpad->v_proj);

    /* Attention output */
    rc |= vk_buffer_create(device, phys_dev, attn_out_size, usage, flags, &out_scratchpad->attn_output);

    /* FFN intermediate buffers */
    rc |= vk_buffer_create(device, phys_dev, ffn_intermediate_size, usage, flags, &out_scratchpad->ffn_gate);
    rc |= vk_buffer_create(device, phys_dev, ffn_intermediate_size, usage, flags, &out_scratchpad->ffn_value);

    /* Ping-pong hidden state buffers */
    rc |= vk_buffer_create(device, phys_dev, hidden_size, usage, flags, &out_scratchpad->layer_hidden[0]);
    rc |= vk_buffer_create(device, phys_dev, hidden_size, usage, flags, &out_scratchpad->layer_hidden[1]);

    /* Layer norm output buffer */
    rc |= vk_buffer_create(device, phys_dev, norm_size, usage, flags, &out_scratchpad->norm_buf);

    /* Residual connection buffer (same size as hidden) */
    rc |= vk_buffer_create(device, phys_dev, hidden_size, usage, flags, &out_scratchpad->residual);

    if (rc != 0) {
        LOG_ERROR("Failed to create one or more scratchpad buffers");
        vk_scratchpad_destroy(device, out_scratchpad);
        return -1;
    }

    LOG_INFO("Created GPU scratchpad: batch=%d, total_size=%zu bytes",
             cfg->max_batch_size,
             q_proj_size + kv_proj_size * 2 + attn_out_size + ffn_intermediate_size * 2 + hidden_size * 3 + norm_size);
    return 0;
}

void vk_scratchpad_destroy(VkDevice device, vk_gpu_scratchpad_t *scratchpad) {
    if (!device || !scratchpad) {
        return;
    }

    vk_buffer_destroy(device, &scratchpad->q_proj);
    vk_buffer_destroy(device, &scratchpad->k_proj);
    vk_buffer_destroy(device, &scratchpad->v_proj);
    vk_buffer_destroy(device, &scratchpad->attn_output);
    vk_buffer_destroy(device, &scratchpad->ffn_gate);
    vk_buffer_destroy(device, &scratchpad->ffn_value);
    vk_buffer_destroy(device, &scratchpad->layer_hidden[0]);
    vk_buffer_destroy(device, &scratchpad->layer_hidden[1]);
    vk_buffer_destroy(device, &scratchpad->norm_buf);
    vk_buffer_destroy(device, &scratchpad->residual);

    memset(scratchpad, 0, sizeof(vk_gpu_scratchpad_t));

    LOG_DEBUG("Destroyed GPU scratchpad");
}

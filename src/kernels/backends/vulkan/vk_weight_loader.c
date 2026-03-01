/**
 * @file vk_weight_loader.c
 * @brief Weight buffer loading via staging (host→device transfer).
 *
 * Implements:
 * - vk_weight_buffer_load (Safetensors weights → device-local buffer)
 */

#include "../../../../include/vk_buffers.h"
#include "../../../../include/log.h"

#include <stdlib.h>
#include <string.h>


/* Forward declare helper from vk_buffers.c */
extern int vk_buffer_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    size_t size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    vk_buffer_t *out_buffer
);

extern void vk_buffer_destroy(VkDevice device, vk_buffer_t *buffer);

extern int vk_buffer_stage_data(
    VkDevice device,
    vk_buffer_t *host_visible_buf,
    const void *data,
    size_t size,
    size_t offset_in_buffer
);

/**
 * Helper: Submit one-time command buffer for transfers.
 */
static int submit_transfer_command(
    VkDevice device,
    VkCommandPool cmd_pool,
    VkQueue queue,
    VkCommandBuffer *out_cmd_buf
) {
    VkCommandBufferAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = cmd_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkResult res = vkAllocateCommandBuffers(device, &alloc_info, out_cmd_buf);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkAllocateCommandBuffers failed: %d", res);
        return -1;
    }

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    res = vkBeginCommandBuffer(*out_cmd_buf, &begin_info);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkBeginCommandBuffer failed: %d", res);
        return -1;
    }

    return 0;
}

/**
 * Helper: End and submit one-time command buffer.
 */
static int end_transfer_command(
    VkDevice device,
    VkCommandPool cmd_pool,
    VkQueue queue,
    VkCommandBuffer cmd_buf
) {
    VkResult res = vkEndCommandBuffer(cmd_buf);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkEndCommandBuffer failed: %d", res);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buf);
        return -1;
    }

    /* Create fence for proper GPU synchronization (prevents RADV context loss) */
    VkFenceCreateInfo fence_info = {0};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = 0;  /* Create unsignaled */

    VkFence fence;
    res = vkCreateFence(device, &fence_info, NULL, &fence);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateFence failed: %d", res);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buf);
        return -1;
    }

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;

    res = vkQueueSubmit(queue, 1, &submit_info, fence);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkQueueSubmit failed: %d", res);
        vkDestroyFence(device, fence, NULL);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buf);
        return -1;
    }

    /* Wait for fence (GPU explicitly signals when transfer is complete) */
    res = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkWaitForFences failed: %d", res);
        vkDestroyFence(device, fence, NULL);
        vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buf);
        return -1;
    }

    /* Cleanup (safe now that GPU has signaled completion via fence) */
    vkDestroyFence(device, fence, NULL);
    vkFreeCommandBuffers(device, cmd_pool, 1, &cmd_buf);
    return 0;
}

/* ========================================================================
 * Weight Buffer Loader
 * ======================================================================== */

int vk_weight_buffer_load(
    const vk_transfer_ctx_t *ctx,
    const void *host_weights,
    size_t weight_size,
    vk_buffer_t *out_device_buffer
) {
    if (!ctx || !host_weights || !out_device_buffer) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return -1;
    }

    if (weight_size == 0) {
        LOG_ERROR("Weight size cannot be 0");
        return -1;
    }

    /* Zero-initialize output */
    memset(out_device_buffer, 0, sizeof(vk_buffer_t));

    /* Create staging buffer (host-visible) */
    vk_buffer_t staging = {0};
    int rc = vk_buffer_create(
        ctx->device, ctx->phys_dev, weight_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging
    );
    if (rc != 0) {
        LOG_ERROR("Failed to create staging buffer for weights");
        return -1;
    }

    /* Copy weights to staging buffer */
    rc = vk_buffer_stage_data(ctx->device, &staging, host_weights, weight_size, 0);
    if (rc != 0) {
        LOG_ERROR("Failed to stage weight data");
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Create device-local buffer */
    rc = vk_buffer_create(
        ctx->device, ctx->phys_dev, weight_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        out_device_buffer
    );
    if (rc != 0) {
        LOG_ERROR("Failed to create device-local weight buffer");
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Record copy command */
    VkCommandBuffer cmd_buf;
    if (submit_transfer_command(ctx->device, ctx->cmd_pool, ctx->queue, &cmd_buf) != 0) {
        vk_buffer_destroy(ctx->device, out_device_buffer);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    VkBufferCopy copy_region = {0};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = weight_size;
    vkCmdCopyBuffer(cmd_buf, staging.buffer, out_device_buffer->buffer, 1, &copy_region);

    if (end_transfer_command(ctx->device, ctx->cmd_pool, ctx->queue, cmd_buf) != 0) {
        vk_buffer_destroy(ctx->device, out_device_buffer);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Cleanup staging buffer */
    vk_buffer_destroy(ctx->device, &staging);

    LOG_INFO("Loaded %zu bytes of weights to device-local buffer", weight_size);
    return 0;
}

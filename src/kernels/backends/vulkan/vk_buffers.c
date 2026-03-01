/**
 * @file vk_buffers.c
 * @brief Core Vulkan buffer primitives: allocation, staging, download.
 *
 * Implements:
 * - vk_buffer_create/destroy (generic buffer allocation)
 * - vk_buffer_stage_data (host→device copy via mapping)
 * - vk_buffer_download (device→host copy via staging)
 * - Memory type selection helpers
 */

#include "../../../../include/vk_buffers.h"
#include "../../../../include/log.h"

#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * Memory Type Selection
 * ======================================================================== */

/**
 * Find memory type index matching required properties.
 *
 * @param phys_dev          Physical device
 * @param type_filter       Memory type bits from VkMemoryRequirements
 * @param properties        Required VkMemoryPropertyFlags
 * @param out_type_index    Output memory type index
 *
 * @return 0 on success, -1 if no suitable memory type found
 */
static int find_memory_type(
    VkPhysicalDevice phys_dev,
    uint32_t type_filter,
    VkMemoryPropertyFlags properties,
    uint32_t *out_type_index
) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            *out_type_index = i;
            return 0;
        }
    }

    LOG_ERROR("Failed to find suitable memory type (filter=0x%x, props=0x%x)", type_filter, properties);
    return -1;
}

/* ========================================================================
 * Core Buffer Primitives
 * ======================================================================== */

int vk_buffer_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    size_t size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    vk_buffer_t *out_buffer
) {
    if (!device || !phys_dev || !out_buffer) {
        LOG_ERROR("Invalid parameters (device, phys_dev, or out_buffer is NULL)");
        return -1;
    }

    if (size == 0) {
        LOG_ERROR("Buffer size cannot be 0");
        return -1;
    }

    /* Zero-initialize output */
    memset(out_buffer, 0, sizeof(vk_buffer_t));

    /* Create VkBuffer */
    VkBufferCreateInfo buf_info = {0};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size;
    buf_info.usage = usage;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateBuffer(device, &buf_info, NULL, &out_buffer->buffer);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateBuffer failed: %d", res);
        return -1;
    }

    /* Get memory requirements */
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, out_buffer->buffer, &mem_reqs);

    /* Find suitable memory type */
    uint32_t mem_type_idx;
    if (find_memory_type(phys_dev, mem_reqs.memoryTypeBits, flags, &mem_type_idx) != 0) {
        vkDestroyBuffer(device, out_buffer->buffer, NULL);
        memset(out_buffer, 0, sizeof(vk_buffer_t));
        return -1;
    }

    /* Allocate memory */
    VkMemoryAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type_idx;

    res = vkAllocateMemory(device, &alloc_info, NULL, &out_buffer->memory);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkAllocateMemory failed: %d (size=%zu)", res, (size_t)mem_reqs.size);
        vkDestroyBuffer(device, out_buffer->buffer, NULL);
        memset(out_buffer, 0, sizeof(vk_buffer_t));
        return -1;
    }

    /* Bind memory to buffer */
    res = vkBindBufferMemory(device, out_buffer->buffer, out_buffer->memory, 0);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkBindBufferMemory failed: %d", res);
        vkFreeMemory(device, out_buffer->memory, NULL);
        vkDestroyBuffer(device, out_buffer->buffer, NULL);
        memset(out_buffer, 0, sizeof(vk_buffer_t));
        return -1;
    }

    out_buffer->size = size;
    out_buffer->memory_flags = flags;

    LOG_DEBUG("Created buffer: size=%zu, usage=0x%x, flags=0x%x", size, usage, flags);
    return 0;
}

void vk_buffer_destroy(VkDevice device, vk_buffer_t *buffer) {
    if (!device || !buffer) {
        return;  /* Safe no-op */
    }

    if (buffer->memory) {
        vkFreeMemory(device, buffer->memory, NULL);
    }
    if (buffer->buffer) {
        vkDestroyBuffer(device, buffer->buffer, NULL);
    }

    memset(buffer, 0, sizeof(vk_buffer_t));
}

/* ========================================================================
 * Staging Utilities
 * ======================================================================== */

int vk_buffer_stage_data(
    VkDevice device,
    vk_buffer_t *host_visible_buf,
    const void *data,
    size_t size,
    size_t offset_in_buffer
) {
    if (!device || !host_visible_buf || !data) {
        LOG_ERROR("Invalid parameters (device, host_visible_buf, or data is NULL)");
        return -1;
    }

    if (size == 0) {
        LOG_DEBUG("vk_buffer_stage_data called with size=0, skipping");
        return 0;
    }

    if (offset_in_buffer + size > host_visible_buf->size) {
        LOG_ERROR("Stage data out of bounds (offset=%zu, size=%zu, buf_size=%zu)",
                  offset_in_buffer, size, host_visible_buf->size);
        return -1;
    }

    /* Map memory */
    void *mapped = NULL;
    VkResult res = vkMapMemory(device, host_visible_buf->memory, offset_in_buffer, size, 0, &mapped);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkMapMemory failed: %d", res);
        return -1;
    }

    /* Copy data */
    memcpy(mapped, data, size);

    /* Unmap */
    vkUnmapMemory(device, host_visible_buf->memory);

    LOG_DEBUG("Staged %zu bytes at offset %zu", size, offset_in_buffer);
    return 0;
}

/**
 * Helper: Submit one-time command buffer for transfers.
 */
static int submit_one_time_command(
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
static int end_one_time_command(
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

    /* Wait for fence (GPU explicitly signals when operation is complete) */
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

int vk_buffer_download(
    const vk_transfer_ctx_t *ctx,
    vk_buffer_t *device_buf,
    size_t size,
    void *out_host_data
) {
    if (!ctx || !device_buf || !out_host_data) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return -1;
    }

    if (size == 0 || size > device_buf->size) {
        LOG_ERROR("Invalid download size (size=%zu, buf_size=%zu)", size, device_buf->size);
        return -1;
    }

    /* Create staging buffer (host-visible) */
    vk_buffer_t staging = {0};
    int rc = vk_buffer_create(
        ctx->device, ctx->phys_dev, size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging
    );
    if (rc != 0) {
        return -1;
    }

    /* Record copy command */
    VkCommandBuffer cmd_buf;
    if (submit_one_time_command(ctx->device, ctx->cmd_pool, ctx->queue, &cmd_buf) != 0) {
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    VkBufferCopy copy_region = {0};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(cmd_buf, device_buf->buffer, staging.buffer, 1, &copy_region);

    if (end_one_time_command(ctx->device, ctx->cmd_pool, ctx->queue, cmd_buf) != 0) {
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Map staging buffer and copy to host */
    void *mapped = NULL;
    VkResult res = vkMapMemory(ctx->device, staging.memory, 0, size, 0, &mapped);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkMapMemory failed: %d", res);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    memcpy(out_host_data, mapped, size);
    vkUnmapMemory(ctx->device, staging.memory);

    vk_buffer_destroy(ctx->device, &staging);

    LOG_DEBUG("Downloaded %zu bytes from device buffer", size);
    return 0;
}

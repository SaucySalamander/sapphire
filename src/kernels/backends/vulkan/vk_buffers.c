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
#include <stdint.h>

#include "vk_mem_alloc.h"

static VmaAllocator g_vk_vma_allocator = NULL;

int vk_buffer_set_vma_allocator(VmaAllocator allocator) {
    if (!allocator) {
        LOG_ERROR("vk_buffer_set_vma_allocator: allocator is NULL");
        return -1;
    }
    g_vk_vma_allocator = allocator;
    return 0;
}

void vk_buffer_clear_vma_allocator(void) {
    g_vk_vma_allocator = NULL;
}

static VmaMemoryUsage select_vma_usage(VkMemoryPropertyFlags flags) {
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        return VMA_MEMORY_USAGE_CPU_TO_GPU;
    }
    if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
        return VMA_MEMORY_USAGE_GPU_ONLY;
    }
    return VMA_MEMORY_USAGE_AUTO;
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

    if (!g_vk_vma_allocator) {
        LOG_ERROR("VMA allocator not configured before vk_buffer_create");
        return -1;
    }

    if (size == 0) {
        LOG_ERROR("Buffer size cannot be 0");
        return -1;
    }

    /* Zero-initialize output */
    memset(out_buffer, 0, sizeof(vk_buffer_t));

    VkBufferCreateInfo buf_info = {0};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size;
    buf_info.usage = usage;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info = {0};
    alloc_info.usage = select_vma_usage(flags);
    alloc_info.requiredFlags = flags;

    VmaAllocationInfo vma_alloc_info = {0};
    VkResult res = vmaCreateBuffer(
        g_vk_vma_allocator,
        &buf_info,
        &alloc_info,
        &out_buffer->buffer,
        &out_buffer->allocation,
        &vma_alloc_info
    );
    if (res != VK_SUCCESS) {
        LOG_ERROR("vmaCreateBuffer failed: %d (size=%zu)", (int)res, size);
        memset(out_buffer, 0, sizeof(vk_buffer_t));
        return -1;
    }

    out_buffer->size = size;
    out_buffer->memory_flags = flags;
    out_buffer->memory = vma_alloc_info.deviceMemory;

    LOG_DEBUG("Created buffer: size=%zu, usage=0x%x, flags=0x%x", size, usage, flags);
    return 0;
}

void vk_buffer_destroy(VkDevice device, vk_buffer_t *buffer) {
    if (!device || !buffer) {
        return;  /* Safe no-op */
    }

    if ((buffer->buffer || buffer->allocation) && g_vk_vma_allocator) {
        vmaDestroyBuffer(g_vk_vma_allocator, buffer->buffer, buffer->allocation);
    } else if (buffer->buffer) {
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

    void *mapped = NULL;
    int rc = vk_buffer_map(device, host_visible_buf, offset_in_buffer, size, &mapped);
    if (rc != 0) {
        LOG_ERROR("vk_buffer_map failed for staging");
        return -1;
    }

    /* Copy data */
    memcpy(mapped, data, size);

    vk_buffer_unmap(device, host_visible_buf);

    LOG_DEBUG("Staged %zu bytes at offset %zu", size, offset_in_buffer);
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

    if (!ctx->transfer_cmd || !ctx->transfer_fence ||
        !ctx->download_staging_buffer || !ctx->download_staging_mapped ||
        ctx->download_staging_size < size) {
        LOG_ERROR("Persistent transfer/download resources are not configured");
        return -1;
    }

    VkResult res = vkWaitForFences(ctx->device, 1, &ctx->transfer_fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkWaitForFences failed: %d", (int)res);
        return -1;
    }
    res = vkResetFences(ctx->device, 1, &ctx->transfer_fence);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkResetFences failed: %d", (int)res);
        return -1;
    }
    res = vkResetCommandBuffer(ctx->transfer_cmd, 0);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkResetCommandBuffer failed: %d", (int)res);
        return -1;
    }

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(ctx->transfer_cmd, &begin_info);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkBeginCommandBuffer failed: %d", (int)res);
        return -1;
    }

    VkBufferCopy copy_region = {0};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(ctx->transfer_cmd,
                    device_buf->buffer,
                    ctx->download_staging_buffer,
                    1,
                    &copy_region);

    VkBufferMemoryBarrier copy_to_host = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = ctx->download_staging_buffer,
        .offset = 0,
        .size = size
    };
    vkCmdPipelineBarrier(ctx->transfer_cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        0, NULL,
        1, &copy_to_host,
        0, NULL);

    res = vkEndCommandBuffer(ctx->transfer_cmd);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkEndCommandBuffer failed: %d", (int)res);
        return -1;
    }

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &ctx->transfer_cmd;
    res = vkQueueSubmit(ctx->queue, 1, &submit_info, ctx->transfer_fence);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkQueueSubmit failed: %d", (int)res);
        return -1;
    }

    res = vkWaitForFences(ctx->device, 1, &ctx->transfer_fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkWaitForFences failed: %d", (int)res);
        return -1;
    }

    memcpy(out_host_data, ctx->download_staging_mapped, size);

    LOG_DEBUG("Downloaded %zu bytes from device buffer", size);
    return 0;
}

int vk_buffer_map(
    VkDevice device,
    vk_buffer_t *buffer,
    size_t offset,
    size_t size,
    void **out_mapped
) {
    (void)device;
    if (!buffer || !out_mapped) {
        LOG_ERROR("vk_buffer_map: invalid parameters");
        return -1;
    }
    if (!g_vk_vma_allocator || !buffer->allocation) {
        LOG_ERROR("vk_buffer_map: VMA allocator/allocation unavailable");
        return -1;
    }
    if (offset > buffer->size) {
        LOG_ERROR("vk_buffer_map: offset out of bounds (offset=%zu, size=%zu)", offset, buffer->size);
        return -1;
    }
    if (size > 0 && (offset + size > buffer->size)) {
        LOG_ERROR("vk_buffer_map: range out of bounds (offset=%zu, size=%zu, buf_size=%zu)",
                  offset, size, buffer->size);
        return -1;
    }

    void *base = NULL;
    VkResult res = vmaMapMemory(g_vk_vma_allocator, buffer->allocation, &base);
    if (res != VK_SUCCESS || !base) {
        LOG_ERROR("vmaMapMemory failed: %d", (int)res);
        return -1;
    }

    *out_mapped = (void *)((uint8_t *)base + offset);
    return 0;
}

void vk_buffer_unmap(
    VkDevice device,
    vk_buffer_t *buffer
) {
    (void)device;
    if (!buffer || !buffer->allocation || !g_vk_vma_allocator) {
        return;
    }
    vmaUnmapMemory(g_vk_vma_allocator, buffer->allocation);
}

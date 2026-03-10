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

    if (!ctx->transfer_cmd || !ctx->transfer_fence ||
        !ctx->upload_staging_buffer || !ctx->upload_staging_mapped ||
        ctx->upload_staging_size < weight_size) {
        LOG_ERROR("Persistent upload staging resources are not configured");
        return -1;
    }

    memcpy(ctx->upload_staging_mapped, host_weights, weight_size);

    /* Create device-local buffer */
    int rc = vk_buffer_create(
        ctx->device, ctx->phys_dev, weight_size,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        out_device_buffer
    );
    if (rc != 0) {
        LOG_ERROR("Failed to create device-local weight buffer");
        return -1;
    }

    VkResult res = vkWaitForFences(ctx->device, 1, &ctx->transfer_fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkWaitForFences failed: %d", (int)res);
        vk_buffer_destroy(ctx->device, out_device_buffer);
        return -1;
    }
    res = vkResetFences(ctx->device, 1, &ctx->transfer_fence);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkResetFences failed: %d", (int)res);
        vk_buffer_destroy(ctx->device, out_device_buffer);
        return -1;
    }
    res = vkResetCommandBuffer(ctx->transfer_cmd, 0);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkResetCommandBuffer failed: %d", (int)res);
        vk_buffer_destroy(ctx->device, out_device_buffer);
        return -1;
    }

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    res = vkBeginCommandBuffer(ctx->transfer_cmd, &begin_info);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkBeginCommandBuffer failed: %d", (int)res);
        vk_buffer_destroy(ctx->device, out_device_buffer);
        return -1;
    }

    VkBufferCopy copy_region = {0};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = weight_size;
    vkCmdCopyBuffer(ctx->transfer_cmd,
                    ctx->upload_staging_buffer,
                    out_device_buffer->buffer,
                    1,
                    &copy_region);

    res = vkEndCommandBuffer(ctx->transfer_cmd);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkEndCommandBuffer failed: %d", (int)res);
        vk_buffer_destroy(ctx->device, out_device_buffer);
        return -1;
    }

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &ctx->transfer_cmd;
    res = vkQueueSubmit(ctx->queue, 1, &submit_info, ctx->transfer_fence);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkQueueSubmit failed: %d", (int)res);
        vk_buffer_destroy(ctx->device, out_device_buffer);
        return -1;
    }

    res = vkWaitForFences(ctx->device, 1, &ctx->transfer_fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkWaitForFences failed: %d", (int)res);
        vk_buffer_destroy(ctx->device, out_device_buffer);
        return -1;
    }

    LOG_INFO("Loaded %zu bytes of weights to device-local buffer", weight_size);
    return 0;
}

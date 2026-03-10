/**
 * @file vk_sync.c
 * @brief Vulkan synchronization primitives for GPU inter-shader coordination.
 *
 * Implements:
 * - vk_buffer_barrier (single-buffer synchronization)
 * - vk_pipeline_barrier_compute (multi-buffer synchronization)
 */

#include "../../../../include/vk_buffers.h"
#include "../../../../include/log.h"

#include <string.h>

/* ========================================================================
 * Pipeline Barrier Helper
 * ======================================================================== */

void vk_buffer_barrier(
    VkCommandBuffer cmd,
    VkBuffer buf,
    VkAccessFlags src,
    VkAccessFlags dst,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage
) {
    if (!cmd || buf == VK_NULL_HANDLE) {
        LOG_ERROR("vk_buffer_barrier: invalid command buffer or buffer handle");
        return;
    }

    VkBufferMemoryBarrier bb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = src,
        .dstAccessMask = dst,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buf,
        .offset = 0,
        .size = VK_WHOLE_SIZE
    };

    vkCmdPipelineBarrier(cmd,
                         srcStage,
                         dstStage,
                         0,
                         0,
                         NULL,
                         1,
                         &bb,
                         0,
                         NULL);
}

void vk_pipeline_barrier_compute(
    VkCommandBuffer cmd_buf,
    const VkBuffer *buffers,
    uint32_t buffer_count,
    const vk_barrier_cfg_t *cfg
) {
    if (!cmd_buf || !buffers || buffer_count == 0 || !cfg) {
        LOG_ERROR("vk_pipeline_barrier_compute: invalid arguments");
        return;
    }

    if (buffer_count <= 8u) {
        VkBufferMemoryBarrier barriers[8] = {0};
        for (uint32_t i = 0; i < buffer_count; i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[i].srcAccessMask = cfg->src_access;
            barriers[i].dstAccessMask = cfg->dst_access;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].buffer = buffers[i];
            barriers[i].offset = 0;
            barriers[i].size = VK_WHOLE_SIZE;
        }

        vkCmdPipelineBarrier(cmd_buf,
                             cfg->src_stage,
                             cfg->dst_stage,
                             0,
                             0,
                             NULL,
                             buffer_count,
                             barriers,
                             0,
                             NULL);
        return;
    }

    for (uint32_t i = 0; i < buffer_count; i++) {
        vk_buffer_barrier(cmd_buf,
                          buffers[i],
                          cfg->src_access,
                          cfg->dst_access,
                          cfg->src_stage,
                          cfg->dst_stage);
    }
}

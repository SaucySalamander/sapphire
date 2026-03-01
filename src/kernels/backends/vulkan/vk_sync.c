/**
 * @file vk_sync.c
 * @brief Vulkan synchronization primitives for GPU inter-shader coordination.
 *
 * Implements:
 * - vk_pipeline_barrier_compute (GPU shader chain synchronization)
 */

#include "../../../../include/vk_buffers.h"
#include "../../../../include/log.h"

#include <string.h>

/* ========================================================================
 * Pipeline Barrier Helper
 * ======================================================================== */

void vk_pipeline_barrier_compute(
    VkCommandBuffer cmd_buf,
    VkPipelineStageFlags src_stage,
    VkPipelineStageFlags dst_stage,
    VkAccessFlags src_access,
    VkAccessFlags dst_access
) {
    if (!cmd_buf) {
        LOG_ERROR("cmd_buf is NULL");
        return;
    }

    VkMemoryBarrier memory_barrier = {0};
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = src_access;
    memory_barrier.dstAccessMask = dst_access;

    vkCmdPipelineBarrier(
        cmd_buf,
        src_stage,
        dst_stage,
        0,  /* No dependency flags */
        1,  /* Memory barrier count */
        &memory_barrier,
        0,  /* No buffer barriers */
        NULL,
        0,  /* No image barriers */
        NULL
    );

    LOG_DEBUG("Inserted pipeline barrier: src_stage=0x%x, dst_stage=0x%x, src_access=0x%x, dst_access=0x%x",
              src_stage, dst_stage, src_access, dst_access);
}

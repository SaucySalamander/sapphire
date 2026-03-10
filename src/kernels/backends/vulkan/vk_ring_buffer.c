/**
 * @file vk_ring_buffer.c
 * @brief Fast persistent mapped streaming buffer with timeline semaphore.
 *
 * Public API remains ring-buffer compatible, but the implementation uses a
 * single persistent HOST_VISIBLE slot optimized for decode (batch=1).
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
 * Single-slot Persistent Streaming Buffer
 * ======================================================================== */

#define VK_STREAMING_SLOTS 1u

int vk_ring_buffer_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    size_t num_slots,
    size_t slot_size,
    vk_ring_buffer_t *out_ring
) {
    if (!device || !phys_dev || !out_ring) {
        LOG_ERROR("Invalid parameters (device, phys_dev, or out_ring is NULL)");
        return -1;
    }

    if (slot_size == 0) {
        LOG_ERROR("Invalid ring buffer dimensions (num_slots=%zu, slot_size=%zu)", num_slots, slot_size);
        return -1;
    }

    /* Zero-initialize output */
    memset(out_ring, 0, sizeof(vk_ring_buffer_t));

    out_ring->num_slots = VK_STREAMING_SLOTS;
    out_ring->slot_size = slot_size;
    out_ring->cpu_write_idx = 0;
    out_ring->gpu_read_idx = 0;
    out_ring->generation_stride = 1;

    out_ring->slots = calloc(out_ring->num_slots, sizeof(vk_buffer_t));
    if (!out_ring->slots) {
        LOG_ERROR("Failed to allocate slot array");
        return -1;
    }

    out_ring->mapped_slots = calloc(out_ring->num_slots, sizeof(void *));
    if (!out_ring->mapped_slots) {
        LOG_ERROR("Failed to allocate mapped slot array");
        free(out_ring->slots);
        memset(out_ring, 0, sizeof(vk_ring_buffer_t));
        return -1;
    }

    out_ring->timeline_semaphore = calloc(out_ring->num_slots, sizeof(VkSemaphore));
    if (!out_ring->timeline_semaphore) {
        LOG_ERROR("Failed to allocate timeline semaphore array");
        free(out_ring->mapped_slots);
        free(out_ring->slots);
        memset(out_ring, 0, sizeof(vk_ring_buffer_t));
        return -1;
    }

    out_ring->timeline_counter = calloc(out_ring->num_slots, sizeof(uint64_t));
    if (!out_ring->timeline_counter) {
        LOG_ERROR("Failed to allocate timeline counter array");
        free(out_ring->timeline_semaphore);
        free(out_ring->mapped_slots);
        free(out_ring->slots);
        memset(out_ring, 0, sizeof(vk_ring_buffer_t));
        return -1;
    }

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    for (size_t i = 0; i < out_ring->num_slots; i++) {
        int rc = vk_buffer_create(device, phys_dev, slot_size, usage, flags, &out_ring->slots[i]);
        if (rc != 0) {
            LOG_ERROR("Failed to create ring buffer slot %zu", i);
            vk_ring_buffer_destroy(device, out_ring);
            return -1;
        }

        void *mapped = NULL;
        if (vk_buffer_map(device, &out_ring->slots[i], 0, out_ring->slots[i].size, &mapped) != 0 || !mapped) {
            LOG_ERROR("Failed to persistently map ring buffer slot %zu", i);
            vk_ring_buffer_destroy(device, out_ring);
            return -1;
        }
        out_ring->mapped_slots[i] = mapped;
    }

    VkSemaphoreTypeCreateInfo timeline_info = {0};
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_info.initialValue = 0;

    VkSemaphoreCreateInfo sem_info = {0};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sem_info.pNext = &timeline_info;

    for (size_t i = 0; i < out_ring->num_slots; i++) {
        VkResult res = vkCreateSemaphore(device, &sem_info, NULL, &out_ring->timeline_semaphore[i]);
        if (res != VK_SUCCESS) {
            LOG_ERROR("Failed to create timeline semaphore for slot %zu: %d", i, res);
            vk_ring_buffer_destroy(device, out_ring);
            return -1;
        }
        out_ring->timeline_counter[i] = 0;
    }

    LOG_INFO("Created streaming buffer: requested_slots=%zu using_slots=%u slot_size=%zu",
             num_slots, (unsigned)VK_STREAMING_SLOTS, slot_size);
    return 0;
}

/* ========================================================================
 * CPU Producer (Hot Loop Operations)
 * ======================================================================== */

int vk_ring_buffer_acquire_write(
    VkDevice device,
    vk_ring_buffer_t *ring,
    void **out_mapped_slot,
    size_t *out_slot_index,
    uint64_t *out_write_counter
) {
    if (!device || !ring || !out_mapped_slot || !out_slot_index || !out_write_counter) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return -1;
    }

    (void)device;
    size_t slot_idx = 0;
    if (!ring->mapped_slots || !ring->mapped_slots[slot_idx]) {
        LOG_ERROR("Persistent mapped slot is unavailable");
        return -1;
    }

    *out_mapped_slot = ring->mapped_slots[slot_idx];
    *out_slot_index = slot_idx;
    *out_write_counter = ring->timeline_counter[slot_idx] + 1;

    return 0;
}

int vk_ring_buffer_commit_write(
    vk_ring_buffer_t *ring,
    uint64_t counter_value
) {
    if (!ring) {
        LOG_ERROR("ring is NULL");
        return -1;
    }

    size_t slot_idx = 0;
    ring->timeline_counter[slot_idx] = counter_value;
    ring->cpu_write_idx++;
    return 0;
}

/* ========================================================================
 * GPU Consumer (Hot Loop Operations)
 * ======================================================================== */

int vk_ring_buffer_acquire_read(
    vk_ring_buffer_t *ring,
    vk_buffer_t **out_slot_buffer,
    VkSemaphore *out_timeline_semaphore,
    uint64_t *out_wait_counter,
    size_t *out_slot_index
) {
    if (!ring || !out_slot_buffer || !out_timeline_semaphore || !out_wait_counter || !out_slot_index) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return -1;
    }

    size_t slot_idx = 0;
    if (ring->timeline_counter[slot_idx] == 0) {
        return -1;
    }

    *out_slot_buffer = &ring->slots[slot_idx];
    *out_timeline_semaphore = ring->timeline_semaphore[slot_idx];
    *out_wait_counter = ring->timeline_counter[slot_idx];
    *out_slot_index = slot_idx;
    return 0;
}

int vk_ring_buffer_commit_read(
    VkDevice device,
    vk_ring_buffer_t *ring
) {
    if (!device || !ring) {
        LOG_ERROR("Invalid parameters (device or ring is NULL)");
        return -1;
    }

    size_t slot_idx = 0;
    VkSemaphoreSignalInfo signal_info = {0};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = ring->timeline_semaphore[slot_idx];
    signal_info.value = ring->timeline_counter[slot_idx];

    VkResult res = vkSignalSemaphore(device, &signal_info);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkSignalSemaphore failed for read commit (slot=%zu): %d", slot_idx, res);
        return -1;
    }

    ring->gpu_read_idx++;
    return 0;
}

/* ========================================================================
 * Shutdown
 * ======================================================================== */

void vk_ring_buffer_destroy(VkDevice device, vk_ring_buffer_t *ring) {
    if (!device || !ring) {
        return;
    }

    if (ring->timeline_semaphore) {
        for (size_t i = 0; i < ring->num_slots; i++) {
            if (ring->timeline_semaphore[i]) {
                vkDestroySemaphore(device, ring->timeline_semaphore[i], NULL);
            }
        }
        free(ring->timeline_semaphore);
    }

    if (ring->mapped_slots && ring->slots) {
        for (size_t i = 0; i < ring->num_slots; i++) {
            if (ring->mapped_slots[i]) {
                vk_buffer_unmap(device, &ring->slots[i]);
            }
        }
        free(ring->mapped_slots);
    }

    if (ring->slots) {
        for (size_t i = 0; i < ring->num_slots; i++) {
            vk_buffer_destroy(device, &ring->slots[i]);
        }
        free(ring->slots);
    }

    if (ring->timeline_counter) {
        free(ring->timeline_counter);
    }

    memset(ring, 0, sizeof(vk_ring_buffer_t));

    LOG_DEBUG("Destroyed streaming buffer");
}

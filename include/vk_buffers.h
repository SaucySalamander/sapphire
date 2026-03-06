/**
 * @file vk_buffers.h
 * @brief Vulkan buffer management for indirect inference pipeline.
 *
 * Provides type-safe buffer abstractions for:
 * - Static weight buffers (device-local, read-only)
 * - Persistent KV cache buffers (device-local, position-tracked)
 * - Ephemeral GPU scratchpad buffers (device-local activations)
 * - Ring buffers for CPU↔GPU producer-consumer streaming (timeline semaphores)
 *
 * All allocation happens at model initialization. Zero malloc/free in inference hot loop.
 */

#ifndef VK_BUFFERS_H
#define VK_BUFFERS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Include Vulkan headers */
#include <vulkan/vulkan.h>

/* Forward declarations for Vulkan Memory Allocator (VMA) opaque handles. */
typedef struct VmaAllocator_T* VmaAllocator;
typedef struct VmaAllocation_T* VmaAllocation;

/* ========================================================================
 * Config Structs (for parameter reduction)
 * ======================================================================== */

/** Transfer context for buffer staging (weight loading, download, etc.) */
typedef struct {
    VkDevice device;
    VkPhysicalDevice phys_dev;
    VkCommandPool cmd_pool;
    VkQueue queue;
    VkCommandBuffer transfer_cmd;
    VkFence transfer_fence;
    VkBuffer upload_staging_buffer;
    size_t upload_staging_size;
    void *upload_staging_mapped;
    VkBuffer download_staging_buffer;
    size_t download_staging_size;
    void *download_staging_mapped;
} vk_transfer_ctx_t;

/** KV cache configuration for allocation */
typedef struct {
    int num_layers;
    int num_kv_heads;
    int max_seq_len;
    int head_dim;
} vk_kv_cache_cfg_t;

/** GPU scratchpad configuration for allocation */
typedef struct {
    int max_batch_size;
    int d_model;
    int d_ff;
    int num_kv_heads;
    int head_dim;
} vk_scratchpad_cfg_t;

/** Multi-buffer pipeline barrier configuration. */
typedef struct {
    VkPipelineStageFlags src_stage;
    VkPipelineStageFlags dst_stage;
    VkAccessFlags src_access;
    VkAccessFlags dst_access;
} vk_barrier_cfg_t;

/* ========================================================================
 * 5.1 Core Buffer Wrapper
 * ======================================================================== */

/**
 * Generic Vulkan buffer with metadata.
 *
 * Ownership: Caller owns buffer; must call vk_buffer_destroy at shutdown.
 * All buffers are allocated at model initialization, never resized.
 */
typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VmaAllocation allocation;
    size_t size;
    VkMemoryPropertyFlags memory_flags;  /* Device-local? Host-visible? Coherent? */
} vk_buffer_t;

/* Set/clear active VMA allocator used by vk_buffer_create/destroy/map helpers. */
int vk_buffer_set_vma_allocator(VmaAllocator allocator);
void vk_buffer_clear_vma_allocator(void);

/* Map/unmap buffer memory through VMA. */
int vk_buffer_map(
    VkDevice device,
    vk_buffer_t *buffer,
    size_t offset,
    size_t size,
    void **out_mapped
);

void vk_buffer_unmap(
    VkDevice device,
    vk_buffer_t *buffer
);

/**
 * Create a Vulkan buffer with specified usage and memory properties.
 *
 * Allocates VkBuffer and binds VkDeviceMemory. Memory type selected based on flags.
 * Called only at model initialization, never during inference loop.
 *
 * @param device      Vulkan logical device
 * @param phys_dev    Physical device (for memory type selection)
 * @param size        Buffer size in bytes
 * @param usage       VkBufferUsageFlags (e.g., VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
 * @param flags       VkMemoryPropertyFlags (e.g., VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
 * @param out_buffer  Output buffer (caller owns)
 *
 * @return 0 on success, -1 on error (errors logged via LOG_ERROR)
 */
int vk_buffer_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    size_t size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags flags,
    vk_buffer_t *out_buffer
);

/**
 * Destroy a Vulkan buffer and free memory.
 *
 * Called only at model shutdown. Safe to call on zero-initialized struct.
 *
 * @param device  Vulkan logical device
 * @param buffer  Buffer to destroy (can be zero-initialized)
 */
void vk_buffer_destroy(VkDevice device, vk_buffer_t *buffer);

/* ========================================================================
 * 5.2 Weight Buffer Loader
 * ======================================================================== */

/**
 * Load model weights from host memory to device-local buffer via staging.
 *
 * Allocates temporary staging buffer (host-visible), copies weights, transfers
 * to device-local buffer via VkCmdCopyBuffer, waits for completion, frees staging.
 *
 * Called once per weight tensor at model initialization.
 *
 * @param ctx              Transfer context (device, phys_dev, cmd_pool, queue)
 * @param host_weights     Host memory pointer to weight data
 * @param weight_size      Size of weight data in bytes
 * @param out_device_buffer  Output device-local buffer (caller owns)
 *
 * @return 0 on success, -1 on error (errors logged via LOG_ERROR)
 */
int vk_weight_buffer_load(
    const vk_transfer_ctx_t *ctx,
    const void *host_weights,
    size_t weight_size,
    vk_buffer_t *out_device_buffer
);

/* ========================================================================
 * 5.3 GPU Scratchpad Buffers (Ephemeral Activations)
 * ======================================================================== */

/**
 * Pre-allocated scratchpad buffers for layer-to-layer activations.
 *
 * All buffers sized for max_batch_size; no resizing during inference.
 * Ping-pong hidden states between layers; reused on every forward pass.
 */
typedef struct {
    vk_buffer_t q_proj;              /* [batch_size, num_attention_heads * head_dim] — sized via d_model bound */
    vk_buffer_t k_proj;              /* [batch_size, num_kv_heads * head_dim] */
    vk_buffer_t v_proj;              /* [batch_size, num_kv_heads * head_dim] */
    vk_buffer_t attn_output;         /* [batch_size, d_model] */
    vk_buffer_t ffn_gate;            /* [batch_size, d_ff] */
    vk_buffer_t ffn_value;           /* [batch_size, d_ff] */
    vk_buffer_t layer_hidden[2];     /* Ping-pong: [batch_size, d_model] */
    vk_buffer_t norm_buf;            /* Layer norm output [batch_size, d_model] */
    vk_buffer_t residual;            /* Residual connection buffer [batch_size, d_model] */
} vk_gpu_scratchpad_t;

/**
 * Allocate all scratchpad buffers for a given batch size and model config.
 *
 * All buffers device-local. Called once at model initialization.
 *
 * @param device    Vulkan logical device
 * @param phys_dev  Physical device
 * @param cfg       Scratchpad configuration (batch, dims, heads)
 * @param out_scratchpad   Output scratchpad (caller owns)
 *
 * @return 0 on success, -1 on error (errors logged via LOG_ERROR)
 */
int vk_scratchpad_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    const vk_scratchpad_cfg_t *cfg,
    vk_gpu_scratchpad_t *out_scratchpad
);

/**
 * Destroy all scratchpad buffers.
 *
 * Called only at model shutdown.
 *
 * @param device      Vulkan logical device
 * @param scratchpad  Scratchpad to destroy
 */
void vk_scratchpad_destroy(VkDevice device, vk_gpu_scratchpad_t *scratchpad);

/* ========================================================================
 * 5.4 KV Cache Buffer (Persistent Across Frames)
 * ======================================================================== */

/**
 * KV cache buffer for all layers with position tracking (Phase 5 ready).
 *
 * Persistent across all forward passes in a session (autoregressive).
 * Position tracked CPU-side; ready for paging/eviction interface in Phase 5.
 */
typedef struct {
    vk_buffer_t kv_data;             /* [num_layers × num_kv_heads × max_seq_len × head_dim] floats */
    size_t num_layers;
    size_t num_kv_heads;
    size_t max_seq_len;
    size_t head_dim;
    int current_seq_pos;             /* Shared position across all layers */
} vk_kv_cache_t;

/**
 * Allocate KV cache matching model configuration.
 *
 * Buffer is device-local. Position tracking is CPU-visible.
 * Called once at model initialization.
 *
 * @param device    Vulkan logical device
 * @param phys_dev  Physical device
 * @param cfg       KV cache configuration (layers, heads, seq_len, head_dim)
 * @param out_cache Output KV cache (caller owns)
 *
 * @return 0 on success, -1 on error (errors logged via LOG_ERROR)
 */
int vk_kv_cache_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    const vk_kv_cache_cfg_t *cfg,
    vk_kv_cache_t *out_cache
);

/**
 * Reset KV cache position to 0 for new session/sequence.
 *
 * @param cache  KV cache to reset
 */
void vk_kv_cache_reset(vk_kv_cache_t *cache);

/**
 * Get current sequence position (CPU side, for Phase 5 paging).
 *
 * @param cache  KV cache
 * @return Current sequence position
 */
int vk_kv_cache_get_seq_pos(const vk_kv_cache_t *cache);

/**
 * Set sequence position (used when loading checkpointed state in Phase 5).
 *
 * @param cache  KV cache
 * @param pos    New sequence position
 * @return 0 on success, -1 if pos exceeds max_seq_len
 */
int vk_kv_cache_set_seq_pos(vk_kv_cache_t *cache, int pos);

/**
 * Destroy KV cache buffer.
 *
 * Called only at shutdown (or before checkpointing in Phase 5).
 *
 * @param device  Vulkan logical device
 * @param cache   KV cache to destroy
 */
void vk_kv_cache_destroy(VkDevice device, vk_kv_cache_t *cache);

/* ========================================================================
 * 5.5 Ring Buffers (Token Input & Output)
 * ======================================================================== */

/**
 * Ring buffer for producer-consumer streaming (CPU↔GPU).
 *
 * Uses timeline semaphores (Vulkan 1.2) for frame synchronization.
 * All slots and semaphores pre-allocated at creation; zero alloc in hot loop.
 */
typedef struct {
    vk_buffer_t *slots;                  /* Array of num_slots pre-allocated buffers */
    void **mapped_slots;                 /* Persistent mapped pointers per slot */
    size_t num_slots;
    size_t slot_size;
    VkSemaphore *timeline_semaphore;    /* Per-slot timeline semaphore (pre-allocated) */
    uint64_t *timeline_counter;         /* Per-slot current counter value (pre-allocated) */
    size_t cpu_write_idx;               /* Current CPU write position (mod num_slots) */
    size_t gpu_read_idx;                /* Current GPU read position (mod num_slots) */
    uint64_t generation_stride;         /* Counter increment per frame */
} vk_ring_buffer_t;

/**
 * Create ring buffer with all slots and timeline semaphores pre-allocated.
 *
 * Called at model initialization, NOT during inference loop.
 * All timeline counters initialized to 0.
 *
 * @param device      Vulkan logical device
 * @param phys_dev    Physical device
 * @param num_slots   Number of ring buffer slots (pipeline depth)
 * @param slot_size   Size of each slot in bytes
 * @param out_ring    Output ring buffer (caller owns)
 *
 * @return 0 on success, -1 on error (errors logged via LOG_ERROR)
 */
int vk_ring_buffer_create(
    VkDevice device,
    VkPhysicalDevice phys_dev,
    size_t num_slots,
    size_t slot_size,
    vk_ring_buffer_t *out_ring
);

/**
 * Acquire write slot for CPU (blocking wait on timeline semaphore).
 *
 * HOT LOOP OPERATION: Zero allocation, zero GPU submission.
 * Waits for GPU to finish reading slot (timeline counter check).
 *
 * @param device            Vulkan logical device
 * @param ring              Ring buffer
 * @param out_mapped_slot   Pointer to host-mapped slot (if host-visible)
 * @param out_slot_index    Slot index acquired
 * @param out_write_counter Counter value this write will signal
 *
 * @return 0 on success (slot acquired), -1 on timeout/error
 */
int vk_ring_buffer_acquire_write(
    VkDevice device,
    vk_ring_buffer_t *ring,
    void **out_mapped_slot,
    size_t *out_slot_index,
    uint64_t *out_write_counter
);

/**
 * Commit write and signal completion (advance write pointer).
 *
 * HOT LOOP OPERATION: Zero allocation, timeline signal + index arithmetic only.
 *
 * @param ring           Ring buffer
 * @param counter_value  Counter value to signal (from acquire_write)
 *
 * @return 0 on success, -1 on error
 */
int vk_ring_buffer_commit_write(
    vk_ring_buffer_t *ring,
    uint64_t counter_value
);

/**
 * Acquire read slot for GPU (non-blocking check).
 *
 * HOT LOOP OPERATION: Zero allocation.
 * Returns semaphore and counter for GPU queue wait (deferred GPU wait).
 *
 * @param ring                    Ring buffer
 * @param out_slot_buffer         Slot buffer for GPU
 * @param out_timeline_semaphore  Semaphore to wait on in GPU queue
 * @param out_wait_counter        Counter value GPU should wait for
 * @param out_slot_index          Slot index acquired
 *
 * @return 0 if read slot available (counter ready), -1 if not yet ready
 */
int vk_ring_buffer_acquire_read(
    vk_ring_buffer_t *ring,
    vk_buffer_t **out_slot_buffer,
    VkSemaphore *out_timeline_semaphore,
    uint64_t *out_wait_counter,
    size_t *out_slot_index
);

/**
 * Commit GPU read (signal GPU done, advance read pointer).
 *
 * HOT LOOP OPERATION: Zero allocation, index arithmetic and timeline signal only.
 *
 * @param device  Vulkan logical device
 * @param ring    Ring buffer
 *
 * @return 0 on success, -1 on error
 */
int vk_ring_buffer_commit_read(
    VkDevice device,
    vk_ring_buffer_t *ring
);

/**
 * Destroy ring buffer and free all slots and semaphores.
 *
 * Called only at model shutdown, NOT during inference.
 * Caller responsible for draining any pending GPU work.
 *
 * @param device  Vulkan logical device
 * @param ring    Ring buffer to destroy
 */
void vk_ring_buffer_destroy(VkDevice device, vk_ring_buffer_t *ring);

/* ========================================================================
 * 5.6 Staging Utilities
 * ======================================================================== */

/**
 * Map host-visible buffer, copy data, unmap.
 *
 * Safe to call repeatedly; handles map/unmap internally.
 *
 * @param device             Vulkan logical device
 * @param host_visible_buf   Host-visible buffer
 * @param data               Data to copy
 * @param size               Size in bytes
 * @param offset_in_buffer   Offset within buffer
 *
 * @return 0 on success, -1 on error (errors logged via LOG_ERROR)
 */
int vk_buffer_stage_data(
    VkDevice device,
    vk_buffer_t *host_visible_buf,
    const void *data,
    size_t size,
    size_t offset_in_buffer
);

/**
 * Copy device buffer to host via staging (for verification/debugging).
 *
 * Allocates temporary staging buffer, transfers via VkCmdCopyBuffer,
 * waits for completion, copies to host, frees staging.
 *
 * @param ctx           Transfer context (device, phys_dev, cmd_pool, queue)
 * @param device_buf    Device buffer to download
 * @param size          Size in bytes
 * @param out_host_data Host memory to copy into (caller-allocated)
 *
 * @return 0 on success, -1 on error (errors logged via LOG_ERROR)
 */
int vk_buffer_download(
    const vk_transfer_ctx_t *ctx,
    vk_buffer_t *device_buf,
    size_t size,
    void *out_host_data
);

/* ========================================================================
 * 5.7 Buffer Barrier Helpers (GPU Inter-Shader Synchronization)
 * ======================================================================== */

/**
 * Insert a buffer-scoped memory dependency.
 *
 * Synchronizes accesses for one specific VkBuffer without introducing
 * global memory barriers.
 */
void vk_buffer_barrier(
    VkCommandBuffer cmd,
    VkBuffer buf,
    VkAccessFlags src,
    VkAccessFlags dst,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage
);

/**
 * Insert buffer-scoped barriers for one or more buffers.
 *
 * Convenience wrapper over vkCmdPipelineBarrier that emits VkBufferMemoryBarrier
 * entries for each buffer in `buffers[0..buffer_count)` with identical access and
 * stage masks.
 *
 * @param cmd_buf     Command buffer
 * @param buffers     Buffer array
 * @param buffer_count Number of buffers in `buffers`
 * @param src_stage   Source pipeline stage
 * @param dst_stage   Destination pipeline stage
 * @param src_access  Source access mask
 * @param dst_access  Destination access mask
 */
void vk_pipeline_barrier_compute(
    VkCommandBuffer cmd_buf,
    const VkBuffer *buffers,
    uint32_t buffer_count,
    const vk_barrier_cfg_t *cfg
);

#ifdef __cplusplus
}
#endif

#endif /* VK_BUFFERS_H */

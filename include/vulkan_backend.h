/**
 * @file vulkan_backend.h
 * @brief Vulkan GPU backend runtime and device initialization.
 *
 * Provides opaque Vulkan context management for GPU-accelerated inference.
 * Handles instance creation, GPU device enumeration, logical device creation,
 * and command infrastructure allocation. Gracefully falls back to CPU on
 * systems without Vulkan support.
 */

#ifndef VULKAN_BACKEND_H
#define VULKAN_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version constants */
#define SAPPHIRE_VK_MAJOR 1
#define SAPPHIRE_VK_MINOR 4

/**
 * Opaque Vulkan backend context.
 *
 * Encapsulates all Vulkan runtime state including device, queues, and
 * command infrastructure. Allocated and owned by vk_backend_init(),
 * must be freed by vk_backend_shutdown().
 *
 * Implementation details hidden from consumers. Stored opaquely in
 * inference_session_t.backend_data.
 */
typedef struct vk_backend_context_t vk_backend_context_t;

/**
 * Initialize Vulkan backend.
 *
 * Performs complete Vulkan bootstrap:
 * 1. Creates VkInstance with optional validation layers (debug builds only)
 * 2. Enumerates physical devices and scores them based on:
 *    - Compute queue family availability (required)
 *    - GPU type (discrete preferred over integrated)
 *    - Compute queue count (prefer 1+ exclusive compute queue)
 * 3. Creates VkDevice on selected physical device with single compute queue
 * 4. Allocates VkCommandPool for async compute command recording
 * 5. Detects GPU capabilities (descriptor indexing, push descriptors)
 * 6. Queries GPU memory available
 *
 * On systems without Vulkan or with no suitable GPU:
 * - Returns NULL and logs detailed reason (INFO/WARN level)
 * - Caller detects NULL and falls back to CPU backend transparently
 *
 * @param ctx_out Output pointer to newly allocated vk_backend_context_t.
 *                Set to NULL on error; caller must check before using.
 *
 * @return 0 on success, -1 on error (detailed errors logged via LOG_*)
 *         Success means ctx_out contains valid, opaque Vulkan context.
 *         Failure sets ctx_out to NULL and is NOT a hard error (CPU fallback is expected).
 */
int vk_backend_init(vk_backend_context_t **ctx_out);

/**
 * Shutdown Vulkan backend.
 *
 * Safely destroys all Vulkan resources in reverse order:
 * 1. VkCommandPool destruction
 * 2. VkDevice destruction
 * 3. VkInstance destruction
 *
 * Idempotent: safe to call multiple times on same context.
 * Does nothing if ctx is NULL (defensive design).
 *
 * All errors are logged; caller is not responsible for error propagation.
 *
 * @param ctx Vulkan context to destroy. If NULL, this is a no-op.
 */
void vk_backend_shutdown(vk_backend_context_t *ctx);

/**
 * Check if Vulkan backend is available and functional.
 *
 * Returns non-zero if context was successfully initialized.
 * Used to determine if GPU compute path can be used.
 *
 * @param ctx Vulkan context to check (may be NULL)
 *
 * @return Non-zero (true) if ctx is valid and Vulkan is available.
 *         Zero (false) if ctx is NULL or Vulkan failed to initialize.
 */
int vk_backend_is_available(const vk_backend_context_t *ctx);

/* ========================================================================
 * Context Accessors (P11-03 Pipeline Integration)
 * ======================================================================== */

#include <vulkan/vulkan.h>

/**
 * Get Vulkan logical device from context.
 *
 * Required for pipeline creation, buffer allocation, and descriptor management.
 *
 * @param ctx Vulkan context (must not be NULL)
 * @return VkDevice handle, or VK_NULL_HANDLE if ctx is NULL
 */
VkDevice vk_backend_get_device(const vk_backend_context_t *ctx);

/**
 * Get Vulkan physical device from context.
 *
 * Required for memory type selection during buffer allocation.
 *
 * @param ctx Vulkan context (must not be NULL)
 * @return VkPhysicalDevice handle, or VK_NULL_HANDLE if ctx is NULL
 */
VkPhysicalDevice vk_backend_get_physical_device(const vk_backend_context_t *ctx);

/**
 * Get Vulkan instance from context.
 *
 * Required for VMA allocator creation.
 *
 * @param ctx Vulkan context (must not be NULL)
 * @return VkInstance handle, or VK_NULL_HANDLE if ctx is NULL
 */
VkInstance vk_backend_get_instance(const vk_backend_context_t *ctx);

/**
 * Get Vulkan compute queue from context.
 *
 * Required for command buffer submission.
 *
 * @param ctx Vulkan context (must not be NULL)
 * @return VkQueue handle, or VK_NULL_HANDLE if ctx is NULL
 */
VkQueue vk_backend_get_compute_queue(const vk_backend_context_t *ctx);

/**
 * Get Vulkan command pool from context.
 *
 * Required for command buffer allocation.
 *
 * @param ctx Vulkan context (must not be NULL)
 * @return VkCommandPool handle, or VK_NULL_HANDLE if ctx is NULL
 */
VkCommandPool vk_backend_get_command_pool(const vk_backend_context_t *ctx);

/**
 * Get compute queue family index from context.
 *
 * Required for buffer memory allocation and queue submit operations.
 *
 * @param ctx Vulkan context (must not be NULL)
 * @return Queue family index, or UINT32_MAX if ctx is NULL
 */
uint32_t vk_backend_get_compute_queue_family_idx(const vk_backend_context_t *ctx);

/**
 * Get minimum storage buffer offset alignment requirement.
 *
 * Required for sub-allocating tensors within large buffers.
 * All descriptor buffer offsets must be aligned to this value to prevent GPU corruption.
 *
 * @param ctx Vulkan context (must not be NULL)
 * @return Alignment in bytes (typically 32-256), or 256 (safe default) if ctx is NULL
 */
size_t vk_backend_get_min_storage_buffer_offset_alignment(const vk_backend_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // VULKAN_BACKEND_H

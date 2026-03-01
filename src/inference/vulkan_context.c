/**
 * @file vulkan_context.c
 * @brief Vulkan backend context initialization and device management.
 *
 * Implements GPU device enumeration, scoring, and selection.
 * Handles instance/device creation with fallback for unavailable Vulkan.
 */

#include "../include/vulkan_backend.h"
#include "../include/log.h"

#include <stdlib.h>
#include <string.h>

/* Include Vulkan headers - they auto-define VK_VERSION_1_0 */
#include <vulkan/vulkan.h>

static int vk_validation_enabled(void) {
    const char *v = getenv("SAPPHIRE_VK_VALIDATION");
    return (v && v[0] == '1') ? 1 : 0;
}

/**
 * Vulkan backend context.
 *
 * Contains all Vulkan runtime state.
 * Opaque to consumers; only exposed through vk_backend_context_t typedef.
 */
typedef struct vk_backend_context_t {
    #ifdef VK_VERSION_1_0
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family_idx;
    VkCommandPool command_pool;
    VkDebugUtilsMessengerEXT debug_messenger;  /* Debug callback for validation layers */

    /* Capability flags for future optimization */
    int supports_descriptor_indexing;
    int supports_push_descriptors;

    /* GPU memory tracking */
    size_t gpu_memory_available;
    size_t gpu_memory_allocated;

    /* Memory alignment requirements (Issue #2 fix) */
    size_t min_storage_buffer_offset_alignment;
    #else
    /* Stub struct when Vulkan unavailable */
    int dummy;
    #endif
} vk_backend_context_t;

/**
 * Vulkan debug messenger callback.
 *
 * Called by validation layers to report errors, warnings, and info.
 */
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data
) {
    (void)type;
    (void)user_data;

    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LOG_WARN("[VULKAN VALIDATION] %s", callback_data->pMessage);
    } else {
        LOG_DEBUG("[VULKAN VALIDATION] %s", callback_data->pMessage);
    }

    return VK_FALSE;
}

/**
 * Device scorer: Rank a physical device for inference.
 *
 * Higher score = better choice. Scoring criteria:
 * - Dedicated compute queue family required (0 score if missing)
 * - Discrete GPU preferred over integrated (add base score)
 * - Additional queues increase score
 *
 * @param device     Physical device to score
 * @param instance   Vulkan instance (for vkGetPhysicalDeviceProperties)
 * @param out_queue_family Assigned output for compute queue family index
 *
 * @return Score (0 = unsuitable, 1-1000 = ranked quality)
 */
static uint32_t score_device(VkPhysicalDevice device, VkInstance instance,
                              uint32_t *out_queue_family) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, NULL);

    if (queue_family_count == 0) {
        LOG_DEBUG("Device %s has no queue families", props.deviceName);
        return 0;
    }

    VkQueueFamilyProperties *queue_families =
        malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
    if (!queue_families) {
        LOG_WARN("Failed to allocate queue family array for device %s", props.deviceName);
        return 0;
    }

    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families);

    /* Find compute queue family */
    uint32_t compute_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family = i;
            break;
        }
    }

    free(queue_families);

    if (compute_queue_family == UINT32_MAX) {
        LOG_DEBUG("Device %s has no compute queue", props.deviceName);
        return 0;
    }

    /* Device is suitable; score by type */
    uint32_t score = 100;  /* Base score for having compute queue */

    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 500;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 200;
    }

    *out_queue_family = compute_queue_family;

    LOG_DEBUG("Device %s: type=%d, score=%u, compute_queue_family=%u",
              props.deviceName, props.deviceType, score, compute_queue_family);

    return score;
}

/**
 * Select best physical device from enumerated list.
 *
 * Scores all devices and selects the best one.
 *
 * @param instance        Vulkan instance
 * @param dev_out         Output: selected physical device
 * @param qf_out          Output: compute queue family index
 *
 * @return 0 on success, -1 if no suitable device found
 */
static int select_physical_device(VkInstance instance,
                                   VkPhysicalDevice *dev_out,
                                   uint32_t *qf_out) {
    uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance, &physical_device_count, NULL);

    if (physical_device_count == 0) {
        LOG_WARN("No physical devices found");
        return -1;
    }

    VkPhysicalDevice *physical_devices = malloc(physical_device_count * sizeof(VkPhysicalDevice));
    if (!physical_devices) {
        LOG_ERROR("Failed to allocate physical device array");
        return -1;
    }

    VkResult res = vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices);
    if (res != VK_SUCCESS) {
        LOG_WARN("vkEnumeratePhysicalDevices failed: %d", res);
        free(physical_devices);
        return -1;
    }

    LOG_DEBUG("Enumerated %u physical devices", physical_device_count);

    /* Score and select best device */
    uint32_t best_score = 0;
    uint32_t best_device_idx = UINT32_MAX;
    uint32_t best_queue_family = UINT32_MAX;

    for (uint32_t i = 0; i < physical_device_count; i++) {
        uint32_t queue_family = UINT32_MAX;
        uint32_t score = score_device(physical_devices[i], instance, &queue_family);

        if (score > best_score) {
            best_score = score;
            best_device_idx = i;
            best_queue_family = queue_family;
        }
    }

    if (best_score == 0) {
        LOG_WARN("No suitable compute devices found");
        free(physical_devices);
        return -1;
    }

    *dev_out = physical_devices[best_device_idx];
    *qf_out = best_queue_family;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(*dev_out, &props);
    LOG_INFO("Selected GPU device: %s", props.deviceName);

    free(physical_devices);
    return 0;
}

/**
 * Create Vulkan logical device and command pool.
 *
 * @param physical_dev  Selected physical device
 * @param queue_family  Compute queue family index
 * @param device_out    Output: created logical device
 * @param queue_out     Output: compute queue
 * @param pool_out      Output: command pool
 *
 * @return 0 on success, -1 on error
 */
static int create_device_and_pool(VkPhysicalDevice physical_dev,
                                   uint32_t queue_family,
                                   VkDevice *device_out,
                                   VkQueue *queue_out,
                                   VkCommandPool *pool_out) {
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    /* Enable timeline semaphore feature (required for ring buffers) */
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .pNext = NULL,
        .timelineSemaphore = VK_TRUE,
    };

    /* Issue #3 fix: Enable descriptor indexing BEFORE device creation */
    VkPhysicalDeviceDescriptorIndexingFeaturesEXT desc_indexing_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
        .pNext = &timeline_features,  /* Chain timeline features */
        .shaderStorageBufferArrayNonUniformIndexing = VK_TRUE,
    };

    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &desc_indexing_features,
    };

    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features2,  /* Enable features via pNext chain */
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };

    VkResult res = vkCreateDevice(physical_dev, &device_info, NULL, device_out);
    if (res != VK_SUCCESS) {
        LOG_WARN("vkCreateDevice failed: %d", res);
        return -1;
    }

    LOG_DEBUG("VkDevice created successfully");

    /* Get compute queue */
    vkGetDeviceQueue(*device_out, queue_family, 0, queue_out);

    /* Create command pool */
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };

    res = vkCreateCommandPool(*device_out, &pool_info, NULL, pool_out);
    if (res != VK_SUCCESS) {
        LOG_WARN("vkCreateCommandPool failed: %d", res);
        vkDestroyDevice(*device_out, NULL);
        return -1;
    }

    LOG_DEBUG("VkCommandPool created successfully");
    return 0;
}

/**
 * Initialize Vulkan backend.
 *
 * Creates instance, selects GPU, creates logical device and command pool.
 *
 * @param ctx_out Output context (set to NULL on failure)
 *
 * @return 0 on success, -1 on error
 */
int vk_backend_init(vk_backend_context_t **ctx_out) {
    if (!ctx_out) {
        LOG_ERROR("ctx_out is NULL");
        return -1;
    }

    *ctx_out = NULL;

    /* Allocate context */
    vk_backend_context_t *ctx = malloc(sizeof(vk_backend_context_t));
    if (!ctx) {
        LOG_ERROR("Failed to allocate Vulkan context");
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));

    /* Create instance (validation optional for performance) */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Sapphire LLM",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Sapphire",
        .engineVersion = VK_MAKE_VERSION(SAPPHIRE_VK_MAJOR, SAPPHIRE_VK_MINOR, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    int enable_validation = vk_validation_enabled();
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    const char *extensions[] = {"VK_EXT_debug_utils"};

    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = enable_validation ? 1u : 0u,
        .ppEnabledLayerNames = enable_validation ? layers : NULL,
        .enabledExtensionCount = enable_validation ? 1u : 0u,
        .ppEnabledExtensionNames = enable_validation ? extensions : NULL,
    };

    VkResult res = vkCreateInstance(&instance_info, NULL, &ctx->instance);
    if (res != VK_SUCCESS) {
        LOG_WARN("vkCreateInstance failed: %d. Vulkan unavailable, will use CPU backend.", res);
        free(ctx);
        return -1;
    }

    LOG_DEBUG("VkInstance created successfully (validation=%s)",
              enable_validation ? "on" : "off");

    /* Setup debug messenger to capture validation layer messages */
    if (enable_validation) {
        VkDebugUtilsMessengerCreateInfoEXT debug_info = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_callback,
        };

        PFN_vkCreateDebugUtilsMessengerEXT create_debug_messenger =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(ctx->instance, "vkCreateDebugUtilsMessengerEXT");

        if (create_debug_messenger) {
            res = create_debug_messenger(ctx->instance, &debug_info, NULL, &ctx->debug_messenger);
            if (res != VK_SUCCESS) {
                LOG_WARN("Failed to create debug messenger: %d", res);
                ctx->debug_messenger = VK_NULL_HANDLE;
            } else {
                LOG_DEBUG("Debug messenger created successfully");
            }
        } else {
            LOG_WARN("VK_EXT_debug_utils not available, validation messages will not be captured");
            ctx->debug_messenger = VK_NULL_HANDLE;
        }
    } else {
        ctx->debug_messenger = VK_NULL_HANDLE;
    }

    /* Select physical device */
    int rc = select_physical_device(ctx->instance, &ctx->physical_device,
                                    &ctx->compute_queue_family_idx);
    if (rc != 0) {
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    /* Query device memory */
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &mem_props);

    /* Find largest device-local memory heap */
    ctx->gpu_memory_available = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
        if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (mem_props.memoryHeaps[i].size > ctx->gpu_memory_available) {
                ctx->gpu_memory_available = mem_props.memoryHeaps[i].size;
            }
        }
    }

    LOG_DEBUG("GPU memory available: %.2f GB", ctx->gpu_memory_available / (1024.0 * 1024.0 * 1024.0));

    /* Create logical device and command pool */
    rc = create_device_and_pool(ctx->physical_device, ctx->compute_queue_family_idx,
                                &ctx->device, &ctx->compute_queue, &ctx->command_pool);
    if (rc != 0) {
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return -1;
    }

    /* Query device capabilities (Vulkan 1.0 API) */
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(ctx->physical_device, &features);
    
    /* We use static descriptor sets (no indexing extensions needed) */
    ctx->supports_descriptor_indexing = 0;
    ctx->supports_push_descriptors = 0;

    /* Issue #2 fix: Query minStorageBufferOffsetAlignment for sub-allocation */
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &props);
    ctx->min_storage_buffer_offset_alignment = props.limits.minStorageBufferOffsetAlignment;
    LOG_DEBUG("minStorageBufferOffsetAlignment: %zu bytes", ctx->min_storage_buffer_offset_alignment);

    LOG_INFO("Vulkan backend initialized successfully");

    *ctx_out = ctx;
    return 0;
}

/**
 * Shutdown Vulkan backend.
 *
 * Destroys resources in reverse order: pool, device, instance.
 *
 * @param ctx Vulkan context (may be NULL)
 */
void vk_backend_shutdown(vk_backend_context_t *ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->command_pool) {
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
        LOG_DEBUG("VkCommandPool destroyed");
    }

    if (ctx->device) {
        vkDestroyDevice(ctx->device, NULL);
        LOG_DEBUG("VkDevice destroyed");
    }

    if (ctx->debug_messenger) {
        PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_messenger =
            (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(ctx->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy_debug_messenger) {
            destroy_debug_messenger(ctx->instance, ctx->debug_messenger, NULL);
            LOG_DEBUG("Debug messenger destroyed");
        }
    }

    if (ctx->instance) {
        vkDestroyInstance(ctx->instance, NULL);
        LOG_DEBUG("VkInstance destroyed");
    }

    free(ctx);
    LOG_INFO("Vulkan backend shut down");
}

/**
 * Check if Vulkan backend is available.
 *
 * @param ctx Context to check
 *
 * @return Non-zero if valid, zero otherwise
 */
int vk_backend_is_available(const vk_backend_context_t *ctx) {
    if (!ctx) {
        return 0;
    }

    return ctx->instance != NULL ? 1 : 0;
}

/* ========================================================================
 * Context Accessors (P11-03 Pipeline Integration)
 * ======================================================================== */

VkDevice vk_backend_get_device(const vk_backend_context_t *ctx) {
    if (!ctx) {
        return VK_NULL_HANDLE;
    }
    return ctx->device;
}

VkPhysicalDevice vk_backend_get_physical_device(const vk_backend_context_t *ctx) {
    if (!ctx) {
        return VK_NULL_HANDLE;
    }
    return ctx->physical_device;
}

VkQueue vk_backend_get_compute_queue(const vk_backend_context_t *ctx) {
    if (!ctx) {
        return VK_NULL_HANDLE;
    }
    return ctx->compute_queue;
}

VkCommandPool vk_backend_get_command_pool(const vk_backend_context_t *ctx) {
    if (!ctx) {
        return VK_NULL_HANDLE;
    }
    return ctx->command_pool;
}

uint32_t vk_backend_get_compute_queue_family_idx(const vk_backend_context_t *ctx) {
    if (!ctx) {
        return UINT32_MAX;
    }
    return ctx->compute_queue_family_idx;
}

/**
 * Get minimum storage buffer offset alignment requirement.
 *
 * Required for sub-allocating tensors within large buffers.
 * All descriptor buffer offsets must be aligned to this value.
 *
 * @param ctx Vulkan context (must not be NULL)
 * @return Alignment in bytes, or 256 (safe default) if ctx is NULL
 */
size_t vk_backend_get_min_storage_buffer_offset_alignment(const vk_backend_context_t *ctx) {
    if (!ctx) {
        return 256;  /* Safe fallback (conservative upper bound) */
    }
    #ifdef VK_VERSION_1_0
    return ctx->min_storage_buffer_offset_alignment;
    #else
    return 256;
    #endif
}

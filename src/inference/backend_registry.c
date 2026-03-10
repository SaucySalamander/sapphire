/**
 * @file backend_registry.c
 * @brief Backend registry implementation.
 *
 * Manages available backend implementations and provides registry functions.
 */

#include "../include/backend.h"
#include "../include/log.h"

#include <stdlib.h>
#include <string.h>

/* Forward declarations of backend implementations */
extern sapphire_backend_t backend_cpu_impl;
extern sapphire_backend_t backend_vulkan_impl;

/**
 * Get backend implementation by type.
 */
sapphire_backend_t* backend_get(sapphire_backend_type_t type) {
    switch (type) {
        case SAPPHIRE_BACKEND_TYPE_CPU:
            return &backend_cpu_impl;
        case SAPPHIRE_BACKEND_TYPE_VULKAN:
            return &backend_vulkan_impl;
        default:
            LOG_ERROR("Unsupported backend type: %d", (int)type);
            return NULL;
    }
}

/**
 * Detect backend from environment or defaults to CPU.
 */
sapphire_backend_type_t backend_detect(void) {
    const char* env = getenv("SAPPHIRE_BACKEND");
    if (!env) {
        // Default to CPU if not specified
        LOG_DEBUG("SAPPHIRE_BACKEND not set, defaulting to CPU");
        return SAPPHIRE_BACKEND_TYPE_CPU;
    }

    if (strcmp(env, "cpu") == 0) {
        LOG_INFO("Backend detection: CPU (from SAPPHIRE_BACKEND=%s)", env);
        return SAPPHIRE_BACKEND_TYPE_CPU;
    } else if (strcmp(env, "vulkan") == 0) {
        LOG_INFO("Backend detection: Vulkan (from SAPPHIRE_BACKEND=%s)", env);
        return SAPPHIRE_BACKEND_TYPE_VULKAN;
    } else {
        LOG_WARN("Unknown SAPPHIRE_BACKEND=%s, defaulting to CPU", env);
        return SAPPHIRE_BACKEND_TYPE_CPU;
    }
}

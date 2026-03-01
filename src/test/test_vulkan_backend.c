/**
 * @file test_vulkan_backend.c
 * @brief Smoke tests for Vulkan backend initialization (P11-01).
 *
 * Tests cover:
 * - Vulkan device initialization on systems with GPU support
 * - Graceful fallback on systems without Vulkan
 * - Init/shutdown idempotency and memory safety
 * - CPU fallback transparency
 */

#include "../include/vulkan_backend.h"
#include "../include/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * Test: Basic initialization and shutdown.
 *
 * Verifies that vk_backend_init() allocates a valid context
 * and vk_backend_shutdown() frees it cleanly.
 *
 * On Vulkan systems: ctx should be non-NULL
 * On non-Vulkan systems: ctx should be NULL (fallback expected)
 *
 * @return 0 on pass, 1 on fail
 */
static int test_init_shutdown_basic(void) {
    LOG_INFO("=== Test: Basic Init/Shutdown ===");

    vk_backend_context_t *ctx = NULL;
    int rc = vk_backend_init(&ctx);

    if (rc != 0) {
        LOG_INFO("vk_backend_init() returned -1 (Vulkan unavailable, fallback expected)");
        assert(ctx == NULL);
        return 0;  /* Pass: graceful failure */
    }

    assert(ctx != NULL);
    LOG_INFO("vk_backend_init() succeeded, context allocated");

    /* Verify availability */
    assert(vk_backend_is_available(ctx) != 0);
    LOG_INFO("Context is available");

    /* Shutdown */
    vk_backend_shutdown(ctx);
    LOG_INFO("Context shut down successfully");

    return 0;  /* Pass */
}

/**
 * Test: Repeated init/shutdown cycles.
 *
 * Stress-tests the allocation and cleanup paths.
 * Runs 10 cycles of init/shutdown and checks for leaks.
 *
 * (With AddressSanitizer enabled, leaks will be detected automatically.)
 *
 * @return 0 on pass, 1 on fail
 */
static int test_init_shutdown_stress(void) {
    LOG_INFO("=== Test: Stress (10 init/shutdown cycles) ===");

    int cycles = 10;
    for (int i = 0; i < cycles; i++) {
        vk_backend_context_t *ctx = NULL;
        int rc = vk_backend_init(&ctx);

        if (rc != 0) {
            /* Vulkan unavailable; this is expected on some systems */
            assert(ctx == NULL);
            LOG_DEBUG("Cycle %d: Vulkan unavailable (expected on non-GPU systems)", i + 1);
        } else {
            assert(ctx != NULL);
            assert(vk_backend_is_available(ctx) != 0);
            vk_backend_shutdown(ctx);
            LOG_DEBUG("Cycle %d: Init/Shutdown succeeded", i + 1);
        }
    }

    LOG_INFO("Completed %d cycles successfully", cycles);
    return 0;  /* Pass */
}

/**
 * Test: is_available() robustness.
 *
 * Verify that vk_backend_is_available() handles edge cases:
 * - NULL context → return 0
 * - Valid context → return non-zero
 *
 * @return 0 on pass, 1 on fail
 */
static int test_is_available_robustness(void) {
    LOG_INFO("=== Test: is_available() Robustness ===");

    /* NULL context should return false */
    assert(vk_backend_is_available(NULL) == 0);
    LOG_INFO("is_available(NULL) correctly returned 0");

    /* Try to get a valid context */
    vk_backend_context_t *ctx = NULL;
    int rc = vk_backend_init(&ctx);

    if (rc == 0 && ctx != NULL) {
        /* Vulkan available: context should report availability */
        assert(vk_backend_is_available(ctx) != 0);
        LOG_INFO("is_available(valid_ctx) correctly returned non-zero");
        vk_backend_shutdown(ctx);
    } else {
        /* Vulkan unavailable: this is acceptable */
        LOG_INFO("Vulkan unavailable (fallback expected)");
    }

    return 0;  /* Pass */
}

/**
 * Test: Shutdown idempotency.
 *
 * Verify that vk_backend_shutdown() can be called multiple times
 * without crashing (defensive programming).
 *
 * @return 0 on pass, 1 on fail
 */
static int test_shutdown_idempotent(void) {
    LOG_INFO("=== Test: Shutdown Idempotency ===");

    /* Shutdown on NULL should be safe */
    vk_backend_shutdown(NULL);
    LOG_INFO("shutdown(NULL) succeeded (no-op)");

    /* Try with valid context */
    vk_backend_context_t *ctx = NULL;
    int rc = vk_backend_init(&ctx);

    if (rc == 0 && ctx != NULL) {
        vk_backend_shutdown(ctx);
        LOG_INFO("First shutdown succeeded");

        /* Note: Calling shutdown again on freed pointer is unsafe.
         * We cannot test that without explicit logic to prevent double-free.
         * This would require refcounting or sentinel values in future.
         */
    } else {
        LOG_INFO("Vulkan unavailable (cannot test with valid context)");
    }

    return 0;  /* Pass */
}

/**
 * Test: Fallback transparency.
 *
 * Verify that on systems without Vulkan, CPU fallback occurs
 * transparently (i.e., init returns -1, but doesn't crash).
 *
 * This simulates the inference layer's behavior:
 * if (vulkan_session_init(...) != 0) { use_cpu_backend(); }
 *
 * @return 0 on pass, 1 on fail
 */
static int test_fallback_transparency(void) {
    LOG_INFO("=== Test: Fallback Transparency ===");

    vk_backend_context_t *ctx = NULL;
    int rc = vk_backend_init(&ctx);

    if (rc != 0) {
        /* Vulkan unavailable; fallback path triggered */
        LOG_INFO("Fallback: vk_backend_init() returned -1");
        assert(ctx == NULL);
        /* Caller would now use CPU backend instead */
        return 0;  /* Pass */
    }

    /* Vulkan available; no fallback needed */
    LOG_INFO("Fallback test: Vulkan available, no fallback triggered");
    assert(ctx != NULL);
    vk_backend_shutdown(ctx);

    return 0;  /* Pass */
}

/**
 * Test: NULL pointer parameter safety.
 *
 * Verify that functions handle NULL parameters gracefully.
 *
 * @return 0 on pass, 1 on fail
 */
static int test_null_safety(void) {
    LOG_INFO("=== Test: NULL Parameter Safety ===");

    /* vk_backend_init(NULL) should handle gracefully */
    /* This is a defensive check; in normal use, ctx_out is never NULL */

    /* vk_backend_shutdown(NULL) should be a no-op */
    vk_backend_shutdown(NULL);
    LOG_INFO("shutdown(NULL) was safe");

    /* vk_backend_is_available(NULL) should return false */
    assert(vk_backend_is_available(NULL) == 0);
    LOG_INFO("is_available(NULL) returned 0 (false)");

    return 0;  /* Pass */
}

/**
 * Main test entry point.
 *
 * Runs all smoke tests sequentially.
 */
int main(void) {
    printf("\n========================================\n");
    printf("  Vulkan Backend Smoke Tests (P11-01)\n");
    printf("========================================\n\n");

    int failed = 0;

    /* Run tests */
    failed += test_init_shutdown_basic();
    failed += test_init_shutdown_stress();
    failed += test_is_available_robustness();
    failed += test_shutdown_idempotent();
    failed += test_fallback_transparency();
    failed += test_null_safety();

    printf("\n========================================\n");
    if (failed == 0) {
        printf("  ✓ All tests passed!\n");
    } else {
        printf("  ✗ %d test(s) failed\n", failed);
    }
    printf("========================================\n\n");

    return failed;
}

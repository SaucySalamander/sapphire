#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "tensor_gemv.h"
#include "tensor.h"

// ============================================================================
// Test: F32 GEMV
// ============================================================================

static void test_tensor_gemv_f32_basic(void) {
    printf("TEST: tensor_gemv F32 basic\n");
    
    int m = 3, n = 4;
    
    // Create weight tensor A [3, 4]
    int shape_A[] = {m, n};
    tensor_t *A = tensor_create(2, shape_A, DTYPE_F32);
    assert(A != NULL);
    
    // Initialize A with simple values
    // A = [[1, 2, 3, 4],
    //      [5, 6, 7, 8],
    //      [9, 10, 11, 12]]
    float A_vals[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    for (int i = 0; i < m * n; i++) {
        tensor_set_f32(A, i, A_vals[i]);
    }
    
    // Create input vector x [4]
    float x[] = {1, 1, 1, 1};
    
    // Initialize GEMV system (use explicit context)
    sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
    assert(ctx != NULL);
    
    // Compute y = A @ x
    float y[m];
    int ret = tensor_gemv_with_ctx(ctx, y, A, x);
    assert(ret == 0);
    
    // Expected: y[0] = 1+2+3+4 = 10
    //           y[1] = 5+6+7+8 = 26
    //           y[2] = 9+10+11+12 = 42
    assert(fabs(y[0] - 10.0f) < 1e-5f);
    assert(fabs(y[1] - 26.0f) < 1e-5f);
    assert(fabs(y[2] - 42.0f) < 1e-5f);
    
    tensor_release(A);
    tensor_gemv_ctx_destroy(ctx);
    
    printf("  ✓ F32 GEMV basic computation correct\n");
}

static void test_tensor_gemv_f32_random(void) {
    printf("TEST: tensor_gemv F32 random matrix\n");
    
    int m = 16, n = 32;
    
    int shape_A[] = {m, n};
    tensor_t *A = tensor_create(2, shape_A, DTYPE_F32);
    assert(A != NULL);
    
    // Initialize with random-like values
    for (int i = 0; i < m * n; i++) {
        tensor_set_f32(A, i, (float)(i % 7) * 0.1f);
    }
    
    // Create input vector
    float *x = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        x[i] = (float)(i % 5) * 0.2f;
    }
    
    // Initialize and compute (explicit context)
    sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
    assert(ctx != NULL);
    
    float *y = (float *)malloc(m * sizeof(float));
    int ret = tensor_gemv_with_ctx(ctx, y, A, x);
    assert(ret == 0);
    
    // Verify manually for first row
    float expected_y0 = 0.0f;
    for (int j = 0; j < n; j++) {
        float a_val = tensor_get_f32(A, j);  // A[0, j]
        expected_y0 += a_val * x[j];
    }
    assert(fabs(y[0] - expected_y0) < 1e-4f);
    
    free(x);
    free(y);
    tensor_release(A);
    tensor_gemv_ctx_destroy(ctx);
    
    printf("  ✓ F32 GEMV random matrix computation correct\n");
}

// ============================================================================
// Test: Tensor GEMV wrapper
// ============================================================================

static void test_tensor_gemv_tensor_wrapper(void) {
    printf("TEST: tensor_gemv_tensor wrapper\n");
    
    int m = 4, n = 5;
    
    // Create tensors
    int shape_A[] = {m, n};
    tensor_t *A = tensor_create(2, shape_A, DTYPE_F32);
    
    int shape_x[] = {n};
    tensor_t *x = tensor_create(1, shape_x, DTYPE_F32);
    
    int shape_y[] = {m};
    tensor_t *y = tensor_create(1, shape_y, DTYPE_F32);
    
    // Initialize A and x with simple values
    for (int i = 0; i < m * n; i++) {
        tensor_set_f32(A, i, 2.0f);  // All A elements = 2
    }
    for (int i = 0; i < n; i++) {
        tensor_set_f32(x, i, 1.0f);  // All x elements = 1
    }
    
    // Compute with explicit context
    sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
    assert(ctx != NULL);
    int ret = tensor_gemv_tensor_with_ctx(ctx, y, A, x);
    assert(ret == 0);
    
    // Expected: each output = 2 * n = 2 * 5 = 10
    for (int i = 0; i < m; i++) {
        float val = tensor_get_f32(y, i);
        assert(fabs(val - 10.0f) < 1e-5f);
    }
    
    tensor_release(A);
    tensor_release(x);
    tensor_release(y);
    tensor_gemv_ctx_destroy(ctx);
    
    printf("  ✓ Tensor wrapper works correctly\n");
}

// ============================================================================
// Test: GEMV accumulation
// ============================================================================

static void test_tensor_gemv_add(void) {
    printf("TEST: tensor_gemv_add accumulation\n");
    
    int m = 3, n = 2;
    
    // Create weight matrix
    int shape_A[] = {m, n};
    tensor_t *A = tensor_create(2, shape_A, DTYPE_F32);

    // A = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    tensor_set_f32(A, 0, 1.0f);  // A[0, 0]
    tensor_set_f32(A, 1, 0.0f);  // A[0, 1]
    tensor_set_f32(A, 2, 0.0f);  // A[1, 0]
    tensor_set_f32(A, 3, 1.0f);  // A[1, 1]
    tensor_set_f32(A, 4, 1.0f);  // A[2, 0]
    tensor_set_f32(A, 5, 1.0f);  // A[2, 1]
    
    float x[] = {2.0f, 3.0f};
    float y[] = {10.0f, 20.0f, 30.0f};  // Initial values
    
    // Compute y += 1.0 * (A @ x)
    // A @ x = [2, 3, 5]
    // y should become [12, 23, 35]
    sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
    assert(ctx != NULL);
    int ret = tensor_gemv_add_with_ctx(ctx, y, A, x, 1.0f);
    assert(ret == 0);
    
    assert(fabs(y[0] - 12.0f) < 1e-5f);
    assert(fabs(y[1] - 23.0f) < 1e-5f);
    assert(fabs(y[2] - 35.0f) < 1e-5f);
    
    tensor_release(A);
    tensor_gemv_ctx_destroy(ctx);
    
    printf("  ✓ GEMV accumulation works correctly\n");
}

static void test_tensor_gemv_add_with_scaling(void) {
    printf("TEST: tensor_gemv_add with scaling\n");
    
    int m = 2, n = 2;
    
    int shape_A[] = {m, n};
    tensor_t *A = tensor_create(2, shape_A, DTYPE_F32);
    
    // A = [[1, 1], [1, 1]]
    tensor_set_f32(A, 0, 1.0f);
    tensor_set_f32(A, 1, 1.0f);
    tensor_set_f32(A, 2, 1.0f);
    tensor_set_f32(A, 3, 1.0f);
    
    float x[] = {2.0f, 2.0f};  // x @ x = [4, 4]
    float y[] = {0.0f, 0.0f};
    
    // Compute y += 0.5 * (A @ x)
    // A @ x = [4, 4]
    // y should become [2, 2]
    sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
    assert(ctx != NULL);
    int ret = tensor_gemv_add_with_ctx(ctx, y, A, x, 0.5f);
    assert(ret == 0);
    
    assert(fabs(y[0] - 2.0f) < 1e-5f);
    assert(fabs(y[1] - 2.0f) < 1e-5f);
    
    tensor_release(A);
    tensor_gemv_ctx_destroy(ctx);
    
    printf("  ✓ GEMV accumulation with scaling works correctly\n");
}

// ============================================================================
// Test: Batched GEMV
// ============================================================================

static void test_tensor_gemv_batch(void) {
    printf("TEST: tensor_gemv_batch\n");
    
    int batch_size = 3;
    int m = 2, n = 2;
    
    int shape_A[] = {m, n};
    tensor_t *A = tensor_create(2, shape_A, DTYPE_F32);
    
    // A = [[1, 2], [3, 4]]
    tensor_set_f32(A, 0, 1.0f);
    tensor_set_f32(A, 1, 2.0f);
    tensor_set_f32(A, 2, 3.0f);
    tensor_set_f32(A, 3, 4.0f);
    
    // X = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    float X[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    float Y[batch_size * m];
    
    sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
    assert(ctx != NULL);
    int ret = tensor_gemv_batch_with_ctx(ctx, Y, A, X, batch_size);
    assert(ret == 0);
    
    // Expected:
    // Y[0,:] = A @ [1, 0] = [1, 3]
    // Y[1,:] = A @ [0, 1] = [2, 4]
    // Y[2,:] = A @ [1, 1] = [3, 7]
    assert(fabs(Y[0] - 1.0f) < 1e-5f && fabs(Y[1] - 3.0f) < 1e-5f);
    assert(fabs(Y[2] - 2.0f) < 1e-5f && fabs(Y[3] - 4.0f) < 1e-5f);
    assert(fabs(Y[4] - 3.0f) < 1e-5f && fabs(Y[5] - 7.0f) < 1e-5f);
    
    tensor_release(A);
    tensor_gemv_ctx_destroy(ctx);
    
    printf("  ✓ Batched GEMV works correctly\n");
}

// ============================================================================
// Test: Error handling
// ============================================================================

static void test_tensor_gemv_error_handling(void) {
    printf("TEST: tensor_gemv error handling\n");
    
    int m = 3, n = 4;
    
    int shape_A[] = {m, n};
    tensor_t *A = tensor_create(2, shape_A, DTYPE_F32);
    float x[n];
    float y[m];
    
    // Test without initialization (should fail for quantized, but succeed for F32)
    // Old behavior: tensor_gemv_cleanup();  // Ensure not initialized
    // We'll just call tensor_gemv_with_ctx with NULL ctx (which is allowed for F32)
    int ret = tensor_gemv_with_ctx(NULL, y, A, x);
    assert(ret == 0);  // F32 should work even without init
    
    // Null pointer checks
    ret = tensor_gemv_with_ctx(NULL, NULL, A, x);
    assert(ret == -1);
    
    ret = tensor_gemv_with_ctx(NULL, y, NULL, x);
    assert(ret == -1);
    
    ret = tensor_gemv_with_ctx(NULL, y, A, NULL);
    assert(ret == -1);
    
    tensor_release(A);
    
    printf("  ✓ Error handling works correctly\n");
}

// ============================================================================
// Test: Init/Cleanup idempotency
// ============================================================================

static void test_tensor_gemv_init_cleanup_idempotent(void) {
    printf("TEST: tensor_gemv_init/cleanup idempotency\n");
    
    // We'll test context creation/destroy idempotency
    sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
    assert(ctx != NULL);
    tensor_gemv_ctx_destroy(ctx);
    tensor_gemv_ctx_destroy(NULL); // destroy NULL-safe
    
    printf("  ✓ Init/cleanup idempotency correct\n");
}

// ============================================================================
// Test: Large matrix
// ============================================================================

static void test_tensor_gemv_large(void) {
    printf("TEST: tensor_gemv large matrix\n");
    
    int m = 256, n = 512;
    
    int shape_A[] = {m, n};
    tensor_t *A = tensor_create(2, shape_A, DTYPE_F32);
    assert(A != NULL);
    
    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(m * sizeof(float));

    // Initialize with small values
    for (int i = 0; i < m * n; i++) {
        tensor_set_f32(A, i, 0.01f);
    }
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
    }
    
    sapphire_context *ctx = tensor_gemv_ctx_create(0, 1024);
    assert(ctx != NULL);
    int ret = tensor_gemv_with_ctx(ctx, y, A, x);
    assert(ret == 0);

    // Each output should be approximately 0.01 * n = 5.12
    for (int i = 0; i < m; i++) {
        assert(fabs(y[i] - 5.12f) < 0.01f);
    }
    
    free(x);
    free(y);
    tensor_release(A);
    tensor_gemv_ctx_destroy(ctx);
    
    printf("  ✓ Large matrix GEMV works\n");
}

// ============================================================================
// Main test runner
// ============================================================================

int main(void) {
    printf("\n");
    printf("============================================================\n");
    printf("                TENSOR GEMV TEST SUITE\n");
    printf("============================================================\n");
    printf("\n");
    
    // F32 GEMV tests
    test_tensor_gemv_f32_basic();
    test_tensor_gemv_f32_random();
    printf("\n");
    
    // Tensor wrapper tests
    test_tensor_gemv_tensor_wrapper();
    printf("\n");
    
    // Accumulation tests
    test_tensor_gemv_add();
    test_tensor_gemv_add_with_scaling();
    printf("\n");
    
    // Batch tests
    test_tensor_gemv_batch();
    printf("\n");
    
    // Error handling
    test_tensor_gemv_error_handling();
    printf("\n");
    
    // Init/cleanup
    test_tensor_gemv_init_cleanup_idempotent();
    printf("\n");
    
    // Large matrix
    test_tensor_gemv_large();
    printf("\n");
    
    printf("============================================================\n");
    printf("                    ALL TESTS PASSED ✓\n");
    printf("============================================================\n");
    printf("\n");
    
    return 0;
}

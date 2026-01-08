#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "activations.h"
#include "test_utils.h"

// Global test counters
int tests_passed = 0;
int tests_failed = 0;

// ============================================================================
// Test: SiLU
// ============================================================================

static void test_silu_scalar(void) {
    printf("TEST: silu scalar values\n");
    
    // Test a few key points
    float result;
    
    // silu(0) = 0
    result = silu(0.0f);
    assert(fabs(result - 0.0f) < 1e-6f);
    
    // silu(1) = 1 * sigmoid(1) = 1 * 0.731... ≈ 0.731
    result = silu(1.0f);
    float expected = 1.0f / (1.0f + expf(-1.0f));
    assert(fabs(result - expected) < 1e-6f);
    
    // silu(-1) = -1 * sigmoid(-1) = -1 * 0.268... ≈ -0.268
    result = silu(-1.0f);
    expected = -1.0f / (1.0f + expf(1.0f));
    assert(fabs(result - expected) < 1e-6f);
    
    // Large positive: silu(10) ≈ 10 (sigmoid(10) ≈ 1)
    result = silu(10.0f);
    assert(result > 9.9f && result < 10.1f);
    
    // Large negative: silu(-10) ≈ 0 (sigmoid(-10) ≈ 0)
    result = silu(-10.0f);
    assert(result > -0.1f && result < 0.1f);
    
    printf("  ✓ SiLU scalar values correct\n");
    tests_passed++;
}

static void test_silu_inplace(void) {
    printf("TEST: silu_inplace array activation\n");
    
    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float expected[] = {
        -2.0f / (1.0f + expf(2.0f)),
        -1.0f / (1.0f + expf(1.0f)),
        0.0f,
        1.0f / (1.0f + expf(-1.0f)),
        2.0f / (1.0f + expf(-2.0f))
    };
    
    silu_inplace(x, 5);
    
    for (int i = 0; i < 5; i++) {
        assert(fabs(x[i] - expected[i]) < 1e-5f);
    }
    
    printf("  ✓ SiLU in-place array activation correct\n");
    tests_passed++;
}

// ============================================================================
// Test: ReLU
// ============================================================================

static void test_relu_scalar(void) {
    printf("TEST: relu scalar values\n");
    
    // Positive values pass through
    assert(fabs(relu(0.5f) - 0.5f) < 1e-6f);
    assert(fabs(relu(1.0f) - 1.0f) < 1e-6f);
    assert(fabs(relu(100.0f) - 100.0f) < 1e-6f);
    
    // Zero
    assert(fabs(relu(0.0f) - 0.0f) < 1e-6f);
    
    // Negative values become zero
    assert(fabs(relu(-0.5f) - 0.0f) < 1e-6f);
    assert(fabs(relu(-100.0f) - 0.0f) < 1e-6f);
    
    printf("  ✓ ReLU scalar values correct\n");
    tests_passed++;
}

static void test_relu_inplace(void) {
    printf("TEST: relu_inplace array activation\n");
    
    float x[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    float expected[] = {0.0f, 0.0f, 0.0f, 1.0f, 3.0f};
    
    relu_inplace(x, 5);
    
    for (int i = 0; i < 5; i++) {
        assert(fabs(x[i] - expected[i]) < 1e-6f);
    }
    
    printf("  ✓ ReLU in-place array activation correct\n");
    tests_passed++;
}

// ============================================================================
// Test: GELU
// ============================================================================

static void test_gelu_scalar(void) {
    printf("TEST: gelu scalar values\n");
    
    // Test a few key points
    float result;
    
    // gelu(0) = 0
    result = gelu(0.0f);
    assert(fabs(result - 0.0f) < 1e-5f);
    
    // gelu(1) should be approximately 0.841 (from reference implementation)
    result = gelu(1.0f);
    assert(result > 0.8f && result < 0.9f);
    
    // gelu(-1) should be approximately -0.159 (negative)
    result = gelu(-1.0f);
    assert(result < 0.0f && result > -0.2f);
    
    // gelu(2) should be approximately 1.954
    result = gelu(2.0f);
    assert(result > 1.9f && result < 2.0f);
    
    printf("  ✓ GELU scalar values correct\n");
    tests_passed++;
}

static void test_gelu_inplace(void) {
    printf("TEST: gelu_inplace array activation\n");
    
    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    
    gelu_inplace(x, 5);
    
    // Verify gelu(0) = 0
    assert(fabs(x[2] - 0.0f) < 1e-5f);
    
    // Verify non-zero values have changed (except 0)
    assert(fabs(x[0] - gelu(-2.0f)) < 1e-6f);  // gelu(-2) applied
    assert(fabs(x[1] - gelu(-1.0f)) < 1e-6f);  // gelu(-1) applied
    assert(fabs(x[3] - gelu(1.0f)) < 1e-6f);   // gelu(1) applied
    assert(fabs(x[4] - gelu(2.0f)) < 1e-6f);   // gelu(2) applied
    
    printf("  ✓ GELU in-place array activation correct\n");
    tests_passed++;
}

// ============================================================================
// Test: Activation properties
// ============================================================================

static void test_activation_continuity(void) {
    printf("TEST: activation function continuity\n");
    
    // Test small step changes
    float x = 0.0f;
    float step = 0.001f;
    
    // SiLU should be continuous
    float silu_curr = silu(x);
    float silu_next = silu(x + step);
    assert(fabs(silu_next - silu_curr) < 0.01f);
    
    // ReLU has a kink at 0, but should still be defined everywhere
    float relu_curr = relu(x);
    float relu_next = relu(x + step);
    assert(relu_curr >= 0.0f && relu_next >= 0.0f);
    
    // GELU should be smooth everywhere
    float gelu_curr = gelu(x);
    float gelu_next = gelu(x + step);
    assert(fabs(gelu_next - gelu_curr) < 0.01f);
    
    printf("  ✓ Activation functions are continuous\n");
    tests_passed++;
}

// ============================================================================
// Test: Activation ranges
// ============================================================================

static void test_activation_ranges(void) {
    printf("TEST: activation output ranges\n");
    
    float test_vals[] = {-10.0f, -5.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 5.0f, 10.0f};
    int n = 9;
    
    for (int i = 0; i < n; i++) {
        float x = test_vals[i];
        
        // SiLU: output has same sign as input (mostly)
        float y_silu = silu(x);
        if (x > 0) assert(y_silu > 0);
        if (x < 0) assert(y_silu < 0);
        if (x == 0) assert(y_silu == 0);
        
        // ReLU: output >= 0 always
        float y_relu = relu(x);
        assert(y_relu >= 0.0f);
        
        // GELU: generally follows sign of input but smooth (not perfectly antisymmetric)
        float y_gelu = gelu(x);
        // For strong negative values, GELU should be negative or close to zero
        if (x < -2.0f) assert(y_gelu < 0.1f);
        // For strong positive values, GELU should be positive
        if (x > 2.0f) assert(y_gelu > 0.1f);
        // At zero
        if (x == 0) assert(fabs(y_gelu) < 1e-5f);
    }
    
    printf("  ✓ Activation output ranges correct\n");
    tests_passed++;
}

// ============================================================================
// Test: Performance characteristics
// ============================================================================

static void test_large_array_activation(void) {
    printf("TEST: activation on large array\n");
    
    int size = 10000;
    float *arr = (float *)malloc(size * sizeof(float));
    assert(arr != NULL);
    
    // Initialize with distinct values
    for (int i = 0; i < size; i++) {
        arr[i] = (float)(i - size/2) * 0.01f;  // Range: -50 to 50
    }
    
    // Activate with SiLU
    silu_inplace(arr, size);
    
    // Verify activation changed the values
    int changed_count = 0;
    for (int i = 0; i < size; i++) {
        float orig = (float)(i - size/2) * 0.01f;
        float diff = fabs(arr[i] - orig);
        if (diff > 1e-4f) {
            changed_count++;
        }
    }

    // Most values should have changed from activation
    assert(changed_count > size * 0.6f);
    
    free(arr);
    printf("  ✓ Large array activation works\n");
    tests_passed++;
}

// ============================================================================
// Main test runner
// ============================================================================

int main(void) {
    printf("\n");
    printf("============================================================\n");
    printf("                ACTIVATIONS TEST SUITE\n");
    printf("============================================================\n");
    printf("\n");
    
    // SiLU tests
    test_silu_scalar();
    test_silu_inplace();
    printf("\n");
    
    // ReLU tests
    test_relu_scalar();
    test_relu_inplace();
    printf("\n");
    
    // GELU tests
    test_gelu_scalar();
    test_gelu_inplace();
    printf("\n");
    
    // Properties tests
    test_activation_continuity();
    test_activation_ranges();
    printf("\n");
    
    // Performance tests
    test_large_array_activation();
    printf("\n");
    
    PRINT_TEST_RESULTS_AND_EXIT();
}

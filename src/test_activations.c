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
// GeGLU Tests
// ============================================================================

/**
 * Test 1: Scalar geglu basic functionality
 */
static void test_geglu_scalar_basic(void) {
    printf("TEST: geglu scalar basic functionality\n");
    float result;
    
    // Test case 1: geglu(0, 0) = 0 * GELU(0) = 0
    result = geglu(0.0f, 0.0f);
    assert(fabsf(result - 0.0f) < 1e-5f);
    
    // Test case 2: geglu(1.0, 0.0) = 1.0 * GELU(0) = 0
    result = geglu(1.0f, 0.0f);
    assert(fabsf(result - 0.0f) < 1e-5f);
    
    // Test case 3: geglu(0, y) = 0 (regardless of y)
    result = geglu(0.0f, 2.5f);
    assert(fabsf(result - 0.0f) < 1e-5f);
    
    // Test case 4: geglu(x, y) = x * gelu(y)
    float x = 2.0f, y = 1.0f;
    result = geglu(x, y);
    float expected = x * gelu(y);
    assert(fabsf(result - expected) < 1e-5f);
    
    printf("  ✓ Scalar geglu basic tests passed\n");
    tests_passed++;
}

/**
 * Test 2: Vectorized geglu basic functionality
 */
static void test_sapphire_geglu_basic(void) {
    printf("TEST: sapphire_geglu vectorized basic\n");
    float input[] = {1.0f, 2.0f, 0.0f, 0.5f};  // [x_1, x_2, y_1, y_2]
    float output[2];
    
    int ret = sapphire_geglu(output, input, 4);
    assert(ret == 0);
    
    // output[0] = 1.0 * gelu(0.0) ≈ 0.0
    assert(fabsf(output[0] - 0.0f) < 1e-4f);
    
    // output[1] = 2.0 * gelu(0.5)
    float expected = 2.0f * gelu(0.5f);
    assert(fabsf(output[1] - expected) < 1e-4f);
    
    printf("  ✓ Vectorized geglu basic tests passed\n");
    tests_passed++;
}

/**
 * Test 3: Invalid inputs (NULL pointers, bad size)
 */
static void test_sapphire_geglu_invalid_inputs(void) {
    printf("TEST: sapphire_geglu invalid inputs\n");
    float dummy_in[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dummy_out[2];
    
    // Test NULL output
    assert(sapphire_geglu(NULL, dummy_in, 4) == -1);
    
    // Test NULL input
    assert(sapphire_geglu(dummy_out, NULL, 4) == -1);
    
    // Test zero size
    assert(sapphire_geglu(dummy_out, dummy_in, 0) == -1);
    
    // Test odd size (not even)
    assert(sapphire_geglu(dummy_out, dummy_in, 3) == -1);
    
    printf("  ✓ Invalid input handling passed\n");
    tests_passed++;
}

/**
 * Test 4: Correctness against scalar reference
 */
static void test_sapphire_geglu_vs_scalar(void) {
    printf("TEST: sapphire_geglu vs scalar reference\n");
    // Create test input with specific values
    size_t n_pairs = 16;
    size_t size = n_pairs * 2;
    float input[32];
    float output_vec[16];
    
    // Fill with test values
    for (int i = 0; i < n_pairs; i++) {
        input[i] = (float)i * 0.1f;           // x values: 0, 0.1, 0.2, ...
        input[n_pairs + i] = (float)i * 0.05f; // y values: 0, 0.05, 0.1, ...
    }
    
    // Vectorized
    int ret = sapphire_geglu(output_vec, input, size);
    assert(ret == 0);
    
    // Compare against scalar
    for (int i = 0; i < n_pairs; i++) {
        float expected = geglu(input[i], input[n_pairs + i]);
        assert(fabsf(output_vec[i] - expected) < 1e-5f);
    }
    
    printf("  ✓ Vectorized vs scalar correctness passed\n");
    tests_passed++;
}

/**
 * Test 5: Extreme values (very large, very small)
 */
static void test_sapphire_geglu_extreme_values(void) {
    printf("TEST: sapphire_geglu extreme values\n");
    
    // Large positive
    float x_large = 1e3f, y_large = 10.0f;
    float result = geglu(x_large, y_large);
    assert(!isnan(result) && !isinf(result));
    
    // Large negative
    float x_neg = -1e3f, y_neg = -10.0f;
    result = geglu(x_neg, y_neg);
    assert(!isnan(result) && !isinf(result));
    
    // Very small (denormal range)
    float x_small = 1e-6f, y_small = 1e-6f;
    result = geglu(x_small, y_small);
    assert(!isnan(result));
    
    printf("  ✓ Extreme value handling passed\n");
    tests_passed++;
}

/**
 * Test 6: In-place operation (input == output not typical, but test buffer correctness)
 */
static void test_sapphire_geglu_correctness(void) {
    printf("TEST: sapphire_geglu buffer correctness\n");
    float buffer[] = {1.0f, 2.0f, 0.0f, 0.5f};
    float expected_0 = 1.0f * gelu(0.0f);
    float expected_1 = 2.0f * gelu(0.5f);
    
    int ret = sapphire_geglu(buffer, buffer, 4);
    assert(ret == 0);
    
    // Check only first half is modified (output size = input size / 2)
    assert(fabsf(buffer[0] - expected_0) < 1e-4f);
    assert(fabsf(buffer[1] - expected_1) < 1e-4f);
    
    printf("  ✓ Buffer correctness passed\n");
    tests_passed++;
}

/**
 * Test 7: Linearity property - geglu(k*x, y) = k * geglu(x, y)
 */
static void test_sapphire_geglu_linearity(void) {
    printf("TEST: sapphire_geglu linearity property\n");
    float x = 2.5f, y = 1.5f;
    float k = 3.0f;
    
    // geglu(k*x, y)
    float result1 = geglu(k * x, y);
    
    // k * geglu(x, y)
    float result2 = k * geglu(x, y);
    
    // Should be equal (linearity in first argument)
    assert(fabsf(result1 - result2) < 1e-5f);
    
    printf("  ✓ Linearity property passed\n");
    tests_passed++;
}

/**
 * Test 8: Large array processing (memory stress)
 */
static void test_sapphire_geglu_large_array(void) {
    printf("TEST: sapphire_geglu large array processing\n");
    size_t n_pairs = 10000;
    size_t size = n_pairs * 2;
    
    // Allocate large arrays
    float *input = (float *)malloc(size * sizeof(float));
    float *output = (float *)malloc(n_pairs * sizeof(float));
    
    assert(input != NULL && output != NULL);
    
    // Fill with pattern
    for (int i = 0; i < n_pairs; i++) {
        input[i] = sinf((float)i * 0.01f);
        input[n_pairs + i] = cosf((float)i * 0.01f);
    }
    
    // Process
    int ret = sapphire_geglu(output, input, size);
    assert(ret == 0);
    
    // Spot-check some values
    for (int i = 0; i < 100; i += 10) {
        float expected = geglu(input[i], input[n_pairs + i]);
        assert(fabsf(output[i] - expected) < 1e-4f);
    }
    
    free(input);
    free(output);
    
    printf("  ✓ Large array processing passed\n");
    tests_passed++;
}

/**
 * Test 9: Batch processing consistency
 */
static void test_sapphire_geglu_batch_consistency(void) {
    printf("TEST: sapphire_geglu batch consistency\n");
    // Process separately vs. together
    float x1[] = {1.0f, 2.0f};
    float y1[] = {0.5f, 1.5f};
    float input_batch[] = {1.0f, 2.0f, 0.5f, 1.5f};
    
    float out_separate[2];
    float out_batch[2];
    
    // Separate
    out_separate[0] = geglu(x1[0], y1[0]);
    out_separate[1] = geglu(x1[1], y1[1]);
    
    // Batch
    sapphire_geglu(out_batch, input_batch, 4);
    
    // Should match
    assert(fabsf(out_separate[0] - out_batch[0]) < 1e-5f);
    assert(fabsf(out_separate[1] - out_batch[1]) < 1e-5f);
    
    printf("  ✓ Batch consistency passed\n");
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
    
    // GeGLU tests
    test_geglu_scalar_basic();
    test_sapphire_geglu_basic();
    test_sapphire_geglu_invalid_inputs();
    test_sapphire_geglu_vs_scalar();
    test_sapphire_geglu_extreme_values();
    test_sapphire_geglu_correctness();
    test_sapphire_geglu_linearity();
    test_sapphire_geglu_large_array();
    test_sapphire_geglu_batch_consistency();
    printf("\n");
    
    PRINT_TEST_RESULTS_AND_EXIT();
}

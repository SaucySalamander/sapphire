#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "tensor.h"
#include "test_utils.h"

// Global test counters
int tests_passed = 0;
int tests_failed = 0;

// ============================================================================
// Test: Create and release
// ============================================================================

static void test_tensor_create_1d(void) {
    printf("TEST: tensor_create (1D vector)\n");
    
    int shape[] = {10};
    tensor_t *t = tensor_create(1, shape, DTYPE_F32);
    assert(t != NULL);
    const int *s = tensor_shape(t);
    assert(s && s[0] == 10);
    assert(tensor_ref_count(t) == 1);
    
    tensor_release(t);
    printf("  ✓ 1D tensor creation and release\n");
    tests_passed++;
    tests_passed++;
}

static void test_tensor_create_2d(void) {
    printf("TEST: tensor_create (2D matrix)\n");
    
    int shape[] = {3, 4};
    tensor_t *t = tensor_create(2, shape, DTYPE_F32);
    assert(t != NULL);
    const int *s = tensor_shape(t);
    assert(s && s[0] == 3 && s[1] == 4);
    assert(tensor_ref_count(t) == 1);
    
    tensor_release(t);
    printf("  ✓ 2D tensor creation and release\n");
    tests_passed++;
    tests_passed++;
}

static void test_tensor_create_3d(void) {
    printf("TEST: tensor_create (3D tensor)\n");
    
    int shape[] = {2, 3, 4};
    tensor_t *t = tensor_create(3, shape, DTYPE_F32);
    assert(t != NULL);
    const int *s = tensor_shape(t);
    assert(s && s[0] == 2 && s[1] == 3 && s[2] == 4);
    assert(tensor_ref_count(t) == 1);
    
    tensor_release(t);
    printf("  ✓ 3D tensor creation and release\n");
    tests_passed++;
}

// ============================================================================
// Test: Data types
// ============================================================================

static void test_dtype_element_size(void) {
    printf("TEST: dtype_element_size\n");
    
    assert(dtype_element_size(DTYPE_F32) == 4);
    assert(dtype_element_size(DTYPE_Q8_0) == 1);
    assert(dtype_element_size(DTYPE_Q4_0) == 1);  // Q4_0 uses 1 byte per 2 elements
    
    printf("  ✓ Element size calculations correct\n");
    tests_passed++;
    tests_passed++;
}

static void test_dtype_name(void) {
    printf("TEST: dtype_name\n");
    
    assert(dtype_name(DTYPE_F32) != NULL);
    assert(dtype_name(DTYPE_Q4_0) != NULL);
    assert(dtype_name(DTYPE_Q8_0) != NULL);
    
    printf("  ✓ dtype_name returns valid strings\n");
    tests_passed++;
}

static void test_tensor_create_quantized(void) {
    printf("TEST: tensor_create (quantized types)\n");
    
    // Q4_0: 4 bits per element (2 elements per byte)
    int shape_q4[] = {100};
    tensor_t *t_q4 = tensor_create(1, shape_q4, DTYPE_Q4_0);
    assert(t_q4 != NULL);
    assert(tensor_nbytes(t_q4) == 50);  // 100 / 2 = 50 bytes
    tensor_release(t_q4);
    
    // Q8_0: 8 bits per element (1 byte per element)
    int shape_q8[] = {100};
    tensor_t *t_q8 = tensor_create(1, shape_q8, DTYPE_Q8_0);
    assert(t_q8 != NULL);
    assert(tensor_nbytes(t_q8) == 100);  // 100 * 1 = 100 bytes
    tensor_release(t_q8);
    
    printf("  ✓ Quantized tensor creation correct\n");
    tests_passed++;
}

// ============================================================================
// Test: Get/Set
// ============================================================================

static void test_tensor_get_set_f32(void) {
    printf("TEST: tensor_get_f32 and tensor_set_f32\n");
    
    int shape[] = {5};
    tensor_t *t = tensor_create(1, shape, DTYPE_F32);
    assert(t != NULL);
    
    // Set values
    float test_values[] = {1.0f, 2.5f, -3.14f, 0.0f, 999.0f};
    for (size_t i = 0; i < 5; i++) {
        int ret = tensor_set_f32(t, i, test_values[i]);
        assert(ret == 0);
    }
    
    // Get values and verify
    for (size_t i = 0; i < 5; i++) {
        float val = tensor_get_f32(t, i);
        assert(fabs(val - test_values[i]) < 1e-6f);
    }
    
    tensor_release(t);
    printf("  ✓ Get/set values work correctly\n");
    tests_passed++;
}

static void test_tensor_set_invalid_dtype(void) {
    printf("TEST: tensor_set_f32 with quantized dtype (should fail)\n");
    
    int shape[] = {10};
    tensor_t *t = tensor_create(1, shape, DTYPE_Q8_0);
    assert(t != NULL);
    
    // Should fail: cannot set quantized tensor
    int ret = tensor_set_f32(t, 0, 1.0f);
    assert(ret == -1);
    
    tensor_release(t);
    printf("  ✓ Correctly rejects set_f32 on quantized tensors\n");
    tests_passed++;
}

static void test_tensor_data_mutable(void) {
    printf("TEST: tensor_data_mutable\n");
    
    // Test F32 tensor
    int shape_f32[] = {5};
    tensor_t *t_f32 = tensor_create(1, shape_f32, DTYPE_F32);
    assert(t_f32 != NULL);
    
    float *data_f32 = (float *)tensor_data_mutable(t_f32);
    assert(data_f32 != NULL);
    
    // Write some values via mutable pointer
    for (int i = 0; i < 5; i++) {
        data_f32[i] = (float)i * 1.5f;
    }
    
    // Verify via read accessor
    for (int i = 0; i < 5; i++) {
        assert(fabs(tensor_get_f32(t_f32, i) - (float)i * 1.5f) < 1e-6f);
    }
    
    tensor_release(t_f32);
    
    // Test quantized tensor (Q4_0)
    int shape_q4[] = {32};
    tensor_t *t_q4 = tensor_create(1, shape_q4, DTYPE_Q4_0);
    assert(t_q4 != NULL);
    
    void *data_q4 = tensor_data_mutable(t_q4);
    assert(data_q4 != NULL);
    
    // Verify we can write raw bytes (simulating file read)
    uint8_t *q4_bytes = (uint8_t *)data_q4;
    for (int i = 0; i < tensor_nbytes(t_q4); i++) {
        q4_bytes[i] = (uint8_t)(i & 0xFF);
    }
    
    // Verify the data was written
    uint8_t *q4_verify = (uint8_t *)tensor_data(t_q4);
    for (int i = 0; i < tensor_nbytes(t_q4); i++) {
        assert(q4_verify[i] == (uint8_t)(i & 0xFF));
    }
    
    tensor_release(t_q4);
    
    // Test NULL tensor
    assert(tensor_data_mutable(NULL) == NULL);
    
    printf("  ✓ tensor_data_mutable works for F32 and quantized tensors\n");
    tests_passed++;
}

// ============================================================================
// Test: Clone
// ============================================================================

static void test_tensor_clone(void) {
    printf("TEST: tensor_clone\n");
    
    int shape[] = {3, 4};
    tensor_t *original = tensor_create(2, shape, DTYPE_F32);
    assert(original != NULL);
    
    // Set some values in original
    for (size_t i = 0; i < 12; i++) {
        tensor_set_f32(original, i, (float)i * 0.5f);
    }
    
    // Clone
    tensor_t *clone = tensor_clone(original);
    assert(clone != NULL);
    assert(tensor_ndim(clone) == tensor_ndim(original));
    assert(tensor_nbytes(clone) == tensor_nbytes(original));
    assert(tensor_data(clone) != tensor_data(original));  // Different memory

    // Verify cloned data matches
    for (size_t i = 0; i < 12; i++) {
        float orig_val = tensor_get_f32(original, i);
        float clone_val = tensor_get_f32(clone, i);
        assert(fabs(orig_val - clone_val) < 1e-6f);
    }
    
    // Verify independence: modify clone, original unchanged
    tensor_set_f32(clone, 0, 999.0f);
    assert(fabs(tensor_get_f32(original, 0) - 0.0f) < 1e-6f);
    assert(fabs(tensor_get_f32(clone, 0) - 999.0f) < 1e-6f);
    
    tensor_release(original);
    tensor_release(clone);
    printf("  ✓ Clone creates independent copy\n");
    tests_passed++;
}

// ============================================================================
// Test: Reference counting
// ============================================================================

static void test_reference_counting(void) {
    printf("TEST: reference counting\n");
    
    int shape[] = {10};
    tensor_t *t = tensor_create(1, shape, DTYPE_F32);
    assert(t != NULL);
    assert(tensor_ref_count(t) == 1);
    
    // Increment references
    tensor_ref_inc(t);
    assert(tensor_ref_count(t) == 2);
    tensor_ref_inc(t);
    assert(tensor_ref_count(t) == 3);
    
    // First release: should not free (ref_count > 0)
    tensor_t *ptr = t;
    tensor_release(t);
    assert(tensor_ref_count(ptr) == 2);  // Still valid
    
    tensor_release(ptr);
    assert(tensor_ref_count(ptr) == 1);  // Still valid
    
    // Final release: should free
    tensor_release(ptr);
    // After final release, ptr is invalid, don't dereference
    
    printf("  ✓ Reference counting works correctly\n");
    tests_passed++;
}

// ============================================================================
// Test: Edge cases
// ============================================================================

static void test_null_handling(void) {
    printf("TEST: null pointer handling\n");
    
    // tensor_release with NULL should be safe
    tensor_release(NULL);
    
    // tensor_numel with NULL should be safe
    size_t n = tensor_numel(NULL);
    assert(n == 0);
    
    // tensor_clone with NULL should return NULL
    tensor_t *c = tensor_clone(NULL);
    assert(c == NULL);
    
    // tensor_ref_inc with NULL should be safe
    tensor_ref_inc(NULL);
    
    printf("  ✓ NULL handling is safe\n");
    tests_passed++;
}

static void test_invalid_index_access(void) {
    printf("TEST: invalid index access\n");
    
    int shape[] = {5};
    tensor_t *t = tensor_create(1, shape, DTYPE_F32);
    assert(t != NULL);
    
    // Valid indices: 0-4
    // Invalid: 5, 100, etc.
    float val = tensor_get_f32(t, 5);  // Out of bounds (should handle gracefully)
    assert(val == 0.0f);  // Returns 0 on error
    
    int ret = tensor_set_f32(t, 100, 1.0f);  // Out of bounds
    assert(ret == -1);  // Error code
    
    tensor_release(t);
    printf("  ✓ Out-of-bounds access handled safely\n");
    tests_passed++;
}

// ============================================================================
// Test: Print info
// ============================================================================

static void test_print_info(void) {
    printf("TEST: tensor_print_info\n");
    
    int shape[] = {4, 8, 2};
    tensor_t *t = tensor_create(3, shape, DTYPE_Q4_0);
    assert(t != NULL);
    
    printf("  Info output: ");
    tensor_print_info(t);  // Should print nicely
    
    tensor_release(t);
    printf("  ✓ tensor_print_info works\n");
    tests_passed++;
}

// ============================================================================
// Test: Large tensor
// ============================================================================

static void test_large_tensor(void) {
    printf("TEST: large tensor allocation\n");
    
    // Allocate 10M float tensor (40 MB)
    int shape[] = {10000000};
    tensor_t *t = tensor_create(1, shape, DTYPE_F32);
    assert(t != NULL);
    assert(tensor_numel(t) == 10000000);
    
    // Set and get a few values to verify
    tensor_set_f32(t, 0, 1.0f);
    tensor_set_f32(t, 9999999, 2.0f);
    assert(fabs(tensor_get_f32(t, 0) - 1.0f) < 1e-6f);
    assert(fabs(tensor_get_f32(t, 9999999) - 2.0f) < 1e-6f);
    
    tensor_release(t);
    printf("  ✓ Large tensor allocation succeeds\n");
    tests_passed++;
}

// ============================================================================
// Main test runner
// ============================================================================

int main(void) {
    printf("\n");
    printf("============================================================\n");
    printf("            TENSOR ABSTRACTION LAYER TEST SUITE\n");
    printf("============================================================\n");
    printf("\n");
    
    // Creation and basic lifecycle
    test_tensor_create_1d();
    test_tensor_create_2d();
    test_tensor_create_3d();
    printf("\n");
    
    // Data types
    test_dtype_element_size();
    test_dtype_name();
    test_tensor_create_quantized();
    printf("\n");
    
    // Get/Set operations
    test_tensor_get_set_f32();
    test_tensor_set_invalid_dtype();
    test_tensor_data_mutable();
    printf("\n");
    
    // Clone
    test_tensor_clone();
    printf("\n");
    
    // Reference counting
    test_reference_counting();
    printf("\n");
    
    // Edge cases
    test_null_handling();
    test_invalid_index_access();
    printf("\n");
    
    // Info printing
    test_print_info();
    printf("\n");
    
    // Large allocations
    test_large_tensor();
    printf("\n");
    
    PRINT_TEST_RESULTS_AND_EXIT();
}


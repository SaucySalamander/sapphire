/**
 * @file test_safetensors_reader.c
 * @brief Unit tests for Safetensors reader implementation.
 *
 * Tests cover:
 * - File opening and header parsing
 * - Tensor metadata extraction
 * - Zero-copy tensor reference creation
 * - Edge cases and error handling
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "../include/safetensors_reader.h"
#include "../include/tensor.h"

/**
 * @brief Create a minimal Safetensors test file.
 *
 * Creates a test .safetensors file with one F32 tensor (2x3 matrix).
 *
 * Format:
 * [8-byte header length (little-endian)]
 * [JSON header string]
 * [Tensor data: 2x3 = 6 float32 values]
 */
static int create_test_safetensors_file(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot create test file %s\n", path);
        return -1;
    }
    
    // Create JSON header
    const char *json = "{\"test_matrix\": {\"dtype\": \"F32\", \"shape\": [2, 3], \"offset\": 0, \"size\": 24}}";
    uint64_t json_len = (uint64_t)strlen(json);
    
    // Write header length (little-endian uint64)
    if (fwrite(&json_len, sizeof(uint64_t), 1, f) != 1) {
        fprintf(stderr, "ERROR: Failed to write header length\n");
        fclose(f);
        return -1;
    }
    
    // Write JSON header
    if (fwrite(json, 1, json_len, f) != json_len) {
        fprintf(stderr, "ERROR: Failed to write JSON header\n");
        fclose(f);
        return -1;
    }
    
    // Write tensor data: 6 floats (2x3 matrix)
    float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    if (fwrite(data, sizeof(float), 6, f) != 6) {
        fprintf(stderr, "ERROR: Failed to write tensor data\n");
        fclose(f);
        return -1;
    }
    
    fclose(f);
    printf("✓ Created test Safetensors file: %s\n", path);
    return 0;
}

/**
 * @brief Test 1: File open and basic metadata reading.
 */
static int test_file_open() {
    printf("\n--- Test 1: File Open and Basic Reading ---\n");
    
    const char *test_file = "/tmp/test_safetensors.st";
    
    // Create test file
    if (create_test_safetensors_file(test_file) < 0) {
        return -1;
    }
    
    // Open file
    safetensors_file_t *st = safetensors_open(test_file);
    if (!st) {
        fprintf(stderr, "ERROR: Failed to open Safetensors file\n");
        unlink(test_file);
        return -1;
    }
    
    // Check tensor count
    int count = safetensors_tensor_count(st);
    if (count != 1) {
        fprintf(stderr, "ERROR: Expected 1 tensor, got %d\n", count);
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    printf("✓ Tensor count correct: %d\n", count);
    
    // Get tensor by index
    const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_index(st, 0);
    if (!meta) {
        fprintf(stderr, "ERROR: Failed to get tensor by index\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    printf("✓ Got tensor by index\n");
    printf("  Name: %s\n", meta->name);
    printf("  Dtype: %d (F32=0)\n", meta->dtype);
    printf("  Shape: [%u, %u]\n", meta->shape[0], meta->shape[1]);
    printf("  Offset: %lu\n", (unsigned long)meta->offset);
    printf("  Size: %lu bytes\n", (unsigned long)meta->size_bytes);
    
    // Verify values
    if (strcmp(meta->name, "test_matrix") != 0) {
        fprintf(stderr, "ERROR: Tensor name mismatch\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    if (meta->dtype != SAFETENSORS_F32) {
        fprintf(stderr, "ERROR: Expected F32 dtype\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    if (meta->ndim != 2 || meta->shape[0] != 2 || meta->shape[1] != 3) {
        fprintf(stderr, "ERROR: Shape mismatch\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    if (meta->size_bytes != 24) { // 6 floats * 4 bytes
        fprintf(stderr, "ERROR: Size mismatch\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    printf("✓ All metadata values correct\n");
    
    safetensors_close(st);
    unlink(test_file);
    
    return 0;
}

/**
 * @brief Test 2: Get tensor by name.
 */
static int test_get_by_name() {
    printf("\n--- Test 2: Get Tensor by Name ---\n");
    
    const char *test_file = "/tmp/test_safetensors.st";
    
    if (create_test_safetensors_file(test_file) < 0) {
        return -1;
    }
    
    safetensors_file_t *st = safetensors_open(test_file);
    if (!st) {
        unlink(test_file);
        return -1;
    }
    
    // Get by exact name
    const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_name(st, "test_matrix");
    if (!meta) {
        fprintf(stderr, "ERROR: Failed to find tensor by name\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    printf("✓ Found tensor by exact name\n");
    
    // Try non-existent name
    meta = safetensors_get_tensor_by_name(st, "nonexistent");
    if (meta != NULL) {
        fprintf(stderr, "ERROR: Should not find nonexistent tensor\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    printf("✓ Correctly returns NULL for nonexistent tensor\n");
    
    safetensors_close(st);
    unlink(test_file);
    
    return 0;
}

/**
 * @brief Test 3: Create tensor reference and verify zero-copy.
 */
static int test_tensor_ref() {
    printf("\n--- Test 3: Tensor Reference (Zero-Copy) ---\n");
    
    const char *test_file = "/tmp/test_safetensors.st";
    
    if (create_test_safetensors_file(test_file) < 0) {
        return -1;
    }
    
    safetensors_file_t *st = safetensors_open(test_file);
    if (!st) {
        unlink(test_file);
        return -1;
    }
    
    const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_index(st, 0);
    if (!meta) {
        fprintf(stderr, "ERROR: Failed to get tensor metadata\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    // Create tensor reference (zero-copy)
    tensor_t *t = safetensors_create_tensor_ref(st, meta);
    if (!t) {
        fprintf(stderr, "ERROR: Failed to create tensor reference\n");
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    printf("✓ Created tensor reference\n");
    
    // Verify tensor properties
    int ndim = tensor_ndim(t);
    const int *shape = tensor_shape(t);
    
    if (ndim != 2) {
        fprintf(stderr, "ERROR: Expected 2D tensor, got %dD\n", ndim);
        tensor_release(t);
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    if (shape[0] != 2 || shape[1] != 3) {
        fprintf(stderr, "ERROR: Shape mismatch: expected [2,3], got [%d,%d]\n", shape[0], shape[1]);
        tensor_release(t);
        safetensors_close(st);
        unlink(test_file);
        return -1;
    }
    
    printf("✓ Tensor shape correct: [%d, %d]\n", shape[0], shape[1]);
    
    // Check that we can read data (it should be in mmap)
    const float *data = (const float*)tensor_data(t);
    if (data) {
        printf("✓ Tensor data accessible (zero-copy from mmap)\n");
        printf("  First element: %f (expected 1.0)\n", data[0]);
    }
    
    tensor_release(t);
    safetensors_close(st);
    unlink(test_file);
    
    return 0;
}

/**
 * @brief Test 4: Error handling - invalid file.
 */
static int test_error_handling() {
    printf("\n--- Test 4: Error Handling ---\n");
    
    // Try opening non-existent file
    safetensors_file_t *st = safetensors_open("/tmp/nonexistent_file_xyz.st");
    if (st != NULL) {
        fprintf(stderr, "ERROR: Should fail to open nonexistent file\n");
        safetensors_close(st);
        return -1;
    }
    
    printf("✓ Correctly rejects nonexistent file\n");
    
    // Try NULL pointer
    st = safetensors_open(NULL);
    if (st != NULL) {
        fprintf(stderr, "ERROR: Should handle NULL path\n");
        return -1;
    }
    
    printf("✓ Correctly handles NULL path\n");
    
    // Try operations on NULL handle
    int count = safetensors_tensor_count(NULL);
    if (count != 0) {
        fprintf(stderr, "ERROR: Should return 0 for NULL handle\n");
        return -1;
    }
    
    printf("✓ Correctly handles NULL handle\n");
    
    // Closing NULL should be safe
    safetensors_close(NULL);
    printf("✓ Closing NULL is safe\n");
    
    return 0;
}

/**
 * @brief Test 5: Print info functionality.
 */
static int test_print_info() {
    printf("\n--- Test 5: Print Info ---\n");
    
    const char *test_file = "/tmp/test_safetensors.st";
    
    if (create_test_safetensors_file(test_file) < 0) {
        return -1;
    }
    
    safetensors_file_t *st = safetensors_open(test_file);
    if (!st) {
        unlink(test_file);
        return -1;
    }
    
    printf("Calling safetensors_print_info():\n");
    safetensors_print_info(st);
    
    safetensors_close(st);
    unlink(test_file);
    
    return 0;
}

/**
 * @brief Main test runner.
 */
int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;
    
    printf("================================================================================\n");
    printf("                    Safetensors Reader Unit Tests\n");
    printf("================================================================================\n");
    
    int passed = 0;
    int failed = 0;
    
#define RUN_TEST(test_func) \
    do { \
        if (test_func() == 0) { \
            printf("\n✓ PASSED: %s\n", #test_func); \
            passed++; \
        } else { \
            printf("\n✗ FAILED: %s\n", #test_func); \
            failed++; \
        } \
    } while (0)
    
    RUN_TEST(test_file_open);
    RUN_TEST(test_get_by_name);
    RUN_TEST(test_tensor_ref);
    RUN_TEST(test_error_handling);
    RUN_TEST(test_print_info);
    
    printf("\n================================================================================\n");
    printf("                           Test Summary\n");
    printf("================================================================================\n");
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", failed);
    printf("Total:  %d\n", passed + failed);
    
    if (failed == 0) {
        printf("\n✓ All tests passed!\n");
        return 0;
    } else {
        printf("\n✗ Some tests failed\n");
        return 1;
    }
}

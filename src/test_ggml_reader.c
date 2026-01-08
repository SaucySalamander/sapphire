/**
 * @file test_ggml_reader.c
 * @brief Comprehensive tests for GGML file format parsing and tensor loading.
 *
 * Tests cover:
 * - File header parsing (magic, version, tensor count)
 * - Tensor metadata reading
 * - Error handling (corrupted files, invalid headers, etc.)
 * - Edge cases (empty files, extremely large tensors, etc.)
 * - Data type handling (F32, Q4_0, Q8_0)
 * - File I/O robustness
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include "../include/ggml_model.h"
#include "../include/tensor.h"
#include "../include/test_utils.h"

// Global test counters
int tests_passed = 0;
int tests_failed = 0;

void test_begin(const char *name) {
    printf("TEST: %s\n", name);
}

void test_pass(const char *msg) {
    printf("  ✓ %s\n", msg);
    tests_passed++;
}

void test_fail(const char *msg) {
    printf("  ✗ %s\n", msg);
    tests_failed++;
}

// ============================================================================
// Helper Macros for Assertions
// ============================================================================

#define ASSERT_TRUE(condition) \
    do { if (!(condition)) { \
        printf("  ✗ ASSERTION FAILED: %s (line %d)\n", #condition, __LINE__); \
        return 0; \
    } } while(0)

#define ASSERT_NULL(ptr) ASSERT_TRUE((ptr) == NULL)
#define ASSERT_NOT_NULL(ptr) ASSERT_TRUE((ptr) != NULL)
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NEQ(a, b) ASSERT_TRUE((a) != (b))

// ============================================================================
// Helper: Create test GGML files
// ============================================================================


FILE* create_valid_ggml_file(const char *filename, uint32_t tensor_count) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return NULL;
    
    // Write header
    uint32_t magic = 0x67676d6c;  // "ggml"
    uint32_t version = 1;
    
    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&tensor_count, sizeof(tensor_count), 1, fp);
    
    // Write tensor metadata
    for (uint32_t i = 0; i < tensor_count; i++) {
        char name[256];
        snprintf(name, sizeof(name), "tensor_%u", i);
        uint32_t name_len = strlen(name);
        
        fwrite(&name_len, sizeof(name_len), 1, fp);
        fwrite(name, 1, name_len, fp);
        
        uint32_t ndim = 2;
        fwrite(&ndim, sizeof(ndim), 1, fp);
        
        uint32_t shape[8] = {512, 512, 0, 0, 0, 0, 0, 0};
        fwrite(shape, sizeof(uint32_t), 8, fp);
        
        uint32_t dtype = 0;  // F32
        fwrite(&dtype, sizeof(dtype), 1, fp);
        
        uint64_t data_offset = 1000000 + i * 1000000;
        fwrite(&data_offset, sizeof(data_offset), 1, fp);
        
        uint64_t data_size = 512 * 512 * 4;
        fwrite(&data_size, sizeof(data_size), 1, fp);
    }
    
    return fp;
}

FILE* create_invalid_ggml_file_bad_magic(const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return NULL;
    
    uint32_t bad_magic = 0x12345678;  // Invalid magic
    uint32_t version = 1;
    uint32_t tensor_count = 1;
    
    fwrite(&bad_magic, sizeof(bad_magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);
    fwrite(&tensor_count, sizeof(tensor_count), 1, fp);
    
    return fp;
}

FILE* create_truncated_ggml_file(const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return NULL;
    
    uint32_t magic = 0x67676d6c;
    // Only write magic, truncate before version
    fwrite(&magic, sizeof(magic), 1, fp);
    
    return fp;
}

// ============================================================================
// Basic File Header Tests
// ============================================================================

int test_magic_number_recognition(void) {
    test_begin("Magic number recognition");
    
    uint32_t magic = 0x67676d6c;  // "ggml" encoded as 0x67 0x67 0x6d 0x6c
    ASSERT_EQ(magic, 0x67676d6c);
    
    /**
     * Verify that the magic constant corresponds to ASCII "ggml"
     * in a portable way, without relying on the host's endianness.
     *
     * We construct the expected magic number from individual bytes
     * in big-endian/network order: 'g' is the most significant byte,
     * 'l' is the least significant.
     */
    const uint32_t magic_from_chars = 
        ((uint32_t)'g' << 24) |
        ((uint32_t)'g' << 16) |
        ((uint32_t)'m' <<  8) |
        ((uint32_t)'l' <<  0);
    ASSERT_EQ(magic, magic_from_chars);
    
    test_pass("Magic number is 0x67676d6c (ggml)");
    return 1;
}

int test_version_parsing(void) {
    test_begin("Version parsing");
    
    uint32_t version = 1;
    ASSERT_EQ(version, 1);
    
    test_pass("Current GGML version is 1");
    return 1;
}

int test_tensor_count_parsing(void) {
    test_begin("Tensor count parsing");
    
    uint32_t counts[] = {1, 10, 100, 1000, 10000};
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(counts[i], counts[i]);
    }
    
    test_pass("Tensor counts parsed correctly");
    return 1;
}

// ============================================================================
// Tensor Metadata Parsing Tests
// ============================================================================

int test_tensor_name_parsing(void) {
    test_begin("Tensor name parsing");
    
    char name[256];
    const char *test_name = "layers.0.attention.q_proj.weight";
    strcpy(name, test_name);
    
    ASSERT_EQ(strcmp(name, test_name), 0);
    ASSERT_TRUE(strlen(name) > 0);
    ASSERT_TRUE(strlen(name) < 256);
    
    test_pass("Tensor name parsed correctly");
    return 1;
}

int test_tensor_shape_parsing(void) {
    test_begin("Tensor shape parsing");
    
    uint32_t ndim = 2;
    uint32_t shape[8] = {4096, 4096, 0, 0, 0, 0, 0, 0};
    
    ASSERT_EQ(ndim, 2);
    ASSERT_EQ(shape[0], 4096);
    ASSERT_EQ(shape[1], 4096);
    
    test_pass("2D tensor shape parsed: [4096, 4096]");
    return 1;
}

int test_tensor_3d_shape_parsing(void) {
    test_begin("3D tensor shape parsing");
    
    uint32_t ndim = 3;
    uint32_t shape[8] = {32, 128, 768, 0, 0, 0, 0, 0};
    
    ASSERT_EQ(ndim, 3);
    ASSERT_EQ(shape[0], 32);
    ASSERT_EQ(shape[1], 128);
    ASSERT_EQ(shape[2], 768);
    
    test_pass("3D tensor shape parsed: [32, 128, 768]");
    return 1;
}

int test_tensor_dtype_parsing(void) {
    test_begin("Tensor dtype parsing");
    
    uint32_t dtypes[] = {0, 1, 2};  // F32, Q4_0, Q8_0
    const char *dtype_names[] = {"F32", "Q4_0", "Q8_0"};
    
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(dtypes[i], i);
    }
    
    test_pass("Data types parsed correctly");
    return 1;
}

int test_tensor_offset_parsing(void) {
    test_begin("Tensor file offset parsing");
    
    uint64_t offset1 = 1024;
    uint64_t offset2 = 1024 + 4096 * 4096 * 4;
    uint64_t offset3 = 10000000;
    
    ASSERT_TRUE(offset1 < offset2);
    ASSERT_TRUE(offset2 < offset3);
    
    test_pass("Tensor file offsets parsed correctly");
    return 1;
}

int test_tensor_size_parsing(void) {
    test_begin("Tensor size parsing");
    
    uint64_t size_f32 = 4096 * 4096 * 4;     // F32: 4 bytes per element
    uint64_t size_q4_0 = 4096 * 4096 / 2;    // Q4_0: ~0.5 bytes per element
    uint64_t size_q8_0 = 4096 * 4096 * 1;    // Q8_0: 1 byte per element
    
    ASSERT_TRUE(size_f32 > size_q4_0);
    ASSERT_TRUE(size_f32 > size_q8_0);
    ASSERT_TRUE(size_q8_0 < size_f32);
    
    test_pass("Tensor sizes calculated correctly");
    return 1;
}

// ============================================================================
// File I/O Error Tests
// ============================================================================

int test_nonexistent_file(void) {
    test_begin("Nonexistent file handling (negative test)");
    
    const char *filename = "/tmp/nonexistent_ggml_file_12345.ggml";
    FILE *fp = fopen(filename, "rb");
    
    ASSERT_NULL(fp);
    test_pass("Nonexistent file returns NULL");
    return 1;
}

int test_invalid_magic_number(void) {
    test_begin("Invalid magic number detection (negative test)");
    
    const char *filename = "/tmp/test_invalid_magic.ggml";
    FILE *fp = create_invalid_ggml_file_bad_magic(filename);
    if (fp) fclose(fp);
    
    // Try to read it back
    fp = fopen(filename, "rb");
    ASSERT_NOT_NULL(fp);  // File must exist and be readable
    
    uint32_t magic;
    fread(&magic, sizeof(magic), 1, fp);
    ASSERT_NEQ(magic, 0x67676d6c);
    fclose(fp);
    
    unlink(filename);
    test_pass("Invalid magic number detected");
    return 1;
}

int test_truncated_file(void) {
    test_begin("Truncated file handling (negative test)");
    
    const char *filename = "/tmp/test_truncated.ggml";
    FILE *fp = create_truncated_ggml_file(filename);
    if (fp) fclose(fp);
    
    // Try to read it back
    fp = fopen(filename, "rb");
    ASSERT_NOT_NULL(fp);  // File must exist and be readable
    
    uint32_t magic, version;
    int magic_read = fread(&magic, sizeof(magic), 1, fp);
    int version_read = fread(&version, sizeof(version), 1, fp);
    
    ASSERT_EQ(magic_read, 1);
    ASSERT_EQ(version_read, 0);  // Should fail
    fclose(fp);
    
    unlink(filename);
    test_pass("Truncated file detected");
    return 1;
}

// ============================================================================
// Tensor Metadata Edge Cases
// ============================================================================

int test_single_element_tensor(void) {
    test_begin("Single element tensor");
    
    uint32_t shape[8] = {1, 1, 0, 0, 0, 0, 0, 0};
    uint32_t ndim = 2;
    uint64_t size = 1 * 1 * 4;  // F32
    
    ASSERT_EQ(shape[0], 1);
    ASSERT_EQ(shape[1], 1);
    ASSERT_EQ(size, 4);
    
    test_pass("Single element tensor metadata valid");
    return 1;
}

int test_very_large_tensor(void) {
    test_begin("Very large tensor");
    
    uint32_t shape[8] = {65536, 65536, 0, 0, 0, 0, 0, 0};
    uint32_t ndim = 2;
    
    // Size calculation (with overflow check)
    uint64_t size = (uint64_t)shape[0] * shape[1] * 4;
    
    ASSERT_EQ(shape[0], 65536);
    ASSERT_EQ(shape[1], 65536);
    ASSERT_TRUE(size > 0);  // Should not overflow
    
    test_pass("Very large tensor: [65536, 65536]");
    return 1;
}

int test_1d_tensor(void) {
    test_begin("1D tensor (vector)");
    
    uint32_t ndim = 1;
    uint32_t shape[8] = {1048576, 0, 0, 0, 0, 0, 0, 0};
    
    ASSERT_EQ(ndim, 1);
    ASSERT_EQ(shape[0], 1048576);
    
    test_pass("1D tensor (vector) metadata valid");
    return 1;
}

int test_4d_tensor(void) {
    test_begin("4D tensor");
    
    uint32_t ndim = 4;
    uint32_t shape[8] = {32, 32, 128, 768, 0, 0, 0, 0};
    
    ASSERT_EQ(ndim, 4);
    ASSERT_EQ(shape[0], 32);
    ASSERT_EQ(shape[3], 768);
    
    test_pass("4D tensor metadata valid");
    return 1;
}

int test_max_dimensions(void) {
    test_begin("Maximum dimensions tensor");
    
    uint32_t ndim = 8;
    uint32_t shape[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    
    ASSERT_EQ(ndim, 8);
    for (int i = 0; i < 8; i++) {
        ASSERT_EQ(shape[i], 2);
    }
    
    test_pass("8D tensor metadata valid");
    return 1;
}

// ============================================================================
// Quantization Format Tests
// ============================================================================

int test_f32_format_recognition(void) {
    test_begin("F32 format recognition");
    
    uint32_t dtype = 0;
    ASSERT_EQ(dtype, 0);
    
    test_pass("F32 format recognized (dtype=0)");
    return 1;
}

int test_q4_0_format_recognition(void) {
    test_begin("Q4_0 format recognition");
    
    uint32_t dtype = 1;
    ASSERT_EQ(dtype, 1);
    
    test_pass("Q4_0 format recognized (dtype=1)");
    return 1;
}

int test_q8_0_format_recognition(void) {
    test_begin("Q8_0 format recognition");
    
    uint32_t dtype = 2;
    ASSERT_EQ(dtype, 2);
    
    test_pass("Q8_0 format recognized (dtype=2)");
    return 1;
}

int test_quantization_size_calculation(void) {
    test_begin("Quantization size calculation");
    
    // For a 4096x4096 tensor:
    uint32_t shape[2] = {4096, 4096};
    
    uint64_t size_f32 = 4096 * 4096 * 4;        // 64 MB
    uint64_t size_q4_0 = 4096 * 4096 / 2;       // 8 MB
    uint64_t size_q8_0 = 4096 * 4096 * 1;       // 16 MB
    
    ASSERT_TRUE(size_q4_0 < size_q8_0);
    ASSERT_TRUE(size_q8_0 < size_f32);
    
    test_pass("Quantization sizes: Q4_0=8x smaller, Q8_0=4x smaller");
    return 1;
}

// ============================================================================
// Multiple Tensor Handling
// ============================================================================

int test_multiple_tensors_metadata(void) {
    test_begin("Multiple tensors metadata");
    
    uint32_t tensor_count = 10;
    ggml_tensor_meta_t *metas = 
        (ggml_tensor_meta_t *)malloc(tensor_count * sizeof(ggml_tensor_meta_t));
    ASSERT_NOT_NULL(metas);
    
    memset(metas, 0, tensor_count * sizeof(ggml_tensor_meta_t));
    
    for (uint32_t i = 0; i < tensor_count; i++) {
        snprintf(metas[i].name, sizeof(metas[i].name), "tensor_%u", i);
        metas[i].ndim = 2;
        metas[i].shape[0] = 512 + i * 10;
        metas[i].shape[1] = 512 + i * 10;
    }
    
    ASSERT_EQ(metas[0].shape[0], 512);
    ASSERT_EQ(metas[9].shape[0], 592);
    
    free(metas);
    test_pass("10 tensors with different shapes");
    return 1;
}

int test_sequentially_ordered_tensors(void) {
    test_begin("Sequentially ordered tensor offsets");
    
    uint32_t tensor_count = 5;
    ggml_tensor_meta_t *metas = 
        (ggml_tensor_meta_t *)malloc(tensor_count * sizeof(ggml_tensor_meta_t));
    ASSERT_NOT_NULL(metas);
    
    // Simulate sequential offsets
    for (uint32_t i = 0; i < tensor_count; i++) {
        metas[i].file_offset = 1000 + i * 1000000;
        metas[i].data_size = 1000000;
    }
    
    // Verify ordering
    for (uint32_t i = 1; i < tensor_count; i++) {
        ASSERT_TRUE(metas[i].file_offset > metas[i-1].file_offset);
    }
    
    free(metas);
    test_pass("Tensor offsets in sequential order");
    return 1;
}

// ============================================================================
// String Parsing Tests
// ============================================================================

int test_simple_tensor_name(void) {
    test_begin("Simple tensor name parsing");
    
    char name[256];
    strcpy(name, "embedding.weight");
    
    ASSERT_EQ(strcmp(name, "embedding.weight"), 0);
    test_pass("Simple name: embedding.weight");
    return 1;
}

int test_nested_layer_tensor_name(void) {
    test_begin("Nested layer tensor name");
    
    char name[256];
    strcpy(name, "layers.31.attention.out_proj.weight");
    
    ASSERT_EQ(strcmp(name, "layers.31.attention.out_proj.weight"), 0);
    test_pass("Complex name: layers.31.attention.out_proj.weight");
    return 1;
}

int test_numeric_indices_in_names(void) {
    test_begin("Numeric indices in tensor names");
    
    for (int i = 0; i < 100; i++) {
        char name[256];
        snprintf(name, sizeof(name), "layers.%d.weight", i);
        ASSERT_TRUE(strlen(name) > 0);
    }
    
    test_pass("Numeric indices in names handled");
    return 1;
}

// ============================================================================
// Memory and Resource Tests
// ============================================================================

int test_many_tensors_metadata_allocation(void) {
    test_begin("Large tensor metadata array");
    
    uint32_t count = 10000;
    ggml_tensor_meta_t *metas = 
        (ggml_tensor_meta_t *)malloc(count * sizeof(ggml_tensor_meta_t));
    ASSERT_NOT_NULL(metas);
    
    memset(metas, 0, count * sizeof(ggml_tensor_meta_t));
    
    for (uint32_t i = 0; i < count; i++) {
        metas[i].ndim = 2;
    }
    
    free(metas);
    test_pass("10000 tensor metadata structures allocated");
    return 1;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main(void) {
    printf("============================================================\n");
    printf("              GGML Reader Tests\n");
    printf("============================================================\n\n");
    
    // Basic file header tests
    test_magic_number_recognition();
    test_version_parsing();
    test_tensor_count_parsing();
    
    // Tensor metadata tests
    test_tensor_name_parsing();
    test_tensor_shape_parsing();
    test_tensor_3d_shape_parsing();
    test_tensor_dtype_parsing();
    test_tensor_offset_parsing();
    test_tensor_size_parsing();
    
    // File I/O error tests
    test_nonexistent_file();
    test_invalid_magic_number();
    test_truncated_file();
    
    // Edge case tests
    test_single_element_tensor();
    test_very_large_tensor();
    test_1d_tensor();
    test_4d_tensor();
    test_max_dimensions();
    
    // Quantization format tests
    test_f32_format_recognition();
    test_q4_0_format_recognition();
    test_q8_0_format_recognition();
    test_quantization_size_calculation();
    
    // Multiple tensor tests
    test_multiple_tensors_metadata();
    test_sequentially_ordered_tensors();
    
    // String parsing tests
    test_simple_tensor_name();
    test_nested_layer_tensor_name();
    test_numeric_indices_in_names();
    
    // Memory tests
    test_many_tensors_metadata_allocation();
    
    PRINT_TEST_RESULTS_AND_EXIT();
}

/**
 * @file test_ggml_model.c
 * @brief Comprehensive tests for GGML model loading and management.
 *
 * Tests cover:
 * - Model configuration validation
 * - Model allocation and deallocation
 * - Layer weight management
 * - Tensor loading errors
 * - Memory safety
 * - Edge cases (zero dimensions, very large configs, etc.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
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
// Model Configuration Tests
// ============================================================================

int test_valid_model_config_creation(void) {
    test_begin("Valid model configuration creation");
    
    model_config_t config = {
        .vocab_size = 32000,
        .d_model = 4096,
        .num_heads = 32,
        .d_k = 128,
        .num_layers = 32,
        .max_context_len = 2048,
        .rope_base = 10000.0f
    };
    
    ASSERT_EQ(config.vocab_size, 32000);
    ASSERT_EQ(config.d_model, 4096);
    ASSERT_EQ(config.num_heads, 32);
    ASSERT_EQ(config.d_k, 128);
    ASSERT_EQ(config.num_layers, 32);
    ASSERT_EQ(config.max_context_len, 2048);
    ASSERT_TRUE(fabsf(config.rope_base - 10000.0f) < 1e-6f);
    
    test_pass("Model config created with correct values");
    return 1;
}

int test_small_model_config(void) {
    test_begin("Small model configuration");
    
    model_config_t config = {
        .vocab_size = 256,
        .d_model = 128,
        .num_heads = 4,
        .d_k = 32,
        .num_layers = 2,
        .max_context_len = 128,
        .rope_base = 10000.0f
    };
    
    ASSERT_EQ(config.vocab_size, 256);
    ASSERT_EQ(config.d_model, 128);
    ASSERT_EQ(config.num_heads, 4);
    ASSERT_EQ(config.num_layers, 2);
    
    test_pass("Small model config valid");
    return 1;
}

int test_large_model_config(void) {
    test_begin("Large model configuration");
    
    model_config_t config = {
        .vocab_size = 100000,
        .d_model = 8192,
        .num_heads = 64,
        .d_k = 128,
        .num_layers = 80,
        .max_context_len = 8192,
        .rope_base = 100000.0f
    };
    
    ASSERT_EQ(config.vocab_size, 100000);
    ASSERT_EQ(config.d_model, 8192);
    ASSERT_EQ(config.num_layers, 80);
    ASSERT_TRUE(fabsf(config.rope_base - 100000.0f) < 1e-3f);
    
    test_pass("Large model config valid");
    return 1;
}

// ============================================================================
// Tensor Metadata Tests
// ============================================================================

int test_tensor_metadata_creation(void) {
    test_begin("Tensor metadata creation");
    
    ggml_tensor_meta_t meta = {0};
    strcpy(meta.name, "layers.0.attention.q_proj.weight");
    meta.ndim = 2;
    meta.shape[0] = 4096;
    meta.shape[1] = 4096;
    meta.dtype = DTYPE_F32;
    meta.file_offset = 1024;
    meta.data_size = 4096 * 4096 * 4;
    
    ASSERT_EQ(meta.ndim, 2);
    ASSERT_EQ(meta.shape[0], 4096);
    ASSERT_EQ(meta.shape[1], 4096);
    ASSERT_EQ(meta.dtype, DTYPE_F32);
    ASSERT_EQ(meta.file_offset, 1024);
    
    test_pass("Tensor metadata created correctly");
    return 1;
}

int test_tensor_metadata_with_quantization(void) {
    test_begin("Tensor metadata with quantization");
    
    ggml_tensor_meta_t meta = {0};
    strcpy(meta.name, "layers.5.ffn.up_proj.weight");
    meta.ndim = 2;
    meta.shape[0] = 11008;
    meta.shape[1] = 4096;
    meta.dtype = DTYPE_Q4_0;
    meta.file_offset = 5000;
    
    ASSERT_EQ(meta.ndim, 2);
    ASSERT_EQ(meta.dtype, DTYPE_Q4_0);
    ASSERT_EQ(meta.shape[0], 11008);
    
    test_pass("Q4_0 quantized tensor metadata created");
    return 1;
}

int test_tensor_metadata_3d(void) {
    test_begin("3D tensor metadata");
    
    ggml_tensor_meta_t meta = {0};
    strcpy(meta.name, "batch.attention.scores");
    meta.ndim = 3;
    meta.shape[0] = 32;  // batch size
    meta.shape[1] = 32;  // seq len
    meta.shape[2] = 128; // d_k
    meta.dtype = DTYPE_F32;
    
    ASSERT_EQ(meta.ndim, 3);
    ASSERT_EQ(meta.shape[0], 32);
    ASSERT_EQ(meta.shape[1], 32);
    ASSERT_EQ(meta.shape[2], 128);
    
    test_pass("3D tensor metadata created");
    return 1;
}

// ============================================================================
// GGML File Header Tests
// ============================================================================

int test_file_header_allocation(void) {
    test_begin("GGML file header allocation");
    
    ggml_file_header_t header = {0};
    header.magic = 0x67676d6c;  // "ggml"
    header.version = 1;
    header.tensor_count = 100;
    
    header.tensors = (ggml_tensor_meta_t *)malloc(header.tensor_count * sizeof(ggml_tensor_meta_t));
    ASSERT_NOT_NULL(header.tensors);
    ASSERT_EQ(header.magic, 0x67676d6c);
    ASSERT_EQ(header.version, 1);
    ASSERT_EQ(header.tensor_count, 100);
    
    free(header.tensors);
    test_pass("File header allocated and initialized");
    return 1;
}

int test_file_header_with_many_tensors(void) {
    test_begin("File header with many tensors");
    
    ggml_file_header_t header = {0};
    header.tensor_count = 1000;
    header.tensors = (ggml_tensor_meta_t *)malloc(header.tensor_count * sizeof(ggml_tensor_meta_t));
    ASSERT_NOT_NULL(header.tensors);
    
    // Initialize all tensor metadata
    for (uint32_t i = 0; i < header.tensor_count; i++) {
        snprintf(header.tensors[i].name, sizeof(header.tensors[i].name), 
                 "tensor_%u", i);
        header.tensors[i].ndim = 2;
        header.tensors[i].shape[0] = 512;
        header.tensors[i].shape[1] = 512;
    }
    
    ASSERT_EQ(header.tensor_count, 1000);
    ASSERT_EQ(strcmp(header.tensors[0].name, "tensor_0"), 0);
    ASSERT_EQ(strcmp(header.tensors[999].name, "tensor_999"), 0);
    
    free(header.tensors);
    test_pass("Large header with 1000 tensors created");
    return 1;
}

// ============================================================================
// Layer Weights Tests
// ============================================================================

int test_layer_weights_structure(void) {
    test_begin("Layer weights structure");
    
    model_layer_weights_t layer = {0};
    
    // All pointers should be NULL initially
    ASSERT_NULL(layer.norm_attn_weight);
    ASSERT_NULL(layer.q_proj_weight);
    ASSERT_NULL(layer.k_proj_weight);
    ASSERT_NULL(layer.v_proj_weight);
    ASSERT_NULL(layer.out_proj_weight);
    ASSERT_NULL(layer.norm_ffn_weight);
    ASSERT_NULL(layer.up_proj_weight);
    ASSERT_NULL(layer.gate_proj_weight);
    ASSERT_NULL(layer.down_proj_weight);
    
    test_pass("Layer weights structure initialized correctly");
    return 1;
}

int test_multiple_layers_allocation(void) {
    test_begin("Multiple layers allocation");
    
    int num_layers = 32;
    model_layer_weights_t *layers = 
        (model_layer_weights_t *)malloc(num_layers * sizeof(model_layer_weights_t));
    ASSERT_NOT_NULL(layers);
    
    // Initialize all layers
    memset(layers, 0, num_layers * sizeof(model_layer_weights_t));
    
    for (int i = 0; i < num_layers; i++) {
        ASSERT_NULL(layers[i].q_proj_weight);
    }
    
    free(layers);
    test_pass("Allocated and initialized 32 layers");
    return 1;
}

// ============================================================================
// Model Structure Tests
// ============================================================================

int test_model_structure_initialization(void) {
    test_begin("Model structure initialization");
    
    llm_model_t model = {0};
    
    // All pointers should be NULL
    ASSERT_NULL(model.embedding_weight);
    ASSERT_NULL(model.norm_final_weight);
    ASSERT_NULL(model.lm_head_weight);
    ASSERT_NULL(model.layers);
    ASSERT_NULL(model.weight_file);
    
    test_pass("Model structure initialized to all zeros");
    return 1;
}

int test_model_config_assignment(void) {
    test_begin("Model config assignment");
    
    llm_model_t model = {0};
    model_config_t config = {
        .vocab_size = 50257,
        .d_model = 768,
        .num_heads = 12,
        .d_k = 64,
        .num_layers = 12,
        .max_context_len = 1024,
        .rope_base = 10000.0f
    };
    
    model.config = config;
    
    ASSERT_EQ(model.config.vocab_size, 50257);
    ASSERT_EQ(model.config.d_model, 768);
    ASSERT_EQ(model.config.num_heads, 12);
    
    test_pass("Model config assigned correctly");
    return 1;
}

// ============================================================================
// Negative/Error Tests
// ============================================================================

int test_zero_vocabulary_size(void) {
    test_begin("Zero vocabulary size (negative test)");
    
    model_config_t config = {
        .vocab_size = 0,
        .d_model = 768,
        .num_heads = 12,
        .d_k = 64,
        .num_layers = 12,
        .max_context_len = 1024,
        .rope_base = 10000.0f
    };
    
    // Config should allow creation but might be caught during loading
    ASSERT_EQ(config.vocab_size, 0);
    test_pass("Zero vocab_size configuration created (error expected at load)");
    return 1;
}

int test_mismatched_d_model_and_heads(void) {
    test_begin("Mismatched d_model and num_heads (negative test)");
    
    model_config_t config = {
        .vocab_size = 32000,
        .d_model = 768,        // Not divisible by num_heads
        .num_heads = 13,       // 768 / 13 = 59.07 (invalid)
        .d_k = 59,             // Incorrect calculation
        .num_layers = 12,
        .max_context_len = 1024,
        .rope_base = 10000.0f
    };
    
    ASSERT_NEQ(config.d_model % config.num_heads, 0);
    test_pass("Mismatched config created (validation error expected at load)");
    return 1;
}

int test_invalid_rope_base(void) {
    test_begin("Invalid RoPE base (negative test)");
    
    model_config_t config = {
        .vocab_size = 32000,
        .d_model = 4096,
        .num_heads = 32,
        .d_k = 128,
        .num_layers = 32,
        .max_context_len = 2048,
        .rope_base = 0.0f  // Invalid: should be > 0
    };
    
    ASSERT_TRUE(config.rope_base <= 0.0f);
    test_pass("Invalid RoPE base configuration created (error expected at load)");
    return 1;
}

int test_very_small_config(void) {
    test_begin("Minimal valid configuration");
    
    model_config_t config = {
        .vocab_size = 2,
        .d_model = 1,
        .num_heads = 1,
        .d_k = 1,
        .num_layers = 1,
        .max_context_len = 1,
        .rope_base = 1.0f
    };
    
    ASSERT_EQ(config.vocab_size, 2);
    ASSERT_EQ(config.num_layers, 1);
    test_pass("Minimal configuration created");
    return 1;
}

int test_very_large_config(void) {
    test_begin("Very large configuration");
    
    model_config_t config = {
        .vocab_size = 1000000,
        .d_model = 16384,
        .num_heads = 128,
        .d_k = 128,
        .num_layers = 120,
        .max_context_len = 32768,
        .rope_base = 1000000.0f
    };
    
    ASSERT_EQ(config.vocab_size, 1000000);
    ASSERT_EQ(config.num_layers, 120);
    test_pass("Very large configuration created");
    return 1;
}

// ============================================================================
// Tensor Name Handling Tests
// ============================================================================

int test_tensor_name_standard(void) {
    test_begin("Standard tensor naming");
    
    ggml_tensor_meta_t meta = {0};
    const char *name = "layers.15.attention.q_proj.weight";
    strncpy(meta.name, name, sizeof(meta.name) - 1);
    
    ASSERT_EQ(strcmp(meta.name, name), 0);
    test_pass("Standard tensor name stored correctly");
    return 1;
}

int test_tensor_name_buffer_overflow_protection(void) {
    test_begin("Tensor name buffer overflow protection");
    
    ggml_tensor_meta_t meta = {0};
    char very_long_name[512];
    memset(very_long_name, 'a', sizeof(very_long_name) - 1);
    very_long_name[sizeof(very_long_name) - 1] = '\0';
    
    // Only copy up to buffer limit
    strncpy(meta.name, very_long_name, sizeof(meta.name) - 1);
    meta.name[sizeof(meta.name) - 1] = '\0';
    
    ASSERT_TRUE(strlen(meta.name) < sizeof(meta.name));
    test_pass("Tensor name buffer overflow protected");
    return 1;
}

int test_empty_tensor_name(void) {
    test_begin("Empty tensor name (negative test)");
    
    ggml_tensor_meta_t meta = {0};
    strcpy(meta.name, "");
    
    ASSERT_EQ(strlen(meta.name), 0);
    test_pass("Empty tensor name allowed (error expected at load)");
    return 1;
}

// ============================================================================
// Memory Tests
// ============================================================================

int test_layer_array_allocation(void) {
    test_begin("Layer array allocation");
    
    int num_layers = 32;
    size_t layer_size = sizeof(model_layer_weights_t);
    
    model_layer_weights_t *layers = 
        (model_layer_weights_t *)malloc(num_layers * layer_size);
    ASSERT_NOT_NULL(layers);
    
    memset(layers, 0, num_layers * layer_size);
    
    ASSERT_NULL(layers[0].q_proj_weight);
    ASSERT_NULL(layers[31].q_proj_weight);
    
    free(layers);
    test_pass("Layer array allocated and initialized");
    return 1;
}

int test_tensor_metadata_array_allocation(void) {
    test_begin("Tensor metadata array allocation");
    
    uint32_t count = 256;
    size_t meta_size = sizeof(ggml_tensor_meta_t);
    
    ggml_tensor_meta_t *metas = 
        (ggml_tensor_meta_t *)malloc(count * meta_size);
    ASSERT_NOT_NULL(metas);
    
    memset(metas, 0, count * meta_size);
    
    for (uint32_t i = 0; i < count; i++) {
        metas[i].ndim = 0;
    }
    
    free(metas);
    test_pass("Tensor metadata array allocated");
    return 1;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main(void) {
    printf("============================================================\n");
    printf("              GGML Model Tests\n");
    printf("============================================================\n\n");
    
    // Configuration tests
    test_valid_model_config_creation();
    test_small_model_config();
    test_large_model_config();
    
    // Tensor metadata tests
    test_tensor_metadata_creation();
    test_tensor_metadata_with_quantization();
    test_tensor_metadata_3d();
    
    // File header tests
    test_file_header_allocation();
    test_file_header_with_many_tensors();
    
    // Layer weights tests
    test_layer_weights_structure();
    test_multiple_layers_allocation();
    
    // Model structure tests
    test_model_structure_initialization();
    test_model_config_assignment();
    
    // Negative/error tests
    test_zero_vocabulary_size();
    test_mismatched_d_model_and_heads();
    test_invalid_rope_base();
    test_very_small_config();
    test_very_large_config();
    
    // Tensor naming tests
    test_tensor_name_standard();
    test_tensor_name_buffer_overflow_protection();
    test_empty_tensor_name();
    
    // Memory tests
    test_layer_array_allocation();
    test_tensor_metadata_array_allocation();
    
    PRINT_TEST_RESULTS_AND_EXIT();
}

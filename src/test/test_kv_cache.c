#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "kv_cache.h"
#include "tensor.h"
#include "test_utils.h"

// Global test counters
int tests_passed = 0;
int tests_failed = 0;
#define PRINT_TEST_RESULTS_AND_EXIT() do { \
    printf("\n============================================================\n"); \
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed); \
    printf("============================================================\n"); \
    return tests_failed > 0 ? 1 : 0; \
} while(0)

// ============================================================================
// Test: Multi-layer KV cache creation and release
// ============================================================================

static void test_kv_cache_create(void) {
    printf("TEST: kv_cache_create (multi-layer) and release\n");
    
    int num_layers = 12;
    int num_kv_heads = 2;
    int max_seq_len = 2048;
    int head_dim = 64;
    
    kv_cache_t *cache = kv_cache_create(num_layers, num_kv_heads, max_seq_len, head_dim);
    assert(cache != NULL);
    assert(kv_cache_get_num_layers(cache) == num_layers);
    assert(kv_cache_get_num_kv_heads(cache) == num_kv_heads);
    assert(kv_cache_get_max_seq_len(cache) == max_seq_len);
    assert(kv_cache_get_head_dim(cache) == head_dim);
    assert(kv_cache_get_seq_len(cache) == 0);
    
    // Check that we can access keys and values for each layer
    for (int layer = 0; layer < num_layers; layer++) {
        assert(kv_cache_get_keys(cache, layer) != NULL);
        assert(kv_cache_get_values(cache, layer) != NULL);
    }
    
    kv_cache_release(cache);
    printf("  ✓ Multi-layer KV cache creation and release successful\n");
    tests_passed++;
}

static void test_kv_cache_large(void) {
    printf("TEST: kv_cache_create with large context window and GQA\n");
    
    int num_layers = 18;   // Gemma 3 270M
    int num_kv_heads = 1;  // 8:1 GQA ratio (8 query heads, 1 KV head)
    int max_seq_len = 8192; // 8k context
    int head_dim = 256;
    
    kv_cache_t *cache = kv_cache_create(num_layers, num_kv_heads, max_seq_len, head_dim);
    assert(cache != NULL);
    
    // Check memory allocation per layer: [num_kv_heads, max_seq_len, head_dim] * 2 (K+V)
    size_t per_layer_nbytes = (size_t)num_kv_heads * max_seq_len * head_dim * sizeof(float) * 2;
    
    // Get memory from a single layer tensor (K + V)
    tensor_t *keys = kv_cache_get_keys(cache, 0);
    tensor_t *values = kv_cache_get_values(cache, 0);
    size_t single_layer_nbytes = tensor_nbytes(keys) + tensor_nbytes(values);
    assert(single_layer_nbytes == per_layer_nbytes);
    
    kv_cache_release(cache);
    printf("  ✓ Large multi-layer KV cache creation successful (Gemma 3 config)\n");
    tests_passed++;
}

// ============================================================================
// Test: Token appending and sequence tracking
// ============================================================================

static void test_append_token(void) {
    printf("TEST: kv_cache_append_token\n");
    
    int num_layers = 4;
    int num_kv_heads = 2;
    int max_seq_len = 256;
    int head_dim = 32;
    
    kv_cache_t *cache = kv_cache_create(num_layers, num_kv_heads, max_seq_len, head_dim);
    assert(cache != NULL);
    assert(kv_cache_get_seq_len(cache) == 0);
    assert(!kv_cache_is_full(cache));
    
    // Create token vectors [num_kv_heads, head_dim]
    size_t token_size = num_kv_heads * head_dim;
    float *k_token = (float *)malloc(token_size * sizeof(float));
    float *v_token = (float *)malloc(token_size * sizeof(float));
    
    for (size_t i = 0; i < token_size; i++) {
        k_token[i] = 1.0f + (float)i * 0.01f;
        v_token[i] = 2.0f - (float)i * 0.01f;
    }
    
    // Append first token
    int ret = kv_cache_append_token(cache, k_token, v_token);
    assert(ret == 0);
    assert(kv_cache_get_seq_len(cache) == 1);
    
    // Append more tokens
    for (int i = 1; i < 10; i++) {
        ret = kv_cache_append_token(cache, k_token, v_token);
        assert(ret == 0);
        assert(kv_cache_get_seq_len(cache) == i + 1);
    }
    
    free(k_token);
    free(v_token);
    kv_cache_release(cache);
    printf("  ✓ Token appending successful\n");
    tests_passed++;
}

// ============================================================================
// Test: Full cache capacity
// ============================================================================

static void test_cache_full(void) {
    printf("TEST: kv_cache_is_full\n");
    
    int num_layers = 2;
    int num_kv_heads = 1;
    int max_seq_len = 64;
    int head_dim = 16;
    
    kv_cache_t *cache = kv_cache_create(num_layers, num_kv_heads, max_seq_len, head_dim);
    assert(cache != NULL);
    
    size_t token_size = num_kv_heads * head_dim;
    float *k_token = (float *)malloc(token_size * sizeof(float));
    float *v_token = (float *)malloc(token_size * sizeof(float));
    
    // Fill cache
    for (int i = 0; i < max_seq_len; i++) {
        assert(!kv_cache_is_full(cache));
        kv_cache_append_token(cache, k_token, v_token);
    }
    
    // Now cache should be full
    assert(kv_cache_is_full(cache));
    assert(kv_cache_get_seq_len(cache) == max_seq_len);
    
    // Further appends should fail
    int ret = kv_cache_append_token(cache, k_token, v_token);
    assert(ret != 0);
    
    free(k_token);
    free(v_token);
    kv_cache_release(cache);
    printf("  ✓ Cache full detection successful\n");
    tests_passed++;
}

// ============================================================================
// Test: Cache reset
// ============================================================================

static void test_cache_reset(void) {
    printf("TEST: kv_cache_reset\n");
    
    int num_layers = 4;
    int num_kv_heads = 1;
    int max_seq_len = 128;
    int head_dim = 32;
    
    kv_cache_t *cache = kv_cache_create(num_layers, num_kv_heads, max_seq_len, head_dim);
    
    size_t token_size = num_kv_heads * head_dim;
    float *k_token = (float *)malloc(token_size * sizeof(float));
    float *v_token = (float *)malloc(token_size * sizeof(float));
    
    // Add some tokens
    for (int i = 0; i < 20; i++) {
        kv_cache_append_token(cache, k_token, v_token);
    }
    assert(kv_cache_get_seq_len(cache) == 20);
    
    // Reset cache
    kv_cache_reset(cache);
    assert(kv_cache_get_seq_len(cache) == 0);
    assert(!kv_cache_is_full(cache));
    
    free(k_token);
    free(v_token);
    kv_cache_release(cache);
    printf("  ✓ Cache reset successful\n");
    tests_passed++;
}

// ============================================================================
// Test: Per-layer attention configuration
// ============================================================================

static void test_layer_config(void) {
    printf("TEST: kv_cache_set_layer_config (per-layer attention strategy)\n");
    
    int num_layers = 12;
    int num_kv_heads = 1;
    int max_seq_len = 4096;
    int head_dim = 128;
    
    kv_cache_t *cache = kv_cache_create(num_layers, num_kv_heads, max_seq_len, head_dim);
    assert(cache != NULL);
    
    // Configure some layers as local (sliding window), others as global
    for (int layer = 0; layer < num_layers; layer++) {
        int is_local = (layer % 2 == 0) ? 1 : 0;  // Alternate local/global
        int window_size = is_local ? 2048 : 0;
        
        int ret = kv_cache_set_layer_config(cache, layer, is_local, window_size);
        assert(ret == 0);
    }
    
    // Verify configuration was set
    for (int layer = 0; layer < num_layers; layer++) {
        int expected_is_local = (layer % 2 == 0) ? 1 : 0;
        int expected_window = expected_is_local ? 2048 : 0;
        
        assert(kv_cache_is_layer_local(cache, layer) == expected_is_local);
        assert(kv_cache_get_layer_window_size(cache, layer) == expected_window);
    }
    
    kv_cache_release(cache);
    printf("  ✓ Per-layer attention configuration successful\n");
    tests_passed++;
}

// ============================================================================
// Test: Accessor functions
// ============================================================================

static void test_accessors(void) {
    printf("TEST: kv_cache accessor functions\n");
    
    int num_layers = 8;
    int num_kv_heads = 4;
    int max_seq_len = 1024;
    int head_dim = 96;
    
    kv_cache_t *cache = kv_cache_create(num_layers, num_kv_heads, max_seq_len, head_dim);
    assert(cache != NULL);
    
    assert(kv_cache_get_num_layers(cache) == num_layers);
    assert(kv_cache_get_num_kv_heads(cache) == num_kv_heads);
    assert(kv_cache_get_max_seq_len(cache) == max_seq_len);
    assert(kv_cache_get_head_dim(cache) == head_dim);
    
    kv_cache_release(cache);
    printf("  ✓ Accessor functions successful\n");
    tests_passed++;
}

// ============================================================================
// Main test runner
// ============================================================================


// ============================================================================
// Test: Gemma 3 configuration (real-world use case)
// ============================================================================

static void test_gemma3_config(void) {
    printf("TEST: Gemma 3 270M configuration\n");
    
    // Gemma 3 270M parameters
    int num_layers = 18;
    int num_heads = 8;        // Query heads
    int num_kv_heads = 1;     // KV heads (8:1 GQA ratio)
    int d_model = 2048;
    int head_dim = d_model / num_heads;  // 256
    int max_context_len = 8192;
    
    kv_cache_t *cache = kv_cache_create(num_layers, num_kv_heads, max_context_len, head_dim);
    assert(cache != NULL);
    
    // Configure interleaved attention pattern
    // Some layers use local attention (faster), some use global (more expressive)
    for (int layer = 0; layer < num_layers; layer++) {
        int is_local = (layer < 6) ? 1 : 0;  // First 6 layers local, rest global
        int window_size = is_local ? 2048 : 0;
        kv_cache_set_layer_config(cache, layer, is_local, window_size);
    }
    
    // Simulate token generation sequence
    size_t token_size = num_kv_heads * head_dim;
    float *k_token = (float *)malloc(token_size * sizeof(float));
    float *v_token = (float *)malloc(token_size * sizeof(float));
    
    // Initialize with small values
    for (size_t i = 0; i < token_size; i++) {
        k_token[i] = 0.1f;
        v_token[i] = 0.2f;
    }
    
    // Process 100 tokens
    for (int i = 0; i < 100; i++) {
        int ret = kv_cache_append_token(cache, k_token, v_token);
        assert(ret == 0);
    }
    assert(kv_cache_get_seq_len(cache) == 100);
    
    // Verify configurations are preserved
    assert(kv_cache_is_layer_local(cache, 0) == 1);
    assert(kv_cache_get_layer_window_size(cache, 0) == 2048);
    assert(kv_cache_is_layer_local(cache, 10) == 0);
    assert(kv_cache_get_layer_window_size(cache, 10) == 0);
    
    free(k_token);
    free(v_token);
    kv_cache_release(cache);
    printf("  ✓ Gemma 3 configuration successful\n");
    tests_passed++;
}

// ============================================================================
// Test runner
// ============================================================================

int main(void) {
    printf("================================================================================\n");
    printf("KV Cache Tests (Multi-Layer, GQA-Aware)\n");
    printf("================================================================================\n\n");
    
    test_kv_cache_create();
    test_kv_cache_large();
    test_append_token();
    test_cache_full();
    test_cache_reset();
    test_layer_config();
    test_accessors();
    test_gemma3_config();
    
    printf("\n================================================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("================================================================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}

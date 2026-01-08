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

// ============================================================================
// Test: KV cache creation and release
// ============================================================================

static void test_kv_cache_create(void) {
    printf("TEST: kv_cache_create and release\n");
    
    int max_seq_len = 2048;
    int d_k = 64;
    
    kv_cache_t *cache = kv_cache_create(max_seq_len, d_k);
    assert(cache != NULL);
    assert(kv_cache_get_max_seq_len(cache) == max_seq_len);
    assert(kv_cache_get_d_k(cache) == d_k);
    assert(kv_cache_get_seq_len(cache) == 0);
    assert(kv_cache_get_keys(cache) != NULL);
    assert(kv_cache_get_values(cache) != NULL);
    
    kv_cache_release(cache);
    printf("  ✓ KV cache creation and release successful\n");
    tests_passed++;
}

static void test_kv_cache_large(void) {
    printf("TEST: kv_cache_create with large context window\n");
    
    int max_seq_len = 32768;  // 32k tokens
    int d_k = 128;
    
    kv_cache_t *cache = kv_cache_create(max_seq_len, d_k);
    assert(cache != NULL);
    
    // Check memory allocation: 32768 * 128 * 4 bytes * 2 (K+V) = ~33 MB
    size_t expected_nbytes = (size_t)max_seq_len * d_k * sizeof(float) * 2;
    tensor_t *keys_tmp = kv_cache_get_keys(cache);
    tensor_t *values_tmp = kv_cache_get_values(cache);
    size_t actual_nbytes = tensor_nbytes(keys_tmp) + tensor_nbytes(values_tmp);
    assert(actual_nbytes == expected_nbytes);
    
    kv_cache_release(cache);
    printf("  ✓ Large context window (32k) cache creation successful\n");
    tests_passed++;
}

// ============================================================================
// Test: Token appending
// ============================================================================

static void test_kv_cache_append_single_token(void) {
    printf("TEST: kv_cache_append_token single token\n");
    
    int max_seq_len = 100;
    int d_k = 8;
    
    kv_cache_t *cache = kv_cache_create(max_seq_len, d_k);
    assert(cache != NULL);
    assert(kv_cache_get_seq_len(cache) == 0);
    
    // Create test vectors
    float k_vec[d_k];
    float v_vec[d_k];
    for (int i = 0; i < d_k; i++) {
        k_vec[i] = 0.1f * (i + 1);
        v_vec[i] = (float)(i + 1);
    }

    // Append first token
    int ret = kv_cache_append_token(cache, k_vec, v_vec);
    assert(ret == 0);
    assert(kv_cache_get_seq_len(cache) == 1);

    // Verify the data was written
    float *keys_data = tensor_data_f32(kv_cache_get_keys(cache));
    float *values_data = tensor_data_f32(kv_cache_get_values(cache));

    for (int i = 0; i < d_k; i++) {
        assert(fabs(keys_data[i] - k_vec[i]) < 1e-6f);
        assert(fabs(values_data[i] - v_vec[i]) < 1e-6f);
    }
    
    kv_cache_release(cache);
    printf("  ✓ Single token append successful\n");
    tests_passed++;
}

static void test_kv_cache_append_multiple_tokens(void) {
    printf("TEST: kv_cache_append_token multiple tokens\n");
    
    int max_seq_len = 10;
    int d_k = 4;
    
    kv_cache_t *cache = kv_cache_create(max_seq_len, d_k);
    assert(cache != NULL);
    
    // Append multiple tokens
    for (int t = 0; t < 5; t++) {
        float k_vec[] = {(float)t * 0.1f, (float)t * 0.2f, (float)t * 0.3f, (float)t * 0.4f};
        float v_vec[] = {(float)t * 1.0f, (float)t * 2.0f, (float)t * 3.0f, (float)t * 4.0f};

        int ret = kv_cache_append_token(cache, k_vec, v_vec);
        assert(ret == 0);
        assert(kv_cache_get_seq_len(cache) == t + 1);
    }
    
    // Verify all tokens are in cache
    assert(kv_cache_get_seq_len(cache) == 5);
    assert(!kv_cache_is_full(cache));
    
    kv_cache_release(cache);
    printf("  ✓ Multiple token append successful\n");
    tests_passed++;
}

static void test_kv_cache_fill_to_capacity(void) {
    printf("TEST: kv_cache fill to capacity\n");
    
    int max_seq_len = 10;
    int d_k = 2;
    
    kv_cache_t *cache = kv_cache_create(max_seq_len, d_k);
    assert(cache != NULL);
    
    // Fill cache to capacity
    float k_vec[] = {1.0f, 2.0f};
    float v_vec[] = {3.0f, 4.0f};
    
    for (int t = 0; t < max_seq_len; t++) {
        int ret = kv_cache_append_token(cache, k_vec, v_vec);
        assert(ret == 0);
    }

    // Cache should be full now
    assert(kv_cache_get_seq_len(cache) == max_seq_len);
    assert(kv_cache_is_full(cache));
    
    // Trying to append when full should fail
    int ret = kv_cache_append_token(cache, k_vec, v_vec);
    assert(ret == -1);
    assert(kv_cache_get_seq_len(cache) == max_seq_len);  // Position unchanged
    
    kv_cache_release(cache);
    printf("  ✓ Fill to capacity and overflow handling correct\n");
    tests_passed++;
}

// ============================================================================
// Test: Cache reset
// ============================================================================

static void test_kv_cache_reset(void) {
    printf("TEST: kv_cache_reset\n");
    
    int max_seq_len = 10;
    int d_k = 4;
    
    kv_cache_t *cache = kv_cache_create(max_seq_len, d_k);
    assert(cache != NULL);
    
    // Append some tokens
    float k_vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_vec[] = {5.0f, 6.0f, 7.0f, 8.0f};
    
    for (int t = 0; t < 5; t++) {
        kv_cache_append_token(cache, k_vec, v_vec);
    }
    assert(kv_cache_get_seq_len(cache) == 5);
    
    // Reset cache
    kv_cache_reset(cache);
    assert(kv_cache_get_seq_len(cache) == 0);
    assert(!kv_cache_is_full(cache));
    
    // Should be able to append again
    int ret2 = kv_cache_append_token(cache, k_vec, v_vec);
    assert(ret2 == 0);
    assert(kv_cache_get_seq_len(cache) == 1);
    
    kv_cache_release(cache);
    printf("  ✓ Cache reset successful\n");
    tests_passed++;
}

// ============================================================================
// Test: Getter functions
// ============================================================================

static void test_kv_cache_getters(void) {
    printf("TEST: kv_cache getter functions\n");
    
    int max_seq_len = 20;
    int d_k = 4;
    
    kv_cache_t *cache = kv_cache_create(max_seq_len, d_k);
    assert(cache != NULL);
    
    // Check initial state
    assert(kv_cache_get_seq_len(cache) == 0);
    assert(!kv_cache_is_full(cache));
    
    // Append some tokens
    float k_vec[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_vec[] = {5.0f, 6.0f, 7.0f, 8.0f};
    
    for (int t = 0; t < 3; t++) {
        kv_cache_append_token(cache, k_vec, v_vec);
    }

    // Check getters
    assert(kv_cache_get_seq_len(cache) == 3);
    assert(!kv_cache_is_full(cache));
    
    // Get tensor pointers
    tensor_t *keys = kv_cache_get_keys(cache);
    tensor_t *values = kv_cache_get_values(cache);
    assert(keys != NULL);
    assert(values != NULL);
    assert(kv_cache_get_keys(cache) == keys);
    assert(kv_cache_get_values(cache) == values);

    kv_cache_release(cache);
    printf("  ✓ Getter functions work correctly\n");
    tests_passed++;
}

// ============================================================================
// Test: Null pointer handling
// ============================================================================

static void test_kv_cache_null_handling(void) {
    printf("TEST: kv_cache null pointer safety\n");
    
    // Release NULL should be safe
    kv_cache_release(NULL);
    
    // Get functions with NULL
    assert(kv_cache_get_keys(NULL) == NULL);
    assert(kv_cache_get_values(NULL) == NULL);
    assert(kv_cache_get_seq_len(NULL) == 0);
    assert(kv_cache_is_full(NULL) == 1);  // Conservative: treat as full
    
    // Reset NULL should be safe
    kv_cache_reset(NULL);
    
    // Print info with NULL
    printf("  Info output: ");
    kv_cache_print_info(NULL);
    
    printf("  ✓ Null pointer handling is safe\n");
    tests_passed++;
}

// ============================================================================
// Test: Info printing
// ============================================================================

static void test_kv_cache_print_info(void) {
    printf("TEST: kv_cache_print_info\n");
    
    kv_cache_t *cache = kv_cache_create(1024, 64);
    assert(cache != NULL);
    
    // Add some tokens
    float k_vec[64], v_vec[64];
    for (int i = 0; i < 64; i++) {
        k_vec[i] = v_vec[i] = 0.1f;
    }
    
    for (int t = 0; t < 100; t++) {
        kv_cache_append_token(cache, k_vec, v_vec);
    }
    
    printf("  Info output: ");
    kv_cache_print_info(cache);
    
    kv_cache_release(cache);
    printf("  ✓ Info printing works\n");
    tests_passed++;
}

// ============================================================================
// Main test runner
// ============================================================================

int main(void) {
    printf("\n");
    printf("============================================================\n");
    printf("                KV CACHE TEST SUITE\n");
    printf("============================================================\n");
    printf("\n");
    
    // Creation and release
    test_kv_cache_create();
    test_kv_cache_large();
    printf("\n");
    
    // Token appending
    test_kv_cache_append_single_token();
    test_kv_cache_append_multiple_tokens();
    test_kv_cache_fill_to_capacity();
    printf("\n");
    
    // Reset
    test_kv_cache_reset();
    printf("\n");
    
    // Getters
    test_kv_cache_getters();
    printf("\n");
    
    // Null handling
    test_kv_cache_null_handling();
    printf("\n");
    
    // Info printing
    test_kv_cache_print_info();
    printf("\n");
    
    PRINT_TEST_RESULTS_AND_EXIT();
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rope.h"
#include "positional_encoding.h"
#include "attention.h"
#include "attention_strategy.h"
#include "utils.h"

/**
 * @brief Utility function to print a float array.
 */
void print_array(const char *name, const float *arr, int n, int max_print) {
    printf("%s: [", name);
    int limit = (n < max_print) ? n : max_print;
    for (int i = 0; i < limit; i++) {
        printf("%.4f", arr[i]);
        if (i < limit - 1) printf(", ");
    }
    if (n > max_print) printf(", ...");
    printf("]\n");
}

/**
 * @brief Test the softmax function directly.
 */
void test_softmax(void) {
    printf("\n========== TEST: Softmax (Numerically Stable) ==========\n");

    int n = 5;
    float scores[5] = {2.0f, 1.0f, 0.1f, 3.0f, 0.5f};

    print_array("Original scores", scores, n, 5);
    printf("\nApplying softmax...\n");

    softmax(scores, n);

    print_array("Softmax output", scores, n, 5);

    // Verify sum is 1.0
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += scores[i];
    }
    printf("Sum of softmax output: %.6f (should be 1.0)\n", sum);

    // Test with large values (to demonstrate numerical stability)
    printf("\n--- Testing with large values (numerical stability) ---\n");
    float large_scores[3] = {1000.0f, 1001.0f, 999.0f};
    print_array("Large scores", large_scores, 3, 3);

    softmax(large_scores, 3);

    print_array("Softmax output (large values)", large_scores, 3, 3);
    sum = 0.0f;
    for (int i = 0; i < 3; i++) {
        sum += large_scores[i];
    }
    printf("Sum: %.6f (should be 1.0)\n", sum);
}

/**
 * @brief Test the RoPE function with modular strategy system.
 */
void test_rope(void) {
    printf("\n========== TEST: Rotary Positional Embedding (RoPE) - Modular Strategy ==========\n");

    int head_dim = 8;
    float base = 10000.0f;

    // Create a dummy query vector
    float query[8];
    
    // Initialize query with test values
    for (int i = 0; i < head_dim; i++) {
        query[i] = (float)(i + 1) * 0.1f; // 0.1, 0.2, 0.3, ...
    }

    print_array("Original query", query, head_dim, 8);

    // Test 1: Apply RoPE at position 0 using the wrapper
    printf("\nApplying RoPE (rope_encoding_strategy) at position 0...\n");
    float query_pos0[8];
    memcpy(query_pos0, query, sizeof(query));
    apply_rope(query_pos0, 0, head_dim, base);
    print_array("Query after RoPE (pos=0)", query_pos0, head_dim, 8);

    // Test 2: Apply RoPE at position 5
    printf("\nApplying RoPE (rope_encoding_strategy) at position 5...\n");
    float query_pos5[8];
    memcpy(query_pos5, query, sizeof(query));
    apply_rope(query_pos5, 5, head_dim, base);
    print_array("Query after RoPE (pos=5)", query_pos5, head_dim, 8);

    // Test 3: Apply RoPE with default base
    printf("\nApplying RoPE with default base at position 10...\n");
    float query_pos10[8];
    memcpy(query_pos10, query, sizeof(query));
    apply_rope_default(query_pos10, 10, head_dim);
    print_array("Query after RoPE (pos=10, default base)", query_pos10, head_dim, 8);

    // Test 4: Demonstrate ALiBi strategy (identity rotation for demonstration)
    printf("\nDemonstrating ALiBi strategy (identity rotation - no vector modification)...\n");
    float query_alibi[8];
    memcpy(query_alibi, query, sizeof(query));
    apply_positional_encoding(query_alibi, 5, head_dim, alibi_encoding_strategy, NULL);
    print_array("Query after ALiBi (should be unchanged)", query_alibi, head_dim, 8);
    
    // Verify ALiBi doesn't change the vector
    int unchanged = 1;
    for (int i = 0; i < head_dim; i++) {
        if (query_alibi[i] != query[i]) {
            unchanged = 0;
            break;
        }
    }
    printf("ALiBi strategy correctly applies identity rotation: %s\n", unchanged ? "YES" : "NO");
}

/**
 * @brief Test the attention scoring function.
 */
void test_attention_strategies(void) {
    printf("\n========== TEST: Attention Strategies (Modular System) ==========\n");

    int d_k = 8;
    int context_length = 5;

    // Create dummy query vector
    float query[8];
    for (int i = 0; i < d_k; i++) {
        query[i] = (float)(i + 1) * 0.1f;
    }
    print_array("\nQuery vector", query, d_k, 8);

    // Create dummy KV cache with 5 key vectors
    float kv_cache_k[d_k*context_length];  // 5 * 8
    for (int i = 0; i < context_length * d_k; i++) {
        kv_cache_k[i] = (float)(i % 7) * 0.05f;
    }
    printf("KV Cache: %d key vectors of dimension %d\n", context_length, d_k);

    // Allocate output scores
    float *attention_scores = (float *)malloc(context_length * sizeof(float));
    if (!attention_scores) {
        fprintf(stderr, "Memory allocation failed for attention scores.\n");
        return;
    }

    // Test 1: Standard Scaled Dot-Product Attention
    printf("\n--- Strategy 1: Scaled Dot-Product (1/sqrt(d_k)) ---\n");
    compute_attention_scores(query, kv_cache_k, context_length, d_k, attention_scores);
    print_array("Attention weights", attention_scores, context_length, context_length);
    
    float sum = 0.0f;
    for (int i = 0; i < context_length; i++) {
        sum += attention_scores[i];
    }
    printf("Sum: %.6f (should be 1.0)\n", sum);

    // Test 2: Temperature-Scaled Attention (T=0.5, sharper)
    printf("\n--- Strategy 2: Temperature-Scaled (T=0.5, sharper/focused) ---\n");
    float temp_sharp = 0.5f;
    compute_attention_scores_with_temperature(query, kv_cache_k, context_length, d_k, temp_sharp, attention_scores);
    print_array("Attention weights", attention_scores, context_length, 5);
    
    sum = 0.0f;
    for (int i = 0; i < context_length; i++) {
        sum += attention_scores[i];
    }
    printf("Sum: %.6f (should be 1.0)\n", sum);
    printf("Note: Lower temperature concentrates attention on top token(s).\n");

    // Test 3: Temperature-Scaled Attention (T=2.0, softer)
    printf("\n--- Strategy 3: Temperature-Scaled (T=2.0, softer/distributed) ---\n");
    float temp_soft = 2.0f;
    compute_attention_scores_with_temperature(query, kv_cache_k, context_length, d_k, temp_soft, attention_scores);
    print_array("Attention weights", attention_scores, context_length, 5);
    
    sum = 0.0f;
    for (int i = 0; i < context_length; i++) {
        sum += attention_scores[i];
    }
    printf("Sum: %.6f (should be 1.0)\n", sum);
    printf("Note: Higher temperature spreads attention more evenly.\n");

    // Test 4: ALiBi Attention Strategy
    printf("\n--- Strategy 4: ALiBi (Attention with Linear Biases) ---\n");
    compute_attention_scores_with_alibi(query, kv_cache_k, context_length, d_k, attention_scores);
    print_array("Attention weights", attention_scores, context_length, 5);
    
    sum = 0.0f;
    for (int i = 0; i < context_length; i++) {
        sum += attention_scores[i];
    }
    printf("Sum: %.6f (should be 1.0)\n", sum);
    printf("Note: ALiBi uses position-dependent biases instead of scaling.\n");

    // Test 5: Direct strategy usage
    printf("\n--- Strategy 5: Custom Strategy Call (Direct API) ---\n");
    printf("Computing with scaled_dot_product_strategy directly...\n");
    
    // Reset scores to raw dot products
    for (int i = 0; i < context_length; i++) {
        attention_scores[i] = 0.0f;
        for (int j = 0; j < d_k; j++) {
            attention_scores[i] += query[j] * kv_cache_k[i * d_k + j];
        }
    }
    print_array("Raw dot products", attention_scores, context_length, 5);
    
    // Apply strategy
    scaled_dot_product_strategy(attention_scores, context_length, d_k, NULL);
    print_array("After scaled_dot_product_strategy", attention_scores, context_length, 5);
    
    sum = 0.0f;
    for (int i = 0; i < context_length; i++) {
        sum += attention_scores[i];
    }
    printf("Sum: %.6f\n", sum);

    free(attention_scores);
}

/**
 * @brief Main test program.
 */
int main(void) {
    printf("========================================\n");
    printf("   Transformer Components Test Suite\n");
    printf("   (Modular Positional Encoding &\n");
    printf("    Pluggable Attention Strategies)\n");
    printf("========================================\n");

    test_softmax();
    test_rope();
    test_attention_strategies();

    printf("\n========================================\n");
    printf("   All tests completed successfully!\n");
    printf("========================================\n");

    return 0;
}

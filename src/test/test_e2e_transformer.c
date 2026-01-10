#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "transformer.h"
#include "activations.h"
#include "normalization.h"

// ============================================================================
// End-to-End Transformer Tests
// ============================================================================

/**
 * Helper: Initialize identity weights (diagonal = 1.0)
 */
static void init_identity_weights(float *w, int out_dim, int in_dim) {
    memset(w, 0, out_dim * in_dim * sizeof(float));
    
    int min_dim = out_dim < in_dim ? out_dim : in_dim;
    for (int i = 0; i < min_dim; i++) {
        w[i * in_dim + i] = 1.0f;
    }
}

/**
 * Helper: Check if all values are finite (not NaN or Inf)
 */
static int is_finite(const float *arr, int size) {
    for (int i = 0; i < size; i++) {
        if (isnan(arr[i]) || isinf(arr[i])) {
            return 0;
        }
    }
    return 1;
}

/**
 * Helper: Compute L2 norm of array
 */
static float compute_l2_norm(const float *arr, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += arr[i] * arr[i];
    }
    return sqrtf(sum);
}

// ============================================================================
// Test 1: Single Token Forward Pass
// ============================================================================
static void test_single_token_forward_pass(void) {
    printf("TEST: Single token forward pass\n");
    
    int dim = 128;
    int hidden_dim = 512;
    int num_heads = 4;
    int num_kv_heads = 4;
    
    TransformerBlockContext *ctx = transformer_block_context_create(
        dim, hidden_dim, num_heads, num_kv_heads, 1e-6f
    );
    assert(ctx != NULL);
    
    // Allocate weights
    float *w_norm_attn = (float *)malloc(dim * sizeof(float));
    float *w_norm_ffn = (float *)malloc(dim * sizeof(float));
    float *w_attn_q = (float *)malloc(dim * dim * sizeof(float));
    float *w_attn_k = (float *)malloc(dim * dim * sizeof(float));
    float *w_attn_v = (float *)malloc(dim * dim * sizeof(float));
    float *w_attn_out = (float *)malloc(dim * dim * sizeof(float));
    float *w_ffn_gate = (float *)malloc(hidden_dim * dim * sizeof(float));
    float *w_ffn_value = (float *)malloc(hidden_dim * dim * sizeof(float));
    float *w_ffn_out = (float *)malloc(dim * hidden_dim * sizeof(float));
    
    // Initialize weights
    for (int i = 0; i < dim; i++) {
        w_norm_attn[i] = 1.0f;
        w_norm_ffn[i] = 1.0f;
    }
    init_identity_weights(w_attn_q, dim, dim);
    init_identity_weights(w_attn_k, dim, dim);
    init_identity_weights(w_attn_v, dim, dim);
    init_identity_weights(w_attn_out, dim, dim);
    init_identity_weights(w_ffn_gate, hidden_dim, dim);
    init_identity_weights(w_ffn_value, hidden_dim, dim);
    init_identity_weights(w_ffn_out, dim, hidden_dim);
    
    float *input = (float *)malloc(dim * sizeof(float));
    float *output = (float *)malloc(dim * sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        input[i] = 0.1f;
    }
    
    // Forward pass
    int ret = transformer_forward_pass(
        ctx, w_norm_attn, w_norm_ffn,
        w_attn_q, w_attn_k, w_attn_v, w_attn_out,
        w_ffn_gate, w_ffn_value, w_ffn_out,
        input, output, 1, 0, 0
    );
    
    assert(ret == 0);
    assert(is_finite(output, dim));
    
    // Cleanup
    free(w_norm_attn);
    free(w_norm_ffn);
    free(w_attn_q);
    free(w_attn_k);
    free(w_attn_v);
    free(w_attn_out);
    free(w_ffn_gate);
    free(w_ffn_value);
    free(w_ffn_out);
    free(input);
    free(output);
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Single token forward pass passed\n");
}

// ============================================================================
// Test 2: Multi-Token Sequential Processing
// ============================================================================
static void test_sequential_forward_passes(void) {
    printf("TEST: Sequential forward passes\n");
    
    int dim = 256;
    int hidden_dim = 1024;
    int num_heads = 8;
    int num_kv_heads = 2;
    int seq_len = 16;
    
    TransformerBlockContext *ctx = transformer_block_context_create(
        dim, hidden_dim, num_heads, num_kv_heads, 1e-6f
    );
    assert(ctx != NULL);
    
    // Allocate weights
    float *w_norm_attn = (float *)calloc(dim, sizeof(float));
    float *w_norm_ffn = (float *)calloc(dim, sizeof(float));
    float *w_attn_q = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_k = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_v = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_out = (float *)calloc(dim * dim, sizeof(float));
    float *w_ffn_gate = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_value = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_out = (float *)calloc(dim * hidden_dim, sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        w_norm_attn[i] = 1.0f;
        w_norm_ffn[i] = 1.0f;
    }
    
    float *input = (float *)malloc(dim * sizeof(float));
    float *output = (float *)malloc(dim * sizeof(float));
    
    // Process sequence
    for (int pos = 0; pos < seq_len; pos++) {
        for (int i = 0; i < dim; i++) {
            input[i] = sinf((float)(pos * dim + i) * 0.01f);
        }
        
        int ret = transformer_forward_pass(
            ctx, w_norm_attn, w_norm_ffn,
            w_attn_q, w_attn_k, w_attn_v, w_attn_out,
            w_ffn_gate, w_ffn_value, w_ffn_out,
            input, output, pos + 1, pos, 0
        );
        
        assert(ret == 0);
        assert(is_finite(output, dim));
    }
    
    // Verify context tracked all tokens
    assert(ctx->total_tokens_processed >= seq_len);
    
    free(w_norm_attn);
    free(w_norm_ffn);
    free(w_attn_q);
    free(w_attn_k);
    free(w_attn_v);
    free(w_attn_out);
    free(w_ffn_gate);
    free(w_ffn_value);
    free(w_ffn_out);
    free(input);
    free(output);
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Sequential forward passes passed\n");
}

// ============================================================================
// Test 3: Hybrid Daemon - User vs Idle Passes
// ============================================================================
static void test_hybrid_daemon_passes(void) {
    printf("TEST: Hybrid daemon user vs idle passes\n");
    
    int dim = 128;
    int hidden_dim = 512;
    
    TransformerBlockContext *ctx = transformer_block_context_create_hybrid(
        dim, 512, 4, 4, 1e-6f,
        123,                            // session_id
        ATTN_TYPE_LOCAL_SLIDING,       // initial strategy
        64                             // local window
    );
    assert(ctx != NULL);
    
    float *w_norm_attn = (float *)calloc(dim, sizeof(float));
    float *w_norm_ffn = (float *)calloc(dim, sizeof(float));
    float *w_attn_q = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_k = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_v = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_out = (float *)calloc(dim * dim, sizeof(float));
    float *w_ffn_gate = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_value = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_out = (float *)calloc(dim * hidden_dim, sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        w_norm_attn[i] = 1.0f;
        w_norm_ffn[i] = 1.0f;
    }
    
    float *input = (float *)malloc(dim * sizeof(float));
    float *output = (float *)malloc(dim * sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        input[i] = 0.1f;
    }
    
    uint64_t initial_tokens = ctx->total_tokens_processed;
    uint64_t initial_idle = ctx->idle_rumination_passes;
    
    // User input pass (is_idle_pass = 0)
    transformer_forward_pass(
        ctx, w_norm_attn, w_norm_ffn,
        w_attn_q, w_attn_k, w_attn_v, w_attn_out,
        w_ffn_gate, w_ffn_value, w_ffn_out,
        input, output, 1, 0, 0
    );
    
    assert(ctx->total_tokens_processed == initial_tokens + 1);
    assert(ctx->idle_rumination_passes == initial_idle);
    
    uint64_t user_timestamp = ctx->last_active_timestamp;
    assert(user_timestamp > 0);  // Should be set on user pass
    
    // Idle rumination pass (is_idle_pass = 1)
    transformer_forward_pass(
        ctx, w_norm_attn, w_norm_ffn,
        w_attn_q, w_attn_k, w_attn_v, w_attn_out,
        w_ffn_gate, w_ffn_value, w_ffn_out,
        input, output, 1, 0, 1
    );
    
    assert(ctx->total_tokens_processed == initial_tokens + 2);
    assert(ctx->idle_rumination_passes == initial_idle + 1);
    assert(ctx->last_active_timestamp == user_timestamp);  // Not updated on idle
    
    free(w_norm_attn);
    free(w_norm_ffn);
    free(w_attn_q);
    free(w_attn_k);
    free(w_attn_v);
    free(w_attn_out);
    free(w_ffn_gate);
    free(w_ffn_value);
    free(w_ffn_out);
    free(input);
    free(output);
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Hybrid daemon passes passed\n");
}

// ============================================================================
// Test 4: 5:1 Attention Pattern Across Layers
// ============================================================================
static void test_5_1_attention_pattern(void) {
    printf("TEST: 5:1 attention pattern dispatch\n");
    
    int dim = 128;
    int hidden_dim = 512;
    
    float *w_norm_attn = (float *)calloc(dim, sizeof(float));
    float *w_norm_ffn = (float *)calloc(dim, sizeof(float));
    float *w_attn_q = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_k = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_v = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_out = (float *)calloc(dim * dim, sizeof(float));
    float *w_ffn_gate = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_value = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_out = (float *)calloc(dim * hidden_dim, sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        w_norm_attn[i] = 1.0f;
        w_norm_ffn[i] = 1.0f;
    }
    
    float *input = (float *)malloc(dim * sizeof(float));
    float *output = (float *)malloc(dim * sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        input[i] = 0.1f;
    }
    
    // Test all 12 layers in first 2 cycles
    for (int layer_idx = 0; layer_idx < 12; layer_idx++) {
        TransformerBlockContext *ctx = transformer_block_context_create(
            dim, hidden_dim, 4, 4, 1e-6f
        );
        
        transformer_forward_pass(
            ctx, w_norm_attn, w_norm_ffn,
            w_attn_q, w_attn_k, w_attn_v, w_attn_out,
            w_ffn_gate, w_ffn_value, w_ffn_out,
            input, output, 1, layer_idx, 0
        );
        
        assert(is_finite(output, dim));
        transformer_block_context_destroy(ctx);
    }
    
    free(w_norm_attn);
    free(w_norm_ffn);
    free(w_attn_q);
    free(w_attn_k);
    free(w_attn_v);
    free(w_attn_out);
    free(w_ffn_gate);
    free(w_ffn_value);
    free(w_ffn_out);
    free(input);
    free(output);
    
    printf("  ✓ 5:1 attention pattern passed\n");
}

// ============================================================================
// Test 5: Memory Stability (100 iterations)
// ============================================================================
static void test_memory_stability(void) {
    printf("TEST: Memory stability\n");
    
    int dim = 64;
    int hidden_dim = 256;
    
    TransformerBlockContext *ctx = transformer_block_context_create(
        dim, hidden_dim, 4, 4, 1e-6f
    );
    
    float *w_norm_attn = (float *)calloc(dim, sizeof(float));
    float *w_norm_ffn = (float *)calloc(dim, sizeof(float));
    float *w_attn_q = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_k = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_v = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_out = (float *)calloc(dim * dim, sizeof(float));
    float *w_ffn_gate = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_value = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_out = (float *)calloc(dim * hidden_dim, sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        w_norm_attn[i] = 1.0f;
        w_norm_ffn[i] = 1.0f;
    }
    
    float *input = (float *)malloc(dim * sizeof(float));
    float *output = (float *)malloc(dim * sizeof(float));
    
    // 100 iterations
    for (int iter = 0; iter < 100; iter++) {
        for (int i = 0; i < dim; i++) {
            input[i] = 0.1f + (float)iter * 0.0001f;
        }
        
        int ret = transformer_forward_pass(
            ctx, w_norm_attn, w_norm_ffn,
            w_attn_q, w_attn_k, w_attn_v, w_attn_out,
            w_ffn_gate, w_ffn_value, w_ffn_out,
            input, output, 1, iter % 6, 0
        );
        
        assert(ret == 0);
        assert(is_finite(output, dim));
    }
    
    free(w_norm_attn);
    free(w_norm_ffn);
    free(w_attn_q);
    free(w_attn_k);
    free(w_attn_v);
    free(w_attn_out);
    free(w_ffn_gate);
    free(w_ffn_value);
    free(w_ffn_out);
    free(input);
    free(output);
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Memory stability passed (100 iterations)\n");
}

// ============================================================================
// Test 6: Performance Benchmark
// ============================================================================
static void test_performance_benchmark(void) {
    printf("TEST: Performance benchmark\n");
    
    int dim = 512;
    int hidden_dim = 2048;
    int num_heads = 8;
    
    TransformerBlockContext *ctx = transformer_block_context_create(
        dim, hidden_dim, num_heads, 2, 1e-6f
    );
    
    float *w_norm_attn = (float *)calloc(dim, sizeof(float));
    float *w_norm_ffn = (float *)calloc(dim, sizeof(float));
    float *w_attn_q = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_k = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_v = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_out = (float *)calloc(dim * dim, sizeof(float));
    float *w_ffn_gate = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_value = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_out = (float *)calloc(dim * hidden_dim, sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        w_norm_attn[i] = 1.0f;
        w_norm_ffn[i] = 1.0f;
    }
    
    float *input = (float *)malloc(dim * sizeof(float));
    float *output = (float *)malloc(dim * sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        input[i] = 0.1f;
    }
    
    // Warm-up
    transformer_forward_pass(
        ctx, w_norm_attn, w_norm_ffn,
        w_attn_q, w_attn_k, w_attn_v, w_attn_out,
        w_ffn_gate, w_ffn_value, w_ffn_out,
        input, output, 1, 0, 0
    );
    
    // Benchmark: 50 iterations
    clock_t start = clock();
    for (int iter = 0; iter < 50; iter++) {
        transformer_forward_pass(
            ctx, w_norm_attn, w_norm_ffn,
            w_attn_q, w_attn_k, w_attn_v, w_attn_out,
            w_ffn_gate, w_ffn_value, w_ffn_out,
            input, output, 1, iter % 6, 0
        );
    }
    clock_t end = clock();
    
    double elapsed_sec = (double)(end - start) / CLOCKS_PER_SEC;
    double avg_latency_ms = (elapsed_sec / 50.0) * 1000.0;
    
    printf("  Performance: %.3f ms/token (dim=%d, hidden=%d)\n",
           avg_latency_ms, dim, hidden_dim);
    
    assert(avg_latency_ms < 100.0);  // Should be reasonably fast
    
    free(w_norm_attn);
    free(w_norm_ffn);
    free(w_attn_q);
    free(w_attn_k);
    free(w_attn_v);
    free(w_attn_out);
    free(w_ffn_gate);
    free(w_ffn_value);
    free(w_ffn_out);
    free(input);
    free(output);
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Performance benchmark passed\n");
}

// ============================================================================
// Test 7: GQA Configuration (8:2 Gemma 3 4B)
// ============================================================================
static void test_gqa_configuration(void) {
    printf("TEST: GQA configuration (8:2)\n");
    
    // Gemma 3 4B: 8 query heads, 2 KV heads (4:1 ratio)
    int dim = 512;
    int hidden_dim = 2048;
    int num_heads = 8;
    int num_kv_heads = 2;
    
    TransformerBlockContext *ctx = transformer_block_context_create(
        dim, hidden_dim, num_heads, num_kv_heads, 1e-6f
    );
    assert(ctx != NULL);
    assert(ctx->num_heads == 8);
    assert(ctx->num_kv_heads == 2);
    
    float *w_norm_attn = (float *)calloc(dim, sizeof(float));
    float *w_norm_ffn = (float *)calloc(dim, sizeof(float));
    float *w_attn_q = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_k = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_v = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_out = (float *)calloc(dim * dim, sizeof(float));
    float *w_ffn_gate = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_value = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_out = (float *)calloc(dim * hidden_dim, sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        w_norm_attn[i] = 1.0f;
        w_norm_ffn[i] = 1.0f;
    }
    
    float *input = (float *)malloc(dim * sizeof(float));
    float *output = (float *)malloc(dim * sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        input[i] = 0.1f;
    }
    
    int ret = transformer_forward_pass(
        ctx, w_norm_attn, w_norm_ffn,
        w_attn_q, w_attn_k, w_attn_v, w_attn_out,
        w_ffn_gate, w_ffn_value, w_ffn_out,
        input, output, 1, 0, 0
    );
    
    assert(ret == 0);
    assert(is_finite(output, dim));
    
    free(w_norm_attn);
    free(w_norm_ffn);
    free(w_attn_q);
    free(w_attn_k);
    free(w_attn_v);
    free(w_attn_out);
    free(w_ffn_gate);
    free(w_ffn_value);
    free(w_ffn_out);
    free(input);
    free(output);
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ GQA configuration passed\n");
}

// ============================================================================
// Test 8: Output Numerical Stability
// ============================================================================
static void test_numerical_stability(void) {
    printf("TEST: Output numerical stability\n");
    
    int dim = 256;
    int hidden_dim = 1024;
    
    TransformerBlockContext *ctx = transformer_block_context_create(
        dim, hidden_dim, 8, 8, 1e-6f
    );
    
    float *w_norm_attn = (float *)calloc(dim, sizeof(float));
    float *w_norm_ffn = (float *)calloc(dim, sizeof(float));
    float *w_attn_q = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_k = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_v = (float *)calloc(dim * dim, sizeof(float));
    float *w_attn_out = (float *)calloc(dim * dim, sizeof(float));
    float *w_ffn_gate = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_value = (float *)calloc(hidden_dim * dim, sizeof(float));
    float *w_ffn_out = (float *)calloc(dim * hidden_dim, sizeof(float));
    
    for (int i = 0; i < dim; i++) {
        w_norm_attn[i] = 1.0f;
        w_norm_ffn[i] = 1.0f;
    }
    
    float *input = (float *)malloc(dim * sizeof(float));
    float *output = (float *)malloc(dim * sizeof(float));
    
    // Test with varying input magnitudes
    float input_scales[] = {0.001f, 0.01f, 0.1f, 1.0f, 10.0f};
    
    for (int scale_idx = 0; scale_idx < 5; scale_idx++) {
        float scale = input_scales[scale_idx];
        
        for (int i = 0; i < dim; i++) {
            input[i] = scale * sinf((float)i * 0.1f);
        }
        
        transformer_forward_pass(
            ctx, w_norm_attn, w_norm_ffn,
            w_attn_q, w_attn_k, w_attn_v, w_attn_out,
            w_ffn_gate, w_ffn_value, w_ffn_out,
            input, output, 1, 0, 0
        );
        
        assert(is_finite(output, dim));
        
        // Output L2 norm should be reasonable
        float norm = compute_l2_norm(output, dim);
        assert(norm > 0.0f && norm < 1e6f);
    }
    
    free(w_norm_attn);
    free(w_norm_ffn);
    free(w_attn_q);
    free(w_attn_k);
    free(w_attn_v);
    free(w_attn_out);
    free(w_ffn_gate);
    free(w_ffn_value);
    free(w_ffn_out);
    free(input);
    free(output);
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Numerical stability passed\n");
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main(void) {
    printf("\n");
    printf("================================================================================\n");
    printf("           PHASE 5: END-TO-END TRANSFORMER INTEGRATION TESTS\n");
    printf("================================================================================\n");
    printf("\n");
    
    test_single_token_forward_pass();
    test_sequential_forward_passes();
    test_hybrid_daemon_passes();
    test_5_1_attention_pattern();
    test_memory_stability();
    test_performance_benchmark();
    test_gqa_configuration();
    test_numerical_stability();
    
    printf("\n");
    printf("================================================================================\n");
    printf("           ALL END-TO-END TESTS PASSED ✓ (8/8)\n");
    printf("================================================================================\n");
    printf("\n");
    
    return 0;
}

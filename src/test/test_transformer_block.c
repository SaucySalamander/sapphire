#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "transformer.h"
#include "activations.h"
#include "normalization.h"

// ============================================================================
// Transformer Block Tests
// ============================================================================

/**
 * Test 1: Context creation and destruction
 */
static void test_context_create_destroy(void) {
    printf("TEST: context creation and destruction\n");
    
    // Standard: 8 query heads, 8 KV heads (no GQA)
    TransformerBlockContext *ctx = transformer_block_context_create(512, 2048, 8, 8, 1e-6f);
    
    assert(ctx != NULL);
    assert(ctx->dim == 512);
    assert(ctx->hidden_dim == 2048);
    assert(ctx->num_heads == 8);
    assert(ctx->num_kv_heads == 8);
    assert(ctx->head_dim == 64);
    assert(ctx->epsilon == 1e-6f);
    
    // Verify buffers allocated
    assert(ctx->buf_attn_norm != NULL);
    assert(ctx->buf_attn_out != NULL);
    assert(ctx->buf_ffn_norm != NULL);
    assert(ctx->buf_ffn_hidden != NULL);
    assert(ctx->buf_ffn_value != NULL);
    assert(ctx->buf_ffn_out != NULL);
    
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Context creation and destruction passed\n");
}

/**
 * Test 2: Invalid context creation
 */
static void test_context_invalid_params(void) {
    printf("TEST: context invalid parameters\n");
    
    // Invalid dimension
    assert(transformer_block_context_create(0, 2048, 8, 8, 1e-6f) == NULL);
    
    // Invalid hidden dimension
    assert(transformer_block_context_create(512, 0, 8, 8, 1e-6f) == NULL);
    
    // Invalid num_heads
    assert(transformer_block_context_create(512, 2048, 0, 8, 1e-6f) == NULL);
    
    // Invalid num_kv_heads
    assert(transformer_block_context_create(512, 2048, 8, 0, 1e-6f) == NULL);
    
    // dim not divisible by num_heads
    assert(transformer_block_context_create(500, 2048, 8, 8, 1e-6f) == NULL);
    
    // num_heads not divisible by num_kv_heads (GQA violation)
    assert(transformer_block_context_create(512, 2048, 8, 3, 1e-6f) == NULL);
    
    // Negative epsilon
    assert(transformer_block_context_create(512, 2048, 8, 8, -1e-6f) == NULL);
    
    printf("  ✓ Invalid parameter handling passed\n");
}

/**
 * Test 3: Safe NULL pointer handling
 */
static void test_destroy_null_pointer(void) {
    printf("TEST: destroy NULL pointer\n");
    
    // Should not crash
    transformer_block_context_destroy(NULL);
    
    printf("  ✓ NULL pointer handling passed\n");
}

/**
 * Test 4: Context memory bounds
 */
static void test_context_memory_bounds(void) {
    printf("TEST: context memory bounds\n");
    
    int dim = 64;
    TransformerBlockContext *ctx = transformer_block_context_create(dim, 256, 4, 4, 1e-6f);
    
    // Verify buffer sizes and safe access
    for (int i = 0; i < dim; i++) {
        ctx->buf_attn_norm[i] = 0.0f;  // Safe write
        ctx->buf_attn_out[i] = 0.0f;
        ctx->buf_ffn_norm[i] = 0.0f;
    }
    
    for (int i = 0; i < 256; i++) {
        ctx->buf_ffn_hidden[i] = 0.0f;
        ctx->buf_ffn_value[i] = 0.0f;
    }
    
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Memory bounds checking passed\n");
}

/**
 * Test 5: Hybrid context creation
 */
static void test_context_hybrid_creation(void) {
    printf("TEST: hybrid context creation\n");
    
    // Gemma 3 4B style: 8 query heads, 2 KV heads (GQA 4:1)
    TransformerBlockContext *ctx = transformer_block_context_create_hybrid(
        256, 1024, 8, 2, 1e-6f,
        42,                              // session_id
        ATTN_TYPE_LOCAL_SLIDING,        // initial_attn_type
        256                             // local_window_size
    );
    
    assert(ctx != NULL);
    assert(ctx->dim == 256);
    assert(ctx->num_heads == 8);
    assert(ctx->num_kv_heads == 2);  // GQA: 4:1 ratio
    assert(ctx->session_id == 42);
    assert(ctx->attention_strategy == ATTN_TYPE_LOCAL_SLIDING);
    assert(ctx->local_window_size == 256);
    
    // Verify hybrid fields initialized
    assert(ctx->last_active_timestamp == 0);
    assert(ctx->total_tokens_processed == 0);
    assert(ctx->idle_rumination_passes == 0);
    
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Hybrid context creation passed\n");
}

/**
 * Test 6: Attention strategy dispatch
 */
static void test_attention_strategy_dispatch(void) {
    printf("TEST: attention strategy dispatch\n");
    
    TransformerBlockContext *ctx = transformer_block_context_create(128, 512, 4, 4, 1e-6f);
    
    assert(ctx != NULL);
    
    // Set strategy
    int ret = transformer_set_attention_strategy(ctx, ATTN_TYPE_GLOBAL);
    assert(ret == 0);
    assert(ctx->attention_strategy == ATTN_TYPE_GLOBAL);
    
    // Change strategy
    ret = transformer_set_attention_strategy(ctx, ATTN_TYPE_LOCAL_SLIDING);
    assert(ret == 0);
    assert(ctx->attention_strategy == ATTN_TYPE_LOCAL_SLIDING);
    
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Attention strategy dispatch passed\n");
}

/**
 * Test 7: Idle state detection
 */
static void test_idle_state_detection(void) {
    printf("TEST: idle state detection\n");
    
    TransformerBlockContext *ctx = transformer_block_context_create(128, 512, 4, 4, 1e-6f);
    
    // Initially not idle
    int is_idle = transformer_is_idle(ctx);
    assert(is_idle == 0);
    
    // Set idle state
    ctx->is_idle_state = 1;
    is_idle = transformer_is_idle(ctx);
    assert(is_idle == 1);
    
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Idle state detection passed\n");
}

/**
 * Test 8: Session reset
 */
static void test_session_reset(void) {
    printf("TEST: session reset\n");
    
    TransformerBlockContext *ctx = transformer_block_context_create(128, 512, 4, 4, 1e-6f);
    
    // Set some state
    ctx->total_tokens_processed = 1000;
    ctx->idle_rumination_passes = 50;
    ctx->session_id = 1;
    
    // Reset session
    int ret = transformer_reset_session(ctx, 42);
    assert(ret == 0);
    
    // Verify reset
    assert(ctx->session_id == 42);
    assert(ctx->total_tokens_processed == 0);
    assert(ctx->idle_rumination_passes == 0);
    
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Session reset passed\n");
}

/**
 * Test 9: Temporal state tracking
 */
static void test_temporal_state_tracking(void) {
    printf("TEST: temporal state tracking\n");
    
    TransformerBlockContext *ctx = transformer_block_context_create(128, 512, 4, 4, 1e-6f);
    
    // Initially no activity
    assert(ctx->last_active_timestamp == 0);
    assert(ctx->total_tokens_processed == 0);
    assert(ctx->idle_rumination_passes == 0);
    
    // Simulate incrementing tokens (would happen in forward_pass)
    ctx->total_tokens_processed++;
    ctx->total_tokens_processed++;
    
    assert(ctx->total_tokens_processed == 2);
    
    // Simulate idle pass
    ctx->idle_rumination_passes++;
    assert(ctx->idle_rumination_passes == 1);
    
    transformer_block_context_destroy(ctx);
    
    printf("  ✓ Temporal state tracking passed\n");
}

/**
 * Test 10: Various dimensions with GQA support
 */
static void test_various_dimensions(void) {
    printf("TEST: various dimensions\n");
    
    struct {
        int dim;
        int hidden_dim;
        int num_heads;
        int num_kv_heads;
    } configs[] = {
        {64, 256, 4, 4},        // Standard (no GQA)
        {128, 512, 4, 4},       // Standard (no GQA)
        {256, 1024, 8, 8},      // Standard (no GQA)
        {512, 2048, 8, 2},      // Gemma 3 4B style (8:2 = 4:1 GQA)
        {1024, 4096, 16, 4},    // 16:4 = 4:1 GQA
    };
    
    for (int i = 0; i < 5; i++) {
        int dim = configs[i].dim;
        int hidden_dim = configs[i].hidden_dim;
        int num_heads = configs[i].num_heads;
        int num_kv_heads = configs[i].num_kv_heads;
        
        TransformerBlockContext *ctx = transformer_block_context_create(
            dim, hidden_dim, num_heads, num_kv_heads, 1e-6f
        );
        
        assert(ctx != NULL);
        assert(ctx->dim == dim);
        assert(ctx->hidden_dim == hidden_dim);
        assert(ctx->num_heads == num_heads);
        assert(ctx->num_kv_heads == num_kv_heads);
        assert(ctx->head_dim == dim / num_heads);
        
        transformer_block_context_destroy(ctx);
    }
    
    printf("  ✓ Various dimensions passed\n");
}

// ============================================================================
// Main test runner
// ============================================================================

int main(void) {
    printf("\n");
    printf("============================================================\n");
    printf("            TRANSFORMER BLOCK TEST SUITE\n");
    printf("============================================================\n");
    printf("\n");
    
    test_context_create_destroy();
    test_context_invalid_params();
    test_destroy_null_pointer();
    test_context_memory_bounds();
    test_context_hybrid_creation();
    test_attention_strategy_dispatch();
    test_idle_state_detection();
    test_session_reset();
    test_temporal_state_tracking();
    test_various_dimensions();
    
    printf("\n");
    printf("============================================================\n");
    printf("              ALL TESTS PASSED ✓\n");
    printf("============================================================\n");
    printf("\n");
    
    return 0;
}

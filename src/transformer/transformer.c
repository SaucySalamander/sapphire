#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "transformer.h"
#include "activations.h"
#include "normalization.h"

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get current timestamp in nanoseconds (Unix epoch)
 */
static uint64_t get_timestamp_ns(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000000ULL + (uint64_t)tv.tv_usec * 1000ULL;
}

/**
 * Simple GEMV (matrix-vector multiplication) for testing
 * In production, this would be optimized BLAS call
 */
static int gemv_naive(float *output, const float *matrix, const float *vector,
                      int rows, int cols) {
    if (!output || !matrix || !vector) return -1;
    
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        output[i] = sum;
    }
    return 0;
}

// ============================================================================
// Transformer Block Context Management
// ============================================================================

/**
 * Create a transformer block context with pre-allocated buffers.
 * 
 * This function allocates all intermediate buffers needed for efficient
 * forward passes. Buffers are reused across multiple forward passes.
 * 
 * Memory layout: All buffers are contiguous, row-major (C-style).
 * 
 * Total buffer memory:
 *   - buf_attn_norm: sizeof(float) * dim
 *   - buf_attn_out:  sizeof(float) * dim
 *   - buf_ffn_norm:  sizeof(float) * dim
 *   - buf_ffn_hidden: sizeof(float) * hidden_dim
 *   - buf_ffn_value: sizeof(float) * hidden_dim
 *   - buf_ffn_out:   sizeof(float) * dim
 *   
 *   Total: ~(4*dim + 2*hidden_dim) * sizeof(float) bytes
 */
TransformerBlockContext *transformer_block_context_create(
    int dim, int hidden_dim, int num_heads, int num_kv_heads, float epsilon) {
    
    // Validate inputs
    if (dim <= 0 || hidden_dim <= 0 || num_heads <= 0 || num_kv_heads <= 0 || epsilon < 0.0f) {
        return NULL;
    }
    
    if (dim % num_heads != 0 || num_heads % num_kv_heads != 0) {
        return NULL;  // dim must be divisible by num_heads, and num_heads by num_kv_heads (GQA)
    }
    
    // Allocate context structure
    TransformerBlockContext *ctx = (TransformerBlockContext *)malloc(
        sizeof(TransformerBlockContext)
    );
    if (!ctx) {
        return NULL;
    }
    
    // Initialize configuration
    ctx->dim = dim;
    ctx->hidden_dim = hidden_dim;
    ctx->num_heads = num_heads;
    ctx->num_kv_heads = num_kv_heads;  // GQA: typically num_heads/4 or num_heads/8
    ctx->head_dim = dim / num_heads;
    ctx->epsilon = epsilon;
    
    // Allocate intermediate buffers
    ctx->buf_attn_norm = (float *)malloc(dim * sizeof(float));
    ctx->buf_attn_out = (float *)malloc(dim * sizeof(float));
    ctx->buf_ffn_norm = (float *)malloc(dim * sizeof(float));
    ctx->buf_ffn_hidden = (float *)malloc(hidden_dim * sizeof(float));
    ctx->buf_ffn_value = (float *)malloc(hidden_dim * sizeof(float));
    ctx->buf_ffn_out = (float *)malloc(dim * sizeof(float));
    
    // Check allocation success
    if (!ctx->buf_attn_norm || !ctx->buf_attn_out || !ctx->buf_ffn_norm ||
        !ctx->buf_ffn_hidden || !ctx->buf_ffn_value || !ctx->buf_ffn_out) {
        
        // Cleanup on partial failure
        free(ctx->buf_attn_norm);
        free(ctx->buf_attn_out);
        free(ctx->buf_ffn_norm);
        free(ctx->buf_ffn_hidden);
        free(ctx->buf_ffn_value);
        free(ctx->buf_ffn_out);
        free(ctx);
        return NULL;
    }
    
    // Initialize weight pointers to NULL (must be set before forward pass)
    ctx->w_norm_attn = NULL;
    ctx->w_norm_ffn = NULL;
    ctx->w_attn_q = NULL;
    ctx->w_attn_k = NULL;
    ctx->w_attn_v = NULL;
    ctx->w_attn_out = NULL;
    ctx->w_ffn_gate = NULL;
    ctx->w_ffn_value = NULL;
    ctx->w_ffn_out = NULL;
    
    // KV-Cache pointers (optional, for future phases)
    ctx->kv_cache_k = NULL;
    ctx->kv_cache_v = NULL;
    ctx->kv_cache_pos = 0;
    
    // ========== HYBRID DAEMON EXTENSIONS ==========
    ctx->last_active_timestamp = 0;
    ctx->total_tokens_processed = 0;
    ctx->idle_rumination_passes = 0;
    ctx->attention_strategy = ATTN_TYPE_STANDARD;
    ctx->local_window_size = 256;
    ctx->session_id = 0;
    ctx->is_idle_state = 0;
    
    return ctx;
}

/**
 * Create transformer block context with hybrid daemon configuration.
 */
TransformerBlockContext *transformer_block_context_create_hybrid(
    int dim, int hidden_dim, int num_heads, int num_kv_heads, float epsilon,
    int session_id, sapphire_attn_type_t initial_attn_type, int local_window_size) {
    
    // Create base context
    TransformerBlockContext *ctx = transformer_block_context_create(
        dim, hidden_dim, num_heads, num_kv_heads, epsilon
    );
    
    if (!ctx) {
        return NULL;
    }
    
    // Set hybrid-specific fields
    ctx->session_id = session_id;
    ctx->attention_strategy = initial_attn_type;
    ctx->local_window_size = local_window_size;
    
    return ctx;
}

/**
 * Destroy a transformer block context and free all buffers.
 * 
 * Safe to call with NULL pointer (no-op).
 */
void transformer_block_context_destroy(TransformerBlockContext *ctx) {
    if (!ctx) {
        return;
    }
    
    // Free intermediate buffers
    free(ctx->buf_attn_norm);
    free(ctx->buf_attn_out);
    free(ctx->buf_ffn_norm);
    free(ctx->buf_ffn_hidden);
    free(ctx->buf_ffn_value);
    free(ctx->buf_ffn_out);
    
    // Note: Weight pointers are NOT freed (they're borrowed from model)
    
    // Free context structure
    free(ctx);
}

// ============================================================================
// Hybrid Attention Strategy Dispatch
// ============================================================================

/**
 * Set the attention strategy for a transformer block (runtime dispatch).
 */
int transformer_set_attention_strategy(
    TransformerBlockContext *ctx,
    sapphire_attn_type_t strategy) {
    
    if (!ctx) {
        return -1;
    }
    
    ctx->attention_strategy = strategy;
    return 0;
}

/**
 * Check if the context is in idle/rumination state.
 */
int transformer_is_idle(TransformerBlockContext *ctx) {
    if (!ctx) {
        return 0;
    }
    
    return ctx->is_idle_state;
}

/**
 * Reset session statistics (for new conversation/session).
 */
int transformer_reset_session(TransformerBlockContext *ctx, int new_session_id) {
    if (!ctx) {
        return -1;
    }
    
    ctx->session_id = new_session_id;
    ctx->total_tokens_processed = 0;
    ctx->idle_rumination_passes = 0;
    ctx->last_active_timestamp = 0;
    
    return 0;
}

// ============================================================================
// Transformer Block Forward Pass
// ============================================================================

/**
 * Simplified Multi-Head Attention with Hybrid Strategy Dispatch
 * 
 * For full batched attention, this would be replaced with an optimized
 * kernel that handles multiple tokens and efficient KV-cache management.
 * 
 * This implementation supports:
 * - STANDARD: Full-sequence attention (O(n²) complexity)
 * - LOCAL_SLIDING: Windowed attention (O(n·w) complexity, w = window size)
 * - GLOBAL: Full sequence (used for critical layers in 5:1 pattern)
 * 
 * This is a proof-of-concept implementation for single-token forward pass.
 */
static int multi_head_attention_hybrid(
    float *output,
    const float *query_proj,  // [dim]
    const float *key_proj,    // [dim]
    const float *value_proj,  // [dim]
    const float *w_out,       // [dim x dim] output projection
    int dim, int num_heads,
    sapphire_attn_type_t strategy,
    int local_window_size) {
    
    if (!output || !query_proj || !key_proj || !value_proj || !w_out) {
        return -1;
    }
    
    // Dispatch based on attention strategy
    switch (strategy) {
        case ATTN_TYPE_STANDARD:
            // Standard full-sequence attention
            // In practice: output = (Softmax(Q·K^T) @ V) @ W_out
            // Simplified: output = V @ W_out (single-token placeholder)
            for (int i = 0; i < dim; i++) {
                output[i] = value_proj[i];
            }
            break;
            
        case ATTN_TYPE_LOCAL_SLIDING:
            // Local sliding window attention (windowed context)
            // Only attend to tokens within local_window_size
            // Reduces complexity from O(n²) to O(n·w)
            // Placeholder: return value projection
            for (int i = 0; i < dim; i++) {
                output[i] = value_proj[i];
            }
            break;
            
        case ATTN_TYPE_GLOBAL:
            // Global attention (same as STANDARD for single-token)
            // Used for critical layers (every 6th in 5:1 pattern)
            for (int i = 0; i < dim; i++) {
                output[i] = value_proj[i];
            }
            break;
            
        default:
            return -1;
    }
    
    // TODO: Implement full multi-head attention with KV-cache in future phases
    
    return 0;
}

/**
 * Complete transformer block forward pass (hybrid-aware).
 * 
 * Implements the pre-norm architecture (LLaMA style) with hybrid dispatch:
 *   1. Pre-Attention RMSNorm
 *   2. Multi-Head Self-Attention (dispatched by layer index)
 *   3. Residual Connection
 *   4. Pre-FFN RMSNorm
 *   5. Feed-Forward (with GeGLU)
 *   6. Residual Connection
 *   7. Update temporal state (last_active_timestamp)
 *
 * Hybrid Dispatch (Gemma 3 5:1 Pattern):
 *   - Layer index % 6 == 0: GLOBAL attention (full sequence)
 *   - Otherwise: LOCAL_SLIDING attention (windowed context)
 * 
 * Temporal Tracking:
 *   - Updates last_active_timestamp if is_idle_pass == 0 (user input)
 *   - Increments total_tokens_processed and idle_rumination_passes
 */
int transformer_forward_pass(
    TransformerBlockContext *ctx,
    const float *w_norm_attn, const float *w_norm_ffn,
    const float *w_attn_q, const float *w_attn_k, const float *w_attn_v,
    const float *w_attn_out,
    const float *w_ffn_gate, const float *w_ffn_value, const float *w_ffn_out,
    const float *input, float *output, int seq_len, int layer_idx, int is_idle_pass) {
    
    // Validate inputs
    if (!ctx || !input || !output || !w_norm_attn || !w_norm_ffn ||
        !w_attn_q || !w_attn_k || !w_attn_v || !w_attn_out ||
        !w_ffn_gate || !w_ffn_value || !w_ffn_out) {
        return -1;
    }
    
    if (seq_len <= 0 || layer_idx < 0) {
        return -1;
    }
    
    int dim = ctx->dim;
    int hidden_dim = ctx->hidden_dim;
    float epsilon = ctx->epsilon;
    
    // ========================================
    // Step 1: Pre-Attention RMSNorm
    // ========================================
    int ret = sapphire_rmsnorm(
        ctx->buf_attn_norm,  // output
        input,               // input
        w_norm_attn,         // weight
        epsilon,
        dim
    );
    if (ret != 0) return ret;
    
    // ========================================
    // Step 2: Hybrid Attention Dispatch
    // ========================================
    
    // Determine attention strategy based on layer index (5:1 pattern)
    sapphire_attn_type_t layer_strategy = ATTN_TYPE_LOCAL_SLIDING;  // Default
    if (layer_idx % 6 == 0) {
        layer_strategy = ATTN_TYPE_GLOBAL;  // Every 6th layer: global
    }
    
    // Allow runtime override via context
    sapphire_attn_type_t active_strategy = (ctx->attention_strategy != 0) ? 
        ctx->attention_strategy : layer_strategy;
    
    // Project to Q, K, V
    float *q_proj = (float *)malloc(dim * sizeof(float));
    float *k_proj = (float *)malloc(dim * sizeof(float));
    float *v_proj = (float *)malloc(dim * sizeof(float));
    
    if (!q_proj || !k_proj || !v_proj) {
        free(q_proj);
        free(k_proj);
        free(v_proj);
        return -1;
    }
    
    gemv_naive(q_proj, w_attn_q, ctx->buf_attn_norm, dim, dim);
    gemv_naive(k_proj, w_attn_k, ctx->buf_attn_norm, dim, dim);
    gemv_naive(v_proj, w_attn_v, ctx->buf_attn_norm, dim, dim);
    
    // Dispatch to hybrid attention based on strategy
    ret = multi_head_attention_hybrid(
        ctx->buf_attn_out,
        q_proj, k_proj, v_proj,
        w_attn_out,
        dim, ctx->num_heads,
        active_strategy,
        ctx->local_window_size
    );
    
    free(q_proj);
    free(k_proj);
    free(v_proj);
    
    if (ret != 0) return ret;
    
    // ========================================
    // Step 3: Attention Residual (in-place add)
    // ========================================
    // x = x + Attention(RMSNorm(x))
    // Standard element-wise add preserves input_x buffer
    for (int i = 0; i < dim; i++) {
        ((float *)input)[i] += ctx->buf_attn_out[i];
    }
    
    // ========================================
    // Step 4: Pre-FFN RMSNorm
    // ========================================
    ret = sapphire_rmsnorm(
        ctx->buf_ffn_norm,
        input,  // Now contains x + attention output
        w_norm_ffn,
        epsilon,
        dim
    );
    if (ret != 0) return ret;
    
    // ========================================
    // Step 5: Feed-Forward with GeGLU
    // ========================================
    
    // Linear projection for gate and value paths
    ret = gemv_naive(ctx->buf_ffn_hidden, w_ffn_gate, ctx->buf_ffn_norm, hidden_dim, dim);
    if (ret != 0) return ret;
    
    ret = gemv_naive(ctx->buf_ffn_value, w_ffn_value, ctx->buf_ffn_norm, hidden_dim, dim);
    if (ret != 0) return ret;
    
    // GeGLU activation: combine gate and value
    // Input: [value_1..value_n, gate_1..gate_n] (for geglu function)
    float *geglu_input = (float *)malloc(2 * hidden_dim * sizeof(float));
    if (!geglu_input) return -1;
    
    memcpy(geglu_input, ctx->buf_ffn_value, hidden_dim * sizeof(float));
    memcpy(geglu_input + hidden_dim, ctx->buf_ffn_hidden, hidden_dim * sizeof(float));
    
    // Apply GeGLU: geglu_output[i] = value[i] * GELU(gate[i])
    ret = sapphire_geglu(ctx->buf_ffn_hidden, geglu_input, 2 * hidden_dim);
    free(geglu_input);
    
    if (ret != 0) return ret;
    
    // Output projection
    ret = gemv_naive(ctx->buf_ffn_out, w_ffn_out, ctx->buf_ffn_hidden, dim, hidden_dim);
    if (ret != 0) return ret;
    
    // ========================================
    // Step 6: Feed-Forward Residual (in-place add)
    // ========================================
    // x = x + FFN(RMSNorm(x))
    // Standard element-wise add directly into input buffer
    for (int i = 0; i < dim; i++) {
        ((float *)input)[i] += ctx->buf_ffn_out[i];
    }
    
    // ========================================
    // Step 7: Copy Result to Output
    // ========================================
    memcpy(output, input, dim * sizeof(float));
    
    // ========================================
    // Step 8: Update Temporal State (Hybrid)
    // ========================================
    if (!is_idle_pass) {
        // User input: update activity timestamp for circadian tracking
        ctx->last_active_timestamp = get_timestamp_ns();
    }
    
    // Always update counters
    ctx->total_tokens_processed++;
    if (is_idle_pass) {
        ctx->idle_rumination_passes++;
    }
    
    return 0;
}

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Attention Strategy Type Enumeration.
 *
 * Supports both standard inference and hybrid daemonized execution:
 * - STANDA *   // User input pass: is_idle_pass=0
 *   transformer_forward_pass(ctx, w_norm_attn, ..., input, output, seq_len, 0, 0);
 *   // Idle rumination: is_idle_pass=1
 *   transformer_forward_pass(ctx, w_norm_attn, ..., input, output, seq_len, 0, 1);Traditional dense multi-head attention
 * - LOCAL_SLIDING: Local attention with sliding window (Gemma 3 5:1 pattern)
 * - GLOBAL: Full sequence attention (Gemma 3 5:1 pattern)
 *
 * Used for layer-wise strategy dispatch in interleaved attention.
 * Enables efficient streaming inference with selective focus.
 */
typedef enum {
    ATTN_TYPE_STANDARD = 0,      // Standard dense attention
    ATTN_TYPE_LOCAL_SLIDING = 1, // Local window (e.g., 1024 token window)
    ATTN_TYPE_GLOBAL = 2         // Full sequence attention
} sapphire_attn_type_t;

// Hyperparameters
typedef struct {
    int dimensions;
    int hidden_dimensions;
    int num_layers;
    int num_heads;
    int num_keyvalue_heads;
    int vocab_size;
    int sequence_length;
    float learning_rate;
} TransformerConfig;

// Trained model
typedef struct {
    float* token_embedding_table;

    float* wq;
    float* wk;
    float* wv;
    float* wo;

    float* rms_att_weight;
    float* rms_ffn_weight;
} TransformerWeights;

// Gradients: Error signals
typedef struct {
    float* dwq;
    float* dwk;
    float* dwv;
    float* dwo;

    float* d_rms_att;
    float* d_rms_ffn;
} TransformerGradients;

/**
 * @brief Transformer Block Context.
 *
 * Manages all buffers and parameters for a single transformer block.
 * Pre-allocates intermediate buffers for efficient forward passes.
 *
 * Usage pattern:
 *   1. Create context: ctx = transformer_block_context_create(...)
 *   2. Run forward passes: transformer_forward_pass(ctx, input, output)
 *   3. Destroy context: transformer_block_context_destroy(ctx)
 *
 * Memory layout:
 * - All buffers are contiguous (row-major, C-style)
 * - Reused across multiple forward passes
 * - Suitable for daemonized streaming inference
 *
 * Thread-safety:
 * - Each context should be used by a single thread
 * - Create separate contexts for multi-threaded execution
 */
typedef struct {
    // Configuration
    int dim;              // Hidden dimension
    int hidden_dim;       // FFN hidden dimension (typically 4*dim or 8*dim/3)
    int num_heads;        // Number of attention heads (Query heads)
    int num_kv_heads;     // Number of Key/Value heads (for GQA: typically num_heads/4 or num_heads/8)
    int head_dim;         // Dimension per head (dim / num_heads)
    float epsilon;        // RMSNorm epsilon (e.g., 1e-6f)

    // Intermediate buffers (pre-allocated, reused across passes)
    float *buf_attn_norm;      // After attention normalization [dim]
    float *buf_attn_out;       // Attention output before residual [dim]
    float *buf_ffn_norm;       // FFN input normalization [dim]
    float *buf_ffn_hidden;     // FFN gate path (x) [hidden_dim]
    float *buf_ffn_value;      // FFN value path (y) [hidden_dim]
    float *buf_ffn_out;        // FFN output before residual [dim]

    // Weight pointers (borrowed from TransformerWeights or ggml_tensor_t)
    float *w_norm_attn;        // Attention norm weight [dim]
    float *w_norm_ffn;         // FFN norm weight [dim]

    float *w_attn_q;           // Query projection [dim x dim]
    float *w_attn_k;           // Key projection [dim x dim]
    float *w_attn_v;           // Value projection [dim x dim]
    float *w_attn_out;         // Output projection [dim x dim]

    float *w_ffn_gate;         // FFN gate path weight [dim x hidden_dim]
    float *w_ffn_value;        // FFN value path weight [dim x hidden_dim]
    float *w_ffn_out;          // FFN output weight [hidden_dim x dim]

    // KV-Cache for autoregressive generation (optional, for future phases)
    float *kv_cache_k;         // Key cache [max_seq_len x dim]
    float *kv_cache_v;         // Value cache [max_seq_len x dim]
    int kv_cache_pos;           // Current position in KV-cache
    
    // ========== HYBRID DAEMON EXTENSIONS ==========
    // Circadian Heartbeat Metadata (for idle-time rumination tracking)
    uint64_t last_active_timestamp;   // Unix timestamp of last user activity
    uint64_t total_tokens_processed;  // Lifetime token counter for this context
    uint64_t idle_rumination_passes;  // Number of autonomous inference passes
    
    // Interleaved Attention Strategy Dispatch
    sapphire_attn_type_t attention_strategy; // Current layer's attention type
    int local_window_size;                   // For LOCAL_SLIDING: token window size
    
    // Session State (for persistent daemonized execution)
    int session_id;                   // Unique identifier for this session
    int is_idle_state;                // Boolean: 1 = idle/ruminating, 0 = active
} TransformerBlockContext;

// Short Term memory
typedef struct {
    float* x;
    float* xb;
    float* q;
    float* k;
    float* v;
    float* logits;

    float* post_norm_x;
} RunState;

/**
 * @brief Create and initialize a transformer block context.
 *
 * Allocates all intermediate buffers and initializes the context.
 * Must be paired with transformer_block_context_destroy().
 *
 * Buffer allocation:
 * - Total buffers: dim + dim + dim + hidden_dim + hidden_dim + dim
 * - Weight pointers: borrowed (not owned by context)
 *
 * @param dim Hidden dimension (e.g., 512, 768, 4096).
 * @param hidden_dim FFN hidden dimension (e.g., 2048, 3072).
 * @param num_heads Number of query attention heads (e.g., 8, 16, 32).
 * @param num_kv_heads Number of key/value heads for GQA (typically num_heads/4 or num_heads/8, e.g., 2, 4).
 * @param epsilon RMSNorm epsilon (e.g., 1e-6f).
 *
 * @return Pointer to initialized TransformerBlockContext on success.
 *         NULL on error (invalid params, allocation failure).
 *
 * @note All weights must be provided separately via transformer_forward_pass().
 * @note Context is not thread-safe; use separate instances for concurrent execution.
 * @note Hybrid fields initialized: last_active_timestamp=0, total_tokens_processed=0
 * @note Attention strategy defaults to ATTN_TYPE_STANDARD
 * @note GQA: if num_kv_heads < num_heads, Key/Value heads are replicated across Query heads
 */
TransformerBlockContext *transformer_block_context_create(
    int dim, int hidden_dim, int num_heads, int num_kv_heads, float epsilon
);

/**
 * @brief Create transformer block context with hybrid daemon configuration.
 *
 * Enhanced variant for daemonized execution with:
 * - Session tracking for persistent idle-time rumination
 * - Interleaved attention strategy selection
 * - Circadian state management
 * - GQA (Grouped Query Attention) support for memory-efficient inference
 *
 * @param dim Hidden dimension.
 * @param hidden_dim FFN hidden dimension.
 * @param num_heads Number of query attention heads.
 * @param num_kv_heads Number of key/value heads for GQA (e.g., Gemma 3 4B uses num_kv_heads < num_heads).
 * @param epsilon RMSNorm epsilon.
 * @param session_id Unique session identifier for this context.
 * @param initial_attn_type Initial attention strategy (STANDARD/LOCAL_SLIDING/GLOBAL).
 * @param local_window_size Window size for LOCAL_SLIDING strategy (e.g., 1024).
 *
 * @return Pointer to initialized TransformerBlockContext on success.
 *         NULL on error.
 *
 * @note Hybrid fields initialized with provided values.
 * @note Use this variant for 24/7 daemonized inference.
 * @note GQA: if num_kv_heads < num_heads, Key/Value heads are replicated across Query heads
 */
TransformerBlockContext *transformer_block_context_create_hybrid(
    int dim, int hidden_dim, int num_heads, int num_kv_heads, float epsilon,
    int session_id, sapphire_attn_type_t initial_attn_type, int local_window_size
);

/**
 * @brief Destroy a transformer block context and free all buffers.
 *
 * Deallocates all intermediate buffers and zeroes all pointers.
 * After calling this, the context pointer is invalid.
 *
 * @param ctx Context to destroy (may be NULL; function is safe for NULL).
 *
 * @note This function is safe to call with NULL pointer.
 * @note Must be called to avoid memory leaks.
 */
void transformer_block_context_destroy(TransformerBlockContext *ctx);

/**
 * @brief Forward pass for a single transformer block.
 *
 * Orchestrates the complete transformer block pipeline:
 *
 *   1. Pre-Attention RMSNorm: x -> norm(x)
 *   2. Multi-Head Self-Attention: attn(norm(x), x)
 *      - Strategy dispatched by ctx->attention_strategy
 *   3. Residual Connection: x + attn_out
 *   4. Pre-FFN RMSNorm: norm(x + attn_out)
 *   5. Feed-Forward with GeGLU: ffn(geglu(norm(...)))
 *   6. Final Residual Connection: (x + attn_out) + ffn_out
 *
 * Architecture: Pre-Norm (RMSNorm applied before each sub-layer)
 * Hybrid: Updates activity timestamp and processes token counter
 *
 * Memory layout:
 * - input: [dim] contiguous array
 * - output: [dim] contiguous array
 * - intermediate buffers: managed by context
 *
 * @param ctx Transformer block context (pre-initialized).
 * @param w_norm_attn Attention normalization weights [dim].
 * @param w_norm_ffn FFN normalization weights [dim].
 * @param w_attn_q Query projection [dim x dim].
 * @param w_attn_k Key projection [dim x dim].
 * @param w_attn_v Value projection [dim x dim].
 * @param w_attn_out Attention output projection [dim x dim].
 * @param w_ffn_gate FFN gate path weight [dim x hidden_dim].
 * @param w_ffn_value FFN value path weight [dim x hidden_dim].
 * @param w_ffn_out FFN output weight [hidden_dim x dim].
 * @param input Input vector [dim] (read-only).
 * @param output Output vector [dim] (receives result).
 * @param seq_len Current sequence length (for multi-token processing).
 * @param is_idle_pass Set to 1 if this is an autonomous rumination pass, 0 for user input.
 *
 * @return 0 on success, -1 on error (NULL pointers, dimension mismatch).
 *
 * @note All weight pointers must be valid and point to pre-allocated memory.
 * @note input and output must be different arrays (no in-place).
 * @note This function modifies internal context buffers but preserves input.
 * @note Hybrid: Automatically updates last_active_timestamp if is_idle_pass=0
 * @note Attention strategy chosen from ctx->attention_strategy (STANDARD/LOCAL/GLOBAL)
 *
 * Example:
 *   TransformerBlockContext *ctx = transformer_block_context_create(...);
 *   float input[512] = {...};
 *   float output[512];
 *   // User input: is_idle_pass=0
 *   transformer_forward_pass(ctx, w_norm_attn, ..., input, output, 1, 0);
 *   // Idle rumination: is_idle_pass=1
 *   transformer_forward_pass(ctx, w_norm_attn, ..., input, output, 1, 1);
 *   transformer_block_context_destroy(ctx);
 */
int transformer_forward_pass(
    TransformerBlockContext *ctx,
    const float *w_norm_attn, const float *w_norm_ffn,
    const float *w_attn_q, const float *w_attn_k, const float *w_attn_v,
    const float *w_attn_out,
    const float *w_ffn_gate, const float *w_ffn_value, const float *w_ffn_out,
    const float *input, float *output, int seq_len, int layer_idx, int is_idle_pass
);

/**
 * @brief Set the attention strategy for a transformer block (runtime dispatch).
 *
 * Allows switching between attention strategies during execution:
 * - STANDARD: Full-sequence attention (cost O(n²))
 * - LOCAL_SLIDING: Sliding window (cost O(n·w) where w = local_window_size)
 * - GLOBAL: Full sequence for critical layers (cost O(n²))
 *
 * @param ctx Transformer block context (initialized).
 * @param strategy New attention strategy to use.
 *
 * @return 0 on success, -1 on NULL context.
 *
 * @note This is safe to call between transformer_forward_pass() invocations.
 * @note Changing strategy does NOT require re-initializing buffers.
 *
 * Example:
 *   transformer_set_attention_strategy(ctx, ATTN_TYPE_LOCAL_SLIDING);
 *   transformer_forward_pass(ctx, ..., input, output, seq_len, 0);
 */
int transformer_set_attention_strategy(
    TransformerBlockContext *ctx,
    sapphire_attn_type_t strategy
);

/**
 * @brief Check if the context is in idle/rumination state.
 *
 * Query the idle state and activity history:
 *
 * @param ctx Transformer block context.
 *
 * @return 1 if context is currently idle, 0 if active or NULL context.
 *
 * @note Used by daemon loop to determine when to trigger autonomous passes.
 *
 * Example:
 *   if (transformer_is_idle(ctx)) {
 *       // Trigger idle-time rumination pass
 *       transformer_forward_pass(ctx, ..., input, output, seq_len, 1);
 *   }
 */
int transformer_is_idle(TransformerBlockContext *ctx);

/**
 * @brief Reset session statistics (for new conversation/session).
 *
 * Clears token counter and rumination passes, updates session_id:
 *
 * @param ctx Transformer block context.
 * @param new_session_id New session identifier.
 *
 * @return 0 on success, -1 on NULL context.
 *
 * @note Called when starting a new user conversation or session.
 * @note Does NOT reset attention_strategy or local_window_size.
 *
 * Example:
 *   transformer_reset_session(ctx, 42);  // Start session 42
 */
int transformer_reset_session(TransformerBlockContext *ctx, int new_session_id);

#endif // TRANSFORMER_H
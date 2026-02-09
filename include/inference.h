/**
 * @file inference.h
 * @brief Inference session and full forward pass interface.
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "model_spec.h"
#include "kv_cache.h"
#include "kernels.h"      /* For kernel_context_t in GEMV operations */

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct inference_session_t inference_session_t;
typedef struct sapphire_tokenizer_t sapphire_tokenizer_t;

typedef struct {
    model_spec_t *spec;
    inference_session_t *session;
    sapphire_tokenizer_t *tokenizer;
    float *logits;          // Buffer for logits
    int max_tokens;
    float temperature;
    int context_len;
} inference_context_t;

/**
 * Create an inference context.
 *
 * @param temperature Sampling temperature.
 * @param max_tokens Maximum number of tokens to generate.
 * @param context_len Context length for the model.
 * @param model_name Name of the model to load.
 * @return Pointer to the created inference context, or NULL on failure.
 */
inference_context_t* create_inference_context(float temperature, int max_tokens, int context_len, const char *model_name);

/**
 * Run inference for a prompt using an existing context.
 * Public because the CLI calls this directly.
 */
int perform_inference(inference_context_t* ctx, const char* prompt, char* output, int output_size);

/**
 * Destroy inference context and free owned resources.
 */
void destroy_inference_context(inference_context_t* ctx);

/**
 * Inference session (manages KV caches and intermediate buffers).
 */
typedef struct inference_session_t {
    model_spec_t *model_spec;
    kv_cache_t *kv_cache;          /**< Global multi-layer KV cache for all layers. */
    
    float *scratch_buffer;          /**< Reusable buffer for temporary tensors. */
    size_t scratch_size;

    /* Padded dimensions (multiples of 8 floats / 32 bytes) to ensure SIMD kernels can
     * safely load 256-bit vectors without overrunning the allocation. Calculated from
     * model->config.d_model, d_inner (query dimension), d_kv (key/value dimension),
     * and the FFN hidden size. */
    int padded_d_model;
    int padded_d_inner;   /**< Padded query projection dimension */
    int padded_d_kv;      /**< Padded key/value projection dimension (may differ in GQA) */
    int padded_d_ff;

    float *attn_scores;            /**< Softmax-normalized attention weights (size = max_context_len) */
    float *attn_scores_raw;        /**< Optional raw QK diagnostics buffer (size = max_context_len) */
    
    /* Precomputed RoPE frequencies for both global and local attention layers (Gemma 3) */
    float *rope_freqs_cos_global;  /**< Global base (e.g. 1M) for RoPE. */
    float *rope_freqs_sin_global;
    float *rope_freqs_cos_local;   /**< Local base (e.g. 10k) for RoPE. */
    float *rope_freqs_sin_local;
    
    kernel_context_t *gemv_ctx;     /**< GEMV context for matrix-vector operations. */
} inference_session_t;

/**
 * Create an inference session (allocate KV caches and buffers).
 *
 * @param model Loaded model.
 * @param max_context_len Maximum sequence length for KV cache.
 * @return Allocated session, or NULL on failure.
 */
inference_session_t* inference_session_create(model_spec_t *spec, int max_context_len);

/**
 * Reset KV caches for a new sequence.
 *
 * @param session Inference session to reset.
 */
void inference_session_reset(inference_session_t *session);

/**
 * Single-token forward pass.
 *
 * @param session Inference session.
 * @param token_id Input token ID.
 * @param token_pos Position of this token in sequence.
 * @param logits Output logits: [vocab_size].
 *
 * @note This performs: embedding lookup → N layers of transformer blocks → logits.
 *       KV caches are updated internally for the next token.
 */
void inference_forward(inference_session_t *session, int token_id, int token_pos, float *logits);

/**
 * Free inference session.
 *
 * @param session Session to free.
 */
void destroy_inference_session(inference_session_t *session);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H

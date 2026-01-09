/**
 * @file inference.h
 * @brief Inference session and full forward pass interface.
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "ggml_model.h"
#include "kv_cache.h"
#include "tensor_gemv.h"  /* For sapphire_context in GEMV operations */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Inference session (manages KV caches and intermediate buffers).
 */
typedef struct {
    llm_model_t *model;
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

    float *attn_scores;            /**< Pre-allocated attention scores buffer (size = max_context_len) */
    
    float *rope_freqs_cos;          /**< RoPE precomputed cos frequencies. */
    float *rope_freqs_sin;          /**< RoPE precomputed sin frequencies. */
    
    sapphire_context *gemv_ctx;     /**< GEMV context for matrix-vector operations. */
} inference_session_t;

/**
 * Create an inference session (allocate KV caches and buffers).
 *
 * @param model Loaded model.
 * @param max_context_len Maximum sequence length for KV cache.
 * @return Allocated session, or NULL on failure.
 */
inference_session_t* inference_session_create(llm_model_t *model, int max_context_len);

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
void inference_session_destroy(inference_session_t *session);

/**
 * Generate tokens using greedy decoding (argmax).
 *
 * @param session Inference session.
 * @param prompt_tokens Initial prompt (array of token IDs).
 * @param prompt_len Length of prompt.
 * @param max_tokens Maximum total tokens to generate.
 * @param output_tokens Output buffer (caller allocates, size >= max_tokens).
 * @return Number of tokens generated (including prompt).
 */
int llm_generate_greedy(inference_session_t *session,
                        const int *prompt_tokens,
                        int prompt_len,
                        int max_tokens,
                        int *output_tokens);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H

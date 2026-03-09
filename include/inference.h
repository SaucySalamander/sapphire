/**
 * @file inference.h
 * @brief Inference session and full forward pass interface.
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "model_spec.h"
#include "kv_cache.h"
#include "kernels.h"      /* For kernel_context_t in GEMV operations */
#include "layer_dispatch.h" /* For sapphire_layer_config_t */
#include "backend.h"      /* For backend abstraction */

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
    int *conversation_tokens; // Persistent token history for interactive chat
    int conversation_len;     // Number of valid tokens in conversation_tokens
    int max_tokens;
    float temperature;
    int context_len;
    /* Wall-clock elapsed seconds for the most recent perform_inference() call. */
    double last_inference_time;
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
 * Save backend session state (KV cache and related sequence state) to disk.
 *
 * Currently implemented for CPU backend sessions.
 *
 * @param session Inference session.
 * @param path    Output file path (e.g., .sapphire state file).
 * @return 0 on success, -1 on error/unsupported backend.
 */
int inference_session_save_state(inference_session_t *session, const char *path);

/**
 * Load backend session state (KV cache and related sequence state) from disk.
 *
 * Currently implemented for CPU backend sessions.
 *
 * @param session Inference session.
 * @param path    Input file path.
 * @return 0 on success, -1 on error/unsupported backend.
 */
int inference_session_load_state(inference_session_t *session, const char *path);

/**
 * Save current context session state to disk.
 *
 * Convenience wrapper around inference_session_save_state(ctx->session, path).
 */
int inference_context_save_state(inference_context_t *ctx, const char *path);

/**
 * Load context session state from disk.
 *
 * Convenience wrapper around inference_session_load_state(ctx->session, path).
 */
int inference_context_load_state(inference_context_t *ctx, const char *path);

/**
 * Destroy inference context and free owned resources.
 */
void destroy_inference_context(inference_context_t* ctx);

/**
 * Inference session (manages compute backend and intermediate state).
 *
 * This structure is backend-agnostic: it delegates all hardware-specific
 * operations (memory allocation, kernel execution, synchronization) to
 * the selected backend implementation via the backend interface.
 *
 * For performance, the CPU backend exposes key pointers directly here
 * (kv_cache, gemv_ctx, etc.) to avoid function call overhead in hot paths.
 * Vulkan backend leaves these NULL.
 */
typedef struct inference_session_t {
    /* Public model and configuration */
    model_spec_t *model_spec;

    /* Backend abstraction */
    sapphire_backend_t *backend;      /**< Pointer to selected hardware backend (CPU, Vulkan, etc.). */
    void *backend_data;               /**< Opaque backend-specific session state. Allocated and owned by backend. */

    /* Shared metadata (used by all backends) */
    sapphire_layer_config_t *layer_configs;  /**< Per-layer type and settings [num_layers]. */
    int num_layers;                           /**< Number of transformer layers. */

    /* CPU backend field pointers (set by CPU backend, NULL for others) */
    /* These are exposed here for performance (avoid function call overhead) */
    kv_cache_t *kv_cache;                    /**< [CPU only] Global multi-layer KV cache for all layers. */
    float *scratch_buffer;                   /**< [CPU only] Reusable buffer for temporary tensors. */
    size_t scratch_size;                     /**< [CPU only] Size of scratch buffer in bytes. */
    float *attn_scores;                      /**< [CPU only] Softmax-normalized attention weights. */
    float *attn_scores_raw;                  /**< [CPU only] Optional raw QK diagnostics buffer. */
    float *rope_freqs_cos_global;            /**< [CPU only] Global RoPE base (e.g., 1M) cosines. */
    float *rope_freqs_sin_global;            /**< [CPU only] Global RoPE sines. */
    float *rope_freqs_cos_local;             /**< [CPU only] Local RoPE base (e.g., 10k) cosines. */
    float *rope_freqs_sin_local;             /**< [CPU only] Local RoPE sines. */
    struct sapphire_context *gemv_ctx;       /**< [CPU only] GEMV context for matrix-vector operations. */
    int padded_d_model;                      /**< [CPU only] Padded hidden size. */
    int padded_d_inner;                      /**< [CPU only] Padded query projection dimension. */
    int padded_d_kv;                         /**< [CPU only] Padded key/value projection dimension. */
    int padded_d_ff;                         /**< [CPU only] Padded feed-forward dimension. */
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
 * Batched forward pass (prefill).
 * 
 * @param session Inference session.
 * @param token_ids Array of input token IDs [batch_size].
 * @param start_pos Starting position in sequence.
 * @param batch_size Number of tokens to process.
 * @param logits Optional output logits for the LAST token in the batch.
 */
void inference_forward_batch(inference_session_t* session, const int* token_ids, int start_pos, int batch_size, float* logits);

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

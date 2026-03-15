#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "llm_model.h"
#include "gemma3_270m_config.h"

struct inference_session_t;
struct sapphire_tokenizer_t;

typedef struct {
    const float* cos;
    const float* sin;
} transformer_rope_t;

/**
 * @brief Target activation point within a transformer layer.
 *
 * Enumerated in order of computation: the pre-attention RMSNorm output is
 * captured first, then the raw attention output, then the pre-FFN RMSNorm
 * output, then the GeGLU intermediate.
 */
typedef enum {
    CAPTURE_TARGET_UNKNOWN    = 0,
    CAPTURE_TARGET_QKV_INPUT  = 1, /**< Pre-attn RMSNorm output (input to Q/K/V projections). */
    CAPTURE_TARGET_OUT_INPUT  = 2, /**< Raw attention output before o_proj. */
    CAPTURE_TARGET_FFN_INPUT  = 3, /**< Pre-FFN RMSNorm output (input to gate/up projections). */
    CAPTURE_TARGET_DOWN_INPUT = 4  /**< GeGLU output (input to down projection). */
} capture_target_t;

/**
 * @brief Single activation capture slot for sapphire_record_pass().
 *
 * One slot per (layer_idx, target) pair. The caller pre-allocates out_vector
 * for each sample and resets captured=0 before each pass.
 */
typedef struct {
    int              layer_idx;  /**< Layer to capture from (0-based). */
    capture_target_t target;     /**< Which activation point to capture. */
    float           *out_vector; /**< Caller-allocated output buffer. */
    uint32_t         out_dim;    /**< Expected vector dimension. */
    int              captured;   /**< Set to 1 by sapphire_record_pass() on success. */
} activation_record_slot_t;

/**
 * @brief Single-pass activation recorder.
 *
 * Runs one complete forward pass over 'text' and fills every slot whose
 * (layer_idx, target) is pending (captured == 0). Capture happens at
 * the final token position, before the layer advances hidden state.
 *
 * Requires CPU backend. Returns 0 on success, -1 on error.
 */
int sapphire_record_pass(struct inference_session_t  *session,
                         struct sapphire_tokenizer_t *tokenizer,
                         const struct model_spec     *spec,
                         const char                  *text,
                         activation_record_slot_t    *slots,
                         int                          slot_count);
/**
 * @brief Forward pass for a single transformer layer.
 * 
 * Orchestrates:
 * 1. Pre-attention RMSNorm
 * 2. Q, K, V Projections
 * 3. Q-Norm and K-Norm (Gemma 3)
 * 4. Query Scaling
 * 5. RoPE application
 * 6. KV-Cache write
 * 7. Multi-head Attention (GQA)
 * 8. Output projection
 * 9. Residual connection
 * 10. Post-attention RMSNorm
 * 11. Feed-forward (GeGLU)
 * 12. Output projection
 * 13. Residual connection
 * 
 * @param session Inference session.
 * @param layer_idx Layer index.
 * @param token_pos Current token position.
 * @param hidden Input/Output hidden state [d_model].
 * @param rope RoPE cosine/sine frequencies.
 * @return 0 on success.
 */
int sapphire_transformer_layer(struct inference_session_t* session, int layer_idx, int token_pos, float* hidden,
                               transformer_rope_t rope);

/**
 * @brief Forward pass for a single transformer layer in batched mode.
 */
int sapphire_transformer_layer_batch(struct inference_session_t* session, int layer_idx, int start_pos, int batch_size, float* hidden,
                                     transformer_rope_t rope);

/**
 * @brief Performs embedding lookup.
 */
void sapphire_embed_lookup(struct inference_session_t* session, int token_id, float* hidden);

/**
 * @brief Performs batched embedding lookup.
 */
void sapphire_embed_lookup_batch(struct inference_session_t* session, const int* token_ids, int batch_size, float* hidden);

/**
 * @brief Performs LM Head calculation and softcapping.
 */
void lm_head(struct inference_session_t* session, const float* hidden, float* logits);

typedef struct {
    const model_spec_t* spec;
    const char* text;
    const char* tensor_name;
    float* out_vector;
    uint32_t out_dim;
} transformer_activation_capture_request_t;

int sapphire_collect_tensor_activation(struct inference_session_t* session,
                                       struct sapphire_tokenizer_t* tokenizer,
                                       const transformer_activation_capture_request_t* request);

typedef struct layer_buffers {
    int pm, pi, pk, pf;
    int batch_size;
    float *residual, *norm_buf, *q_proj, *k_proj, *v_proj;
    float *attn_out, *ffn_gate_buf, *ffn_value_buf, *geglu_buf;
    float* weight_scratch;
} layer_buffers_t;

/**
 * @brief Context for a single transformer layer forward pass.
 * Used to keep function signatures clean and within project specifications.
 */
typedef struct transformer_layer_ctx {
    struct inference_session_t* session;
    model_layer_weights_t* layer;
    gemma3_270m_config_t* config;
    int layer_idx;
    int token_pos;
    int batch_size;
    int d_model;
    int head_dim;
} transformer_layer_ctx_t;

/**
 * @brief Computes the Attention stage of a transformer layer (softmax attention).
 * includes Norm -> Projections -> QK Norm -> RoPE -> Attention -> Output Projection -> Post Norm
 */
void compute_attention_stage(layer_buffers_t buf,
                             transformer_layer_ctx_t* ctx,
                             float* hidden,
                             const float* rope_cos,
                             const float* rope_sin);

/**
 * @brief Computes a linear attention stage (LoLCATs-style linearized attention).
 * 
 * Placeholder for linearized attention computation. Similar to softmax attention
 * but replaces softmax with linear (kernel-based) attention mechanism.
 * 
 * @param buf Layer buffers (pre-allocated, same as softmax path).
 * @param ctx Transformer layer context.
 * @param hidden Input/output hidden state [d_model].
 * @param rope_cos RoPE cosine frequencies.
 * @param rope_sin RoPE sine frequencies.
 */
void compute_linear_attention_stage(layer_buffers_t buf,
                                    transformer_layer_ctx_t* ctx,
                                    float* hidden,
                                    const float* rope_cos,
                                    const float* rope_sin);

/**
 * @brief Computes an SSM (state space model) stage (Mamba-style recurrence).
 * 
 * Placeholder for SSM computation. Replaces attention with state-space model
 * recurrence with convolutional kernel, as in Mamba or similar architectures.
 * 
 * @param buf Layer buffers (pre-allocated).
 * @param ctx Transformer layer context.
 * @param hidden Input/output hidden state [d_model].
 */
void compute_ssm_stage(layer_buffers_t buf,
                       transformer_layer_ctx_t* ctx,
                       float* hidden);

/**
 * @brief Computes the FFN stage of a transformer layer.
 * Shared across all layer types (softmax, linear attention, SSM).
 */
void compute_ffn_stage(layer_buffers_t buf,
                       transformer_layer_ctx_t* ctx,
                       float* hidden);

/**
 * @brief Finalizes layer output (post-FFN norm and residual).
 * Shared across all layer types.
 */
void finalize_layer_output(layer_buffers_t buf,
                           transformer_layer_ctx_t* ctx,
                           float* hidden);

#endif // TRANSFORMER_H

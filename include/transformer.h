#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "llm_model.h"
#include "gemma3_270m_config.h"

struct inference_session_t;

typedef struct {
    const float* cos;
    const float* sin;
} transformer_rope_t;

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

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "llm_model.h"
#include "gemma3_270m_config.h"

struct inference_session_t;

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
 * @param rope_cos Cosine frequencies for RoPE.
 * @param rope_sin Sine frequencies for RoPE.
 * @return 0 on success.
 */
int sapphire_transformer_layer(struct inference_session_t* session, int layer_idx, int token_pos, float* hidden,
                               const float* rope_cos, const float* rope_sin);

/**
 * @brief Performs embedding lookup.
 */
void sapphire_embed_lookup(struct inference_session_t* session, int token_id, float* hidden);

/**
 * @brief Performs LM Head calculation and softcapping.
 */
void lm_head(struct inference_session_t* session, const float* hidden, float* logits);

typedef struct layer_buffers {
    int pm, pi, pk, pf;
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
    int d_model;
    int head_dim;
} transformer_layer_ctx_t;

/**
 * @brief Computes the Attention stage of a transformer layer.
 * includes Norm -> Projections -> QK Norm -> RoPE -> Attention -> Output Projection -> Post Norm
 */
void compute_attention_stage(layer_buffers_t buf,
                             transformer_layer_ctx_t* ctx,
                             float* hidden,
                             const float* rope_cos,
                             const float* rope_sin);

#endif // TRANSFORMER_H

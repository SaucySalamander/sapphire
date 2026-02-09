#ifndef ATTENTION_H
#include <stdbool.h>
#include "transformer.h"
#include "llm_model.h"
#include "gemma3_270m_config.h"

#define ATTN_DEBUG_MAX_TOPK 16

/**
 * @brief Configuration for attention debugging.
 */
typedef struct {
    int initialized;
    int enabled;
    int layer_filter;
    int head_filter;
    int token_limit;
    int top_k;
    int max_print;
} attention_debug_config_t;

/**
 * @brief Get the attention debug configuration from environment variables.
 */
const attention_debug_config_t* get_attention_debug_config(void);

/**
 * @brief Check if debugging should be logged for a specific token/layer/head.
 */
bool attention_debug_should_log(const attention_debug_config_t* cfg, int layer_idx, int head_idx, int token_pos);

struct inference_session_t;

/**
 * @brief Forward pass for multi-head attention (GQA supported).
 * 
 * @param session Inference session.
 * @param layer_idx Current layer index.
 * @param token_pos Current token position.
 * @param q_proj Pre-calculated Query projections (RoPE already applied).
 * @param attn_out Output buffer for attention results [d_model].
 * @return 0 on success.
 */
int sapphire_attention_forward(struct inference_session_t* session, int layer_idx, int token_pos,
                               float* q_proj, float* attn_out);

/**
 * @brief Compute attention stage for a transformer layer.
 * 
 * Performs the complete attention computation including:
 * - Pre-attention RMSNorm
 * - Query, Key, Value projections with RoPE
 * - QK-Normalization (Gemma 3 specific)
 * - Softmax attention with optional softcapping
 * - Output projection
 * - Post-attention RMSNorm and residual connection
 * 
 * @param buf Layer buffer with scratch memory and cached outputs
 * @param ctx Layer context (session, layer weights, config, indices)
 * @param hidden Input hidden state [d_model]
 * @param rope_cos Precomputed RoPE cosine values
 * @param rope_sin Precomputed RoPE sine values
 */
void compute_attention_stage(layer_buffers_t buf,
                             transformer_layer_ctx_t* ctx,
                             float* hidden,
                             const float* rope_cos,
                             const float* rope_sin);

#endif // ATTENTION_H

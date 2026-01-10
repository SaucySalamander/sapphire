#ifndef ATTENTION_H
#include <stdbool.h>

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

/**
 * @brief Dump detailed attention information to stderr.
 */
void attention_debug_dump(
    const attention_debug_config_t* cfg,
    float* raw_scores_buf,
    const float* head_q,
    const float* k_base,
    const float* softmax_scores,
    int head_dim,
    int attn_len,
    int layer_idx,
    int head_idx,
    int kv_head_idx,
    int token_pos,
    int window_start,
    bool is_global_layer);

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

#endif // ATTENTION_H

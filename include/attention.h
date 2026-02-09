/**
 * @file attention.h
 * @brief Multi-head and Grouped-Query Attention (GQA) implementation.
 *
 * This module is the exclusive owner of QK-Normalization, Scaling, and
 * Softcapping logic required for Gemma 3.
 */

#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdbool.h>
#include "llm_model.h"
#include "attention_strategy.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// ATTENTION STRATEGIES (Scaling & Normalization)
// ============================================================================
// Moved to attention_strategy.h



/**
 * @brief ALiBi (Attention with Linear Biases) placeholder strategy.
 */
void alibi_attention_strategy(
    float *scores,
    int context_length,
    int d_k,
    void *user_data
);

// ============================================================================
// CORE ATTENTION INTERFACE
// ============================================================================

struct inference_session_t;
typedef struct transformer_layer_ctx transformer_layer_ctx_t;
typedef struct layer_buffers layer_buffers_t;

/**
 * @brief Forward pass for multi-head attention (GQA supported).
 */
int sapphire_attention_forward(struct inference_session_t* session, int layer_idx, int token_pos,
                               float* q_proj, float* attn_out);

// ============================================================================
// DEBUGGING & INSTRUMENTATION
// ============================================================================

#define ATTN_DEBUG_MAX_TOPK 16

typedef struct {
    int initialized;
    int enabled;
    int layer_filter;
    int head_filter;
    int token_limit;
    int top_k;
    int max_print;
} attention_debug_config_t;

const attention_debug_config_t* get_attention_debug_config(void);
bool attention_debug_should_log(const attention_debug_config_t* cfg, int layer_idx, int head_idx, int token_pos);

#ifdef __cplusplus
}
#endif

#endif // ATTENTION_H

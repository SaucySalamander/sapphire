#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
void sapphire_lm_head(struct inference_session_t* session, float* hidden, float* logits);

#endif // TRANSFORMER_H

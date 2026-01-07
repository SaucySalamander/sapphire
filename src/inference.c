/**
 * @file inference.c
 * @brief Inference session and full forward pass implementation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/inference.h"
#include "../include/ggml_model.h"
#include "../include/tensor.h"
#include "../include/kv_cache.h"
#include "../include/rope.h"
#include "../include/attention.h"
#include "../include/activations.h"
#include "../include/normalization.h"
#include "../include/tensor_gemv.h"

/**
 * Precompute RoPE frequencies (cos and sin for all positions and dimensions).
 */
static int precompute_rope_freqs(float *freqs_cos, float *freqs_sin,
                                  int d_model, int max_context_len, float rope_base) {
    if (!freqs_cos || !freqs_sin) return -1;
    
    int d_k = d_model;  // For simplicity, assuming no head dimension reduction
    
    for (int pos = 0; pos < max_context_len; pos++) {
        for (int dim = 0; dim < d_k; dim += 2) {
            float freq = 1.0f / powf(rope_base, (float)dim / (float)d_k);
            float angle = (float)pos * freq;
            
            freqs_cos[pos * d_k + dim] = cosf(angle);
            freqs_cos[pos * d_k + dim + 1] = cosf(angle);
            
            freqs_sin[pos * d_k + dim] = sinf(angle);
            freqs_sin[pos * d_k + dim + 1] = sinf(angle);
        }
    }
    
    return 0;
}

/**
 * Create an inference session.
 */
inference_session_t* inference_session_create(llm_model_t *model, int max_context_len) {
    if (!model) {
        fprintf(stderr, "ERROR: inference_session_create requires model\n");
        return NULL;
    }
    
    inference_session_t *session = (inference_session_t *)malloc(sizeof(inference_session_t));
    if (!session) {
        fprintf(stderr, "ERROR: Failed to allocate inference session\n");
        return NULL;
    }
    
    session->model = model;
    session->scratch_size = model->config.d_model * 4;  // 4x hidden dim scratch buffer
    session->scratch_buffer = (float *)malloc(session->scratch_size * sizeof(float));
    
    if (!session->scratch_buffer) {
        fprintf(stderr, "ERROR: Failed to allocate scratch buffer\n");
        free(session);
        return NULL;
    }
    
    // Allocate KV caches (one per layer)
    session->layer_kv_caches = (kv_cache_t **)malloc(model->config.num_layers * sizeof(kv_cache_t *));
    if (!session->layer_kv_caches) {
        fprintf(stderr, "ERROR: Failed to allocate KV cache array\n");
        free(session->scratch_buffer);
        free(session);
        return NULL;
    }
    
    for (int i = 0; i < model->config.num_layers; i++) {
        session->layer_kv_caches[i] = kv_cache_create(
            max_context_len,
            model->config.d_k
        );
        if (!session->layer_kv_caches[i]) {
            fprintf(stderr, "ERROR: Failed to create KV cache for layer %d\n", i);
            for (int j = 0; j < i; j++) {
                kv_cache_release(session->layer_kv_caches[j]);
            }
            free(session->layer_kv_caches);
            free(session->scratch_buffer);
            free(session);
            return NULL;
        }
    }
    
    // Precompute RoPE frequencies
    int freq_size = max_context_len * model->config.d_model;
    session->rope_freqs_cos = (float *)malloc(freq_size * sizeof(float));
    session->rope_freqs_sin = (float *)malloc(freq_size * sizeof(float));
    
    if (!session->rope_freqs_cos || !session->rope_freqs_sin) {
        fprintf(stderr, "ERROR: Failed to allocate RoPE frequencies\n");
        if (session->rope_freqs_cos) free(session->rope_freqs_cos);
        if (session->rope_freqs_sin) free(session->rope_freqs_sin);
        for (int i = 0; i < model->config.num_layers; i++) {
            kv_cache_release(session->layer_kv_caches[i]);
        }
        free(session->layer_kv_caches);
        free(session->scratch_buffer);
        free(session);
        return NULL;
    }
    
    if (precompute_rope_freqs(session->rope_freqs_cos, session->rope_freqs_sin,
                              model->config.d_model, max_context_len, model->config.rope_base) < 0) {
        fprintf(stderr, "ERROR: Failed to precompute RoPE frequencies\n");
        free(session->rope_freqs_cos);
        free(session->rope_freqs_sin);
        for (int i = 0; i < model->config.num_layers; i++) {
            kv_cache_release(session->layer_kv_caches[i]);
        }
        free(session->layer_kv_caches);
        free(session->scratch_buffer);
        free(session);
        return NULL;
    }
    
    fprintf(stderr, "INFO: Inference session created with %d layers, context_len=%d\n",
            model->config.num_layers, max_context_len);
    
    return session;
}

/**
 * Reset KV caches for a new sequence.
 */
void inference_session_reset(inference_session_t *session) {
    if (!session) return;
    
    for (int i = 0; i < session->model->config.num_layers; i++) {
        kv_cache_reset(session->layer_kv_caches[i]);
    }
}

/**
 * Single-token forward pass (placeholder for now - stub implementation).
 *
 * This is a placeholder that demonstrates the structure. Full implementation
 * requires integration with attention, normalization, and projection operators.
 */
void inference_forward(inference_session_t *session, int token_id, int token_pos, float *logits) {
    if (!session || !logits) {
        fprintf(stderr, "ERROR: inference_forward requires session and logits\n");
        return;
    }
    
    llm_model_t *model = session->model;
    
    // 1. Token embedding lookup
    // (This requires embedding matrix access and token_id validation)
    if (!model->embedding_weight) {
        fprintf(stderr, "ERROR: Model has no embedding weights\n");
        memset(logits, 0, model->config.vocab_size * sizeof(float));
        return;
    }
    
    float *hidden = session->scratch_buffer;  // Use scratch for hidden state
    
    // Simplified: assume embedding row lookup (in practice, use tensor_gemv)
    // hidden = embedding[token_id, :]
    // For now, set to small value
    for (int i = 0; i < model->config.d_model; i++) {
        hidden[i] = 0.1f * (i % 10);
    }
    
    // 2. Transformer block forward passes
    float *layer_in = hidden;
    float *layer_out = session->scratch_buffer + model->config.d_model;
    
    for (int layer_idx = 0; layer_idx < model->config.num_layers; layer_idx++) {
        model_layer_weights_t *layer = &model->layers[layer_idx];
        
        // Placeholder: implement actual layer forward pass
        // This would include:
        // - Layer norm
        // - Attention (Q, K, V projections, attention computation, output projection)
        // - Feed-forward network
        // - Residual connections
        
        // For now, copy input to output
        memcpy(layer_out, layer_in, model->config.d_model * sizeof(float));
        
        // Swap buffers
        float *tmp = layer_in;
        layer_in = layer_out;
        layer_out = tmp;
    }
    
    // 3. Final layer norm
    // (Apply to layer_in)
    
    // 4. LM head projection to logits
    if (!model->lm_head_weight) {
        fprintf(stderr, "ERROR: Model has no LM head weights\n");
        memset(logits, 0, model->config.vocab_size * sizeof(float));
        return;
    }
    
    // Simplified: set logits to small random values
    for (int i = 0; i < model->config.vocab_size; i++) {
        logits[i] = 0.01f * ((i * 7) % 100);
    }
}

/**
 * Free inference session.
 */
void inference_session_destroy(inference_session_t *session) {
    if (!session) return;
    
    if (session->layer_kv_caches) {
        for (int i = 0; i < session->model->config.num_layers; i++) {
            if (session->layer_kv_caches[i]) {
                kv_cache_release(session->layer_kv_caches[i]);
            }
        }
        free(session->layer_kv_caches);
    }
    
    if (session->scratch_buffer) {
        free(session->scratch_buffer);
    }
    
    if (session->rope_freqs_cos) {
        free(session->rope_freqs_cos);
    }
    
    if (session->rope_freqs_sin) {
        free(session->rope_freqs_sin);
    }
    
    free(session);
}

/**
 * Generate tokens using greedy decoding (argmax).
 */
int llm_generate_greedy(inference_session_t *session,
                        const int *prompt_tokens,
                        int prompt_len,
                        int max_tokens,
                        int *output_tokens) {
    if (!session || !prompt_tokens || !output_tokens) {
        fprintf(stderr, "ERROR: llm_generate_greedy requires session, prompt, and output\n");
        return 0;
    }
    
    inference_session_reset(session);
    
    float *logits = (float *)malloc(session->model->config.vocab_size * sizeof(float));
    if (!logits) {
        fprintf(stderr, "ERROR: Failed to allocate logits buffer\n");
        return 0;
    }
    
    int token_count = 0;
    
    // Process prompt
    for (int i = 0; i < prompt_len && token_count < max_tokens; i++) {
        inference_forward(session, prompt_tokens[i], i, logits);
        output_tokens[token_count++] = prompt_tokens[i];
    }
    
    // Generate new tokens
    int last_token = prompt_tokens[prompt_len - 1];
    for (int i = prompt_len; i < max_tokens; i++) {
        inference_forward(session, last_token, i, logits);
        
        // Greedy: pick argmax
        int next_token = 0;
        float max_logit = logits[0];
        for (int j = 1; j < session->model->config.vocab_size; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token = j;
            }
        }
        
        output_tokens[token_count++] = next_token;
        last_token = next_token;
        
        // Simple stopping condition: if token is very small ID (likely end-of-sequence)
        if (next_token < 2) {
            break;
        }
    }
    
    free(logits);
    return token_count;
}

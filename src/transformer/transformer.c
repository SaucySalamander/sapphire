#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/gemma3_270m_config.h"
#include "activations.h"
#include "attention.h"
#include "inference.h"
#include "normalization.h"
#include "rope.h"
#include "tensor.h"
#include "utils.h"
#include "log.h"

// Helper to handle BF16 norm weights on the fly
static const float* get_norm_weights(const tensor_t* weight, float* scratch, int n) {
    if (!weight) return NULL;
    if (tensor_dtype(weight) == DTYPE_BF16) {
        bf16_to_f32_vec(scratch, (const uint16_t*)tensor_data(weight), n);
        return scratch;
    }
    return (const float*)tensor_data(weight);
}

/**
 * @brief Forward pass for a single transformer layer.
 */
int sapphire_transformer_layer(struct inference_session_t* session, int layer_idx, int token_pos, float* hidden,
                               const float* rope_cos, const float* rope_sin) {
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;
    model_layer_weights_t* layer = &model->layers[layer_idx];
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;
    int norm_weights_are_deltas = 0;

    int d_model = config->hidden_size;
    int head_dim = config->head_dim > 0 ? config->head_dim : (d_model / config->num_attention_heads);

    // Buffers from scratch space based on inference.c layout:
    // 0: hidden (passed in)
    // 1: residual
    // 2: norm_buf
    // 3: q_proj
    // 4: k_proj
    // 5: v_proj
    // 6: attn_out
    // 7: ffn_gate
    // 8: ffn_value
    // 9: geglu_input

    int pm = session->padded_d_model;
    int pi = session->padded_d_inner;
    int pk = session->padded_d_kv;
    int pf = session->padded_d_ff;

    float* residual = session->scratch_buffer + pm;
    float* norm_buf = session->scratch_buffer + 2 * pm;
    float* q_proj = session->scratch_buffer + 3 * pm;
    float* k_proj = q_proj + pi;
    float* v_proj = k_proj + pk;
    float* attn_out = v_proj + pk;
    float* ffn_gate_buf = attn_out + pi;
    float* ffn_value_buf = ffn_gate_buf + pf;
    float* geglu_buf = ffn_value_buf + pf;

    // 1. Residual Connection (Save hidden to residual)
    vec_copy(residual, hidden, d_model);

    // 2. Pre-attention RMSNorm
    const float* norm_attn_data = get_norm_weights(layer->norm_attn_weight, q_proj, d_model);
    if (norm_weights_are_deltas) {
        sapphire_rmsnorm_delta(norm_buf, hidden, norm_attn_data, 1e-6f, d_model);
    } else {
        sapphire_rmsnorm(norm_buf, hidden, norm_attn_data, 1e-6f, d_model);
    }

    // 3. Projections (Q, K, V)
    tensor_gemv_with_ctx(session->gemv_ctx, q_proj, layer->q_proj_weight, norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, k_proj, layer->k_proj_weight, norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, v_proj, layer->v_proj_weight, norm_buf);

    if (log_get_level() == LOG_LEVEL_DEBUG) {
        float r_q = 0, r_k = 0, r_v = 0;
        int n_q = config->num_attention_heads * head_dim;
        int n_kv = config->num_key_value_heads * head_dim;
        vec_stats(q_proj, n_q, NULL, NULL, &r_q);
        vec_stats(k_proj, n_kv, NULL, NULL, &r_k);
        vec_stats(v_proj, n_kv, NULL, NULL, &r_v);
        LOG_DEBUG("Layer %d Proj RMS: Q=%.3f K=%.3f V=%.3f", layer_idx, r_q, r_k, r_v);
    }

    // 5. RoPE application
    // Use the explicitly provided RoPE frequencies (toggled in inference.c based on layer type)
    for (int h = 0; h < config->num_attention_heads; h++) rope_apply_fast(q_proj + h * head_dim, token_pos, head_dim, rope_cos, rope_sin);
    for (int h = 0; h < config->num_key_value_heads; h++) rope_apply_fast(k_proj + h * head_dim, token_pos, head_dim, rope_cos, rope_sin);

    // 6. QK-Normalization (Gemma 3: After RoPE, integrated via apply_qk_norm)
    float* q_scale_ptr = NULL;
    float* k_scale_ptr = NULL;

    if (layer->q_norm_weight) {
        int q_norm_len = tensor_shape(layer->q_norm_weight)[0];
        const float* raw = get_norm_weights(layer->q_norm_weight, ffn_gate_buf, q_norm_len);
        int expected_q = config->num_attention_heads * head_dim;
        if (q_norm_len == expected_q) {
            q_scale_ptr = (float*)raw;  // per-head gamma
        } else if (q_norm_len == head_dim) {
            // Broadcast single-head gamma to all Q heads (common for small Gemma configs)
            float head_gamma[head_dim];
            memcpy(head_gamma, raw, head_dim * sizeof(float));
            // reuse ffn_gate_buf as expanded gamma scratch (size pf >= expected_q)
            q_scale_ptr = ffn_gate_buf;
            for (int h = 0; h < config->num_attention_heads; h++) {
                memcpy(q_scale_ptr + h * head_dim, head_gamma, head_dim * sizeof(float));
            }
            LOG_DEBUG("Layer %d q_norm len=%d; broadcasting gamma to %d heads", layer_idx, q_norm_len, config->num_attention_heads);
        } else {
            LOG_WARN("Layer %d q_norm len=%d expected=%d; disabling QK-Norm for Q", layer_idx, q_norm_len, expected_q);
            q_scale_ptr = NULL;
        }
    }

    if (layer->k_norm_weight) {
        int k_norm_len = tensor_shape(layer->k_norm_weight)[0];
        const float* raw = get_norm_weights(layer->k_norm_weight, ffn_value_buf, k_norm_len);
        int expected_k = config->num_key_value_heads * head_dim;
        if (k_norm_len == expected_k) {
            k_scale_ptr = (float*)raw;  // per-KV-head gamma
        } else if (k_norm_len == head_dim) {
            // Broadcast single-head gamma to all KV heads (GQA)
            float head_gamma[head_dim];
            memcpy(head_gamma, raw, head_dim * sizeof(float));
            // reuse ffn_value_buf as expanded gamma scratch (size pf >= expected_k)
            k_scale_ptr = ffn_value_buf;
            for (int h = 0; h < config->num_key_value_heads; h++) {
                memcpy(k_scale_ptr + h * head_dim, head_gamma, head_dim * sizeof(float));
            }
            LOG_DEBUG("Layer %d k_norm len=%d; broadcasting gamma to %d KV heads", layer_idx, k_norm_len, config->num_key_value_heads);
        } else {
            LOG_WARN("Layer %d k_norm len=%d expected=%d; disabling QK-Norm for K", layer_idx, k_norm_len, expected_k);
            k_scale_ptr = NULL;
        }
    }

    // Apply normalization if we have scales
    if (q_scale_ptr || k_scale_ptr) {
        apply_qk_norm(q_proj, k_proj, q_scale_ptr, k_scale_ptr, head_dim, config->num_attention_heads, config->num_key_value_heads);
    }

    // 5a. Query Scaling
    // UPDATE: Removed "16.0x Scaling Rule" based on new Gemma 3 technical report.
    // QK-Norm + Standard 1/sqrt(d) attention scaling is sufficient.
    // We rely on attention.c to handle the scaling.

    // 6. KV-Cache Commit
    kv_cache_write_token(session->kv_cache, layer_idx, token_pos, k_proj, v_proj);

    // 7. Attention Forward Pass
    sapphire_attention_forward(session, layer_idx, token_pos, q_proj, attn_out);

    if (log_get_level() == LOG_LEVEL_DEBUG) {
        float r_a = 0;
        vec_stats(attn_out, config->num_attention_heads * head_dim, NULL, NULL, &r_a);
        LOG_DEBUG("Layer %d AttnOut RMS: %.3f", layer_idx, r_a);
    }

    // 8. Output projection
    tensor_gemv_with_ctx(session->gemv_ctx, hidden, layer->out_proj_weight, attn_out);

    // FIX (Run 32): Removed Residual Dampening (Run 28 Logic: No Brakes)
    // vec_scale(hidden, residual_scale, d_model);

    if (getenv("SAPPHIRE_LOG_TENSORS") && token_pos == 0) {
        float r_o = 0;
        vec_stats(hidden, d_model, NULL, NULL, &r_o);
        fprintf(stderr, "DEBUG: Layer %d OutProj RMS: %.3f\n", layer_idx, r_o);
    }

    // TEMP SUSH MECHANISM (Attention)
    // 8a. Post-Attention RMSNorm (Gemma 3) & Residual Sum
    // hidden currently contains OutProj output (Attn Result)
    if (layer->norm_attn_post_weight) {
        const float* norm_post_w = get_norm_weights(layer->norm_attn_post_weight, q_proj, d_model);
        if (norm_weights_are_deltas)
            sapphire_rmsnorm_delta(norm_buf, hidden, norm_post_w, 1e-6f, d_model);
        else
            sapphire_rmsnorm(norm_buf, hidden, norm_post_w, 1e-6f, d_model);

        vec_add(residual, norm_buf, d_model);
    } else {
        vec_add(residual, hidden, d_model);
    }

    // 9. FFN stage (Sequential)
    // Use RESIDUAL (Current State) for FFN Norm
    const float* norm_ffn_data = NULL;
    if (layer->norm_ffn_weight) {
        int n_norm = tensor_shape(layer->norm_ffn_weight)[0];
        if (n_norm == d_model) {
            norm_ffn_data = get_norm_weights(layer->norm_ffn_weight, q_proj, d_model);
        } else {
            fprintf(stderr, "WARN: layer %d norm_ffn_weight len=%d expected=%d; using weightless RMSNorm\n", layer_idx, n_norm, d_model);
        }
    }

    if (norm_ffn_data) {
        if (norm_weights_are_deltas)
            sapphire_rmsnorm_delta(norm_buf, residual, norm_ffn_data, 1e-6f, d_model);
        else
            sapphire_rmsnorm(norm_buf, residual, norm_ffn_data, 1e-6f, d_model);
    } else {
        // Fallback: weightless RMSNorm (normalize without learned gamma)
        // Reuse q_proj scratch to hold ones if needed for sapphire_rmsnorm path
        // but sapphire_rmsnorm_no_weight expects in-place operation; use the out-variant instead
        // Create an implicit ones vector in q_proj and call sapphire_rmsnorm
        for (int i = 0; i < d_model; i++) q_proj[i] = 1.0f;
        sapphire_rmsnorm(norm_buf, residual, q_proj, 1e-6f, d_model);
    }

    tensor_gemv_with_ctx(session->gemv_ctx, ffn_gate_buf, layer->gate_proj_weight, norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, ffn_value_buf, layer->up_proj_weight, norm_buf);

    if (getenv("SAPPHIRE_LOG_TENSORS") && token_pos == 0) {
        float r_g = 0, r_u = 0;
        vec_stats(ffn_gate_buf, config->intermediate_size, NULL, NULL, &r_g);
        vec_stats(ffn_value_buf, config->intermediate_size, NULL, NULL, &r_u);
        fprintf(stderr, "DEBUG: Layer %d FFN Proj RMS: G=%.3f U=%.3f\n", layer_idx, r_g, r_u);
    }

    // Activations
    memcpy(geglu_buf, ffn_value_buf, config->intermediate_size * sizeof(float));
    memcpy(geglu_buf + config->intermediate_size, ffn_gate_buf, config->intermediate_size * sizeof(float));
    sapphire_geglu(ffn_gate_buf, geglu_buf, 2 * config->intermediate_size);

    if (getenv("SAPPHIRE_LOG_TENSORS") && token_pos == 0) {
        float r_a = 0;
        vec_stats(ffn_gate_buf, config->intermediate_size, NULL, NULL, &r_a);
        fprintf(stderr, "DEBUG: Layer %d Activation RMS: %.3f\n", layer_idx, r_a);
    }

    tensor_gemv_with_ctx(session->gemv_ctx, hidden, layer->down_proj_weight, ffn_gate_buf);

    // 9a. Post-FFN RMSNorm (Gemma 3)
    // hidden currently contains DownProj output (FFN Result)
    if (layer->norm_ffn_post_weight) {
        int n_post = tensor_shape(layer->norm_ffn_post_weight)[0];
        const float* norm_post_w = NULL;
        if (n_post == d_model) {
            norm_post_w = get_norm_weights(layer->norm_ffn_post_weight, q_proj, d_model);
            if (token_pos == 0 && layer_idx == 0) {
                fprintf(stderr, "DEBUG: Layer 0 has norm_ffn_post_weight (ptr=%p). Applying Post-FFN Norm.\n", norm_post_w);
            }
            if (norm_weights_are_deltas)
                sapphire_rmsnorm_delta(norm_buf, hidden, norm_post_w, 1e-6f, d_model);
            else
                sapphire_rmsnorm(norm_buf, hidden, norm_post_w, 1e-6f, d_model);

            vec_add(residual, norm_buf, d_model);
        } else {
            fprintf(stderr, "WARN: layer %d norm_ffn_post_weight len=%d expected=%d; using weightless RMSNorm\n", layer_idx, n_post, d_model);
            // Fallback: compute weightless normalized output
            for (int i = 0; i < d_model; i++) q_proj[i] = 1.0f;
            sapphire_rmsnorm(norm_buf, hidden, q_proj, 1e-6f, d_model);
            vec_add(residual, norm_buf, d_model);
        }
    } else {
        if (token_pos == 0 && layer_idx == 0) {
            fprintf(stderr, "DEBUG: Layer 0 MISSING norm_ffn_post_weight. Adding raw FFN output.\n");
        }
        vec_add(residual, hidden, d_model);
    }

    // Copy computed residual back to hidden (Layer Output)
    vec_copy(hidden, residual, d_model);

    return 0;
}

void sapphire_embed_lookup(struct inference_session_t* session, int token_id, float* hidden) {
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;

    const uint16_t* embed_table_bf16 = (const uint16_t*)tensor_data(model->embedding_weight);
    bf16_to_f32_vec(hidden, embed_table_bf16 + (token_id * config->hidden_size), config->hidden_size);

    if (getenv("SAPPHIRE_LOG_TENSORS")) {
        float mn, mx, rms;
        vec_stats(hidden, config->hidden_size, &mn, &mx, &rms);
        fprintf(stderr, "DEBUG[EMBED]: token=%d before_scale rms=%.4f\n", token_id, rms);
    }

    // Gemma 3 requires input embeddings to be scaled by sqrt(d_model)
    // ENABLED: RMS=0.04 (raw) * 25.3 = ~1.0 (correct for transformer inputs)
    if (!getenv("SAPPHIRE_NO_EMBED_SCALE")) {
        float embed_scale = sqrtf((float)config->hidden_size);
        vec_scale(hidden, embed_scale, config->hidden_size);

        if (getenv("SAPPHIRE_LOG_TENSORS")) {
            float mn, mx, rms;
            vec_stats(hidden, config->hidden_size, &mn, &mx, &rms);
            fprintf(stderr, "DEBUG[EMBED]: token=%d after_scale rms=%.4f scale=%.4f\n", token_id, rms, embed_scale);
        }
    }
}

void sapphire_lm_head(struct inference_session_t* session, float* hidden, float* logits) {
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;
    int norm_weights_are_deltas = 0;

    float* scratch_norm = session->scratch_buffer + 2 * session->padded_d_model;  // reuse norm_buf

    // 1. Final RMSNorm
    // Reuse q_proj area.
    float* tmp_w = session->scratch_buffer + 3 * session->padded_d_model;
    const float* final_w = get_norm_weights(model->norm_final_weight, tmp_w, config->hidden_size);

    if (norm_weights_are_deltas)
        sapphire_rmsnorm_delta(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);
    else
        sapphire_rmsnorm(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);

    if (getenv("SAPPHIRE_LOG_TENSORS")) {
        float mn, mx, rms;
        vec_stats(scratch_norm, config->hidden_size, &mn, &mx, &rms);
        fprintf(stderr, "DEBUG: Final Norm Out: min=%.3f max=%.3f rms=%.3f\n", mn, mx, rms);
    }

    // 2. Output Projection (Weight Tying)
    // Perform projection first to maintain numerical stability in the hidden state
    tensor_gemv_with_ctx(session->gemv_ctx, logits, model->embedding_weight, scratch_norm);

    // Final logit scaling: optionally applied via env var.
    // Use SAPPHIRE_FINAL_LOGIT_SCALE to match external runtimes if required
    // (for example, set to 1.0 / sqrt(d_model) or 1.0 / 25.3 depending on model).
    {
        const char* env = getenv("SAPPHIRE_FINAL_LOGIT_SCALE");
        if (env && env[0] != '\0') {
            float final_scale = strtof(env, NULL);
            if (final_scale != 1.0f) {
                vec_scale(logits, final_scale, config->vocab_size);
                fprintf(stderr, "DEBUG: Applied SAPPHIRE_FINAL_LOGIT_SCALE=%.6f to logits\n", final_scale);
            } else {
                fprintf(stderr, "DEBUG: SAPPHIRE_FINAL_LOGIT_SCALE set to 1.0 (no-op)\n");
            }
        }
    }

    // 3a. Final Logit Soft-Capping (Gemma family behavior)
    // Apply if model config specifies a positive final_logit_softcap or when
    // SAPPHIRE_FINAL_LOGIT_SOFTCAP is set (env overrides model config). The
    // transform applied is: logits := cap * tanh(logits / cap)
    {
        float cap = config->final_logit_softcapping;
        const char* env_cap = getenv("SAPPHIRE_FINAL_LOGIT_SOFTCAP");
        if (env_cap && env_cap[0] != '\0') {
            cap = strtof(env_cap, NULL);
        }
        if (cap > 0.0f) {
            int V = config->vocab_size;
            for (int i = 0; i < V; i++) {
                logits[i] = cap * tanhf(logits[i] / cap);
            }
            fprintf(stderr, "DEBUG: Applied final logit softcap (cap=%.6f) to logits\n", cap);
        }
    }

    // 4. Final Logit Soft-Capping
    // UPDATE: Removed per new Gemma 3 technical report ("replace... with QK-norm").
    // We rely on stable internal activations.
}

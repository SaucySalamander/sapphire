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

    /*
     * Dedicated scratch region for dequantized weight rows returned by
     * `get_norm_weights`. Using a tail-allocated buffer prevents accidental
     * aliasing with activation buffers like `q_proj`, `ffn_gate_buf`, etc.
     */
    size_t scratch_floats = session->scratch_size / sizeof(float);
    int max_needed = d_model;
    int q_needed = config->num_attention_heads * head_dim;
    int k_needed = config->num_key_value_heads * head_dim;
    if (q_needed > max_needed) max_needed = q_needed;
    if (k_needed > max_needed) max_needed = k_needed;

    float* weight_scratch = NULL;
    if ((size_t)max_needed <= scratch_floats) {
        weight_scratch = session->scratch_buffer + (scratch_floats - max_needed);
    } else {
        LOG_WARN("Insufficient scratch for weight_scratch: need=%d have=%zu; falling back to q_proj (may alias)",
                 max_needed, scratch_floats);
        weight_scratch = q_proj; /* best-effort fallback to previous behavior */
    }

    // 1. Residual Connection (Save hidden to residual)
    vec_copy(residual, hidden, d_model);

    // DEBUG: log residual RMS at layer entry on first token
    if (layer_idx == 0 && getenv("SAPPHIRE_DEBUG_RMS")) {
        float mn, mx, rms;
        vec_stats(residual, d_model, &mn, &mx, &rms);
        LOG_DEBUG("Layer 0 ENTRY residual RMS=%.2f (embedding input)", rms);
    }

    // 2. Pre-attention RMSNorm
    if (getenv("SAPPHIRE_DEBUG_LAYER_NORMS")) {
        if (layer->norm_attn_weight) {
            int _len = tensor_shape(layer->norm_attn_weight)[0];
            LOG_DEBUG("Layer %d norm_attn_weight ptr=%p len=%d", layer_idx, layer->norm_attn_weight, _len);
        } else {
            LOG_DEBUG("Layer %d norm_attn_weight MISSING (NULL)", layer_idx);
        }
    }
    const float* norm_attn_data = get_norm_weights(layer->norm_attn_weight, weight_scratch, d_model);
    sapphire_rmsnorm(norm_buf, hidden, norm_attn_data, 1e-6f, d_model);
    
    // 3. Projections (Q, K, V)
    tensor_gemv_with_ctx(session->gemv_ctx, q_proj, layer->q_proj_weight, norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, k_proj, layer->k_proj_weight, norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, v_proj, layer->v_proj_weight, norm_buf);

    /* Debug: print computed projection shapes and buffer paddings to help
     * diagnose head-dimension / GQA mismatches (visible when
     * SAPPHIRE_DEBUG_DUMPS or DEBUG log level enabled). */
    if (getenv("SAPPHIRE_DEBUG_DUMPS") || log_get_level() == LOG_LEVEL_DEBUG) {
        int n_q = config->num_attention_heads * head_dim;
        int n_kv = config->num_key_value_heads * head_dim;
        LOG_DEBUG("DEBUG_SHAPE Layer %d head_dim=%d num_heads=%d num_kv_heads=%d n_q=%d n_kv=%d padded_d_model=%d padded_d_inner=%d padded_d_kv=%d",
                  layer_idx, head_dim, config->num_attention_heads, config->num_key_value_heads,
                  n_q, n_kv, pm, pi, pk);
    }

    // 4. QK-Normalization
    float* q_scale_ptr = NULL;
    float* k_scale_ptr = NULL;

    if (layer->q_norm_weight) {
        int q_norm_len = tensor_shape(layer->q_norm_weight)[0];
        const float* raw = get_norm_weights(layer->q_norm_weight, weight_scratch, q_norm_len);
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
        const float* raw = get_norm_weights(layer->k_norm_weight, weight_scratch, k_norm_len);
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
    if (log_get_level() == LOG_LEVEL_DEBUG) {
        float r_q = 0, r_k = 0, r_v = 0;
        int n_q = config->num_attention_heads * head_dim;
        int n_kv = config->num_key_value_heads * head_dim;
        vec_stats(q_proj, n_q, NULL, NULL, &r_q);
        vec_stats(k_proj, n_kv, NULL, NULL, &r_k);
        vec_stats(v_proj, n_kv, NULL, NULL, &r_v);
        LOG_DEBUG("Layer %d Proj RMS: Q=%.3f K=%.3f V=%.3f", layer_idx, r_q, r_k, r_v);
    }

    // Targeted dumps for Layer 0, first token
    if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
        int n_print = 8;
        int n_q = config->num_attention_heads * head_dim;
        int n_kv = config->num_key_value_heads * head_dim;
        int qN = n_q < n_print ? n_q : n_print;
        int kvN = n_kv < n_print ? n_kv : n_print;
        LOG_DEBUG("DUMP L0 PROJ: first %d Q values:", qN);
        for (int i = 0; i < qN; ++i) LOG_DEBUG("DUMP L0 PROJ Q[%d]=%.9g", i, q_proj[i]);
        LOG_DEBUG("DUMP L0 PROJ: first %d K values:", kvN);
        for (int i = 0; i < kvN; ++i) LOG_DEBUG("DUMP L0 PROJ K[%d]=%.9g", i, k_proj[i]);
        LOG_DEBUG("DUMP L0 PROJ: first %d V values:", kvN);
        for (int i = 0; i < kvN; ++i) LOG_DEBUG("DUMP L0 PROJ V[%d]=%.9g", i, v_proj[i]);
    }

    // 5. RoPE application
    // Use the explicitly provided RoPE frequencies (toggled in inference.c based on layer type)
    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        for (int h = 0; h < config->num_attention_heads; h++) {
            float mn, mx, rms;
            vec_stats(q_proj + h * head_dim, head_dim, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG_RMS Layer %d Head %d Q before RoPE: min=%.6f max=%.6f rms=%.6f", layer_idx, h, mn, mx, rms);
            if (rms < 0.01f || rms > 100.0f) LOG_WARN("DEBUG_RMS WARNING Layer %d Head %d Q RMS out-of-range=%.6f", layer_idx, h, rms);
        }
        for (int h = 0; h < config->num_key_value_heads; h++) {
            float mn, mx, rms;
            vec_stats(k_proj + h * head_dim, head_dim, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG_RMS Layer %d KVHead %d K before RoPE: min=%.6f max=%.6f rms=%.6f", layer_idx, h, mn, mx, rms);
            if (rms < 0.01f || rms > 100.0f) LOG_WARN("DEBUG_RMS WARNING Layer %d KVHead %d K RMS out-of-range=%.6f", layer_idx, h, rms);
        }
    }

    for (int h = 0; h < config->num_attention_heads; h++) rope_apply_fast(q_proj + h * head_dim, token_pos, head_dim, rope_cos, rope_sin);
    for (int h = 0; h < config->num_key_value_heads; h++) rope_apply_fast(k_proj + h * head_dim, token_pos, head_dim, rope_cos, rope_sin);

    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        for (int h = 0; h < config->num_attention_heads; h++) {
            float mn, mx, rms;
            vec_stats(q_proj + h * head_dim, head_dim, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG_RMS Layer %d Head %d Q after RoPE: min=%.6f max=%.6f rms=%.6f", layer_idx, h, mn, mx, rms);
            if (rms < 0.01f || rms > 100.0f) LOG_WARN("DEBUG_RMS WARNING Layer %d Head %d Q RMS out-of-range=%.6f", layer_idx, h, rms);
        }
        for (int h = 0; h < config->num_key_value_heads; h++) {
            float mn, mx, rms;
            vec_stats(k_proj + h * head_dim, head_dim, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG_RMS Layer %d KVHead %d K after RoPE: min=%.6f max=%.6f rms=%.6f", layer_idx, h, mn, mx, rms);
            if (rms < 0.01f || rms > 100.0f) LOG_WARN("DEBUG_RMS WARNING Layer %d KVHead %d K RMS out-of-range=%.6f", layer_idx, h, rms);
        }
    }

    // 6. KV-Cache Commit
    kv_cache_write_token(session->kv_cache, layer_idx, token_pos, k_proj, v_proj);

    // 7. Attention Forward Pass
    sapphire_attention_forward(session, layer_idx, token_pos, q_proj, attn_out);

    if (log_get_level() == LOG_LEVEL_DEBUG) {
        float r_a = 0;
        vec_stats(attn_out, config->num_attention_heads * head_dim, NULL, NULL, &r_a);
        LOG_DEBUG("Layer %d AttnOut RMS: %.3f", layer_idx, r_a);
    }

    if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
        int N = config->num_attention_heads * head_dim;
        int n_print = N < 16 ? N : 16;
        {
            float _mn,_mx,_rms;
            vec_stats(attn_out, N, &_mn, &_mx, &_rms);
            LOG_DEBUG("DUMP L0 ATTN_OUT: first %d vals (RMS=%.6f):", n_print, _rms);
        }
        for (int i = 0; i < n_print; ++i) LOG_DEBUG("DUMP L0 ATTN_OUT[%d]=%.9g", i, attn_out[i]);
    }

    // 8. Output projection
    tensor_gemv_with_ctx(session->gemv_ctx, hidden, layer->out_proj_weight, attn_out);

    if (getenv("SAPPHIRE_LOG_TENSORS") && token_pos == 0) {
        float r_o = 0;
        vec_stats(hidden, d_model, NULL, NULL, &r_o);
        LOG_DEBUG("Layer %d OutProj RMS: %.3f", layer_idx, r_o);
    }

    // 9. Post-Attention RMSNorm (Gemma 3) & Residual Sum
    if (getenv("SAPPHIRE_DEBUG_LAYER_NORMS")) {
        if (layer->norm_attn_post_weight) {
            int _len = tensor_shape(layer->norm_attn_post_weight)[0];
            LOG_DEBUG("Layer %d norm_attn_post_weight ptr=%p len=%d", layer_idx, layer->norm_attn_post_weight, _len);
        } else {
            LOG_DEBUG("Layer %d norm_attn_post_weight MISSING (NULL)", layer_idx);
        }
    }
    if (layer->norm_attn_post_weight) {
        const float* norm_post_w = get_norm_weights(layer->norm_attn_post_weight, weight_scratch, d_model);
        if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
            int n_print = d_model < 8 ? d_model : 8;
            LOG_DEBUG("DUMP L0 NORM_ATT_POST first %d vals:", n_print);
            for (int i = 0; i < n_print; ++i) LOG_DEBUG("DUMP L0 NORM_ATT_POST[%d]=%.9g", i, norm_post_w[i]);
        }
        sapphire_rmsnorm_delta(norm_buf, hidden, norm_post_w, 1e-6f, d_model);

        if (getenv("SAPPHIRE_DEBUG_RMS")) {
            float rms_norm = 0;
            vec_stats(norm_buf, d_model, NULL, NULL, &rms_norm);
            LOG_DEBUG("Layer %d post-attn norm_buf RMS=%.3f before residual add", layer_idx, rms_norm);
        }

        vec_add(residual, norm_buf, d_model);
        LOG_DEBUG("Layer %d applied post-attention norm and added to residual", layer_idx);
    }
    
    vec_copy(hidden, residual, d_model);

    // 10. FFN stage (Sequential)
    const float* norm_ffn_data = NULL;
    if (layer->norm_ffn_weight) {
        int n_norm = tensor_shape(layer->norm_ffn_weight)[0];
        if (n_norm == d_model) {
            norm_ffn_data = get_norm_weights(layer->norm_ffn_weight, weight_scratch, d_model);
        } else {
            LOG_WARN("layer %d norm_ffn_weight len=%d expected=%d; using weightless RMSNorm", layer_idx, n_norm, d_model);
        }
    }

    if (getenv("SAPPHIRE_DEBUG_LAYER_NORMS")) {
        if (layer->norm_ffn_weight) {
            int _len = tensor_shape(layer->norm_ffn_weight)[0];
            LOG_DEBUG("Layer %d norm_ffn_weight ptr=%p len=%d", layer_idx, layer->norm_ffn_weight, _len);
        } else {
            LOG_DEBUG("Layer %d norm_ffn_weight MISSING (NULL)", layer_idx);
        }
    }

    sapphire_rmsnorm(norm_buf, residual, norm_ffn_data, 1e-6f, d_model);

    /* 11. GEMMA3: GeGLU FFN. Compute gate and value projections, apply GeGLU,
     * then project down. */
    tensor_gemv_with_ctx(session->gemv_ctx, ffn_gate_buf, layer->gate_proj_weight, norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, ffn_value_buf, layer->up_proj_weight, norm_buf);

    if (getenv("SAPPHIRE_LOG_TENSORS") && token_pos == 0) {
        float r_g = 0, r_u = 0;
        vec_stats(ffn_gate_buf, config->intermediate_size, NULL, NULL, &r_g);
        vec_stats(ffn_value_buf, config->intermediate_size, NULL, NULL, &r_u);
        LOG_DEBUG("Layer %d FFN Proj RMS: G=%.3f U=%.3f", layer_idx, r_g, r_u);
    }

    // Targeted dump: GEGLU inputs for Layer 0 Token 0
    if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
        int n_print = config->intermediate_size < 16 ? config->intermediate_size : 16;
        int nonfinite_gate = 0, nonfinite_val = 0;
        for (int i = 0; i < config->intermediate_size; ++i) {
            if (!isfinite(ffn_gate_buf[i])) nonfinite_gate++;
            if (!isfinite(ffn_value_buf[i])) nonfinite_val++;
        }
        LOG_DEBUG("DUMP GEGLU L0: non-finite Gate=%d Value=%d size=%d", nonfinite_gate, nonfinite_val, config->intermediate_size);
        for (int i = 0; i < n_print; ++i) {
            LOG_DEBUG("DUMP GEGLU L0 Gate[%d]=%.9g Value[%d]=%.9g", i, ffn_gate_buf[i], i, ffn_value_buf[i]);
        }
    }

    // Combine Gate (apply GELU on gate) and Value via GeGLU into `ffn_gate_buf` output
    // Optimized in-place GeGLU: apply GELU to gate buffer then element-wise multiply
    // by the value buffer to avoid extra memcpy and temporary buffer usage.
    gelu_inplace(ffn_gate_buf, config->intermediate_size);
    for (int _i = 0; _i < config->intermediate_size; ++_i) {
        ffn_gate_buf[_i] *= ffn_value_buf[_i];
    }

    if (getenv("SAPPHIRE_LOG_TENSORS") && token_pos == 0) {
        float r_a = 0;
        vec_stats(ffn_gate_buf, config->intermediate_size, NULL, NULL, &r_a);
        LOG_DEBUG("Layer %d Activation RMS: %.3f", layer_idx, r_a);
    }

    tensor_gemv_with_ctx(session->gemv_ctx, hidden, layer->down_proj_weight, ffn_gate_buf);

    // 12. Post-FFN RMSNorm (Gemma 3)
    // hidden currently contains DownProj output (FFN Result)
    if (getenv("SAPPHIRE_DEBUG_LAYER_NORMS")) {
        if (layer->norm_ffn_post_weight) {
            int _len = tensor_shape(layer->norm_ffn_post_weight)[0];
            LOG_DEBUG("Layer %d norm_ffn_post_weight ptr=%p len=%d", layer_idx, layer->norm_ffn_post_weight, _len);
        } else {
            LOG_DEBUG("Layer %d norm_ffn_post_weight MISSING (NULL)", layer_idx);
        }
    }

    if (layer->norm_ffn_post_weight) {
        int n_post = tensor_shape(layer->norm_ffn_post_weight)[0];
        const float* norm_post_w = NULL;
        if (n_post == d_model) {
            norm_post_w = get_norm_weights(layer->norm_ffn_post_weight, weight_scratch, d_model);
            /* Debug: log norm weight statistics and scratch pointers to detect
             * whether weights are unexpectedly large or if scratch buffers
             * are aliasing/corrupting data. */
            if (layer_idx == 0 && token_pos == 0) {
                float w_mn = 0.0f, w_mx = 0.0f, w_rms = 0.0f;
                vec_stats(norm_post_w, d_model, &w_mn, &w_mx, &w_rms);
                int w_dtype = layer->norm_ffn_post_weight ? tensor_dtype(layer->norm_ffn_post_weight) : -1;
                LOG_DEBUG("DEBUG: Layer %d Norm FFN Post weights ptr=%p dtype=%d len=%d RMS=%.6f min=%.6f max=%.6f",
                          layer_idx, (void*)layer->norm_ffn_post_weight, w_dtype, n_post, w_rms, w_mn, w_mx);
                LOG_DEBUG("DEBUG: scratch pointers: attn_out=%p norm_buf=%p q_proj=%p ffn_gate_buf=%p ffn_value_buf=%p residual=%p hidden=%p",
                          (void*)attn_out, (void*)norm_buf, (void*)q_proj, (void*)ffn_gate_buf, (void*)ffn_value_buf, (void*)residual, (void*)hidden);
            }
            if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
                int n_print = d_model < 64 ? d_model : 64;
                LOG_DEBUG("DUMP L0 NORM_FFN_POST first %d vals:", n_print);
                for (int i = 0; i < n_print; ++i) {
                    LOG_DEBUG("DUMP L0 NORM_FFN_POST[%d]=%.9g", i, norm_post_w[i]);
                }
            }
            if (token_pos == 0 && layer_idx == 0) {
                LOG_DEBUG("Layer 0 has norm_ffn_post_weight (ptr=%p). Applying Post-FFN Norm.", norm_post_w);
            }
            sapphire_rmsnorm_delta(norm_buf, hidden, norm_post_w, 1e-6f, d_model);

            if (getenv("SAPPHIRE_DEBUG_RMS")) {
                float rms_norm = 0;
                vec_stats(norm_buf, d_model, NULL, NULL, &rms_norm);
                LOG_DEBUG("Layer %d post-ffn norm_buf RMS=%.3f before residual add", layer_idx, rms_norm);
            }

            vec_add(residual, norm_buf, d_model);
        } else {
            LOG_WARN("layer %d norm_ffn_post_weight len=%d expected=%d; using weightless RMSNorm", layer_idx, n_post, d_model);
            // Fallback: compute weightless normalized output
            for (int i = 0; i < d_model; i++) q_proj[i] = 1.0f;
            sapphire_rmsnorm(norm_buf, hidden, q_proj, 1e-6f, d_model);
            vec_add(residual, norm_buf, d_model);
        }
    } else {
        if (token_pos == 0 && layer_idx == 0) {
            LOG_DEBUG("Layer 0 MISSING norm_ffn_post_weight. Adding raw FFN output.");
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
        LOG_DEBUG("DEBUG[EMBED]: token=%d before_scale rms=%.4f", token_id, rms);
    }

    // Gemma 3 requires input embeddings to be scaled by sqrt(d_model)
    // ENABLED: RMS=0.04 (raw) * 25.3 = ~1.0 (correct for transformer inputs)
    if (!getenv("SAPPHIRE_NO_EMBED_SCALE")) {
        float embed_scale = sqrtf((float)config->hidden_size);
        vec_scale(hidden, embed_scale, config->hidden_size);

        if (getenv("SAPPHIRE_LOG_TENSORS")) {
            float mn, mx, rms;
            vec_stats(hidden, config->hidden_size, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG[EMBED]: token=%d after_scale rms=%.4f scale=%.4f", token_id, rms, embed_scale);
        }
    }
}

void sapphire_lm_head(struct inference_session_t* session, float* hidden, float* logits) {
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;
    int norm_weights_are_deltas = 1;

    float* scratch_norm = session->scratch_buffer + 2 * session->padded_d_model;  // reuse norm_buf

    // 1. Final RMSNorm
    // Reuse q_proj area.
    float* tmp_w = session->scratch_buffer + 3 * session->padded_d_model;
    const float* final_w = get_norm_weights(model->norm_final_weight, tmp_w, config->hidden_size);

    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        float mn_b, mx_b, rms_b;
        vec_stats(hidden, config->hidden_size, &mn_b, &mx_b, &rms_b);
        LOG_DEBUG("DEBUG_RMS FinalNorm BEFORE: min=%.6f max=%.6f rms=%.6f", mn_b, mx_b, rms_b);
        if (rms_b < 0.01f || rms_b > 100.0f) LOG_WARN("DEBUG_RMS WARNING FinalNorm BEFORE RMS out-of-range=%.6f", rms_b);
    }

    if (norm_weights_are_deltas)
        sapphire_rmsnorm_delta(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);
    else
        sapphire_rmsnorm(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);

    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        float mn_a, mx_a, rms_a;
        vec_stats(scratch_norm, config->hidden_size, &mn_a, &mx_a, &rms_a);
        LOG_DEBUG("DEBUG_RMS FinalNorm AFTER: min=%.6f max=%.6f rms=%.6f", mn_a, mx_a, rms_a);
        if (rms_a < 0.01f || rms_a > 100.0f) LOG_WARN("DEBUG_RMS WARNING FinalNorm AFTER RMS out-of-range=%.6f", rms_a);
    }

    if (log_get_level() == LOG_LEVEL_DEBUG) {
        float mn, mx, rms;
        vec_stats(scratch_norm, config->hidden_size, &mn, &mx, &rms);
        LOG_DEBUG("Final Norm Out: min=%.3f max=%.3f rms=%.3f", mn, mx, rms);
    }

    // 2. Output Projection (Weight Tying)
    // Perform projection first to maintain numerical stability in the hidden state
    tensor_gemv_with_ctx(session->gemv_ctx, logits, model->embedding_weight, scratch_norm);

    /* Optional BF16->F32 GEMV verification (set SAPPHIRE_BF16_VERIFY="N" where N=checks) */
    const char *bfenv = getenv("SAPPHIRE_BF16_VERIFY");
    if (bfenv && bfenv[0] != '\0') {
        int checks = atoi(bfenv);
        if (checks <= 0) checks = 3;
        int V = config->vocab_size;
        if (checks > V) checks = V;

        const uint16_t *embed_bf16 = (const uint16_t *)tensor_data(model->embedding_weight);
        int d = config->hidden_size;
        float *tmp_row = tmp_w; /* reuse tmp_w buffer for BF16->F32 conversion */

        /* Sample evenly across the vocabulary for broad coverage */
        int stride = V / checks;
        if (stride <= 0) stride = 1;

        double sum_diff = 0.0;
        double sum_abs = 0.0;
        double max_abs = 0.0;
        int max_tid = -1;
        int fail_count = 0;
        const double fail_thresh = 1e-3; /* unexpected large mismatch */

        for (int c = 0; c < checks; c++) {
            int tid = (c * stride) % V;
            bf16_to_f32_vec(tmp_row, embed_bf16 + (size_t)tid * d, d);
            float dot = 0.0f;
            for (int j = 0; j < d; j++) dot += tmp_row[j] * scratch_norm[j];
            float gemv_v = logits[tid];
            float diff = gemv_v - dot;
            double ad = fabs((double)diff);
            sum_diff += diff;
            sum_abs += ad;
            if (ad > max_abs) { max_abs = ad; max_tid = tid; }
            if (ad > fail_thresh) fail_count++;

            /* Log first few individual checks to aid debugging */
            if (c < 10) {
                LOG_DEBUG("BF16_VERIFY sample[%d]=%d gemv=%.6f dot=%.6f diff=%.6f", c, tid, gemv_v, dot, diff);
            }
        }

        double avg_diff = sum_diff / (double)checks;
        double avg_abs = sum_abs / (double)checks;
        LOG_DEBUG("BF16_VERIFY summary: samples=%d stride=%d avg_diff=%.9f avg_abs=%.9f max_abs=%.9f max_tid=%d fails(>|%.1e|)=%d",
                  checks, stride, avg_diff, avg_abs, max_abs, max_tid, fail_thresh, fail_count);
    }

    /* Debug: Log logits stats and top-k for numeric inspection */
    if (log_get_level() == LOG_LEVEL_DEBUG) {
        int V = config->vocab_size;
        float lmn = 0.0f, lmx = 0.0f, lrms = 0.0f;
        vec_stats(logits, V, &lmn, &lmx, &lrms);
        LOG_DEBUG("Logits: min=%.6f max=%.6f rms=%.6f (vocab=%d)", lmn, lmx, lrms, V);

        /* compute top-k (k=10) simple selection */
        const int K = 10;
        float topv[K];
        int topi[K];
        for (int t = 0; t < K; t++) { topv[t] = -INFINITY; topi[t] = -1; }

        for (int i = 0; i < V; i++) {
            float v = logits[i];
            /* find smallest in topv */
            int min_idx = 0;
            float min_val = topv[0];
            for (int t = 1; t < K; t++) {
                if (topv[t] < min_val) { min_val = topv[t]; min_idx = t; }
            }
            if (v > min_val) {
                topv[min_idx] = v;
                topi[min_idx] = i;
            }
        }

        /* sort top-k descending (simple insertion sort for K small) */
        for (int a = 0; a < K; a++) {
            for (int b = a + 1; b < K; b++) {
                if (topv[b] > topv[a]) {
                    float tv = topv[a]; topv[a] = topv[b]; topv[b] = tv;
                    int ti = topi[a]; topi[a] = topi[b]; topi[b] = ti;
                }
            }
        }

        /* log top-k */
        char buf[256];
        int off = 0;
        off += snprintf(buf + off, sizeof(buf) - off, "Top-%d:", K);
        for (int t = 0; t < K; t++) {
            if (topi[t] >= 0) off += snprintf(buf + off, sizeof(buf) - off, " %d(%.4f)", topi[t], topv[t]);
        }
        LOG_DEBUG("%s", buf);
    }

    // Final logit scaling: optionally applied via env var.
    // Use SAPPHIRE_FINAL_LOGIT_SCALE to match external runtimes if required
    // (for example, set to 1.0 / sqrt(d_model) or 1.0 / 25.3 depending on model).
    {
        const char* env = getenv("SAPPHIRE_FINAL_LOGIT_SCALE");
        if (env && env[0] != '\0') {
            float final_scale = strtof(env, NULL);
            if (final_scale != 1.0f) {
                vec_scale(logits, final_scale, config->vocab_size);
                LOG_DEBUG("Applied SAPPHIRE_FINAL_LOGIT_SCALE=%.6f to logits", final_scale);
            } else {
                LOG_DEBUG("SAPPHIRE_FINAL_LOGIT_SCALE set to 1.0 (no-op)");
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
            LOG_DEBUG("Applied final logit softcap (cap=%.6f) to logits", cap);
        }
    }

    // 4. Final Logit Soft-Capping
    // UPDATE: Removed per new Gemma 3 technical report ("replace... with QK-norm").
    // We rely on stable internal activations.
}

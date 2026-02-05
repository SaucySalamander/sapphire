#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/gemma3_270m_config.h"
#include "activations.h"
#include "attention.h"
#include "inference.h"
#include "log.h"
#include "normalization.h"
#include "rope.h"
#include "tensor.h"
#include "utils.h"

// Helper to handle BF16 norm weights on the fly
static const float* get_norm_weights(const tensor_t* weight, float* scratch, int n) {
    if (!weight) return NULL;
    if (tensor_dtype(weight) == DTYPE_BF16) {
        bf16_to_f32_vec(scratch, (const uint16_t*)tensor_data(weight), n);
        return scratch;
    }
    return (const float*)tensor_data(weight);
}

typedef struct {
    int pm, pi, pk, pf;
    float *residual, *norm_buf, *q_proj, *k_proj, *v_proj;
    float *attn_out, *ffn_gate_buf, *ffn_value_buf, *geglu_buf;
    float* weight_scratch;
} layer_buffers_t;

layer_buffers_t init_layer_buffers(struct inference_session_t* session,
                                   gemma3_270m_config_t* config,
                                   int d_model,
                                   int head_dim) {
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

    layer_buffers_t buf = {0};

    buf.pm = session->padded_d_model;
    buf.pi = session->padded_d_inner;
    buf.pk = session->padded_d_kv;
    buf.pf = session->padded_d_ff;

    buf.residual = session->scratch_buffer + buf.pm;
    buf.norm_buf = session->scratch_buffer + 2 * buf.pm;
    buf.q_proj = session->scratch_buffer + 3 * buf.pm;
    buf.k_proj = buf.q_proj + buf.pi;
    buf.v_proj = buf.k_proj + buf.pk;
    buf.attn_out = buf.v_proj + buf.pk;
    buf.ffn_gate_buf = buf.attn_out + buf.pi;
    buf.ffn_value_buf = buf.ffn_gate_buf + buf.pf;
    buf.geglu_buf = buf.ffn_value_buf + buf.pf;

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

    buf.weight_scratch = NULL;
    if ((size_t)max_needed <= scratch_floats) {
        buf.weight_scratch = session->scratch_buffer + (scratch_floats - max_needed);
    } else {
        LOG_WARN("Insufficient scratch for weight_scratch: need=%d have=%zu; falling back to q_proj (may alias)",
                 max_needed, scratch_floats);
        buf.weight_scratch = buf.q_proj; /* best-effort fallback to previous behavior */
    }
    return buf;
}

void compute_attention_stage(layer_buffers_t buf,
                             struct inference_session_t* session,
                             model_layer_weights_t* layer,
                             gemma3_270m_config_t* config,
                             float* hidden,
                             int layer_idx,
                             int token_pos,
                             int d_model,
                             int head_dim,
                             const float* rope_cos,
                             const float* rope_sin) {
    // 1. Residual Connection (Save hidden to residual)
    vec_copy(buf.residual, hidden, d_model);

    /* Capture embedding RMS (original hidden) for comparator-friendly logs.
     * Compute only when debug log level is enabled to avoid extra cost. */
    float __embed_rms = 0.0f;
    if (layer_idx == 0 && token_pos == 0 && log_get_level() == LOG_LEVEL_DEBUG) {
        float __mn_e = 0.0f, __mx_e = 0.0f;
        vec_stats(buf.residual, d_model, &__mn_e, &__mx_e, &__embed_rms);
        LOG_DEBUG("Embedding RMS: %.6f", __embed_rms);
    }

    // DEBUG: log residual RMS at layer entry on first token
    if (layer_idx == 0 && getenv("SAPPHIRE_DEBUG_RMS")) {
        float mn, mx, rms;
        vec_stats(buf.residual, d_model, &mn, &mx, &rms);
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
    const float* norm_attn_data = get_norm_weights(layer->norm_attn_weight, buf.weight_scratch, d_model);

    if (getenv("SAPPHIRE_DEBUG_LOGITS") && layer_idx == 0 && token_pos == 0) {
        fprintf(stderr, "[DEBUG_L0_NORM_ATTN_WEIGHTS] First 10 values:");
        for (int i = 0; i < 10 && i < d_model; i++) {
            fprintf(stderr, " %.6f", norm_attn_data[i]);
        }
        fprintf(stderr, "\n");
    }

    sapphire_rmsnorm_delta(buf.norm_buf, hidden, norm_attn_data, 1e-6f, d_model);

    // 3. Projections (Q, K, V)
    tensor_gemv_with_ctx(session->gemv_ctx, buf.q_proj, layer->q_proj_weight, buf.norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, buf.k_proj, layer->k_proj_weight, buf.norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, buf.v_proj, layer->v_proj_weight, buf.norm_buf);

    /* Debug: print computed projection shapes and buffer paddings to help
     * diagnose head-dimension / GQA mismatches (visible when
     * SAPPHIRE_DEBUG_DUMPS or DEBUG log level enabled). */
    if (getenv("SAPPHIRE_DEBUG_DUMPS") || log_get_level() == LOG_LEVEL_DEBUG) {
        int n_q = config->num_attention_heads * head_dim;
        int n_kv = config->num_key_value_heads * head_dim;
        LOG_DEBUG("DEBUG_SHAPE Layer %d head_dim=%d num_heads=%d num_kv_heads=%d n_q=%d n_kv=%d padded_d_model=%d padded_d_inner=%d padded_d_kv=%d",
                  layer_idx, head_dim, config->num_attention_heads, config->num_key_value_heads,
                  n_q, n_kv, buf.pm, buf.pi, buf.pk);
    }

    // 4. QK-Normalization
    float* q_scale_ptr = NULL;
    float* k_scale_ptr = NULL;

    if (layer->q_norm_weight) {
        int q_norm_len = tensor_shape(layer->q_norm_weight)[0];
        const float* raw = get_norm_weights(layer->q_norm_weight, buf.weight_scratch, q_norm_len);
        int expected_q = config->num_attention_heads * head_dim;
        if (q_norm_len == expected_q) {
            q_scale_ptr = (float*)raw;  // per-head gamma
        } else if (q_norm_len == head_dim) {
            // Broadcast single-head gamma to all Q heads (common for small Gemma configs)
            float head_gamma[head_dim];
            memcpy(head_gamma, raw, head_dim * sizeof(float));
            // reuse ffn_gate_buf as expanded gamma scratch (size pf >= expected_q)
            q_scale_ptr = buf.ffn_gate_buf;
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
        const float* raw = get_norm_weights(layer->k_norm_weight, buf.weight_scratch, k_norm_len);
        int expected_k = config->num_key_value_heads * head_dim;
        if (k_norm_len == expected_k) {
            k_scale_ptr = (float*)raw;  // per-KV-head gamma
        } else if (k_norm_len == head_dim) {
            // Broadcast single-head gamma to all KV heads (GQA)
            float head_gamma[head_dim];
            memcpy(head_gamma, raw, head_dim * sizeof(float));
            // reuse ffn_value_buf as expanded gamma scratch (size pf >= expected_k)
            k_scale_ptr = buf.ffn_value_buf;
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
        apply_qk_norm(buf.q_proj, buf.k_proj, q_scale_ptr, k_scale_ptr, head_dim, config->num_attention_heads, config->num_key_value_heads);
    }
    if (log_get_level() == LOG_LEVEL_DEBUG) {
        float r_q = 0, r_k = 0, r_v = 0;
        int n_q = config->num_attention_heads * head_dim;
        int n_kv = config->num_key_value_heads * head_dim;
        vec_stats(buf.q_proj, n_q, NULL, NULL, &r_q);
        vec_stats(buf.k_proj, n_kv, NULL, NULL, &r_k);
        vec_stats(buf.v_proj, n_kv, NULL, NULL, &r_v);
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
        for (int i = 0; i < qN; ++i) LOG_DEBUG("DUMP L0 PROJ Q[%d]=%.9g", i, buf.q_proj[i]);
        LOG_DEBUG("DUMP L0 PROJ: first %d K values:", kvN);
        for (int i = 0; i < kvN; ++i) LOG_DEBUG("DUMP L0 PROJ K[%d]=%.9g", i, buf.k_proj[i]);
        LOG_DEBUG("DUMP L0 PROJ: first %d V values:", kvN);
        for (int i = 0; i < kvN; ++i) LOG_DEBUG("DUMP L0 PROJ V[%d]=%.9g", i, buf.v_proj[i]);
    }

    // 5. RoPE application
    // Use the explicitly provided RoPE frequencies (toggled in inference.c based on layer type)
    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        for (int h = 0; h < config->num_attention_heads; h++) {
            float mn, mx, rms;
            vec_stats(buf.q_proj + h * head_dim, head_dim, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG_RMS Layer %d Head %d Q before RoPE: min=%.6f max=%.6f rms=%.6f", layer_idx, h, mn, mx, rms);
            if (rms < 0.01f || rms > 100.0f) LOG_WARN("DEBUG_RMS WARNING Layer %d Head %d Q RMS out-of-range=%.6f", layer_idx, h, rms);
        }
        for (int h = 0; h < config->num_key_value_heads; h++) {
            float mn, mx, rms;
            vec_stats(buf.k_proj + h * head_dim, head_dim, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG_RMS Layer %d KVHead %d K before RoPE: min=%.6f max=%.6f rms=%.6f", layer_idx, h, mn, mx, rms);
            if (rms < 0.01f || rms > 100.0f) LOG_WARN("DEBUG_RMS WARNING Layer %d KVHead %d K RMS out-of-range=%.6f", layer_idx, h, rms);
        }
    }

    for (int h = 0; h < config->num_attention_heads; h++) rope_apply_fast(buf.q_proj + h * head_dim, token_pos, head_dim, rope_cos, rope_sin);
    for (int h = 0; h < config->num_key_value_heads; h++) rope_apply_fast(buf.k_proj + h * head_dim, token_pos, head_dim, rope_cos, rope_sin);

    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        for (int h = 0; h < config->num_attention_heads; h++) {
            float mn, mx, rms;
            vec_stats(buf.q_proj + h * head_dim, head_dim, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG_RMS Layer %d Head %d Q after RoPE: min=%.6f max=%.6f rms=%.6f", layer_idx, h, mn, mx, rms);
            if (rms < 0.01f || rms > 100.0f) LOG_WARN("DEBUG_RMS WARNING Layer %d Head %d Q RMS out-of-range=%.6f", layer_idx, h, rms);
        }
        for (int h = 0; h < config->num_key_value_heads; h++) {
            float mn, mx, rms;
            vec_stats(buf.k_proj + h * head_dim, head_dim, &mn, &mx, &rms);
            LOG_DEBUG("DEBUG_RMS Layer %d KVHead %d K after RoPE: min=%.6f max=%.6f rms=%.6f", layer_idx, h, mn, mx, rms);
            if (rms < 0.01f || rms > 100.0f) LOG_WARN("DEBUG_RMS WARNING Layer %d KVHead %d K RMS out-of-range=%.6f", layer_idx, h, rms);
        }
    }

    // 6. KV-Cache Commit
    kv_cache_write_token(session->kv_cache, layer_idx, token_pos, buf.k_proj, buf.v_proj);
    // 7. Attention Forward Pass
    sapphire_attention_forward(session, layer_idx, token_pos, buf.q_proj, buf.attn_out);

    if (log_get_level() == LOG_LEVEL_DEBUG) {
        float r_a = 0;
        vec_stats(buf.attn_out, config->num_attention_heads * head_dim, NULL, NULL, &r_a);
        LOG_DEBUG("Layer %d AttnOut RMS: %.3f", layer_idx, r_a);
    }

    if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
        int N = config->num_attention_heads * head_dim;
        int n_print = N < 16 ? N : 16;
        {
            float _mn, _mx, _rms;
            vec_stats(buf.attn_out, N, &_mn, &_mx, &_rms);
            LOG_DEBUG("DUMP L0 ATTN_OUT: first %d vals (RMS=%.6f):", n_print, _rms);
        }
        for (int i = 0; i < n_print; ++i) LOG_DEBUG("DUMP L0 ATTN_OUT[%d]=%.9g", i, buf.attn_out[i]);
    }

    // 8. Output projection
    tensor_gemv_with_ctx(session->gemv_ctx, hidden, layer->out_proj_weight, buf.attn_out);
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
        const float* norm_post_w = get_norm_weights(layer->norm_attn_post_weight, buf.weight_scratch, d_model);
        if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
            int n_print = d_model < 8 ? d_model : 8;
            LOG_DEBUG("DUMP L0 NORM_ATT_POST first %d vals:", n_print);
            for (int i = 0; i < n_print; ++i) LOG_DEBUG("DUMP L0 NORM_ATT_POST[%d]=%.9g", i, norm_post_w[i]);
        }

        if (getenv("SAPPHIRE_DEBUG_LOGITS") && layer_idx == 0 && token_pos == 0) {
            fprintf(stderr, "[DEBUG_L0_NORM_ATTN_POST_WEIGHTS] First 10 values:");
            for (int i = 0; i < 10 && i < d_model; i++) {
                fprintf(stderr, " %.6f", norm_post_w[i]);
            }
            fprintf(stderr, "\n");
        }

        sapphire_rmsnorm_delta(buf.norm_buf, hidden, norm_post_w, 1e-6f, d_model);

        if (getenv("SAPPHIRE_DEBUG_RMS")) {
            float rms_norm = 0;
            vec_stats(buf.norm_buf, d_model, NULL, NULL, &rms_norm);
            LOG_DEBUG("Layer %d post-attn norm_buf RMS=%.3f before residual add", layer_idx, rms_norm);
        }

        vec_add(buf.residual, buf.norm_buf, d_model);
        LOG_DEBUG("Layer %d applied post-attention norm and added to residual", layer_idx);
    }

    vec_copy(hidden, buf.residual, d_model);
}
void compute_ffn_stage(layer_buffers_t buf,
                       struct inference_session_t* session,
                       model_layer_weights_t* layer,
                       gemma3_270m_config_t* config,
                       float* hidden,
                       int layer_idx,
                       int token_pos,
                       int d_model,
                       int head_dim) {
    // 10. FFN stage (Sequential)
    const float* norm_ffn_data = NULL;
    if (layer->norm_ffn_weight) {
        int n_norm = tensor_shape(layer->norm_ffn_weight)[0];
        if (n_norm == d_model) {
            norm_ffn_data = get_norm_weights(layer->norm_ffn_weight, buf.weight_scratch, d_model);
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

    sapphire_rmsnorm_delta(buf.norm_buf, buf.residual, norm_ffn_data, 1e-6f, d_model);

    /* 11. GEMMA3: GeGLU FFN. Compute gate and value projections, apply GeGLU,
     * then project down. */
    tensor_gemv_with_ctx(session->gemv_ctx, buf.ffn_gate_buf, layer->gate_proj_weight, buf.norm_buf);
    tensor_gemv_with_ctx(session->gemv_ctx, buf.ffn_value_buf, layer->up_proj_weight, buf.norm_buf);

    if (getenv("SAPPHIRE_LOG_TENSORS") && token_pos == 0) {
        float r_g = 0, r_u = 0;
        vec_stats(buf.ffn_gate_buf, config->intermediate_size, NULL, NULL, &r_g);
        vec_stats(buf.ffn_value_buf, config->intermediate_size, NULL, NULL, &r_u);
        LOG_DEBUG("Layer %d FFN Proj RMS: G=%.3f U=%.3f", layer_idx, r_g, r_u);
    }

    // Targeted dump: GEGLU inputs for Layer 0 Token 0
    if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
        int n_print = config->intermediate_size < 16 ? config->intermediate_size : 16;
        int nonfinite_gate = 0, nonfinite_val = 0;
        for (int i = 0; i < config->intermediate_size; ++i) {
            if (!isfinite(buf.ffn_gate_buf[i])) nonfinite_gate++;
            if (!isfinite(buf.ffn_value_buf[i])) nonfinite_val++;
        }
        LOG_DEBUG("DUMP GEGLU L0: non-finite Gate=%d Value=%d size=%d", nonfinite_gate, nonfinite_val, config->intermediate_size);
        for (int i = 0; i < n_print; ++i) {
            LOG_DEBUG("DUMP GEGLU L0 Gate[%d]=%.9g Value[%d]=%.9g", i, buf.ffn_gate_buf[i], i, buf.ffn_value_buf[i]);
        }
    }

    // Combine Gate (apply GELU on gate) and Value via GeGLU into `ffn_gate_buf` output
    // Optimized in-place GeGLU: apply GELU to gate buffer then element-wise multiply
    // by the value buffer to avoid extra memcpy and temporary buffer usage.
    gelu_inplace(buf.ffn_gate_buf, config->intermediate_size);
    for (int _i = 0; _i < config->intermediate_size; ++_i) {
        buf.ffn_gate_buf[_i] *= buf.ffn_value_buf[_i];
    }

    if (getenv("SAPPHIRE_LOG_TENSORS") && token_pos == 0) {
        float r_a = 0;
        vec_stats(buf.ffn_gate_buf, config->intermediate_size, NULL, NULL, &r_a);
        LOG_DEBUG("Layer %d Activation RMS: %.3f", layer_idx, r_a);
    }

    tensor_gemv_with_ctx(session->gemv_ctx, hidden, layer->down_proj_weight, buf.ffn_gate_buf);
}

void finalize_layer_output(layer_buffers_t buf,
                           struct inference_session_t* session,
                           model_layer_weights_t* layer,
                           gemma3_270m_config_t* config,
                           float* hidden,
                           int layer_idx,
                           int token_pos,
                           int d_model,
                           int head_dim) {
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
            norm_post_w = get_norm_weights(layer->norm_ffn_post_weight, buf.weight_scratch, d_model);
            /* Debug: log norm weight statistics and scratch pointers to detect
             * whether weights are unexpectedly large or if scratch buffers
             * are aliasing/corrupting data. */
            if (layer_idx == 0 && token_pos == 0) {
                float w_mn = 0.0f, w_mx = 0.0f, w_rms = 0.0f;
                vec_stats(norm_post_w, d_model, &w_mn, &w_mx, &w_rms);
                int w_dtype = layer->norm_ffn_post_weight ? tensor_dtype(layer->norm_ffn_post_weight) : -1;
                LOG_DEBUG("DEBUG: Layer %d Norm FFN Post weights ptr=%p dtype=%d len=%d RMS=%.6f min=%.6f max=%.6f",
                          layer_idx, (void*)layer->norm_ffn_post_weight, w_dtype, n_post, w_rms, w_mn, w_mx);
                LOG_DEBUG("DEBUG: scratch pointers: attn_out=%p norm_buf=%p q_proj=%p ffn_gate_buf=%p ffn_value_buf=%p residual=%p",
                          buf.attn_out, buf.norm_buf, buf.q_proj, buf.ffn_gate_buf, buf.ffn_value_buf, buf.residual);
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

            if (getenv("SAPPHIRE_DEBUG_LOGITS") && layer_idx == 17 && token_pos == 0) {
                fprintf(stderr, "[DEBUG_L17_FFN_PRE_NORM] hidden RMS before ffn post-norm:");
                float mn, mx, rms;
                vec_stats(hidden, d_model, &mn, &mx, &rms);
                fprintf(stderr, " min=%.6f max=%.6f rms=%.6f\n", mn, mx, rms);
                fprintf(stderr, "[DEBUG_L17_NORM_FFN_POST_WEIGHTS] First 10 values:");
                for (int i = 0; i < 10 && i < d_model; i++) {
                    fprintf(stderr, " %.6f", norm_post_w[i]);
                }
                fprintf(stderr, "\n");
            }

            sapphire_rmsnorm_delta(buf.norm_buf, hidden, norm_post_w, 1e-6f, d_model);

            if (getenv("SAPPHIRE_DEBUG_LOGITS") && layer_idx == 17 && token_pos == 0) {
                fprintf(stderr, "[DEBUG_L17_FFN_POST_NORM] norm_buf RMS after ffn post-norm:");
                float mn, mx, rms;
                vec_stats(buf.norm_buf, d_model, &mn, &mx, &rms);
                fprintf(stderr, " min=%.6f max=%.6f rms=%.6f\n", mn, mx, rms);
            }

            if (getenv("SAPPHIRE_DEBUG_RMS")) {
                float rms_norm = 0;
                vec_stats(buf.norm_buf, d_model, NULL, NULL, &rms_norm);
                LOG_DEBUG("Layer %d post-ffn norm_buf RMS=%.3f before residual add", layer_idx, rms_norm);
            }

            vec_add(buf.residual, buf.norm_buf, d_model);
        } else {
            LOG_WARN("layer %d norm_ffn_post_weight len=%d expected=%d; using weightless RMSNorm", layer_idx, n_post, d_model);
            // Fallback: compute weightless normalized output
            for (int i = 0; i < d_model; i++) buf.q_proj[i] = 1.0f;
            sapphire_rmsnorm(buf.norm_buf, hidden, buf.q_proj, 1e-6f, d_model);
            vec_add(buf.residual, buf.norm_buf, d_model);
        }
    } else {
        if (token_pos == 0 && layer_idx == 0) {
            LOG_DEBUG("Layer 0 MISSING norm_ffn_post_weight. Adding raw FFN output.");
        }
        vec_add(buf.residual, hidden, d_model);
    }

    // Copy computed residual back to hidden (Layer Output)
    vec_copy(hidden, buf.residual, d_model);
    /* Comparator-friendly per-layer output RMS (env-var controlled).
     * Print after layer output is finalized. */
    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        float __mn_o = 0.0f, __mx_o = 0.0f, __rms_o = 0.0f;
        vec_stats(hidden, d_model, &__mn_o, &__mx_o, &__rms_o);
        LOG_DEBUG("Layer %d Output RMS: min=%.6f max=%.6f rms=%.6f", layer_idx, __mn_o, __mx_o, __rms_o);
    }
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

    layer_buffers_t buf = init_layer_buffers(session, config, d_model, head_dim);

    compute_attention_stage(buf, session, layer, config, hidden, layer_idx, token_pos, d_model, head_dim, rope_cos, rope_sin);
    compute_ffn_stage(buf, session, layer, config, hidden, layer_idx, token_pos, d_model, head_dim);
    finalize_layer_output(buf, session, layer, config, hidden, layer_idx, token_pos, d_model, head_dim);

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

    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
        float weight_rms, weight_min, weight_max;
        vec_stats(final_w, config->hidden_size, &weight_min, &weight_max, &weight_rms);
        fprintf(stderr, "[DEBUG_FINAL_NORM_WEIGHTS_STATS] min=%.6f max=%.6f rms=%.6f\n", weight_min, weight_max, weight_rms);

        // Compute (1+weight) stats
        float one_plus_w_vals[640];
        for (int i = 0; i < config->hidden_size; i++) {
            one_plus_w_vals[i] = 1.0f + final_w[i];
        }
        float opw_rms, opw_min, opw_max;
        vec_stats(one_plus_w_vals, config->hidden_size, &opw_min, &opw_max, &opw_rms);
        fprintf(stderr, "[DEBUG_1PLUS_WEIGHTS_STATS] min=%.6f max=%.6f rms=%.6f\n", opw_min, opw_max, opw_rms);
    }

    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        float mn_b, mx_b, rms_b;
        vec_stats(hidden, config->hidden_size, &mn_b, &mx_b, &rms_b);
        LOG_DEBUG("DEBUG_RMS FinalNorm BEFORE: min=%.6f max=%.6f rms=%.6f", mn_b, mx_b, rms_b);
        if (rms_b < 0.01f || rms_b > 100.0f) LOG_WARN("DEBUG_RMS WARNING FinalNorm BEFORE RMS out-of-range=%.6f", rms_b);
    }

    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
        float mn_b, mx_b, rms_b;
        vec_stats(hidden, config->hidden_size, &mn_b, &mx_b, &rms_b);
        fprintf(stderr, "[DEBUG_HIDDEN_BEFORE_FINAL_NORM] min=%.6f max=%.6f rms=%.6f\n", mn_b, mx_b, rms_b);
        fprintf(stderr, "[DEBUG_HIDDEN_BEFORE_FINAL_NORM] first 5: %.6f %.6f %.6f %.6f %.6f\n",
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);

        // Compute what RMSNorm should output
        float sum_sq = 0.0f;
        for (int i = 0; i < config->hidden_size; i++) {
            sum_sq += hidden[i] * hidden[i];
        }
        float rms_actual = sqrtf(sum_sq / (float)config->hidden_size + 1e-6f);
        fprintf(stderr, "[DEBUG_RMS_CALC] sum_sq=%.2e, computed_rms=%.6f (should match rms_b=%.6f)\n", sum_sq, rms_actual, rms_b);
    }

    if (norm_weights_are_deltas) {
        if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
            fprintf(stderr, "[DEBUG_FINAL_NORM_CALL] Calling sapphire_rmsnorm_delta\n");
            fprintf(stderr, "[DEBUG_FINAL_NORM_CALL] hidden input first 5: %.6f %.6f %.6f %.6f %.6f\n",
                    hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
            fprintf(stderr, "[DEBUG_FINAL_NORM_WEIGHTS] weight first 5: %.6f %.6f %.6f %.6f %.6f\n",
                    final_w[0], final_w[1], final_w[2], final_w[3], final_w[4]);

            // Manually compute what first element should be after norm
            float sum_sq = 0.0f;
            for (int i = 0; i < config->hidden_size; i++) {
                sum_sq += hidden[i] * hidden[i];
            }
            float rms = sqrtf(sum_sq / (float)config->hidden_size + 1e-6f);
            float expected_0 = (hidden[0] / rms) * (1.0f + final_w[0]);
            fprintf(stderr, "[DEBUG_EXPECTED_OUTPUT] hidden[0]=%.6f, rms=%.6f, weight[0]=%.6f, expected_output[0]=%.6f (%.6f / %.6f) * (1 + %.6f)\n",
                    hidden[0], rms, final_w[0], expected_0, hidden[0], rms, final_w[0]);
        }

        sapphire_rmsnorm_delta(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);

        if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
            fprintf(stderr, "[DEBUG_FINAL_NORM_CALL] scratch_norm output first 5: %.6f %.6f %.6f %.6f %.6f\n",
                    scratch_norm[0], scratch_norm[1], scratch_norm[2], scratch_norm[3], scratch_norm[4]);
        }
    } else
        sapphire_rmsnorm(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);

    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        float mn_a, mx_a, rms_a;
        vec_stats(scratch_norm, config->hidden_size, &mn_a, &mx_a, &rms_a);
        LOG_DEBUG("DEBUG_RMS FinalNorm AFTER: min=%.6f max=%.6f rms=%.6f", mn_a, mx_a, rms_a);
        if (rms_a < 0.01f || rms_a > 100.0f) LOG_WARN("DEBUG_RMS WARNING FinalNorm AFTER RMS out-of-range=%.6f", rms_a);
    }

    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
        float mn_a, mx_a, rms_a;
        vec_stats(scratch_norm, config->hidden_size, &mn_a, &mx_a, &rms_a);
        fprintf(stderr, "[DEBUG_HIDDEN_AFTER_FINAL_NORM] min=%.6f max=%.6f rms=%.6f\n", mn_a, mx_a, rms_a);
    }

    if (log_get_level() == LOG_LEVEL_DEBUG) {
        float mn, mx, rms;
        vec_stats(scratch_norm, config->hidden_size, &mn, &mx, &rms);
        LOG_DEBUG("Final Norm Out: min=%.3f max=%.3f rms=%.3f", mn, mx, rms);
    }

    // Debug: dump first few values of scratch_norm before GEMV
    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
        fprintf(stderr, "[DEBUG_SCRATCH_NORM] First 10 values before GEMV: ");
        for (int i = 0; i < 10 && i < config->hidden_size; i++) {
            fprintf(stderr, "%.6f ", scratch_norm[i]);
        }
        fprintf(stderr, "\n");
    }

    // 2. Output Projection (Weight Tying)
    // Perform projection first to maintain numerical stability in the hidden state
    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
        const int* emb_shape = tensor_shape(model->embedding_weight);
        if (emb_shape) {
            fprintf(stderr, "[DEBUG_EMBEDDING_WEIGHT] shape=[%d, %d] dtype=%d\n",
                    emb_shape[0], emb_shape[1], (int)tensor_dtype(model->embedding_weight));
        }
        fprintf(stderr, "[DEBUG_PRE_GEMV] x (scratch_norm) first 5 values: %.6f %.6f %.6f %.6f %.6f\n",
                scratch_norm[0], scratch_norm[1], scratch_norm[2], scratch_norm[3], scratch_norm[4]);
    }

    tensor_gemv_with_ctx(session->gemv_ctx, logits, model->embedding_weight, scratch_norm);

    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
        fprintf(stderr, "[DEBUG_POST_GEMV] logits first 5 values: %.6f %.6f %.6f %.6f %.6f\n",
                logits[0], logits[1], logits[2], logits[3], logits[4]);
    }

    // DEBUG: Show logits statistics
    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
// Inline BF16 converter
#define LOCAL_BF16_TO_F32(bf16_val) ({                \
    uint32_t f32_bits = ((uint32_t)(bf16_val)) << 16; \
    float f;                                          \
    memcpy(&f, &f32_bits, sizeof(float));             \
    f;                                                \
})

        float min_l, max_l, rms_l;
        vec_stats(logits, config->vocab_size, &min_l, &max_l, &rms_l);

        // Find top 5 logits
        float top_vals[5] = {-1e9, -1e9, -1e9, -1e9, -1e9};
        int top_ids[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < config->vocab_size; i++) {
            for (int j = 0; j < 5; j++) {
                if (logits[i] > top_vals[j]) {
                    for (int k = 4; k > j; k--) {
                        top_vals[k] = top_vals[k - 1];
                        top_ids[k] = top_ids[k - 1];
                    }
                    top_vals[j] = logits[i];
                    top_ids[j] = i;
                    break;
                }
            }
        }

        fprintf(stderr, "[FINAL_LOGITS] min=%.6f max=%.6f rms=%.6f\n", min_l, max_l, rms_l);
        fprintf(stderr, "[FINAL_LOGITS_TOP5]");
        for (int i = 0; i < 5; i++) {
            fprintf(stderr, " %d:%.6f", top_ids[i], top_vals[i]);
        }
        fprintf(stderr, "\n");

        // Debug: Manually compute logit for token 106 and token 818
        if (config->vocab_size > 818) {
            const uint16_t* embed_bf16 = (const uint16_t*)tensor_data(model->embedding_weight);
            const int* embed_shape = tensor_shape(model->embedding_weight);
            int vocab_size = embed_shape[0];
            int hidden_size = embed_shape[1];

            float manual_logit_106 = 0.0f;
            float manual_logit_818 = 0.0f;

            // Token 106
            if (106 < vocab_size) {
                for (int j = 0; j < hidden_size && j < config->hidden_size; j++) {
                    uint16_t w_bf16 = embed_bf16[106 * hidden_size + j];
                    float w_f32 = LOCAL_BF16_TO_F32(w_bf16);
                    manual_logit_106 += w_f32 * scratch_norm[j];
                }
            }

            // Token 818
            if (818 < vocab_size) {
                for (int j = 0; j < hidden_size && j < config->hidden_size; j++) {
                    uint16_t w_bf16 = embed_bf16[818 * hidden_size + j];
                    float w_f32 = LOCAL_BF16_TO_F32(w_bf16);
                    manual_logit_818 += w_f32 * scratch_norm[j];
                }
            }

            fprintf(stderr, "[DEBUG_MANUAL_LOGITS] token_106=%.6f(computed) vs %.6f(actual), token_818=%.6f(computed) vs %.6f(actual)\n",
                    manual_logit_106, logits[106], manual_logit_818, logits[818]);

            // Also check embedding weight values for these tokens
            if (106 < vocab_size && 818 < vocab_size) {
                fprintf(stderr, "[DEBUG_EMBED_WEIGHTS] token_106[0:5]:");
                for (int j = 0; j < 5 && j < hidden_size; j++) {
                    fprintf(stderr, " %.6f", LOCAL_BF16_TO_F32(embed_bf16[106 * hidden_size + j]));
                }
                fprintf(stderr, "\n");
                fprintf(stderr, "[DEBUG_EMBED_WEIGHTS] token_818[0:5]:");
                for (int j = 0; j < 5 && j < hidden_size; j++) {
                    fprintf(stderr, " %.6f", LOCAL_BF16_TO_F32(embed_bf16[818 * hidden_size + j]));
                }
                fprintf(stderr, "\n");
            }
        }

#undef LOCAL_BF16_TO_F32
    }

    /* Optional BF16->F32 GEMV verification (set SAPPHIRE_BF16_VERIFY="N" where N=checks) */
    const char* bfenv = getenv("SAPPHIRE_BF16_VERIFY");
    if (bfenv && bfenv[0] != '\0') {
        int checks = atoi(bfenv);
        if (checks <= 0) checks = 3;
        int V = config->vocab_size;
        if (checks > V) checks = V;

        const uint16_t* embed_bf16 = (const uint16_t*)tensor_data(model->embedding_weight);
        int d = config->hidden_size;
        float* tmp_row = tmp_w; /* reuse tmp_w buffer for BF16->F32 conversion */

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
            if (ad > max_abs) {
                max_abs = ad;
                max_tid = tid;
            }
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
        for (int t = 0; t < K; t++) {
            topv[t] = -INFINITY;
            topi[t] = -1;
        }

        for (int i = 0; i < V; i++) {
            float v = logits[i];
            /* find smallest in topv */
            int min_idx = 0;
            float min_val = topv[0];
            for (int t = 1; t < K; t++) {
                if (topv[t] < min_val) {
                    min_val = topv[t];
                    min_idx = t;
                }
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
                    float tv = topv[a];
                    topv[a] = topv[b];
                    topv[b] = tv;
                    int ti = topi[a];
                    topi[a] = topi[b];
                    topi[b] = ti;
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

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

void compute_ffn_stage(layer_buffers_t buf,
                       transformer_layer_ctx_t* ctx,
                       float* hidden) {
    // 10. FFN stage (Sequential)
    const float* norm_ffn_data = NULL;
    if (ctx->layer->norm_ffn_weight) {
        int n_norm = tensor_shape(ctx->layer->norm_ffn_weight)[0];
        if (n_norm == ctx->d_model) {
            norm_ffn_data = get_norm_weights(ctx->layer->norm_ffn_weight, buf.weight_scratch, ctx->d_model);
        } else {
            LOG_WARN("layer %d norm_ffn_weight len=%d expected=%d; using weightless RMSNorm", ctx->layer_idx, n_norm, ctx->d_model);
        }
    }

    if (getenv("SAPPHIRE_DEBUG_LAYER_NORMS")) {
        if (ctx->layer->norm_ffn_weight) {
            int _len = tensor_shape(ctx->layer->norm_ffn_weight)[0];
            LOG_DEBUG("Layer %d norm_ffn_weight ptr=%p len=%d", ctx->layer_idx, ctx->layer->norm_ffn_weight, _len);
        } else {
            LOG_DEBUG("Layer %d norm_ffn_weight MISSING (NULL)", ctx->layer_idx);
        }
    }

    rmsnorm_delta(buf.norm_buf, buf.residual, norm_ffn_data, 1e-6f, ctx->d_model);

    /* 11. GEMMA3: GeGLU FFN. Compute gate and value projections, apply GeGLU,
     * then project down. */
    tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.ffn_gate_buf, ctx->layer->gate_proj_weight, buf.norm_buf);
    tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.ffn_value_buf, ctx->layer->up_proj_weight, buf.norm_buf);

    if (getenv("SAPPHIRE_LOG_TENSORS") && ctx->token_pos == 0) {
        float r_g = 0, r_u = 0;
        vec_stats(buf.ffn_gate_buf, ctx->config->intermediate_size, NULL, NULL, &r_g);
        vec_stats(buf.ffn_value_buf, ctx->config->intermediate_size, NULL, NULL, &r_u);
        LOG_DEBUG("Layer %d FFN Proj RMS: G=%.3f U=%.3f", ctx->layer_idx, r_g, r_u);
    }

    // Targeted dump: GEGLU inputs for Layer 0 Token 0
    if (ctx->layer_idx == 0 && ctx->token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
        int n_print = ctx->config->intermediate_size < 16 ? ctx->config->intermediate_size : 16;
        int nonfinite_gate = 0, nonfinite_val = 0;
        for (int i = 0; i < ctx->config->intermediate_size; ++i) {
            if (!isfinite(buf.ffn_gate_buf[i])) nonfinite_gate++;
            if (!isfinite(buf.ffn_value_buf[i])) nonfinite_val++;
        }
        LOG_DEBUG("DUMP GEGLU L0: non-finite Gate=%d Value=%d size=%d", nonfinite_gate, nonfinite_val, ctx->config->intermediate_size);
        for (int i = 0; i < n_print; ++i) {
            LOG_DEBUG("DUMP GEGLU L0 Gate[%d]=%.9g Value[%d]=%.9g", i, buf.ffn_gate_buf[i], i, buf.ffn_value_buf[i]);
        }
    }

    // Combine Gate (apply GELU on gate) and Value via GeGLU into `ffn_gate_buf` output
    // Optimized in-place GeGLU: apply GELU to gate buffer then element-wise multiply
    // by the value buffer to avoid extra memcpy and temporary buffer usage.
    gelu_inplace(buf.ffn_gate_buf, ctx->config->intermediate_size);
    for (int _i = 0; _i < ctx->config->intermediate_size; ++_i) {
        buf.ffn_gate_buf[_i] *= buf.ffn_value_buf[_i];
    }

    if (getenv("SAPPHIRE_LOG_TENSORS") && ctx->token_pos == 0) {
        float r_a = 0;
        vec_stats(buf.ffn_gate_buf, ctx->config->intermediate_size, NULL, NULL, &r_a);
        LOG_DEBUG("Layer %d Activation RMS: %.3f", ctx->layer_idx, r_a);
    }

    tensor_gemv_with_ctx(ctx->session->gemv_ctx, hidden, ctx->layer->down_proj_weight, buf.ffn_gate_buf);
}

static void debug_log_ffn_norm(const transformer_layer_ctx_t* ctx, const layer_buffers_t* buf, const float* norm_post_w, const float* hidden) {
    if (ctx->layer_idx == 0 && ctx->token_pos == 0) {
        float mn, mx, rms;
        vec_stats(norm_post_w, ctx->d_model, &mn, &mx, &rms);
        LOG_DEBUG("DEBUG: Layer %d Norm FFN Post weights ptr=%p len=%d RMS=%.6f",
                  ctx->layer_idx, (void*)ctx->layer->norm_ffn_post_weight, ctx->d_model, rms);
        LOG_DEBUG("DEBUG: scratch pointers: attn_out=%p norm_buf=%p residual=%p",
                  buf->attn_out, buf->norm_buf, buf->residual);
    }
    if (getenv("SAPPHIRE_DEBUG_LOGITS") && ctx->layer_idx == 17 && ctx->token_pos == 0) {
        float mn, mx, rms;
        vec_stats(hidden, ctx->d_model, &mn, &mx, &rms);
        fprintf(stderr, "[DEBUG_L17_FFN_PRE_NORM] hidden RMS before ffn post-norm: min=%.6f max=%.6f rms=%.6f\n", mn, mx, rms);
    }
}

void finalize_layer_output(layer_buffers_t buf,
                           transformer_layer_ctx_t* ctx,
                           float* hidden) {
    if (getenv("SAPPHIRE_DEBUG_LAYER_NORMS") && ctx->layer->norm_ffn_post_weight) {
        int _len = tensor_shape(ctx->layer->norm_ffn_post_weight)[0];
        LOG_DEBUG("Layer %d norm_ffn_post_weight ptr=%p len=%d", ctx->layer_idx, ctx->layer->norm_ffn_post_weight, _len);
    }

    if (ctx->layer->norm_ffn_post_weight) {
        int n_post = tensor_shape(ctx->layer->norm_ffn_post_weight)[0];
        if (n_post == ctx->d_model) {
            const float* norm_post_w = get_norm_weights(ctx->layer->norm_ffn_post_weight, buf.weight_scratch, ctx->d_model);
            debug_log_ffn_norm(ctx, &buf, norm_post_w, hidden);
            rmsnorm_delta(buf.norm_buf, hidden, norm_post_w, 1e-6f, ctx->d_model);

            if (getenv("SAPPHIRE_DEBUG_LOGITS") && ctx->layer_idx == 17 && ctx->token_pos == 0) {
                float mn, mx, rms;
                vec_stats(buf.norm_buf, ctx->d_model, &mn, &mx, &rms);
                fprintf(stderr, "[DEBUG_L17_FFN_POST_NORM] norm_buf RMS after ffn post-norm: min=%.6f max=%.6f rms=%.6f\n", mn, mx, rms);
            }
            vec_add(buf.residual, buf.norm_buf, ctx->d_model);
        } else {
            LOG_WARN("using weightless RMSNorm for layer %d", ctx->layer_idx);
            for (int i = 0; i < ctx->d_model; i++) buf.q_proj[i] = 1.0f;
            rmsnorm(buf.norm_buf, hidden, buf.q_proj, 1e-6f, ctx->d_model);
            vec_add(buf.residual, buf.norm_buf, ctx->d_model);
        }
    } else {
        vec_add(buf.residual, hidden, ctx->d_model);
    }
    vec_copy(hidden, buf.residual, ctx->d_model);

    /* Comparator-friendly per-layer output RMS (env-var controlled).
     * Print after layer output is finalized. */
    if (getenv("SAPPHIRE_DEBUG_RMS")) {
        float __mn_o = 0.0f, __mx_o = 0.0f, __rms_o = 0.0f;
        vec_stats(hidden, ctx->d_model, &__mn_o, &__mx_o, &__rms_o);
        LOG_DEBUG("Layer %d Output RMS: min=%.6f max=%.6f rms=%.6f", ctx->layer_idx, __mn_o, __mx_o, __rms_o);
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

    transformer_layer_ctx_t ctx = {
        .session = session,
        .layer = layer,
        .config = config,
        .layer_idx = layer_idx,
        .token_pos = token_pos,
        .d_model = d_model,
        .head_dim = head_dim
    };

    compute_attention_stage(buf, &ctx, hidden, rope_cos, rope_sin);
    compute_ffn_stage(buf, &ctx, hidden);
    finalize_layer_output(buf, &ctx, hidden);

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

/**
 * @brief Language model head: final norm and logit projection.
 * 
 * Applies RMSNorm (delta variant) to the final hidden state using the model's
 * final normalization weights, then projects to vocabulary logits via matrix-vector
 * multiplication with the embedding weight matrix.
 * 
 * @param session Inference session containing model and scratch buffers.
 * @param hidden Final layer output [d_model], will be normalized in-place via scratch.
 * @param logits Output logits buffer [vocab_size], populated with unnormalized scores.
 */
void lm_head(struct inference_session_t* session, float* hidden, float* logits) {
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;
    int norm_weights_are_deltas = 1;
    float* scratch_norm = session->scratch_buffer + 2 * session->padded_d_model;
    float* tmp_w = session->scratch_buffer + 3 * session->padded_d_model;
    const float* final_w = get_norm_weights(model->norm_final_weight, tmp_w, config->hidden_size);

    if (norm_weights_are_deltas) {
        rmsnorm_delta(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);
    } else {
        rmsnorm(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);
    }
    tensor_gemv_with_ctx(session->gemv_ctx, logits, model->embedding_weight, scratch_norm);
}

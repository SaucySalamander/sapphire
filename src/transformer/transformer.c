#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/gemma3_270m_config.h"
#include "attention.h"
#include "inference.h"
#include "kernels.h"
#include "log.h"
#include "rope.h"
#include "tensor.h"
#include "utils.h"



layer_buffers_t init_layer_buffers(const struct inference_session_t* session,
                                   const gemma3_270m_config_t* config,
                                   int d_model,
                                   int head_dim,
                                   int batch_size) {
    // Buffers from scratch space based on inference.c layout:
    // Layout is (Batch x Dim) for each buffer type.
    // 0: hidden (passed in, at session->scratch_buffer)
    // 1: residual
    // 2: norm_buf
    // ...
    int max_batch = 32;

    layer_buffers_t buf = {0};

    buf.pm = session->padded_d_model;
    buf.pi = session->padded_d_inner;
    buf.pk = session->padded_d_kv;
    buf.pf = session->padded_d_ff;
    buf.batch_size = batch_size;

    size_t batch_m = (size_t)max_batch * buf.pm;
    size_t batch_i = (size_t)max_batch * buf.pi;
    size_t batch_k = (size_t)max_batch * buf.pk;
    size_t batch_f = (size_t)max_batch * buf.pf;

    buf.residual = session->scratch_buffer + batch_m;
    buf.norm_buf = session->scratch_buffer + 2 * batch_m;
    buf.q_proj = session->scratch_buffer + 3 * batch_m;
    buf.k_proj = buf.q_proj + batch_i;
    buf.v_proj = buf.k_proj + batch_k;
    buf.attn_out = buf.v_proj + batch_k;
    buf.ffn_gate_buf = buf.attn_out + batch_i;
    buf.ffn_value_buf = buf.ffn_gate_buf + batch_f;
    buf.geglu_buf = buf.ffn_value_buf + batch_f;

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
        LOG_WARN("Insufficient scratch for weight_scratch: need=%d have=%zu; falling back to q_proj",
                 max_needed, scratch_floats);
        buf.weight_scratch = buf.q_proj;
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

    for (int b = 0; b < ctx->batch_size; b++) {
        rmsnorm_delta(buf.norm_buf + b * buf.pm, buf.residual + b * buf.pm, norm_ffn_data, 1e-6f, ctx->d_model);
    }

    /* 11. GEMMA3: GeGLU FFN. Compute gate and value projections, apply GeGLU,
     * then project down. */
    if (ctx->batch_size == 1) {
        tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.ffn_gate_buf, ctx->layer->gate_proj_weight, buf.norm_buf);
        tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.ffn_value_buf, ctx->layer->up_proj_weight, buf.norm_buf);
    } else {
        tensor_gemm_with_ctx(ctx->session->gemv_ctx, buf.ffn_gate_buf, ctx->layer->gate_proj_weight, buf.norm_buf, ctx->batch_size, buf.pf);
        tensor_gemm_with_ctx(ctx->session->gemv_ctx, buf.ffn_value_buf, ctx->layer->up_proj_weight, buf.norm_buf, ctx->batch_size, buf.pf);
    }

    if (getenv("SAPPHIRE_LOG_TENSORS") && ctx->token_pos == 0) {
        float r_g = 0, r_u = 0;
        vec_stats(buf.ffn_gate_buf, ctx->config->intermediate_size, NULL, NULL, &r_g);
        vec_stats(buf.ffn_value_buf, ctx->config->intermediate_size, NULL, NULL, &r_u);
        LOG_DEBUG("Layer %d FFN Proj RMS: G=%.3f U=%.3f", ctx->layer_idx, r_g, r_u);
    }

    // Combine Gate (apply GELU on gate) and Value via GeGLU into `ffn_gate_buf` output
    for (int b = 0; b < ctx->batch_size; b++) {
        float* g = buf.ffn_gate_buf + b * buf.pf;
        const float* u = buf.ffn_value_buf + b * buf.pf;
        gelu_inplace(g, ctx->config->intermediate_size);
        for (int _i = 0; _i < ctx->config->intermediate_size; ++_i) {
            g[_i] *= u[_i];
        }
    }

    if (getenv("SAPPHIRE_LOG_TENSORS") && ctx->token_pos == 0) {
        float r_a = 0;
        vec_stats(buf.ffn_gate_buf, ctx->config->intermediate_size, NULL, NULL, &r_a);
        LOG_DEBUG("Layer %d Activation RMS: %.3f", ctx->layer_idx, r_a);
    }

    if (ctx->batch_size == 1) {
        tensor_gemv_with_ctx(ctx->session->gemv_ctx, hidden, ctx->layer->down_proj_weight, buf.ffn_gate_buf);
    } else {
        tensor_gemm_with_ctx(ctx->session->gemv_ctx, hidden, ctx->layer->down_proj_weight, buf.ffn_gate_buf, ctx->batch_size, ctx->d_model);
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
            
            for (int b = 0; b < ctx->batch_size; b++) {
                float* h_b = hidden + b * ctx->d_model;
                float* r_b = buf.residual + b * buf.pm;
                float* n_b = buf.norm_buf + b * buf.pm;
                
                rmsnorm_delta(n_b, h_b, norm_post_w, 1e-6f, ctx->d_model);
                vec_add(r_b, n_b, ctx->d_model);
                vec_copy(h_b, r_b, ctx->d_model);
            }
        } else {
            LOG_WARN("using weightless RMSNorm for layer %d", ctx->layer_idx);
            float* ones = (float*)malloc(ctx->d_model * sizeof(float));
            if (!ones) {
                LOG_ERROR("Failed to allocate 'ones' buffer in finalize_layer_output");
                return;
            }
            for (int i = 0; i < ctx->d_model; i++) ones[i] = 1.0f;
            
            for (int b = 0; b < ctx->batch_size; b++) {
                float* h_b = hidden + b * ctx->d_model;
                float* r_b = buf.residual + b * buf.pm;
                float* n_b = buf.norm_buf + b * buf.pm;
                
                rmsnorm(n_b, h_b, ones, 1e-6f, ctx->d_model);
                vec_add(r_b, n_b, ctx->d_model);
                vec_copy(h_b, r_b, ctx->d_model);
            }
            free(ones);
        }
    } else {
        for (int b = 0; b < ctx->batch_size; b++) {
            float* h_b = hidden + b * ctx->d_model;
            float* r_b = buf.residual + b * buf.pm;
            vec_add(r_b, h_b, ctx->d_model);
            vec_copy(h_b, r_b, ctx->d_model);
        }
    }

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
                               transformer_rope_t rope) {
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;
    model_layer_weights_t* layer = &model->layers[layer_idx];
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;

    int d_model = config->hidden_size;
    int head_dim = config->head_dim > 0 ? config->head_dim : (d_model / config->num_attention_heads);

    layer_buffers_t buf = init_layer_buffers(session, config, d_model, head_dim, 1);

    transformer_layer_ctx_t ctx = {
        .session = session,
        .layer = layer,
        .config = config,
        .layer_idx = layer_idx,
        .token_pos = token_pos,
        .batch_size = 1,
        .d_model = d_model,
        .head_dim = head_dim
    };

    compute_attention_stage(buf, &ctx, hidden, rope.cos, rope.sin);
    compute_ffn_stage(buf, &ctx, hidden);
    finalize_layer_output(buf, &ctx, hidden);

    return 0;
}

int sapphire_transformer_layer_batch(struct inference_session_t* session, int layer_idx, int start_pos, int batch_size, float* hidden,
                                     transformer_rope_t rope) {
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;
    model_layer_weights_t* layer = &model->layers[layer_idx];
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;

    int d_model = config->hidden_size;
    int head_dim = config->head_dim > 0 ? config->head_dim : (d_model / config->num_attention_heads);

    layer_buffers_t buf = init_layer_buffers(session, config, d_model, head_dim, batch_size);

    transformer_layer_ctx_t ctx = {
        .session = session,
        .layer = layer,
        .config = config,
        .layer_idx = layer_idx,
        .token_pos = start_pos,
        .batch_size = batch_size,
        .d_model = d_model,
        .head_dim = head_dim
    };

    compute_attention_stage(buf, &ctx, hidden, rope.cos, rope.sin);
    compute_ffn_stage(buf, &ctx, hidden);
    finalize_layer_output(buf, &ctx, hidden);

    return 0;
}

void sapphire_embed_lookup(struct inference_session_t* session, int token_id, float* hidden) {
    const llm_model_t* model = (const llm_model_t*)session->model_spec->llm_model;
    const gemma3_270m_config_t* config = (const gemma3_270m_config_t*)session->model_spec->variant_config;

    const uint16_t* embed_table_bf16 = (const uint16_t*)tensor_data(model->embedding_weight);
    bf16_to_f32_vec(hidden, embed_table_bf16 + (token_id * config->hidden_size), config->hidden_size);

    if (getenv("SAPPHIRE_LOG_TENSORS")) {
        float mn, mx, rms;
        vec_stats(hidden, config->hidden_size, &mn, &mx, &rms);
        LOG_DEBUG("DEBUG[EMBED]: token=%d before_scale rms=%.4f", token_id, rms);
    }

    // Gemma 3 requires input embeddings to be scaled by sqrt(d_model)
    if (!getenv("SAPPHIRE_NO_EMBED_SCALE")) {
        float embed_scale = sqrtf((float)config->hidden_size);
        vec_scale(hidden, embed_scale, config->hidden_size);
    }
}

void sapphire_embed_lookup_batch(struct inference_session_t* session, const int* token_ids, int batch_size, float* hidden) {
    const llm_model_t* model = (const llm_model_t*)session->model_spec->llm_model;
    const gemma3_270m_config_t* config = (const gemma3_270m_config_t*)session->model_spec->variant_config;
    const uint16_t* embed_table_bf16 = (const uint16_t*)tensor_data(model->embedding_weight);
    float embed_scale = getenv("SAPPHIRE_NO_EMBED_SCALE") ? 1.0f : sqrtf((float)config->hidden_size);

    for (int i = 0; i < batch_size; i++) {
        float* h_i = hidden + i * config->hidden_size;
        bf16_to_f32_vec(h_i, embed_table_bf16 + (token_ids[i] * config->hidden_size), config->hidden_size);
        if (embed_scale != 1.0f) {
            vec_scale(h_i, embed_scale, config->hidden_size);
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
void lm_head(struct inference_session_t* session, const float* hidden, float* logits) {
    const llm_model_t* model = (const llm_model_t*)session->model_spec->llm_model;
    const gemma3_270m_config_t* config = (const gemma3_270m_config_t*)session->model_spec->variant_config;
    
    // We use a temporary slice from the end of the scratch buffer for normalization
    // to avoid clobbering the batched hidden states.
    int pad_m = session->padded_d_model;
    float* scratch_norm = session->scratch_buffer + (session->scratch_size / sizeof(float)) - (size_t)2 * pad_m;
    float* tmp_w = scratch_norm + pad_m;
    
    const float* final_w = get_norm_weights(model->norm_final_weight, tmp_w, config->hidden_size);

    // Gemma 3 uses delta variant for normalization weights
    rmsnorm_delta(scratch_norm, hidden, final_w, 1e-6f, config->hidden_size);
    tensor_gemv_with_ctx(session->gemv_ctx, logits, model->embedding_weight, scratch_norm);
}

/**
 * @brief Computes a linear attention stage (LoLCATs-style linearized attention).
 * 
 * PLACEHOLDER: Currently a stub that logs a warning.
 * TODO: Implement linearized attention using kernel-based approximation.
 * 
 * @param buf Layer buffers (pre-allocated).
 * @param ctx Transformer layer context.
 * @param hidden Input/output hidden state [d_model].
 * @param rope_cos RoPE cosine frequencies.
 * @param rope_sin RoPE sine frequencies.
 */
void compute_linear_attention_stage(layer_buffers_t buf,
                                    transformer_layer_ctx_t* ctx,
                                    float* hidden,
                                    const float* rope_cos,
                                    const float* rope_sin) {
    LOG_WARN("compute_linear_attention_stage called for layer %d (not yet implemented)", ctx->layer_idx);
    /* TODO: Implement LoLCATs linearized attention */
    (void)buf;
    (void)rope_cos;
    (void)rope_sin;
}

/**
 * @brief Computes an SSM (state space model) stage (Mamba-style recurrence).
 * 
 * PLACEHOLDER: Currently a stub that logs a warning.
 * TODO: Implement SSM forward pass with state-space dynamics and convolutional kernel.
 * 
 * @param buf Layer buffers (pre-allocated).
 * @param ctx Transformer layer context.
 * @param hidden Input/output hidden state [d_model].
 */
void compute_ssm_stage(layer_buffers_t buf,
                       transformer_layer_ctx_t* ctx,
                       float* hidden) {
    LOG_WARN("compute_ssm_stage called for layer %d (not yet implemented)", ctx->layer_idx);
    /* TODO: Implement Mamba-style SSM forward pass */
    (void)buf;
}

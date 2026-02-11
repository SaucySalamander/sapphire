#include "attention.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/gemma3_270m_config.h"
#include "inference.h"
#include "llm_model.h"
#include "kernels.h"
#include "transformer.h"
#include "rope.h"
#include "utils.h"
#include "log.h"

static attention_debug_config_t g_attn_debug_cfg = {0};

static int attention_parse_env_int(const char* name, int default_value) {
    const char* env = getenv(name);
    if (!env || env[0] == '\0') {
        return default_value;
    }
    char* endptr = NULL;
    long parsed = strtol(env, &endptr, 10);
    if (endptr == env) {
        return default_value;
    }
    return (int)parsed;
}

static void attention_debug_config_init(void) {
    if (g_attn_debug_cfg.initialized) {
        return;
    }

    const char* env = getenv("SAPPHIRE_DEBUG_ATTENTION");
    g_attn_debug_cfg.enabled = (env && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    g_attn_debug_cfg.layer_filter = attention_parse_env_int("SAPPHIRE_DEBUG_ATTENTION_LAYER", -1);
    g_attn_debug_cfg.head_filter = attention_parse_env_int("SAPPHIRE_DEBUG_ATTENTION_HEAD", -1);
    g_attn_debug_cfg.token_limit = attention_parse_env_int("SAPPHIRE_DEBUG_ATTENTION_TOKENS", 1);
    g_attn_debug_cfg.top_k = attention_parse_env_int("SAPPHIRE_DEBUG_ATTENTION_TOPK", 5);
    g_attn_debug_cfg.max_print = attention_parse_env_int("SAPPHIRE_DEBUG_ATTENTION_PRINT", 8);

    if (g_attn_debug_cfg.token_limit <= 0) {
        g_attn_debug_cfg.token_limit = -1;  // unlimited tokens
    }
    if (g_attn_debug_cfg.top_k <= 0) {
        g_attn_debug_cfg.top_k = 5;
    }
    if (g_attn_debug_cfg.top_k > ATTN_DEBUG_MAX_TOPK) {
        g_attn_debug_cfg.top_k = ATTN_DEBUG_MAX_TOPK;
    }
    if (g_attn_debug_cfg.max_print <= 0) {
        g_attn_debug_cfg.max_print = 8;
    }

    g_attn_debug_cfg.initialized = 1;

    if (g_attn_debug_cfg.enabled) {
        fprintf(stderr,
                "INFO: Attention diagnostics enabled (layer=%d head=%d tokens=%d topk=%d window_print=%d)\n",
                g_attn_debug_cfg.layer_filter,
                g_attn_debug_cfg.head_filter,
                g_attn_debug_cfg.token_limit,
                g_attn_debug_cfg.top_k,
                g_attn_debug_cfg.max_print);
    }
}

const attention_debug_config_t* get_attention_debug_config(void) {
    attention_debug_config_init();
    return &g_attn_debug_cfg;
}

bool attention_debug_should_log(const attention_debug_config_t* cfg, int layer_idx, int head_idx, int token_pos) {
    if (!cfg || !cfg->enabled) {
        return false;
    }
    if (cfg->layer_filter >= 0 && layer_idx != cfg->layer_filter) {
        return false;
    }
    if (cfg->head_filter >= 0 && head_idx != cfg->head_filter) {
        return false;
    }
    if (cfg->token_limit >= 0 && token_pos >= cfg->token_limit) {
        return false;
    }
    return true;
}

int sapphire_attention_forward(struct inference_session_t* session, int layer_idx, int token_pos,
                               float* q_proj, float* attn_out, float* scores_buf) {
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;

    int seq_len = token_pos + 1;  // Current token pos is 0-indexed, so length is pos + 1

    // Use the model's head dimension (d_k) if set, otherwise fallback to d_model / num_heads
    int head_dim = config->head_dim;
    int max_seq = kv_cache_get_max_seq_len(session->kv_cache);

    // Determine if this is a global layer (full context) or local layer (sliding window)
    bool is_global_layer = false;
    /* Use loaded bitmask if available: bit i set => full (global) attention for layer i */
    if (config->layer_types_mask) {
        is_global_layer = (((config->layer_types_mask >> (unsigned long long)layer_idx) & 1ULL) != 0ULL);
    } else {
        // Fallback to standard 5:1 hybrid pattern if flags not provided in config.json
        is_global_layer = ((layer_idx + 1) % 6 == 0);
    }

    const int swa_window = (config->sliding_window > 0) ? config->sliding_window : 1024;

    const float* cached_k_data = tensor_data(kv_cache_get_keys(session->kv_cache, layer_idx));
    const float* cached_v_data = tensor_data(kv_cache_get_values(session->kv_cache, layer_idx));
    int cache_stride = max_seq * head_dim;

    // Zero attn_out - ensure we zero the full projection width (num_heads * head_dim)
    memset(attn_out, 0, config->num_attention_heads * head_dim * sizeof(float));

    int window_start = 0;
    int attn_len = seq_len;
    if (!is_global_layer && seq_len > swa_window) {
        window_start = seq_len - swa_window;
        attn_len = swa_window;
    }

    float head_scalar = (config->query_pre_attn_scalar > 0.0f) ? (1.0f / sqrtf(config->query_pre_attn_scalar)) : (1.0f / sqrtf((float)head_dim));

    int group_size = config->num_attention_heads / config->num_key_value_heads;

    for (int h = 0; h < config->num_attention_heads; h++) {
        // GQA: Map query head 'h' to KV head 'h / group_size'
        int h_kv = h / group_size;

        const float* k_base = cached_k_data + h_kv * cache_stride + window_start * head_dim;
        const float* v_base = cached_v_data + h_kv * cache_stride + window_start * head_dim;
        const float* head_q = q_proj + h * head_dim;
        float* scores = scores_buf;

        // Step 1: Compute scores (Dot product)
        for (int t = 0; t < attn_len; t++) {
            float raw_dot = vec_dot(head_q, k_base + t * head_dim, head_dim);
            scores[t] = raw_dot * head_scalar;
        }

        // Step 2: Softmax
        softmax(scores, attn_len);

        // Step 3: Accumulate Attention Output
        float* h_out = attn_out + h * head_dim;
        for (int t = 0; t < attn_len; t++) {
            float alpha = scores[t];
            const float* v_vec = v_base + t * head_dim;
            for (int d = 0; d < head_dim; d++) {
                h_out[d] += alpha * v_vec[d];
            }
        }
    }

    return 0;
}

typedef struct {
    struct inference_session_t* session;
    int layer_idx;
    int start_pos;
    float* q_proj;
    float* attn_out;
    int q_stride;
    int max_seq;
} parallel_attn_args_t;

static void parallel_attn_fn(void* arg, int idx) {
    parallel_attn_args_t* a = (parallel_attn_args_t*)arg;
    float* scores_buf = a->session->attn_scores + (size_t)idx * a->max_seq;
    sapphire_attention_forward(a->session, a->layer_idx, a->start_pos + idx, 
                               a->q_proj + (size_t)idx * a->q_stride, 
                               a->attn_out + (size_t)idx * a->q_stride,
                               scores_buf);
}

int sapphire_attention_forward_batch(struct inference_session_t* session, int layer_idx, int start_pos, int batch_size,
                                     float* q_proj, float* attn_out) {
    const gemma3_270m_config_t* config = (const gemma3_270m_config_t*)session->model_spec->variant_config;
    int head_dim = config->head_dim;
    int q_stride = config->num_attention_heads * head_dim;
    int max_seq = kv_cache_get_max_seq_len(session->kv_cache);

    parallel_attn_args_t args = {
        .session = session,
        .layer_idx = layer_idx,
        .start_pos = start_pos,
        .q_proj = q_proj,
        .attn_out = attn_out,
        .q_stride = q_stride,
        .max_seq = max_seq
    };

    if (session->gemv_ctx && batch_size > 1) {
        return kernel_parallel_for(session->gemv_ctx, parallel_attn_fn, &args, batch_size);
    }

    /* Fallback for single batch or unthreaded context */
    for (int b = 0; b < batch_size; b++) {
        float* token_scores = session->attn_scores + (size_t)b * max_seq;
        sapphire_attention_forward(session, layer_idx, start_pos + b, 
                                   q_proj + (size_t)b * q_stride, 
                                   attn_out + (size_t)b * q_stride,
                                   token_scores);
    }

    return 0;
}

// ============================================================================
// Gemma 3 QK-Normalization Internal Helpers
// ============================================================================

static void qk_norm_apply(float* data, const float* scale, int head_dim, int num_heads) {
    if (!data || !scale) return;
    for (int h = 0; h < num_heads; h++) {
        rmsnorm_delta(data + h * head_dim, data + h * head_dim, scale + h * head_dim, 1e-6f, head_dim);
    }
}

static const float* load_query_vector(layer_buffers_t buf, const model_layer_weights_t* layer,
                                 const gemma3_270m_config_t* config, int head_dim, int layer_idx) {
        int q_norm_len = tensor_shape(layer->q_norm_weight)[0];
        const float* raw = get_norm_weights(layer->q_norm_weight, buf.weight_scratch, q_norm_len);
        int expected_q = config->num_attention_heads * head_dim;
        
        if (q_norm_len == expected_q) {
            return raw;  // per-head gamma
        } else if (q_norm_len == head_dim) {
            // Broadcast single-head gamma to all Q heads
            float* q_scale_ptr = buf.ffn_gate_buf; 
            for (int h = 0; h < config->num_attention_heads; h++) {
                 // Note: raw is likely in mmapped memory, so unsafe to modify.
                 // We copy TO scratch.
                 memcpy(q_scale_ptr + h * head_dim, raw, head_dim * sizeof(float));
            }
            return q_scale_ptr;
        } else {
            LOG_WARN("Layer %d q_norm len=%d expected=%d; disabling QK-Norm for Q", layer_idx, q_norm_len, expected_q);
            return NULL;
        }
}

static const float* load_key_vector(layer_buffers_t buf, const model_layer_weights_t* layer,
                               const gemma3_270m_config_t* config, int head_dim, int layer_idx) {
        int k_norm_len = tensor_shape(layer->k_norm_weight)[0];
        const float* raw = get_norm_weights(layer->k_norm_weight, buf.weight_scratch, k_norm_len);
        int expected_k = config->num_key_value_heads * head_dim;
        if (k_norm_len == expected_k) {
            return raw;
        } else if (k_norm_len == head_dim) {
             // reuse ffn_value_buf
            float* k_scale_ptr = buf.ffn_value_buf;
            for (int h = 0; h < config->num_key_value_heads; h++) {
                 memcpy(k_scale_ptr + h * head_dim, raw, head_dim * sizeof(float));
            }
            return k_scale_ptr;
        } else {
            LOG_WARN("Layer %d k_norm len=%d expected=%d; disabling QK-Norm for K", layer_idx, k_norm_len, expected_k);
            return NULL;
        }
}

static void apply_qk_norm_from_layer(layer_buffers_t buf,
                                           const model_layer_weights_t* layer,
                                           const gemma3_270m_config_t* config,
                                           int head_dim,
                                           int layer_idx) {
    const float* q_scale = NULL;
    const float* k_scale = NULL;

    if (layer->q_norm_weight) {
        q_scale = load_query_vector(buf, layer, config, head_dim, layer_idx);
    }

    if (layer->k_norm_weight) {
        k_scale = load_key_vector(buf, layer, config, head_dim, layer_idx);
    }

    qk_norm_apply(buf.q_proj, q_scale, head_dim, config->num_attention_heads);
    qk_norm_apply(buf.k_proj, k_scale, head_dim, config->num_key_value_heads);
}


static void attention_apply_rope_cache(layer_buffers_t buf, transformer_layer_ctx_t* ctx, const float* rope_cos, const float* rope_sin) {
    for (int b = 0; b < ctx->batch_size; b++) {
        layer_buffers_t b_buf = buf;
        b_buf.q_proj = buf.q_proj + b * buf.pi;
        b_buf.k_proj = buf.k_proj + b * buf.pk;
        apply_qk_norm_from_layer(b_buf, ctx->layer, ctx->config, ctx->head_dim, ctx->layer_idx);
        
        int pos = ctx->token_pos + b;
        for (int h = 0; h < ctx->config->num_attention_heads; h++) {
            rope_apply_fast(b_buf.q_proj + h * ctx->head_dim, pos, ctx->head_dim, rope_cos, rope_sin);
        }
        for (int h = 0; h < ctx->config->num_key_value_heads; h++) {
            rope_apply_fast(b_buf.k_proj + h * ctx->head_dim, pos, ctx->head_dim, rope_cos, rope_sin);
        }
        
        kv_cache_write_token(ctx->session->kv_cache, ctx->layer_idx, pos, b_buf.k_proj, buf.v_proj + b * buf.pk);
    }
}

void compute_attention_stage(layer_buffers_t buf,
                             transformer_layer_ctx_t* ctx,
                             float* hidden,
                             const float* rope_cos,
                             const float* rope_sin) {
    // 1. Residual Connection (Save hidden to residual)
    for (int b = 0; b < ctx->batch_size; b++) {
        vec_copy(buf.residual + b * buf.pm, hidden + b * ctx->d_model, ctx->d_model);
    }

    // 2. Pre-attention RMSNorm
    const float* norm_attn_data = get_norm_weights(ctx->layer->norm_attn_weight, buf.weight_scratch, ctx->d_model);
    for (int b = 0; b < ctx->batch_size; b++) {
        rmsnorm_delta(buf.norm_buf + b * buf.pm, hidden + b * ctx->d_model, norm_attn_data, 1e-6f, ctx->d_model);
    }

    // 3. Projections (Q, K, V)
    if (ctx->batch_size == 1) {
        tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.q_proj, ctx->layer->q_proj_weight, buf.norm_buf);
        tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.k_proj, ctx->layer->k_proj_weight, buf.norm_buf);
        tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.v_proj, ctx->layer->v_proj_weight, buf.norm_buf);
    } else {
        tensor_gemm_with_ctx(ctx->session->gemv_ctx, buf.q_proj, ctx->layer->q_proj_weight, buf.norm_buf, ctx->batch_size, buf.pi);
        tensor_gemm_with_ctx(ctx->session->gemv_ctx, buf.k_proj, ctx->layer->k_proj_weight, buf.norm_buf, ctx->batch_size, buf.pk);
        tensor_gemm_with_ctx(ctx->session->gemv_ctx, buf.v_proj, ctx->layer->v_proj_weight, buf.norm_buf, ctx->batch_size, buf.pk);
    }

    // 4. QK-Normalization, RoPE, and KV-Cache
    attention_apply_rope_cache(buf, ctx, rope_cos, rope_sin);

    // 5. Attention Forward Pass
    if (ctx->batch_size == 1) {
        sapphire_attention_forward(ctx->session, ctx->layer_idx, ctx->token_pos, buf.q_proj, buf.attn_out, ctx->session->attn_scores);
    } else {
        sapphire_attention_forward_batch(ctx->session, ctx->layer_idx, ctx->token_pos, ctx->batch_size, buf.q_proj, buf.attn_out);
    }

    // 6. Output projection
    if (ctx->batch_size == 1) {
        tensor_gemv_with_ctx(ctx->session->gemv_ctx, hidden, ctx->layer->out_proj_weight, buf.attn_out);
    } else {
        tensor_gemm_with_ctx(ctx->session->gemv_ctx, hidden, ctx->layer->out_proj_weight, buf.attn_out, ctx->batch_size, ctx->d_model);
    }

    // 7. Post-Attention RMSNorm (Gemma 3) & Residual Sum
    if (ctx->layer->norm_attn_post_weight) {
        const float* norm_post_w = get_norm_weights(ctx->layer->norm_attn_post_weight, buf.weight_scratch, ctx->d_model);

        for (int b = 0; b < ctx->batch_size; b++) {
            float* h_b = hidden + b * ctx->d_model;
            float* r_b = buf.residual + b * buf.pm;
            float* n_b = buf.norm_buf + b * buf.pm;
            rmsnorm_delta(n_b, h_b, norm_post_w, 1e-6f, ctx->d_model);
            vec_add(r_b, n_b, ctx->d_model);
            vec_copy(h_b, r_b, ctx->d_model);
        }
    } else {
        for (int b = 0; b < ctx->batch_size; b++) {
            float* h_b = hidden + b * ctx->d_model;
            float* r_b = buf.residual + b * buf.pm;
            vec_add(r_b, h_b, ctx->d_model);
            vec_copy(h_b, r_b, ctx->d_model);
        }
    }
}

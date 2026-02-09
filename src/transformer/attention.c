#include "attention.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/gemma3_270m_config.h"
#include "attention_strategy.h"
#include "inference.h"
#include "llm_model.h"
#include "normalization.h"
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
                               float* q_proj, float* attn_out) {
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;

    int seq_len = token_pos + 1;  // Current token pos is 0-indexed, so length is pos + 1

    // Use the model's head dimension (d_k) if set, otherwise fallback to d_model / num_heads
    int head_dim = config->head_dim;

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
    int cache_stride = kv_cache_get_max_seq_len(session->kv_cache) * head_dim;

    // Zero attn_out - ensure we zero the full projection width (num_heads * head_dim)
    // In hybrid models like Gemma 3, this may be larger than d_model.
    memset(attn_out, 0, config->num_attention_heads * head_dim * sizeof(float));

    int window_start = 0;
    int attn_len = seq_len;
    if (!is_global_layer && seq_len > swa_window) {
        window_start = seq_len - swa_window;
        attn_len = swa_window;
    }

    /*
     * Gemma 3 Attention Scaling: 
     * Dot product is scaled by query_pre_attn_scalar ** -0.5.
     * If not provided, fallback to 1/sqrt(head_dim).
     */
    float head_scalar = 1.0f;
    if (config->query_pre_attn_scalar > 0.0f) {
        head_scalar = 1.0f / sqrtf(config->query_pre_attn_scalar);
    } else {
        head_scalar = 1.0f / sqrtf((float)head_dim);
    }

    int group_size = config->num_attention_heads / config->num_key_value_heads;

    for (int h = 0; h < config->num_attention_heads; h++) {
        // GQA: Map query head 'h' to KV head 'h / group_size'
        int h_kv = h / group_size;

        const float* k_base = cached_k_data + h_kv * cache_stride + window_start * head_dim;
        const float* v_base = cached_v_data + h_kv * cache_stride + window_start * head_dim;
        const float* head_q = q_proj + h * head_dim;
        float* scores = session->attn_scores;

        // Step 1: Compute scores (Dot product)
        for (int t = 0; t < attn_len; t++) {
            float raw_dot = vec_dot(head_q, k_base + t * head_dim, head_dim);
            scores[t] = raw_dot * head_scalar;
        }

        // Step 2: Attention Softcap
        // UPDATE: Removed per new Gemma 3 technical report ("replace... with QK-norm").
        /*
        float attn_softcap = 50.0f;
        float inv_softcap = 1.0f / attn_softcap;
        for (int t = 0; t < attn_len; t++) {
            scores[t] = attn_softcap * tanhf(scores[t] * inv_softcap);
        }
        */

        // Step 3: Softmax
        softmax(scores, attn_len);

        // Step 4: Accumulate Attention Output
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

void compute_attention_stage(layer_buffers_t buf,
                             transformer_layer_ctx_t* ctx,
                             float* hidden,
                             const float* rope_cos,
                             const float* rope_sin) {
    // 1. Residual Connection (Save hidden to residual)
    vec_copy(buf.residual, hidden, ctx->d_model);

    // 2. Pre-attention RMSNorm
    const float* norm_attn_data = get_norm_weights(ctx->layer->norm_attn_weight, buf.weight_scratch, ctx->d_model);
    rmsnorm_delta(buf.norm_buf, hidden, norm_attn_data, 1e-6f, ctx->d_model);

    // 3. Projections (Q, K, V)
    tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.q_proj, ctx->layer->q_proj_weight, buf.norm_buf);
    tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.k_proj, ctx->layer->k_proj_weight, buf.norm_buf);
    tensor_gemv_with_ctx(ctx->session->gemv_ctx, buf.v_proj, ctx->layer->v_proj_weight, buf.norm_buf);

    // 4. QK-Normalization
    qk_norm_from_layer(buf, ctx->layer, ctx->config, ctx->head_dim, ctx->layer_idx);

    // 5. RoPE application
    // Use the explicitly provided RoPE frequencies (toggled in inference.c based on layer type)
    for (int h = 0; h < ctx->config->num_attention_heads; h++) rope_apply_fast(buf.q_proj + h * ctx->head_dim, ctx->token_pos, ctx->head_dim, rope_cos, rope_sin);
    for (int h = 0; h < ctx->config->num_key_value_heads; h++) rope_apply_fast(buf.k_proj + h * ctx->head_dim, ctx->token_pos, ctx->head_dim, rope_cos, rope_sin);

    // 6. KV-Cache Commit
    kv_cache_write_token(ctx->session->kv_cache, ctx->layer_idx, ctx->token_pos, buf.k_proj, buf.v_proj);
    // 7. Attention Forward Pass
    sapphire_attention_forward(ctx->session, ctx->layer_idx, ctx->token_pos, buf.q_proj, buf.attn_out);

    // 8. Output projection
    tensor_gemv_with_ctx(ctx->session->gemv_ctx, hidden, ctx->layer->out_proj_weight, buf.attn_out);

    // 9. Post-Attention RMSNorm (Gemma 3) & Residual Sum
    if (ctx->layer->norm_attn_post_weight) {
        const float* norm_post_w = get_norm_weights(ctx->layer->norm_attn_post_weight, buf.weight_scratch, ctx->d_model);

        rmsnorm_delta(buf.norm_buf, hidden, norm_post_w, 1e-6f, ctx->d_model);

        vec_add(buf.residual, buf.norm_buf, ctx->d_model);
    }

    vec_copy(hidden, buf.residual, ctx->d_model);
}
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

void attention_debug_dump(
    const attention_debug_config_t* cfg,
    float* raw_scores_buf,
    const float* head_q,
    const float* k_base,
    const float* softmax_scores,
    int head_dim,
    int attn_len,
    int layer_idx,
    int head_idx,
    int kv_head_idx,
    int token_pos,
    int window_start,
    bool is_global_layer) {
    if (!cfg || !cfg->enabled || !softmax_scores || attn_len <= 0) {
        return;
    }

    float w_min = softmax_scores[0];
    float w_max = softmax_scores[0];
    double entropy = 0.0;
    double sum_w = 0.0;
    double com = 0.0; /* center-of-mass (index weighted) */
    for (int i = 0; i < attn_len; i++) {
        float w = softmax_scores[i];
        if (w < w_min) w_min = w;
        if (w > w_max) w_max = w;
        sum_w += (double)w;
        com += (double)i * (double)w;
        if (w > 0.0f) {
            entropy -= (double)w * (double)logf(fmaxf(w, 1e-9f));
        }
    }
    float com_pos = (sum_w > 0.0) ? (float)(com / sum_w) : -1.0f;

    const float scale_factor = (1.0f / sqrtf((float)head_dim));
    fprintf(stderr,
            "ATTN DEBUG: token=%d layer=%d head=%d (kv=%d) len=%d window=[%d,%d) strategy=%s\n",
            token_pos,
            layer_idx,
            head_idx,
            kv_head_idx,
            attn_len,
            window_start,
            window_start + attn_len,
            is_global_layer ? "global" : "local");
    fprintf(stderr,
            "             weights: min=%.6f max=%.6f entropy=%.4f sum=%.6f COM=%.3f\n",
            w_min,
            w_max,
            (float)entropy,
            (float)sum_w,
            com_pos);

    if (fabs(sum_w - 1.0) > 1e-3) {
        fprintf(stderr, "             WARNING: softmax sum deviates from 1.0 by %.6f\n", (float)(sum_w - 1.0));
    }

    if (raw_scores_buf != NULL) {
        float raw_min = raw_scores_buf[0];
        float raw_max = raw_scores_buf[0];
        double sum_sq = 0.0;
        for (int i = 0; i < attn_len; i++) {
            float v = raw_scores_buf[i];
            if (v < raw_min) raw_min = v;
            if (v > raw_max) raw_max = v;
            sum_sq += (double)v * (double)v;
        }
        float raw_rms = sqrtf((float)(sum_sq / (double)attn_len));
        fprintf(stderr,
                "             raw QK: min=%.3f max=%.3f rms=%.3f scale=%.5f\n",
                raw_min,
                raw_max,
                raw_rms,
                scale_factor);
    }

    const int print_len = (attn_len < cfg->max_print) ? attn_len : cfg->max_print;
    if (print_len > 0) {
        fprintf(stderr, "             first %d weights:", print_len);
        for (int i = 0; i < print_len; i++) {
            fprintf(stderr, " %.5f", softmax_scores[i]);
        }
        fprintf(stderr, "\n");
    }

    int top_k = cfg->top_k;
    if (top_k > attn_len) {
        top_k = attn_len;
    }
    if (top_k > 0) {
        float top_vals[ATTN_DEBUG_MAX_TOPK];
        int top_idx[ATTN_DEBUG_MAX_TOPK];
        for (int i = 0; i < top_k; i++) {
            top_vals[i] = -FLT_MAX;
            top_idx[i] = -1;
        }
        for (int i = 0; i < attn_len; i++) {
            float w = softmax_scores[i];
            int min_slot = 0;
            float min_val = top_vals[0];
            for (int t = 1; t < top_k; t++) {
                if (top_vals[t] < min_val) {
                    min_val = top_vals[t];
                    min_slot = t;
                }
            }
            if (w > min_val) {
                top_vals[min_slot] = w;
                top_idx[min_slot] = i;
            }
        }
        for (int i = 0; i < top_k - 1; i++) {
            for (int j = i + 1; j < top_k; j++) {
                if (top_vals[j] > top_vals[i]) {
                    float tmp_val = top_vals[i];
                    top_vals[i] = top_vals[j];
                    top_vals[j] = tmp_val;
                    int tmp_idx = top_idx[i];
                    top_idx[i] = top_idx[j];
                    top_idx[j] = tmp_idx;
                }
            }
        }
        fprintf(stderr, "             top-%d spans:\n", top_k);
        for (int i = 0; i < top_k; i++) {
            if (top_idx[i] < 0) {
                continue;
            }
            const int rel = top_idx[i];
            const int abs_pos = window_start + rel;
            const float raw_val = (raw_scores_buf != NULL) ? raw_scores_buf[rel] : 0.0f;
            fprintf(stderr,
                    "               #%d rel=%4d abs=%4d weight=%.5f raw=%.3f scaled=%.3f\n",
                    i + 1,
                    rel,
                    abs_pos,
                    top_vals[i],
                    raw_val,
                    raw_val * scale_factor);
        }
    }
}

int sapphire_attention_forward(struct inference_session_t* session, int layer_idx, int token_pos,
                               float* q_proj, float* attn_out) {
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;
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

    const attention_debug_config_t* debug_cfg = get_attention_debug_config();

    // Scaling: Gemma 3 uses query_pre_attn_scalar**-0.5 (HF: self.scaling = config.query_pre_attn_scalar**-0.5).
    // If not present, preserve prior behavior: 1.0 when QK-norm present, otherwise 1/sqrt(d_k).
    bool has_qk_norm = (model->layers[layer_idx].q_norm_weight != NULL) &&
                       (model->layers[layer_idx].k_norm_weight != NULL);
    float head_scalar = 1.0f;
    if (config->query_pre_attn_scalar > 0.0f) {
        /*
         * HuggingFace Gemma3Attention uses:
         *   self.scaling = config.query_pre_attn_scalar ** -0.5
         * and applies it to the Q·K dot product.
         */
        head_scalar = 1.0f / sqrtf(config->query_pre_attn_scalar);
        if (getenv("SAPPHIRE_LOG_TENSORS") && layer_idx == 0 && token_pos == 0) {
            fprintf(stderr, "DEBUG: Using query_pre_attn_scalar=%.6f -> head_scalar=%.6f\n", config->query_pre_attn_scalar, head_scalar);
        }
    } else {
        head_scalar = has_qk_norm ? 1.0f : (1.0f / sqrtf((float)head_dim));
    }

    int group_size = config->num_attention_heads / config->num_key_value_heads;

    /* Optional KV readback verification (compare reads against last-written snapshots) */
    const char* kvrb_env = getenv("SAPPHIRE_DEBUG_KV_READBACK");
    int kvrb_enabled = (kvrb_env && kvrb_env[0] != '\0' && strcmp(kvrb_env, "0") != 0) ? 1 : 0;

    for (int h = 0; h < config->num_attention_heads; h++) {
        // GQA: Map query head 'h' to KV head 'h / group_size'
        int h_kv = h / group_size;

        const float* k_base = cached_k_data + h_kv * cache_stride + window_start * head_dim;
        const float* v_base = cached_v_data + h_kv * cache_stride + window_start * head_dim;
        float* head_q = q_proj + h * head_dim;
        float* scores = session->attn_scores;

        // Step 1: Compute scores (Dot product)
        for (int t = 0; t < attn_len; t++) {
            /* compute raw dot-product first, then apply head_scalar so we can
             * trace both values independently for Layer 0 debugging */
            float raw_dot = vec_dot(head_q, k_base + t * head_dim, head_dim);
            float scaled = raw_dot * head_scalar;
            scores[t] = scaled;
            if (session->attn_scores_raw) {
                session->attn_scores_raw[t] = scaled;
            }

            /* Targeted tracing: when tracing enabled, log head_scalar and a
             * few raw vs scaled values for Layer 0 Token 0 across heads. */
            if (getenv("SAPPHIRE_LOG_TENSORS") && layer_idx == 0 && token_pos < 5) {
                if (t == 0) {
                    fprintf(stderr, "ATTN_TRACE: L%d H%d head_scalar=%.6f raw_dot[0]=%.6f scaled[0]=%.6f has_qk_norm=%d\n",
                            layer_idx, h, head_scalar, raw_dot, scaled, has_qk_norm ? 1 : 0);
                }
                if (t < 5) {
                    const float* k_ptr_debug = k_base + t * head_dim;
                    fprintf(stderr, "ATTN_TRACE: L%d H%d t=%d raw=%.6f K0=%.6f scaled=%.6f\n",
                            layer_idx, h, t, raw_dot, k_ptr_debug[0], scaled);
                }
            }
        }

        /* KV readback verification: compare what we read against last-written snapshots */
        if (kvrb_enabled) {
            for (int t = 0; t < attn_len; t++) {
                int abs_pos = window_start + t;
                kv_cache_verify_entry(session->kv_cache, layer_idx, abs_pos,
                                      k_base + t * head_dim,
                                      v_base + t * head_dim);
            }
        }

        // DEBUG: Check for Attention Saturation
        if (getenv("SAPPHIRE_LOG_TENSORS") && layer_idx == 0 && h == 0 && token_pos < 5) {
            float s_min = scores[0], s_max = scores[0], s_sum = 0, s_sq = 0;
            for (int t = 0; t < attn_len; t++) {
                if (scores[t] < s_min) s_min = scores[t];
                if (scores[t] > s_max) s_max = scores[t];
                s_sum += scores[t];
                s_sq += scores[t] * scores[t];
            }
                float s_mean = s_sum / attn_len;
                float s_rms = sqrtf(s_sq / attn_len);
                fprintf(stderr, "DEBUG: Layer 0 Head 0 Raw Scores (len=%d): min=%.2f max=%.2f mean=%.2f rms=%.2f (Softcap=%.1f head_scalar=%.3f has_qk_norm=%d)\n",
                    attn_len, s_min, s_max, s_mean, s_rms, config->attn_logit_softcapping, head_scalar, has_qk_norm ? 1 : 0);
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
        /* Optional targeted dump: log min/max of attention scores before and after softmax
           for Layer 0 Token 0 to diagnose cold-start spikes */
        if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
            float pre_min = scores[0], pre_max = scores[0];
            for (int tt = 0; tt < attn_len; ++tt) {
                if (scores[tt] < pre_min) pre_min = scores[tt];
                if (scores[tt] > pre_max) pre_max = scores[tt];
            }
            LOG_DEBUG("DUMP ATTN L0 Token0 Head%d PRE-softmax min=%.6f max=%.6f", h, pre_min, pre_max);
        }
        softmax(scores, attn_len);
        if (layer_idx == 0 && token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
            float post_min = scores[0], post_max = scores[0];
            for (int tt = 0; tt < attn_len; ++tt) {
                if (scores[tt] < post_min) post_min = scores[tt];
                if (scores[tt] > post_max) post_max = scores[tt];
            }
            LOG_DEBUG("DUMP ATTN L0 Token0 Head%d POST-softmax min=%.6f max=%.6f", h, post_min, post_max);
        }

        // Debug Log
        if (attention_debug_should_log(debug_cfg, layer_idx, h, token_pos)) {
            attention_debug_dump(debug_cfg, session->attn_scores_raw, head_q, k_base, scores,
                                 head_dim, attn_len, layer_idx, h, h_kv, token_pos,
                                 window_start, is_global_layer);
        }

        // Step 4: Accumulate Attention Output
        float* h_out = attn_out + h * head_dim;
        for (int t = 0; t < attn_len; t++) {
            float alpha = scores[t];
            const float* v_vec = v_base + t * head_dim;
            for (int d = 0; d < head_dim; d++) {
                h_out[d] += alpha * v_vec[d];
            }
        }
        /* Targeted elementwise dumps for external comparison tools */
        if (token_pos == 0) {
            int dump_layers[] = {4, 13, 14, 15, 17};
            const int dump_count = sizeof(dump_layers) / sizeof(dump_layers[0]);
            for (int di = 0; di < dump_count; ++di) {
                if (layer_idx == dump_layers[di]) {
                    fprintf(stderr, "ATTN_VEC: L%d H%d token=%d Q_vec:", layer_idx, h, token_pos);
                    for (int d = 0; d < head_dim; ++d) {
                        fprintf(stderr, " %.6f", head_q[d]);
                    }
                    fprintf(stderr, "\n");

                    int rel = token_pos - window_start;
                    if (rel >= 0 && rel < attn_len) {
                        const float* k_tok = k_base + rel * head_dim;
                        fprintf(stderr, "ATTN_VEC: L%d H%d token=%d K_vec:", layer_idx, h, token_pos);
                        for (int d = 0; d < head_dim; ++d) {
                            fprintf(stderr, " %.6f", k_tok[d]);
                        }
                        fprintf(stderr, "\n");
                    }
                    break;
                }
            }
        }
    }

    return 0;
}

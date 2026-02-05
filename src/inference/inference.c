/**
 * @file inference.c
 * @brief Inference session and full forward pass implementation.
 */

#include "../include/inference.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/activations.h"
#include "../include/attention.h"
#include "../include/gemma3_270m_config.h"
#include "../include/ggml_model.h"
#include "../include/kv_cache.h"
#include "../include/log.h"
#include "../include/model_spec_loader.h"
#include "../include/normalization.h"
#include "../include/rope.h"
#include "../include/tensor.h"
#include "../include/tensor_gemv.h"
#include "../include/transformer.h"
#include "../include/utils.h"
#include "tokenizer.h"

static int g_attn_debug_raw_warned = 0;

/* Note: build_gemma3_prompt moved to tokenizer module (see include/tokenizer.h)
    to centralize tokenization / prompt construction logic. The function intelligently
    detects whether the model is IT or base and calls the appropriate builder. */

/**
 * @brief Initialize inference context
 */
inference_context_t* create_inference_context(float temperature, int max_tokens, int context_len, const char* model_name) {
    // Allow max_tokens == 0 for diagnostic-only initialization runs
    if (!model_name || temperature < 0.0f || max_tokens < 0 || context_len <= 0) return NULL;

    // Get the model specification for the requested model
    model_spec_t* spec = get_model_spec(model_name);
    if (!spec) {
        LOG_ERROR("Failed to get spec for model: %s", model_name);
        return NULL;
    }

    // Construct the model directory path: ./models/{model_name}
    char model_dir[512];
    snprintf(model_dir, sizeof(model_dir), "./models/%s", model_name);

    // Allocate model structure
    llm_model_t* model = (llm_model_t*)malloc(sizeof(llm_model_t));
    if (!model) {
        LOG_ERROR("Failed to allocate model structure");
        return NULL;
    }
    memset(model, 0, sizeof(llm_model_t));

    spec->llm_model = model;

    // Trigger the loader hooks to populate model and config
    char error_msg[512] = {0};
    if (spec->loader_hooks && spec->loader_hooks->populate_from_files) {
        int rc = spec->loader_hooks->populate_from_files(model_dir, spec, error_msg, sizeof(error_msg));
        if (rc != 0) {
            LOG_ERROR("Failed to populate model from files: %s", error_msg);
            free(model);
            return NULL;
        }
    } else {
        LOG_ERROR("Model spec has no loader hooks");
        free(model);
        return NULL;
    }

    // Run postprocessing hook if available
    if (spec->loader_hooks && spec->loader_hooks->postprocess_model) {
        spec->loader_hooks->postprocess_model(spec);
    }

    inference_context_t* ctx = (inference_context_t*)malloc(sizeof(inference_context_t));
    if (!ctx) {
        LOG_ERROR("Failed to allocate inference context");
        llm_model_destroy(model);
        return NULL;
    }
    memset(ctx, 0, sizeof(inference_context_t));

    ctx->spec = spec;
    ctx->context_len = context_len;
    ctx->max_tokens = max_tokens;
    ctx->temperature = temperature;

    // Allocate logits buffer
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)spec->variant_config;
    ctx->logits = (float*)malloc(config->vocab_size * sizeof(float));
    if (!ctx->logits) {
        fprintf(stderr, "ERROR: Failed to allocate logits buffer\n");
        free(ctx);
        return NULL;
    }

    // Create a single inference session for convenience (1:many supported via model_spec)
    ctx->session = inference_session_create(spec, context_len);
    if (!ctx->session) {
        fprintf(stderr, "ERROR: Failed to create inference session\n");
        free(ctx->logits);
        free(ctx);
        return NULL;
    }

    // Load tokenizer from model directory or spec; require successful load (no fallback)
    if (!ctx->tokenizer) {
        // Prefer tokenizer already loaded and stored in the spec
        if (spec->tokenizer_handle) {
            ctx->tokenizer = (sapphire_tokenizer_t*)spec->tokenizer_handle;
            LOG_INFO("Using tokenizer from spec for model %s", spec->model_id);
        } else {
            LOG_INFO("Loading tokenizer from %s", model_dir);
            sapphire_tokenizer_t* tk = tokenizer_load(model_dir);
            if (!tk) {
                LOG_ERROR("Failed to load tokenizer from %s", model_dir);
                // Clean up allocated resources and return error (no fallback)
                if (ctx->session) destroy_inference_session(ctx->session);
                if (ctx->logits) free(ctx->logits);
                free(ctx);
                if (spec->llm_model) {
                    llm_model_destroy_ex((const struct model_spec*)spec);
                    spec->llm_model = NULL;
                } else {
                    free(model);
                }
                return NULL;
            }
            // Store tokenizer in spec for ownership and reuse
            spec->tokenizer_handle = (void*)tk;
            ctx->tokenizer = tk;
        }
    }

    return ctx;
}

/**
 * @brief Destroy inference context
 */
void destroy_inference_context(inference_context_t* ctx) {
    if (!ctx) return;

    if (ctx->session) {
        destroy_inference_session(ctx->session);
    }

    if (ctx->logits) {
        free(ctx->logits);
    }

    // Tokenizer ownership: prefer spec-owned tokenizer. Only free ctx->tokenizer
    // if the spec did not take ownership.
    if (ctx->tokenizer) {
        if (ctx->spec && ctx->spec->tokenizer_handle) {
            // tokenizer is owned by spec; it will be freed below.
        } else {
            tokenizer_free(ctx->tokenizer);
        }
    }

    // Clean up model-owned resources (tensors and any open file handles).
    // The safetensors file handle is owned by the `llm_model_t` and is closed
    // inside `llm_model_destroy()`, so destroy the model here if present.
    if (ctx->spec && ctx->spec->llm_model) {
        llm_model_destroy_ex((const struct model_spec*)ctx->spec);
        ctx->spec->llm_model = NULL;
    }
    if (ctx->spec && ctx->spec->tokenizer_handle) {
        tokenizer_free((sapphire_tokenizer_t*)ctx->spec->tokenizer_handle);
        ctx->spec->tokenizer_handle = NULL;
    }

    free(ctx);
}

int perform_inference(inference_context_t* ctx, const char* prompt, char* output, int output_size) {
    if (!ctx || !prompt || !output) return -1;

    llm_model_t* model = (llm_model_t*)ctx->spec->llm_model;
    inference_session_t* session = ctx->session;
    gemma3_270m_config_t* config = (gemma3_270m_config_t*)ctx->spec->variant_config;

    printf("\n[Generating response...]\n");
    fflush(stdout);

    LOG_DEBUG("User prompt: '%s'", prompt);

    // 1. Build prompt (intelligently selects IT or base format based on model)
    int* tokens = malloc((ctx->context_len) * sizeof(int));
    int prompt_len = build_gemma3_prompt(ctx->spec, prompt, tokens, ctx->context_len);

    if (prompt_len <= 0) {
        LOG_ERROR("Failed to build prompt");
        free(tokens);
        return -1;
    }

    LOG_DEBUG("Built prompt with %d tokens", prompt_len);

    /* Log the entire token list. To avoid enormous logs, cap the printed
     * tokens to `MAX_SHOW_LIMIT`. This prints a single DEBUG line with the
     * token ids in order. */
    {
        int max_show = prompt_len;
        const int MAX_SHOW_LIMIT = 10000; /* safety cap */
        if (max_show > MAX_SHOW_LIMIT) max_show = MAX_SHOW_LIMIT;
        size_t buf_sz = (size_t)max_show * 12 + 64;
        char *tbuf = (char*)malloc(buf_sz);
        if (tbuf) {
            char *p = tbuf;
            size_t rem = buf_sz;
            int w = snprintf(p, rem, "Tokens (%d):", prompt_len);
            if (w < 0) w = 0;
            p += w; rem -= w;
            for (int i = 0; i < max_show; ++i) {
                int n = snprintf(p, rem, " %d", tokens[i]);
                if (n < 0 || (size_t)n >= rem) break;
                p += n; rem -= n;
            }
            if (max_show < prompt_len) snprintf(p, rem, " ...(truncated)");
            LOG_DEBUG("%s", tbuf);
            free(tbuf);
        } else {
            /* Fallback: log only the first token if allocation fails */
            LOG_DEBUG("First token (alloc fail): %d", prompt_len > 0 ? tokens[0] : -1);
        }
    }

    if (prompt_len <= 20) {
        printf("DEBUG: Token sequence: [");
        for (int i = 0; i < prompt_len; i++) {
            printf("%d", tokens[i]);
            if (i < prompt_len - 1) printf(", ");
        }
        printf("]\n");
    }

    // DEBUG: Show tokenized prompt
    printf("DEBUG: Prompt '%s' tokenized to %d tokens: [", prompt, prompt_len);
    for (int i = 0; i < prompt_len && i < 20; i++) {
        printf("%d%s", tokens[i], i < prompt_len - 1 ? ", " : "");
    }
    if (prompt_len > 20) printf("...");
    printf("]\n");
    printf("DEBUG: First few decoded tokens: ");
    for (int i = 0; i < prompt_len && i < 5; i++) {
        printf("'%s' ", decode(ctx->tokenizer, tokens[i]));
    }
    printf("\n");

    // Reset KV cache for a clean slate
    inference_session_reset(session);

    // 2. Pre-fill the KV cache with the prompt (Prompt Processing)
    for (int i = 0; i < prompt_len - 1; i++) {
        inference_forward(session, tokens[i], i, ctx->logits);
    }

    int total_tokens = prompt_len;
    int last_token = tokens[prompt_len - 1];
    int generated_count = 0;

    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
        fprintf(stderr, "[DEBUG_INIT] prompt_len=%d, total_tokens=%d, last_token=%d\n", prompt_len, total_tokens, last_token);
    }

    printf("\n[Response]\n");

    // 3. Generation loop (Auto-regressive)
    while (total_tokens < ctx->context_len && generated_count < ctx->max_tokens) {
        // Forward pass for the current token
        int cur_pos = total_tokens - 1;
        inference_forward(session, last_token, cur_pos, ctx->logits);

        if (generated_count == 0 && getenv("SAPPHIRE_DEBUG_LOGITS")) {
            fprintf(stderr, "[DEBUG_GEN_LOOP] generated_count=%d, cur_pos=%d, total_tokens=%d\n", generated_count, cur_pos, total_tokens);
        }

        // Sample next token (Greedy or Temperature)
        // Adjust logits by temperature BEFORE passing to sampler (Phase 8 Stability)
        if (ctx->temperature > 0.0f) {
            float inv_temp = 1.0f / ctx->temperature;
            vec_scale(ctx->logits, inv_temp, config->vocab_size);
        }

        // FIX (Phase 8): Logit Collapse Diagnostic
        // Check if the entropy of the distribution has collapsed
        if (generated_count == 0 || generated_count % 10 == 0) {
            float min_l, max_l, rms_l;
            vec_stats(ctx->logits, config->vocab_size, &min_l, &max_l, &rms_l);
            if (rms_l < 0.5f) {
                fprintf(stderr, "WARN: Logit RMS Low (%.4f) - Possible Probability Collapse / Uniform Distribution\n", rms_l);
            }
        }

        // FINITE/EXPLOSION CHECK: detect NaN/Inf or very large logits
        {
            int has_nonfinite = 0;
            float max_abs = 0.0f;
            for (int i = 0; i < config->vocab_size; i++) {
                float v = ctx->logits[i];
                if (!isfinite(v)) {
                    has_nonfinite = 1;
                    break;
                }
                float av = fabsf(v);
                if (av > max_abs) max_abs = av;
            }
            if (has_nonfinite) {
                fprintf(stderr, "WARN: Non-finite logits detected before sampling (token=%d)\n", generated_count);
            } else if (max_abs > 1e6f) {
                fprintf(stderr, "WARN: Very large logits detected (max_abs=%.2e) - possible internal amplification\n", max_abs);
            }
        }

        // Optional detailed logits/softmax debugging
        int debug_logits = getenv("SAPPHIRE_DEBUG_LOGITS") != NULL;
        if (debug_logits && ctx->temperature <= 0.0f) {
            // For greedy sampling, compute softmax on a temporary copy to inspect probabilities
            float *tmp = (float*)malloc(sizeof(float) * config->vocab_size);
            if (tmp) {
                memcpy(tmp, ctx->logits, sizeof(float) * config->vocab_size);
                softmax(tmp, config->vocab_size);
                float sum_exp = 0.0f; // already normalized
                for (int i = 0; i < config->vocab_size; ++i) sum_exp += tmp[i];
                float ent = sampling_entropy_from_unnormalized(tmp, config->vocab_size, sum_exp);
                float topk_mass = sampling_topk_mass_from_unnormalized(tmp, config->vocab_size, 10, sum_exp);
                float mn, mx, rms;
                vec_stats(ctx->logits, config->vocab_size, &mn, &mx, &rms);
                fprintf(stderr, "DEBUG_LOGITS pre-softmax: min=%.3f max=%.3f rms=%.3f\n", mn, mx, rms);
                fprintf(stderr, "DEBUG_LOGITS greedy softmax: entropy=%.4f topk_mass=%.4f\n", ent, topk_mass);
                free(tmp);
            }
        }

        // Use actual temperature for sample_temperature (already scaled logits by 1/T above)
        // When temperature <= 0, sample_temperature will use greedy selection
        int debug_logits_pre_select = getenv("SAPPHIRE_DEBUG_LOGITS") != NULL;
        if (debug_logits_pre_select && generated_count == 0) {
            // Print actual logits for key tokens BEFORE sampling
            fprintf(stderr, "[DEBUG_LOGITS_PRE_SELECT] token_106=%.6f token_818=%.6f token_236776=%.6f\n",
                    ctx->logits[106], ctx->logits[818], ctx->logits[236776]);
            // Find argmax
            int max_idx = 0;
            float max_logit = ctx->logits[0];
            for (int i = 1; i < config->vocab_size; ++i) {
                if (ctx->logits[i] > max_logit) {
                    max_logit = ctx->logits[i];
                    max_idx = i;
                }
            }
            fprintf(stderr, "[DEBUG_LOGITS_PRE_SELECT] argmax=token_%d with logit=%.6f\n", max_idx, max_logit);
        }
        
        int next_token = sample_temperature(ctx->logits, config->vocab_size, ctx->temperature);

        // After temperature sampling (when temperature>0) the `ctx->logits` buffer
        // contains unnormalized exp(logit) values. If debugging is enabled, compute
        // summary stats (entropy, top-k mass) and print top-k probabilities.
        if (debug_logits && ctx->temperature > 0.0f) {
            // sum of exp-values
            double sum_exp = 0.0;
            for (int i = 0; i < config->vocab_size; ++i) sum_exp += (double)ctx->logits[i];
            float sumf = (float)sum_exp;
            float ent = sampling_entropy_from_unnormalized(ctx->logits, config->vocab_size, sumf);
            int topk = 10;
            float topk_mass = sampling_topk_mass_from_unnormalized(ctx->logits, config->vocab_size, topk, sumf);

            float mn, mx, rms;
            vec_stats(ctx->logits, config->vocab_size, &mn, &mx, &rms);
            fprintf(stderr, "DEBUG_LOGITS post-exp: min=%.3f max=%.3f rms=%.3f sum_exp=%.3e entropy=%.4f top%d_mass=%.4f\n",
                    mn, mx, rms, sum_exp, ent, topk, topk_mass);

            // Print top-k ids and probabilities
            int K = topk;
            float top_vals[16];
            int top_ids[16];
            if (K > 16) K = 16;
            for (int i = 0; i < K; ++i) { top_vals[i] = 0.0f; top_ids[i] = -1; }
            for (int i = 0; i < config->vocab_size; ++i) {
                float v = ctx->logits[i];
                for (int j = 0; j < K; ++j) {
                    if (v > top_vals[j]) {
                        for (int t = K - 1; t > j; --t) { top_vals[t] = top_vals[t-1]; top_ids[t] = top_ids[t-1]; }
                        top_vals[j] = v;
                        top_ids[j] = i;
                        break;
                    }
                }
            }
            fprintf(stderr, "DEBUG_LOGITS Top-%d probs: ", K);
            for (int i = 0; i < K; ++i) {
                float prob = (sumf > 0.0f && top_ids[i] >= 0) ? top_vals[i] / sumf : 0.0f;
                fprintf(stderr, "%d(%.6f) ", top_ids[i], prob);
            }
            fprintf(stderr, "\n");
        }

        // DEBUG: Show what token was selected and top-3 logits
        if (generated_count < 10 || generated_count % 20 == 0) {
            // Find top 3 logits
            float top_val[3] = {-1e9, -1e9, -1e9};
            int top_id[3] = {0, 0, 0};
            // Calculate sum for normalization (since logits are now exponentials)
            float sum_exp = 0.0f;
            for (int i = 0; i < config->vocab_size; i++) sum_exp += ctx->logits[i];

            for (int i = 0; i < config->vocab_size; i++) {
                if (ctx->logits[i] > top_val[0]) {
                    top_val[2] = top_val[1];
                    top_id[2] = top_id[1];
                    top_val[1] = top_val[0];
                    top_id[1] = top_id[0];
                    top_val[0] = ctx->logits[i];
                    top_id[0] = i;
                } else if (ctx->logits[i] > top_val[1]) {
                    top_val[2] = top_val[1];
                    top_id[2] = top_id[1];
                    top_val[1] = ctx->logits[i];
                    top_id[1] = i;
                } else if (ctx->logits[i] > top_val[2]) {
                    top_val[2] = ctx->logits[i];
                    top_id[2] = i;
                }
            }
            printf("\nDEBUG[%d]: Top-3: (%d:'%s' %.2f) (%d:'%s' %.2f) (%d:'%s' %.2f) -> selected %d:'%s'\n",
                   generated_count,
                   top_id[0], decode(ctx->tokenizer, top_id[0]), top_val[0] / sum_exp,
                   top_id[1], decode(ctx->tokenizer, top_id[1]), top_val[1] / sum_exp,
                   top_id[2], decode(ctx->tokenizer, top_id[2]), top_val[2] / sum_exp,
                   next_token, decode(ctx->tokenizer, next_token));
        }

        // Gemma 3 stop tokens:
        // - Token 1: <eos> (explicit end-of-sequence)
        // - Token 106: <end_of_turn> (marks end of model's turn in chat format)
        if (next_token == 1 || next_token == 106) break;

        // --- STREAMING DECODE WITH SPIECE HANDLING ---
        const char* token_str = decode(ctx->tokenizer, next_token);
        if (token_str) {
            // Gemma 3 uses SPIECE prefix: 0xE2 0x96 0x81 (UTF-8 for U+2581 = ▁)
            // Replace with space for readable output
            int j = 0;
            if ((unsigned char)token_str[0] == 0xE2 && (unsigned char)token_str[1] == 0x96 && (unsigned char)token_str[2] == 0x81) {
                printf(" ");
                j = 3;  // Skip the 3-byte SPIECE prefix
            }
            // Print the rest of the token
            for (; token_str[j] != '\0'; j++) {
                printf("%c", token_str[j]);
            }
            fflush(stdout);
        }

        // Store for next iteration
        tokens[total_tokens++] = next_token;
        last_token = next_token;
        generated_count++;
    }

    // 4. Final Detokenization into the output buffer for the caller
    detokenize(ctx->tokenizer, tokens + prompt_len, generated_count, output, output_size);

    printf("\n\nDEBUG: Generated %d tokens\n", generated_count);

    free(tokens);

    return 0;
}

// Forward declarations

/**
 * Create an inference session.
 */
inference_session_t* inference_session_create(model_spec_t* spec, int max_context_len) {
    if (!spec) {
        fprintf(stderr, "ERROR: inference_session_create requires model\n");
        return NULL;
    }
    LOG_DEBUG("Creating inference session for model: %s", spec->model_id);
    LOG_DEBUG("Spec details: tensor_map_size=%d %s", spec->tensor_map_size, spec->variant_config != NULL ? "with variant_config" : "no variant_config");

    llm_model_t* model = (llm_model_t*)spec->llm_model;
    if (!model) {
        fprintf(stderr, "ERROR: Model is NULL in inference_session_create\n");
        return NULL;
    }

    gemma3_270m_config_t* config = (gemma3_270m_config_t*)spec->variant_config;
    if (!config) {
        fprintf(stderr, "ERROR: Model config is NULL in inference_session_create\n");
        return NULL;
    }

    inference_session_t* session = (inference_session_t*)malloc(sizeof(inference_session_t));
    if (!session) {
        fprintf(stderr, "ERROR: Failed to allocate inference session\n");
        return NULL;
    }

    session->model_spec = spec;
    session->kv_cache = NULL;
    session->scratch_buffer = NULL;
    session->attn_scores = NULL;
    session->attn_scores_raw = NULL;
    session->rope_freqs_cos_global = NULL;
    session->rope_freqs_sin_global = NULL;
    session->rope_freqs_cos_local = NULL;
    session->rope_freqs_sin_local = NULL;
    session->gemv_ctx = NULL;

    // Initialize d_inner (attention hidden dimension)
    // In hybrid architectures like Gemma 3, d_inner may differ from d_model
    int d_inner = config->num_attention_heads * config->head_dim;
    if (d_inner <= 0) {
        LOG_ERROR("Unable to determine d_inner (query projection dimension) %d x %d", config->num_attention_heads, config->head_dim);
        destroy_inference_session(session);
        return NULL;
    }

    // Initialize d_kv (key/value projection dimension)
    // In GQA architectures, this may differ from d_inner (query dimension)
    int d_kv = config->num_key_value_heads * config->head_dim;
    if (d_kv <= 0) {
        LOG_ERROR("Unable to determine d_kv (key/value projection dimension)");
        destroy_inference_session(session);
        return NULL;
    }

    int num_heads = d_inner / config->head_dim;
    int num_kv_heads = d_kv / config->head_dim;
    LOG_INFO("Inferred num_heads = %d, num_kv_heads = %d (head_dim=%d)",
             num_heads, num_kv_heads, config->head_dim);

    int d_ff = config->intermediate_size;

    LOG_INFO("Model dims: d_model=%d d_inner=%d d_kv=%d d_k=%d d_ff=%d heads=%d kv_heads=%d",
             config->hidden_size,
             d_inner,
             d_kv,
             config->head_dim,
             d_ff,
             num_heads,
             num_kv_heads);

    // Compute SIMD-friendly padded dimensions (round up to multiple of 8 floats => 32 bytes)
    // This ensures all buffers used as inputs to SIMD kernels can safely over-read
    // without triggering address sanitizer errors or segfaults. AVX2 loads read 256 bits
    // at a time without bounds checking, so we must guarantee sufficient padding.
    int d_model = config->hidden_size;

    // Choose SIMD lane count from kernel capabilities for a representative weight dtype.
    int simd_lanes = 1;
    if (model->layers && config->num_hidden_layers > 0) {
        model_layer_weights_t* lay0 = &model->layers[0];
        const tensor_t* rep = NULL;
        if (lay0->q_proj_weight)
            rep = lay0->q_proj_weight;
        else if (lay0->k_proj_weight)
            rep = lay0->k_proj_weight;
        if (rep) {
            simd_lanes = tensor_gemv_simd_lane_count_for_dtype(tensor_dtype(rep));
        }
    }
    if (simd_lanes <= 0) simd_lanes = 1;

    int pad_m = (d_model + simd_lanes - 1) & ~(simd_lanes - 1);
    int pad_inner = (d_inner + simd_lanes - 1) & ~(simd_lanes - 1);
    int pad_kv = (d_kv + simd_lanes - 1) & ~(simd_lanes - 1);
    int pad_ff = (d_ff + simd_lanes - 1) & ~(simd_lanes - 1);

    // Scratch buffer layout: all buffers are padded to their respective dimensions
    // In hybrid architectures (e.g., Gemma 3), d_inner may differ from d_model.
    // In GQA (Grouped-Query Attention), d_kv may differ from d_inner.
    // - hidden (pad_m):       embedding lookup result, input to transformer layers
    // - residual (pad_m):     residual pathway, accumulated across blocks
    // - norm_buf (pad_m):     output of RMSNorm, input to FFN/attention
    // - q_proj (pad_inner):   query projection (query head dimension)
    // - k_proj (pad_kv):      key projection (KV head dimension, may be smaller in GQA)
    // - v_proj (pad_kv):      value projection (KV head dimension, may be smaller in GQA)
    // - attn_out (pad_inner):  attention output (num_heads * head_dim)
    // - ffn_gate (pad_ff):    gate/activation branch of FFN
    // - ffn_value (pad_ff):   value branch of FFN
    // - geglu_input (2*pad_ff): concatenated buffer for GEGLU activation
    // Total: 3*pad_m + 2*pad_inner + 2*pad_kv + 6*pad_ff floats
    size_t scratch_floats = (size_t)3 * pad_m + (size_t)2 * pad_inner + (size_t)2 * pad_kv + (size_t)6 * pad_ff;
    scratch_floats += 32;

    // Allocate aligned scratch buffer (32-byte aligned for AVX2)
    // posix_memalign ensures the buffer starts at a 32-byte boundary,
    // and the padding ensures each slice can safely be read by SIMD kernels.
    void* aligned_ptr = NULL;
    size_t align_bytes = (size_t)simd_lanes * sizeof(float);
    if (align_bytes < 16) align_bytes = 16;
    if (posix_memalign(&aligned_ptr, align_bytes, scratch_floats * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: Failed to allocate aligned scratch buffer\n");
        destroy_inference_session(session);
        return NULL;
    }
    session->scratch_buffer = (float*)aligned_ptr;
    session->scratch_size = scratch_floats;
    session->padded_d_model = pad_m;
    session->padded_d_inner = pad_inner;
    session->padded_d_kv = pad_kv;
    session->padded_d_ff = pad_ff;
    memset(session->scratch_buffer, 0, scratch_floats * sizeof(float));

    // Pre-allocate attention scores buffer (max_context_len)
    session->attn_scores = (float*)malloc(max_context_len * sizeof(float));
    if (session->attn_scores) {
        memset(session->attn_scores, 0, max_context_len * sizeof(float));
    }
    if (!session->attn_scores) {
        fprintf(stderr, "ERROR: Failed to allocate attention score buffer\n");
        destroy_inference_session(session);
        return NULL;
    }

    // Create a global multi-layer KV cache
    session->kv_cache = kv_cache_create(
        config->num_hidden_layers,
        num_kv_heads,
        max_context_len,
        config->head_dim);
    if (!session->kv_cache) {
        fprintf(stderr, "ERROR: Failed to create global KV cache\n");
        destroy_inference_session(session);
        return NULL;
    }

    // Precompute RoPE frequencies for both possible bases (Gemma 3)
    int freq_size = max_context_len * config->head_dim;
    session->rope_freqs_cos_global = (float*)malloc(freq_size * sizeof(float));
    session->rope_freqs_sin_global = (float*)malloc(freq_size * sizeof(float));
    session->rope_freqs_cos_local = (float*)malloc(freq_size * sizeof(float));
    session->rope_freqs_sin_local = (float*)malloc(freq_size * sizeof(float));

    if (!session->rope_freqs_cos_global || !session->rope_freqs_sin_global ||
        !session->rope_freqs_cos_local || !session->rope_freqs_sin_local) {
        fprintf(stderr, "ERROR: Failed to allocate RoPE frequencies\n");
        destroy_inference_session(session);
        return NULL;
    }

    // Precompute for Global base (e.g. 1M)
    float base_global = config->rope_theta;
    if (rope_precompute_freqs(session->rope_freqs_cos_global, session->rope_freqs_sin_global,
                              config->head_dim, max_context_len, base_global) < 0) {
        LOG_ERROR("Failed to precompute global RoPE frequencies");
        destroy_inference_session(session);
        return NULL;
    }

    // Precompute for Local base (e.g. 10k)
    float base_local = config->rope_local_base_freq;
    if (rope_precompute_freqs(session->rope_freqs_cos_local, session->rope_freqs_sin_local,
                              config->head_dim, max_context_len, base_local) < 0) {
        LOG_ERROR("Failed to precompute local RoPE frequencies");
        destroy_inference_session(session);
        return NULL;
    }

    // Create GEMV context for matrix-vector operations (LM head, etc.)
    session->gemv_ctx = tensor_gemv_ctx_create(0, 1024);  // 0 = auto-detect threads, 1024 = chunk size
    if (!session->gemv_ctx) {
        LOG_ERROR("Failed to create GEMV context");
        destroy_inference_session(session);
        return NULL;
    }

    LOG_INFO("Inference session created with %d layers, context_len=%d",
             config->num_hidden_layers, max_context_len);

    return session;
}

/**
 * Reset KV caches for a new sequence.
 */
void inference_session_reset(inference_session_t* session) {
    if (!session || !session->kv_cache) return;

    kv_cache_reset(session->kv_cache);
}

/**
 * Single-token forward pass for autoregressive generation.
 *
 * Implements the full transformer forward pipeline:
 * 1. Token embedding lookup (row slice from embedding matrix)
 * 2. Transformer layer stack (18 layers for Gemma 3) with 5:1 interleave
 *    - RMSNorm pre-attention
 *    - Multi-head self-attention with GQA (16 query, 4 KV heads)
 *    - Residual connection
 *    - RMSNorm pre-FFN
 *    - Feed-forward with GeGLU gating
 *    - Residual connection
 * 3. Final RMSNorm
 * 4. LM head (weight-tied with embeddings)
 *
 * Performance target: ~2.67ms/token at -O3 with AVX2
 */
void inference_forward(inference_session_t* session, int token_id, int token_pos, float* logits) {
    if (!session || !logits) {
        LOG_ERROR("inference_forward requires session and logits buffer");
        return;
    }

    gemma3_270m_config_t* config = (gemma3_270m_config_t*)session->model_spec->variant_config;
    llm_model_t* model = (llm_model_t*)session->model_spec->llm_model;

    // 1. Embedding lookup
    sapphire_embed_lookup(session, token_id, session->scratch_buffer);

    // Debug: log embeddings
    if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
        float mn, mx, rms;
        vec_stats(session->scratch_buffer, config->hidden_size, &mn, &mx, &rms);
        fprintf(stderr, "[DEBUG_EMBEDDING] pos=%d min=%.6f max=%.6f rms=%.6f first5: %.6f %.6f %.6f %.6f %.6f\n", 
                token_pos, mn, mx, rms,
                session->scratch_buffer[0], session->scratch_buffer[1], session->scratch_buffer[2], 
                session->scratch_buffer[3], session->scratch_buffer[4]);
    }

    // Targeted dump: embeddings for token 0
    if (token_pos == 0 && getenv("SAPPHIRE_DEBUG_DUMPS")) {
        float mn, mx, rms;
        int d_model = config->hidden_size;
        vec_stats(session->scratch_buffer, d_model, &mn, &mx, &rms);
        LOG_DEBUG("DUMP EMBED: token=%d stats min=%.6f max=%.6f rms=%.6f", token_id, mn, mx, rms);
        int nonfinite = 0;
        int N = d_model < 16 ? d_model : 16;
        for (int i = 0; i < d_model; ++i) {
            if (!isfinite(session->scratch_buffer[i])) nonfinite++;
        }
        LOG_DEBUG("DUMP EMBED: non-finite count=%d (first %d values):", nonfinite, N);
        for (int i = 0; i < N; ++i) {
            LOG_DEBUG("DUMP EMBED: idx=%d val=%.9g", i, session->scratch_buffer[i]);
        }
    }

    // 2. Transformer layers
    for (int l = 0; l < config->num_hidden_layers; l++) {
        // Determine RoPE Strategy for this layer
        // Pattern: 5 Sliding Window (Local) : 1 Global Attention
        // Global Layers (5, 11, 17) use Base 1,000,000
        // Local Layers use Base 10,000
        bool is_global_layer = false;
        /* Use loaded bitmask if available: bit i set => full (global) attention for layer i */
        if (config->layer_types_mask) {
            is_global_layer = ((config->layer_types_mask >> l) & 1ULL) != 0ULL;
        } else {
            is_global_layer = ((l + 1) % 6 == 0);
        }

        const float* f_cos = is_global_layer ? session->rope_freqs_cos_global : session->rope_freqs_cos_local;
        const float* f_sin = is_global_layer ? session->rope_freqs_sin_global : session->rope_freqs_sin_local;

        // Debug: log layer input/output for critical positions
        if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
            float mn, mx, rms;
            vec_stats(session->scratch_buffer, config->hidden_size, &mn, &mx, &rms);
            fprintf(stderr, "[DEBUG_LAYER_INPUT] layer=%d pos=%d min=%.6f max=%.6f rms=%.6f\n", l, token_pos, mn, mx, rms);
        }

        sapphire_transformer_layer(session, l, token_pos, session->scratch_buffer, f_cos, f_sin);
        
        // Debug: log layer output
        if (getenv("SAPPHIRE_DEBUG_LOGITS")) {
            float mn, mx, rms;
            vec_stats(session->scratch_buffer, config->hidden_size, &mn, &mx, &rms);
            fprintf(stderr, "[DEBUG_LAYER_OUTPUT] layer=%d pos=%d min=%.6f max=%.6f rms=%.6f first5: %.6f %.6f %.6f %.6f %.6f\n", l, token_pos, mn, mx, rms,
                    session->scratch_buffer[0], session->scratch_buffer[1], session->scratch_buffer[2], session->scratch_buffer[3], session->scratch_buffer[4]);
        }
    }

    // 3. Final norm & LM Head
    sapphire_lm_head(session, session->scratch_buffer, logits);
}

/**
 * Free inference session.
 */
void destroy_inference_session(inference_session_t* session) {
    if (!session) return;

    if (session->kv_cache) {
        kv_cache_release(session->kv_cache);
    }

    if (session->scratch_buffer) {
        free(session->scratch_buffer);
    }

    if (session->attn_scores) {
        free(session->attn_scores);
    }

    if (session->attn_scores_raw) {
        free(session->attn_scores_raw);
    }

    if (session->rope_freqs_cos_global) {
        free(session->rope_freqs_cos_global);
    }

    if (session->rope_freqs_sin_global) {
        free(session->rope_freqs_sin_global);
    }

    if (session->rope_freqs_cos_local) {
        free(session->rope_freqs_cos_local);
    }

    if (session->rope_freqs_sin_local) {
        free(session->rope_freqs_sin_local);
    }

    if (session->gemv_ctx) {
        tensor_gemv_ctx_destroy(session->gemv_ctx);
    }

    free(session);
}

/* build_gemma3_prompt moved to tokenizer module (include/tokenizer.h).
   The tokenizer now owns prompt construction and tokenization helpers.
   The smart selector function detects IT vs base model and calls the right builder. */
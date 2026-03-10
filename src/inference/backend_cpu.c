/**
 * @file backend_cpu.c
 * @brief CPU backend implementation.
 *
 * Implements the CPU hardware backend using AVX2/AVX512 SIMD kernels.
 */

#include "../include/backend.h"
#include "../include/inference.h"
#include "../include/backend_cpu.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "../include/attention.h"
#include "../include/gemma3_270m_config.h"
#include "../include/ggml_model.h"
#include "../include/kv_cache.h"
#include "../include/layer_config_loader.h"
#include "../include/log.h"
#include "../include/rope.h"
#include "../include/tensor.h"
#include "../include/kernels.h"
#include "../include/transformer.h"
#include "../include/utils.h"

/* Forward declarations of external functions from inference.c */
extern void lm_head(inference_session_t* session, const float* hidden, float* logits);
extern void sapphire_embed_lookup_batch(inference_session_t* session, const int* token_ids, int batch_size, float* dest);

/**
 * Allocate and initialize CPU session buffers.
 *
 * Allocates:
 * - Scratch buffer (SIMD-aligned) for temporary tensors
 * - Attention scores matrices
 * - KV cache for all layers
 *
 * Returns 0 on success, -1 on error (errors logged internally).
 */
static int allocate_session_buffers(inference_session_t* session, const gemma3_270m_config_t* config, llm_model_t* model, int max_context_len) {
    if (!session || !config || !model) return -1;

    // Compute model dimensions
    int d_inner = config->num_attention_heads * config->head_dim;
    if (d_inner <= 0) {
        LOG_ERROR("Unable to determine d_inner (query projection dimension) %d x %d", config->num_attention_heads, config->head_dim);
        return -1;
    }

    int d_kv = config->num_key_value_heads * config->head_dim;
    if (d_kv <= 0) {
        LOG_ERROR("Unable to determine d_kv (key/value projection dimension)");
        return -1;
    }

    int num_heads = d_inner / config->head_dim;
    int num_kv_heads = d_kv / config->head_dim;
    LOG_INFO("Inferred num_heads = %d, num_kv_heads = %d (head_dim=%d)",
             num_heads, num_kv_heads, config->head_dim);

    int d_ff = config->intermediate_size;
    int d_model = config->hidden_size;

    LOG_INFO("Model dims: d_model=%d d_inner=%d d_kv=%d d_k=%d d_ff=%d heads=%d kv_heads=%d",
             d_model, d_inner, d_kv, config->head_dim, d_ff, num_heads, num_kv_heads);

    // Compute SIMD-friendly padded dimensions
    int simd_lanes = 1;
    if (model->layers && config->num_hidden_layers > 0) {
        const model_layer_weights_t* lay0 = &model->layers[0];
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

    // Scratch buffer layout for batch processing (maximum 32 tokens)
    int max_batch = 32;
    size_t scratch_floats = (size_t)max_batch * ((size_t)3 * pad_m + (size_t)2 * pad_inner + (size_t)2 * pad_kv + (size_t)6 * pad_ff);
    scratch_floats += 32;

    // Allocate aligned scratch buffer (32-byte aligned for AVX2)
    void* aligned_ptr = NULL;
    size_t align_bytes = (size_t)simd_lanes * sizeof(float);
    if (align_bytes < 16) align_bytes = 16;
    if (posix_memalign(&aligned_ptr, align_bytes, scratch_floats * sizeof(float)) != 0) {
        LOG_ERROR("Failed to allocate aligned scratch buffer");
        return -1;
    }

    backend_cpu_session_data_t* cpu_data = (backend_cpu_session_data_t*)session->backend_data;
    cpu_data->scratch_buffer = (float*)aligned_ptr;
    cpu_data->scratch_size = scratch_floats * sizeof(float);
    cpu_data->padded_d_model = pad_m;
    cpu_data->padded_d_inner = pad_inner;
    cpu_data->padded_d_kv = pad_kv;
    cpu_data->padded_d_ff = pad_ff;
    memset(cpu_data->scratch_buffer, 0, scratch_floats * sizeof(float));

    // Pre-allocate attention scores buffer
    cpu_data->attn_scores = (float*)malloc((size_t)max_batch * max_context_len * sizeof(float));
    if (!cpu_data->attn_scores) {
        LOG_ERROR("Failed to allocate attention score buffer");
        return -1;
    }
    memset(cpu_data->attn_scores, 0, (size_t)max_batch * max_context_len * sizeof(float));

    // Create global multi-layer KV cache
    cpu_data->kv_cache = kv_cache_create(
        config->num_hidden_layers,
        num_kv_heads,
        max_context_len,
        config->head_dim);
    if (!cpu_data->kv_cache) {
        LOG_ERROR("Failed to create global KV cache");
        return -1;
    }

    /* Expose CPU backend fields directly in session for zero-overhead access */
    session->kv_cache = cpu_data->kv_cache;
    session->scratch_buffer = cpu_data->scratch_buffer;
    session->scratch_size = cpu_data->scratch_size;
    session->attn_scores = cpu_data->attn_scores;
    session->attn_scores_raw = cpu_data->attn_scores_raw;
    session->rope_freqs_cos_global = cpu_data->rope_freqs_cos_global;
    session->rope_freqs_sin_global = cpu_data->rope_freqs_sin_global;
    session->rope_freqs_cos_local = cpu_data->rope_freqs_cos_local;
    session->rope_freqs_sin_local = cpu_data->rope_freqs_sin_local;
    session->gemv_ctx = cpu_data->gemv_ctx; /* Updated after creation below */
    session->padded_d_model = cpu_data->padded_d_model;
    session->padded_d_inner = cpu_data->padded_d_inner;
    session->padded_d_kv = cpu_data->padded_d_kv;
    session->padded_d_ff = cpu_data->padded_d_ff;

    return 0;
}

/**
 * Precompute RoPE frequencies for both global and local bases (Gemma 3).
 *
 * Returns 0 on success, -1 on error (errors logged internally).
 */
static int precompute_rope_frequencies(inference_session_t* session, const gemma3_270m_config_t* config, int max_context_len) {
    if (!session || !config) return -1;

    backend_cpu_session_data_t* cpu_data = (backend_cpu_session_data_t*)session->backend_data;

    int freq_size = max_context_len * config->head_dim;
    cpu_data->rope_freqs_cos_global = (float*)malloc(freq_size * sizeof(float));
    cpu_data->rope_freqs_sin_global = (float*)malloc(freq_size * sizeof(float));
    cpu_data->rope_freqs_cos_local = (float*)malloc(freq_size * sizeof(float));
    cpu_data->rope_freqs_sin_local = (float*)malloc(freq_size * sizeof(float));

    if (!cpu_data->rope_freqs_cos_global || !cpu_data->rope_freqs_sin_global ||
        !cpu_data->rope_freqs_cos_local || !cpu_data->rope_freqs_sin_local) {
        LOG_ERROR("Failed to allocate RoPE frequency buffers");
        return -1;
    }

    // Precompute for Global base (e.g., 1M)
    float base_global = config->rope_theta;
    if (rope_precompute_freqs(cpu_data->rope_freqs_cos_global, cpu_data->rope_freqs_sin_global,
                              config->head_dim, max_context_len, base_global) < 0) {
        LOG_ERROR("Failed to precompute global RoPE frequencies");
        return -1;
    }

    // Precompute for Local base (e.g., 10k)
    float base_local = config->rope_local_base_freq;
    if (rope_precompute_freqs(cpu_data->rope_freqs_cos_local, cpu_data->rope_freqs_sin_local,
                              config->head_dim, max_context_len, base_local) < 0) {
        LOG_ERROR("Failed to precompute local RoPE frequencies");
        return -1;
    }

    return 0;
}

/**
 * CPU backend: Initialize session.
 */
static int cpu_session_init(inference_session_t* session, const model_spec_t* spec, int max_context_len) {
    if (!session || !spec) {
        LOG_ERROR("cpu_session_init: invalid parameters");
        return -1;
    }

    llm_model_t* model = (llm_model_t*)spec->llm_model;
    if (!model) {
        LOG_ERROR("Model is NULL in cpu_session_init");
        return -1;
    }

    const gemma3_270m_config_t* config = (const gemma3_270m_config_t*)spec->variant_config;
    if (!config) {
        LOG_ERROR("Model config is NULL in cpu_session_init");
        return -1;
    }

    // Allocate CPU backend session data
    backend_cpu_session_data_t* cpu_data = (backend_cpu_session_data_t*)malloc(sizeof(backend_cpu_session_data_t));
    if (!cpu_data) {
        LOG_ERROR("Failed to allocate CPU backend session data");
        return -1;
    }
    memset(cpu_data, 0, sizeof(backend_cpu_session_data_t));
    session->backend_data = (void*)cpu_data;

    // Allocate all buffers
    if (allocate_session_buffers(session, config, model, max_context_len) != 0) {
        LOG_ERROR("Failed to allocate session buffers");
        return -1;
    }

    // Precompute RoPE frequencies
    if (precompute_rope_frequencies(session, config, max_context_len) != 0) {
        LOG_ERROR("Failed to precompute RoPE frequencies");
        return -1;
    }

    // Create kernel execution context for matrix-vector operations
    cpu_data->gemv_ctx = tensor_gemv_ctx_create(0, 1024);  // 0 = auto-detect threads, 1024 = chunk size
    if (!cpu_data->gemv_ctx) {
        LOG_ERROR("Failed to create GEMV context");
        return -1;
    }

    // Initialize persistent worker threads
    if (kernel_ctx_init(cpu_data->gemv_ctx) != 0) {
        LOG_ERROR("Failed to initialize GEMV worker threads");
        return -1;
    }

    /* Expose gemv_ctx in session after creation */
    session->gemv_ctx = cpu_data->gemv_ctx;

    LOG_INFO("CPU backend initialized: context_len=%d, scratch_size=%zu bytes",
             max_context_len, cpu_data->scratch_size);

    return 0;
}

/**
 * CPU backend: Destroy session.
 */
static void cpu_session_destroy(inference_session_t* session) {
    if (!session || !session->backend_data) return;

    backend_cpu_session_data_t* cpu_data = (backend_cpu_session_data_t*)session->backend_data;

    if (cpu_data->kv_cache) {
        kv_cache_release(cpu_data->kv_cache);
    }

    if (cpu_data->scratch_buffer) {
        free(cpu_data->scratch_buffer);
    }

    if (cpu_data->attn_scores) {
        free(cpu_data->attn_scores);
    }

    if (cpu_data->attn_scores_raw) {
        free(cpu_data->attn_scores_raw);
    }

    if (cpu_data->rope_freqs_cos_global) {
        free(cpu_data->rope_freqs_cos_global);
    }

    if (cpu_data->rope_freqs_sin_global) {
        free(cpu_data->rope_freqs_sin_global);
    }

    if (cpu_data->rope_freqs_cos_local) {
        free(cpu_data->rope_freqs_cos_local);
    }

    if (cpu_data->rope_freqs_sin_local) {
        free(cpu_data->rope_freqs_sin_local);
    }

    if (cpu_data->gemv_ctx) {
        tensor_gemv_ctx_destroy(cpu_data->gemv_ctx);
    }

    free(cpu_data);
    session->backend_data = NULL;
}

/**
 * CPU backend: Execute forward batch.
 *
 * Implements layer-by-layer transformer computation with SIMD kernels.
 */
static int cpu_forward_batch(inference_session_t* session, const int* token_ids,
                             int start_pos, int batch_size, float* logits) {
    if (!session || !session->backend_data) {
        LOG_ERROR("cpu_forward_batch: invalid session");
        return -1;
    }

    backend_cpu_session_data_t* cpu_data = (backend_cpu_session_data_t*)session->backend_data;
    const gemma3_270m_config_t* config = (const gemma3_270m_config_t*)session->model_spec->variant_config;

    // 1. Embedding lookup
    sapphire_embed_lookup_batch(session, token_ids, batch_size, cpu_data->scratch_buffer);

    // 2. Transformer layers with layer-type dispatch
    for (int l = 0; l < config->num_hidden_layers; l++) {
        sapphire_layer_config_t* layer_cfg = &session->layer_configs[l];
        bool is_global = layer_cfg->config.attention.is_global;
        const float* f_cos = is_global ? cpu_data->rope_freqs_cos_global : cpu_data->rope_freqs_cos_local;
        const float* f_sin = is_global ? cpu_data->rope_freqs_sin_global : cpu_data->rope_freqs_sin_local;

        /* Dispatch based on layer type */
        if (layer_cfg->type == LAYER_TYPE_ATTENTION_SOFTMAX || layer_cfg->type == LAYER_TYPE_ATTENTION_LINEAR) {
            sapphire_transformer_layer_batch(session, l, start_pos, batch_size, cpu_data->scratch_buffer, (transformer_rope_t){f_cos, f_sin});
        } else {
            // For non-attention layers, fall back to single-token mode
            for (int b = 0; b < batch_size; b++) {
                sapphire_transformer_layer(session, l, start_pos + b, cpu_data->scratch_buffer + b * config->hidden_size, (transformer_rope_t){f_cos, f_sin});
            }
        }
    }

    // 3. Final norm & LM Head (for the last token in the batch)
    if (logits) {
        const float* last_hidden = cpu_data->scratch_buffer + (batch_size - 1) * config->hidden_size;
        lm_head(session, (float*)last_hidden, logits);
    }

    return 0;
}

/**
 * CPU backend: Reset session for a new sequence.
 */
static void cpu_reset(inference_session_t* session) {
    if (!session || !session->backend_data) return;

    backend_cpu_session_data_t* cpu_data = (backend_cpu_session_data_t*)session->backend_data;
    if (cpu_data->kv_cache) {
        kv_cache_reset(cpu_data->kv_cache);
    }
}

/**
 * CPU backend implementation (static instance).
 */
sapphire_backend_t backend_cpu_impl = {
    .type = SAPPHIRE_BACKEND_TYPE_CPU,
    .name = "cpu",
    .session_init = cpu_session_init,
    .session_destroy = cpu_session_destroy,
    .forward_batch = cpu_forward_batch,
    .reset = cpu_reset,
};

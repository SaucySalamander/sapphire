/**
 * @file inference.c
 * @brief Inference session and full forward pass implementation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/inference.h"
#include "../include/ggml_model.h"
#include "../include/tensor.h"
#include "../include/kv_cache.h"
#include "../include/rope.h"
#include "../include/attention.h"
#include "../include/activations.h"
#include "../include/normalization.h"
#include "../include/tensor_gemv.h"
#include "../include/transformer.h"

// Forward declarations
static int validate_gemv_dims(const tensor_t *A, int expected_out, int expected_in, const char *name);

/**
 * Precompute RoPE frequencies (cos and sin for all positions and dimensions).
 */
static int precompute_rope_freqs(float *freqs_cos, float *freqs_sin,
                                  int d_k, int max_context_len, float rope_base) {
    if (!freqs_cos || !freqs_sin) return -1;
    
    /**
     * RoPE (Rotary Position Encoding) formula:
     * For each position pos and pair index m (where m = 0, 1, 2, ..., d_k/2-1):
     *   freq = 1.0 / rope_base^(2m / d_k)
     *   angle = pos * freq
     * The same angle is applied to both elements in the pair (dim, dim+1).
     */
    for (int pos = 0; pos < max_context_len; pos++) {
        for (int m = 0; m < d_k / 2; m++) {
            float freq = 1.0f / powf(rope_base, 2.0f * (float)m / (float)d_k);
            float angle = (float)pos * freq;
            
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            
            // Store same angle for both elements in the pair
            int dim = 2 * m;
            freqs_cos[pos * d_k + dim] = cos_val;
            freqs_cos[pos * d_k + dim + 1] = cos_val;
            
            freqs_sin[pos * d_k + dim] = sin_val;
            freqs_sin[pos * d_k + dim + 1] = sin_val;
        }
    }
    
    return 0;
}

/**
 * Create an inference session.
 */
inference_session_t* inference_session_create(llm_model_t *model, int max_context_len) {
    if (!model) {
        fprintf(stderr, "ERROR: inference_session_create requires model\n");
        return NULL;
    }
    
    inference_session_t *session = (inference_session_t *)malloc(sizeof(inference_session_t));
    if (!session) {
        fprintf(stderr, "ERROR: Failed to allocate inference session\n");
        return NULL;
    }
    
    session->model = model;
    
    // Initialize d_inner (attention hidden dimension)
    // In hybrid architectures like Gemma 3, d_inner may differ from d_model
    int d_inner = model->config.d_inner;
    
    // Auto-detect d_inner from the actual loaded tensor shape if not explicitly set
    if (d_inner <= 0 && model->config.num_layers > 0 && model->layers) {
        model_layer_weights_t *lay0 = &model->layers[0];
        if (lay0->q_proj_weight && tensor_ndim(lay0->q_proj_weight) == 2) {
            const int *q_shape = tensor_shape(lay0->q_proj_weight);
            if (q_shape) {
                // q_proj_weight is [d_inner, d_model], so first dimension is d_inner
                d_inner = q_shape[0];
                model->config.d_inner = d_inner;  // Save it back to config for consistency
                fprintf(stderr, "INFO: Auto-detected d_inner = %d from q_proj_weight\n", d_inner);
            }
        }
    }
    
    // Fallback to d_model if still not set
    if (d_inner <= 0) {
        d_inner = model->config.d_model;
        model->config.d_inner = d_inner;
    }
    
    // Initialize d_kv (key/value projection dimension)
    // In GQA architectures, this may differ from d_inner (query dimension)
    int d_kv = model->config.d_kv;
    
    // Auto-detect d_kv from the actual loaded tensor shape if not explicitly set
    if (d_kv <= 0 && model->config.num_layers > 0 && model->layers) {
        model_layer_weights_t *lay0 = &model->layers[0];
        if (lay0->k_proj_weight && tensor_ndim(lay0->k_proj_weight) == 2) {
            const int *k_shape = tensor_shape(lay0->k_proj_weight);
            if (k_shape) {
                // k_proj_weight is [d_kv, d_model], so first dimension is d_kv
                d_kv = k_shape[0];
                model->config.d_kv = d_kv;  // Save it back to config
                fprintf(stderr, "INFO: Auto-detected d_kv = %d from k_proj_weight (GQA)\n", d_kv);
            }
        }
    }
    
    // Fallback to d_inner if still not set
    if (d_kv <= 0) {
        d_kv = d_inner;
        model->config.d_kv = d_kv;
    }
    
    // Determine FFN hidden size (d_ff). Try to infer from loaded model weights
    int d_ff = 1760;  // default for Gemma 3
    if (model->config.num_layers > 0 && model->layers) {
        model_layer_weights_t *lay0 = &model->layers[0];
        const tensor_t *cand = NULL;
        if (lay0->up_proj_weight) cand = lay0->up_proj_weight;
        else if (lay0->gate_proj_weight) cand = lay0->gate_proj_weight;
        else if (lay0->down_proj_weight) cand = lay0->down_proj_weight;
        if (cand && tensor_ndim(cand) == 2) {
            const int *s = tensor_shape(cand);
            if (s) {
                // Candidate matrix could be [d_model, d_ff] or [d_ff, d_model]
                if (s[0] != model->config.d_model) d_ff = s[0];
                else if (s[1] != model->config.d_model) d_ff = s[1];
            }
        }
    }

    // Compute SIMD-friendly padded dimensions (round up to multiple of 8 floats => 32 bytes)
    // This ensures all buffers used as inputs to SIMD kernels can safely over-read
    // without triggering address sanitizer errors or segfaults. AVX2 loads read 256 bits
    // at a time without bounds checking, so we must guarantee sufficient padding.
    int d_model = model->config.d_model;
    int pad_m = (d_model + 7) & ~7;      // Round up to nearest multiple of 8 floats (32 bytes)
    int pad_inner = (d_inner + 7) & ~7;  // Same for query projection dimension (may differ in hybrid arch)
    int pad_kv = (d_kv + 7) & ~7;        // Same for key/value projection dimension (may differ in GQA)
    int pad_ff = (d_ff + 7) & ~7;        // Same for FFN hidden dimension

    // Scratch buffer layout: all buffers are padded to their respective dimensions
    // In hybrid architectures (e.g., Gemma 3), d_inner may differ from d_model.
    // In GQA (Grouped-Query Attention), d_kv may differ from d_inner.
    // - hidden (pad_m):       embedding lookup result, input to transformer layers
    // - residual (pad_m):     residual pathway, accumulated across blocks
    // - norm_buf (pad_m):     output of RMSNorm, input to FFN/attention
    // - q_proj (pad_inner):   query projection (query head dimension)
    // - k_proj (pad_kv):      key projection (KV head dimension, may be smaller in GQA)
    // - v_proj (pad_kv):      value projection (KV head dimension, may be smaller in GQA)
    // - attn_out (pad_m):     attention output (back to embedding dimension)
    // - ffn_gate (pad_ff):    gate/activation branch of FFN
    // - ffn_value (pad_ff):   value branch of FFN
    // - geglu_input (2*pad_ff): concatenated buffer for GEGLU activation
    // Total: 4*pad_m + pad_inner + 2*pad_kv + 4*pad_ff floats
    size_t scratch_floats = (size_t)4 * pad_m + (size_t)pad_inner + (size_t)2 * pad_kv + (size_t)4 * pad_ff;

    // Allocate aligned scratch buffer (32-byte aligned for AVX2)
    // posix_memalign ensures the buffer starts at a 32-byte boundary,
    // and the padding ensures each slice can safely be read by SIMD kernels.
    void *aligned_ptr = NULL;
    if (posix_memalign(&aligned_ptr, 32, scratch_floats * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: Failed to allocate aligned scratch buffer\n");
        free(session);
        return NULL;
    }
    session->scratch_buffer = (float *)aligned_ptr;
    session->scratch_size = scratch_floats;
    session->padded_d_model = pad_m;
    session->padded_d_inner = pad_inner;
    session->padded_d_kv = pad_kv;
    session->padded_d_ff = pad_ff;
    memset(session->scratch_buffer, 0, scratch_floats * sizeof(float));

    // Pre-allocate attention scores buffer (max_context_len)
    session->attn_scores = (float *)malloc(max_context_len * sizeof(float));
    if (!session->attn_scores) {
        fprintf(stderr, "ERROR: Failed to allocate attention score buffer\n");
        free(session->scratch_buffer);
        free(session);
        return NULL;
    }

    // Create a global multi-layer KV cache
    // Default to num_heads as num_kv_heads if not explicitly set (for backward compatibility)
    int num_kv_heads = model->config.num_kv_heads > 0 ? model->config.num_kv_heads : model->config.num_heads;
    
    session->kv_cache = kv_cache_create(
        model->config.num_layers,
        num_kv_heads,
        max_context_len,
        model->config.d_k
    );
    if (!session->kv_cache) {
        fprintf(stderr, "ERROR: Failed to create global KV cache\n");
        free(session->attn_scores);
        free(session->scratch_buffer);
        free(session);
        return NULL;
    }
    
    // Precompute RoPE frequencies
    int d_k = model->config.d_k;
    int freq_size = max_context_len * d_k;
    session->rope_freqs_cos = (float *)malloc(freq_size * sizeof(float));
    session->rope_freqs_sin = (float *)malloc(freq_size * sizeof(float));
    
    if (!session->rope_freqs_cos || !session->rope_freqs_sin) {
        fprintf(stderr, "ERROR: Failed to allocate RoPE frequencies\n");
        if (session->rope_freqs_cos) free(session->rope_freqs_cos);
        if (session->rope_freqs_sin) free(session->rope_freqs_sin);
        kv_cache_release(session->kv_cache);
        free(session->attn_scores);
        free(session->scratch_buffer);
        free(session);
        return NULL;
    }
    
    if (precompute_rope_freqs(session->rope_freqs_cos, session->rope_freqs_sin,
                              d_k, max_context_len, model->config.rope_base) < 0) {
        fprintf(stderr, "ERROR: Failed to precompute RoPE frequencies\n");
        free(session->rope_freqs_cos);
        free(session->rope_freqs_sin);
        kv_cache_release(session->kv_cache);
        free(session->attn_scores);
        free(session->scratch_buffer);
        free(session);
        return NULL;
    }
    
    // Create GEMV context for matrix-vector operations (LM head, etc.)
    session->gemv_ctx = tensor_gemv_ctx_create(0, 1024);  // 0 = auto-detect threads, 1024 = chunk size
    if (!session->gemv_ctx) {
        fprintf(stderr, "ERROR: Failed to create GEMV context\n");
        free(session->rope_freqs_cos);
        free(session->rope_freqs_sin);
        kv_cache_release(session->kv_cache);
        free(session->attn_scores);
        free(session->scratch_buffer);
        free(session);
        return NULL;
    }
    
    fprintf(stderr, "INFO: Inference session created with %d layers, context_len=%d\n",
            model->config.num_layers, max_context_len);
    
    return session;
}

/**
 * Reset KV caches for a new sequence.
 */
void inference_session_reset(inference_session_t *session) {
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
void inference_forward(inference_session_t *session, int token_id, int token_pos, float *logits) {
    if (!session || !logits) {
        fprintf(stderr, "ERROR: inference_forward requires session and logits\n");
        return;
    }
    
    llm_model_t *model = session->model;
    
    // Validate token ID
    if (token_id < 0 || token_id >= model->config.vocab_size) {
        fprintf(stderr, "ERROR: Token ID %d out of range [0, %d)\n", token_id, model->config.vocab_size);
        memset(logits, 0, model->config.vocab_size * sizeof(float));
        return;
    }
    
    if (!model->embedding_weight || !model->lm_head_weight || !model->norm_final_weight) {
        fprintf(stderr, "ERROR: Model missing required weights\n");
        memset(logits, 0, model->config.vocab_size * sizeof(float));
        return;
    }
    
    // Buffer layout in scratch (padded to avoid SIMD over-read):
    // [0 ... pad_m-1]: hidden state
    // [pad_m ... 2*pad_m-1]: residual
    // [2*pad_m ... 3*pad_m-1]: norm_buf
    // [3*pad_m ... 4*pad_m-1]: attn_out
    // [4*pad_m ... 4*pad_m+pad_inner-1]: q_proj
    // [4*pad_m+pad_inner ... 4*pad_m+pad_inner+pad_kv-1]: k_proj
    // [4*pad_m+pad_inner+pad_kv ... 4*pad_m+pad_inner+2*pad_kv-1]: v_proj
    // [4*pad_m+pad_inner+2*pad_kv ... 4*pad_m+pad_inner+2*pad_kv+pad_ff-1]: ffn_gate
    // [4*pad_m+pad_inner+2*pad_kv+pad_ff ... 4*pad_m+pad_inner+2*pad_kv+2*pad_ff-1]: ffn_value
    // [4*pad_m+pad_inner+2*pad_kv+2*pad_ff ... 4*pad_m+pad_inner+2*pad_kv+4*pad_ff-1]: geglu_input
    int pad_m = session->padded_d_model;
    int pad_inner = session->padded_d_inner;
    int pad_kv = session->padded_d_kv;
    int pad_ff = session->padded_d_ff;
    float *hidden = session->scratch_buffer;
    float *residual = session->scratch_buffer + pad_m;
    float *norm_buf = session->scratch_buffer + 2 * pad_m;
    
    // === STAGE 1: Token Embedding Lookup ===
    // The embedding table is stored in BF16 format for Gemma 3 (2 bytes per value)
    // Convert BF16 to F32 for computation
    const uint16_t *embed_table_bf16 = (const uint16_t *)tensor_data(model->embedding_weight);
    if (!embed_table_bf16) {
        fprintf(stderr, "ERROR: Cannot access embedding table\n");
        memset(logits, 0, model->config.vocab_size * sizeof(float));
        return;
    }
    
    // Convert BF16 embedding row to F32
    for (int d = 0; d < model->config.d_model; d++) {
        uint16_t bf16_val = embed_table_bf16[token_id * model->config.d_model + d];
        uint32_t f32_bits = ((uint32_t)bf16_val) << 16;
        hidden[d] = *(float *)&f32_bits;
    }
    
    // === STAGE 2: Transformer Stack (18 layers) ===
    for (int layer_idx = 0; layer_idx < model->config.num_layers; layer_idx++) {
        model_layer_weights_t *layer = &model->layers[layer_idx];
        
        // Check all required weights exist
        if (!layer->norm_attn_weight || !layer->q_proj_weight || 
            !layer->k_proj_weight || !layer->v_proj_weight ||
            !layer->out_proj_weight || !layer->norm_ffn_weight ||
            !layer->up_proj_weight || !layer->down_proj_weight) {
            fprintf(stderr, "ERROR: Layer %d has missing weights\n", layer_idx);
            memset(logits, 0, model->config.vocab_size * sizeof(float));
            return;
        }
        
        // Apply 5:1 Attention Pattern (Hybrid Species Logic from Phase 6)
        // 5 local sliding window layers, 1 global attention layer
        // int attn_type = ((layer_idx % 6) == 5) ? ATTN_TYPE_GLOBAL : ATTN_TYPE_LOCAL_SLIDING;
        
        // --- Pre-Attention RMSNorm ---
        float *norm_attn_data = (float *)tensor_data(layer->norm_attn_weight);
        if (!norm_attn_data || sapphire_rmsnorm(residual, hidden, norm_attn_data,
                                                1e-6f, model->config.d_model) != 0) {
            fprintf(stderr, "ERROR: Pre-attention RMSNorm failed at layer %d\n", layer_idx);
            memset(logits, 0, model->config.vocab_size * sizeof(float));
            return;
        }
        
        // --- Multi-Head Self-Attention (GQA: 16 query heads, 4 KV heads) ---
        // Compute Q, K, V projections and apply attention with KV cache
        int d_k = model->config.d_k;  // 64
        
        // Use scratch buffer slices (already allocated and padded) to avoid per-layer
        // mallocs and to provide safe SIMD over-read padding.
        // Buffer layout accounts for different dimensions in hybrid and GQA architectures:
        // - q_proj uses pad_inner (query dimension)
        // - k_proj, v_proj use pad_kv (KV dimension, may be smaller due to GQA)
        float *attn_out = norm_buf + pad_m;
        float *q_proj = attn_out + pad_m;
        float *k_proj = q_proj + pad_inner;
        float *v_proj = k_proj + pad_kv;
        float *ffn_gate_buf = v_proj + pad_kv;
        float *ffn_value_buf = ffn_gate_buf + pad_ff;
        float *geglu_input_buf = ffn_value_buf + pad_ff;
        
        float *attn_scores = session->attn_scores;  // Pre-allocated (size = max_context_len)
        
        // Q, K, V projections using tensor_gemv_with_ctx (validate shapes first)
        // In hybrid architectures, projections go from d_model to d_inner (Q) or d_kv (K/V)
        if (validate_gemv_dims(layer->q_proj_weight, model->config.d_inner, model->config.d_model, "q_proj_weight") != 0 ||
            tensor_gemv_with_ctx(session->gemv_ctx, q_proj, layer->q_proj_weight, residual) != 0) {
            fprintf(stderr, "ERROR: Q projection GEMV failed\n");
            goto attn_cleanup;
        }
        
        // In GQA, K/V projections go to d_kv (which is smaller than d_inner)
        if (validate_gemv_dims(layer->k_proj_weight, model->config.d_kv, model->config.d_model, "k_proj_weight") != 0 ||
            tensor_gemv_with_ctx(session->gemv_ctx, k_proj, layer->k_proj_weight, residual) != 0) {
            fprintf(stderr, "ERROR: K projection GEMV failed\n");
            goto attn_cleanup;
        }
        
        if (validate_gemv_dims(layer->v_proj_weight, model->config.d_kv, model->config.d_model, "v_proj_weight") != 0 ||
            tensor_gemv_with_ctx(session->gemv_ctx, v_proj, layer->v_proj_weight, residual) != 0) {
            fprintf(stderr, "ERROR: V projection GEMV failed\n");
            goto attn_cleanup;
        }
        
        // Add K, V to the global KV cache (for this layer)
        // The KV cache expects:
        // - k_token: [num_kv_heads, head_dim] flattened = [d_kv]
        // - v_token: [num_kv_heads, head_dim] flattened = [d_kv]
        // These are properly sized based on the GQA configuration (model->config.d_kv)
        if (kv_cache_append_token(session->kv_cache, k_proj, v_proj) != 0) {
            fprintf(stderr, "ERROR: KV cache append failed at layer %d\n", layer_idx);
            goto attn_cleanup;
        }
        
        // Compute attention scores: softmax(Q · K^T / sqrt(d_k))
        int seq_len = kv_cache_get_seq_len(session->kv_cache);
        const float *cached_k_data = tensor_data(kv_cache_get_keys(session->kv_cache, layer_idx));
        const float *cached_v_data = tensor_data(kv_cache_get_values(session->kv_cache, layer_idx));
        
        compute_attention_scores(q_proj, cached_k_data, seq_len, d_k, session->attn_scores);
        
        // Attention output: scores @ V using GEMV
        // Create a temporary tensor for V cache to use with tensor_gemv_with_ctx
        // V cache is [seq_len, d_model], need to compute scores @ V = attn_scores @ V_cache
        // But attn_scores is [seq_len], so we need to do a special GEMV: 
        // attn_out[i] = sum_j(attn_scores[j] * V[j, i])
        // This is equivalent to attn_scores @ V_cache (scores is row vector)
        // Zero the first d_model elements of the padded attn_out buffer
        memset(attn_out, 0, model->config.d_model * sizeof(float));
        for (int i = 0; i < model->config.d_model; i++) {
            for (int j = 0; j < seq_len; j++) {
                attn_out[i] += session->attn_scores[j] * cached_v_data[j * model->config.d_model + i];
            }
        }
        
        // Output projection using tensor_gemv_with_ctx
        // In hybrid architectures, out_proj goes from d_inner back to d_model
        if (validate_gemv_dims(layer->out_proj_weight, model->config.d_model, model->config.d_inner, "out_proj_weight") != 0 ||
            tensor_gemv_with_ctx(session->gemv_ctx, hidden, layer->out_proj_weight, attn_out) != 0) {
            fprintf(stderr, "ERROR: Output projection failed\n");
            goto attn_cleanup;
        }
        
        // Add attention residual
        for (int i = 0; i < model->config.d_model; i++) {
            hidden[i] += residual[i];
        }
        
        attn_cleanup:
         
         // --- Pre-FFN RMSNorm ---
         float *norm_ffn_data = (float *)tensor_data(layer->norm_ffn_weight);
         if (!norm_ffn_data || sapphire_rmsnorm(norm_buf, hidden, norm_ffn_data,
                                                1e-6f, model->config.d_model) != 0) {
             fprintf(stderr, "ERROR: Pre-FFN RMSNorm failed at layer %d\n", layer_idx);
             memset(logits, 0, model->config.vocab_size * sizeof(float));
             return;
         }
        
        // --- GeGLU FFN (Gate + Value projections with GELU activation) ---
        // GeGLU: output = down(GELU(gate_proj(norm_buf)) * value_proj(norm_buf))
        // For Gemma 3: d_ff = 1760 (expansion factor 2.75x: 640 * 2.75 = 1760)
        
        int d_ff = 1760;
        // ffn_gate_buf and ffn_value_buf are already defined from scratch buffer
        
        // Gate projection: gate = gate_proj(norm_buf) using tensor_gemv_with_ctx
        if (tensor_gemv_with_ctx(session->gemv_ctx, ffn_gate_buf, layer->gate_proj_weight, norm_buf) != 0) {
            fprintf(stderr, "ERROR: Gate projection GEMV failed at layer %d\n", layer_idx);
            memset(logits, 0, model->config.vocab_size * sizeof(float));
            return;
        }
        
        // Up projection (value path): up = up_proj(norm_buf) using tensor_gemv_with_ctx
        if (tensor_gemv_with_ctx(session->gemv_ctx, ffn_value_buf, layer->up_proj_weight, norm_buf) != 0) {
            fprintf(stderr, "ERROR: Up projection GEMV failed at layer %d\n", layer_idx);
            memset(logits, 0, model->config.vocab_size * sizeof(float));
            return;
        }
        
        // Apply GeGLU activation: geglu_out[i] = GELU(gate[i]) * value[i]
        // sapphire_geglu expects interleaved input: [val_0, val_1, ..., val_{d_ff-1}, gate_0, gate_1, ..., gate_{d_ff-1}]
        // geglu_input_buf is pre-allocated in scratch buffer to avoid per-token malloc
        memcpy(geglu_input_buf, ffn_value_buf, d_ff * sizeof(float));
        memcpy(geglu_input_buf + d_ff, ffn_gate_buf, d_ff * sizeof(float));
        
        // Call sapphire_geglu to compute GELU(gate) * value in-place (writes into ffn_gate_buf)
        if (sapphire_geglu(ffn_gate_buf, geglu_input_buf, 2 * d_ff) != 0) {
             fprintf(stderr, "ERROR: GeGLU activation failed at layer %d\n", layer_idx);
             memset(logits, 0, model->config.vocab_size * sizeof(float));
             return;
         }
        
        // Down projection: residual = down_proj @ gated_output using tensor_gemv_with_ctx
        if (tensor_gemv_with_ctx(session->gemv_ctx, residual, layer->down_proj_weight, ffn_gate_buf) != 0) {
            fprintf(stderr, "ERROR: Down projection GEMV failed at layer %d\n", layer_idx);
            memset(logits, 0, model->config.vocab_size * sizeof(float));
            return;
        }
        
        // --- Add FFN Residual ---
        for (int i = 0; i < model->config.d_model; i++) {
            hidden[i] = hidden[i] + residual[i];
        }
    }
    
    // === STAGE 3: Final RMSNorm ===
    float *norm_final_data = (float *)tensor_data(model->norm_final_weight);
    if (!norm_final_data || sapphire_rmsnorm(residual, hidden, norm_final_data,
                                             1e-6f, model->config.d_model) != 0) {
        fprintf(stderr, "ERROR: Final RMSNorm failed\n");
        memset(logits, 0, model->config.vocab_size * sizeof(float));
        return;
    }
    
    // === STAGE 4: LM Head (Weight-Tied with Embedding) ===
    // For Gemma 3, lm_head is weight-tied with embedding
    // logits = final_norm @ embedding_table.T
    // embedding_table shape: [vocab_size, d_model] = [262144, 640]
    
    // Use tensor_gemv_with_ctx for efficient matrix-vector product
    int ret = tensor_gemv_with_ctx(session->gemv_ctx, logits, model->embedding_weight, residual);
    if (ret != 0) {
        fprintf(stderr, "ERROR: LM head GEMV failed with code %d\n", ret);
        memset(logits, 0, model->config.vocab_size * sizeof(float));
        return;
    }
}

/**
 * Free inference session.
 */
void inference_session_destroy(inference_session_t *session) {
    if (!session) return;
    
    if (session->kv_cache) {
        kv_cache_release(session->kv_cache);
    }
    
    if (session->scratch_buffer) {
        free(session->scratch_buffer);
    }
    
    if (session->rope_freqs_cos) {
        free(session->rope_freqs_cos);
    }
    
    if (session->rope_freqs_sin) {
        free(session->rope_freqs_sin);
    }
    
    if (session->gemv_ctx) {
        tensor_gemv_ctx_destroy(session->gemv_ctx);
    }
    
    free(session);
}

/**
 * Generate tokens using greedy decoding (argmax).
 */
int llm_generate_greedy(inference_session_t *session,
                        const int *prompt_tokens,
                        int prompt_len,
                        int max_tokens,
                        int *output_tokens) {
    if (!session || !prompt_tokens || !output_tokens) {
        fprintf(stderr, "ERROR: llm_generate_greedy requires session, prompt, and output\n");
        return 0;
    }

    if (prompt_len <= 0) {
        fprintf(stderr, "ERROR: llm_generate_greedy requires prompt_len > 0\n");
        return 0;
    }
    
    inference_session_reset(session);
    
    float *logits = (float *)malloc(session->model->config.vocab_size * sizeof(float));
    if (!logits) {
        fprintf(stderr, "ERROR: Failed to allocate logits buffer\n");
        return 0;
    }
    
    int token_count = 0;
    
    // Process prompt
    for (int i = 0; i < prompt_len && token_count < max_tokens; i++) {
        inference_forward(session, prompt_tokens[i], i, logits);
        output_tokens[token_count++] = prompt_tokens[i];
    }
    
    // Generate new tokens
    // Safe to access prompt_tokens[prompt_len - 1] because prompt_len > 0 is guaranteed above
    int last_token = prompt_tokens[prompt_len - 1];
    for (int i = prompt_len; i < max_tokens; i++) {
        inference_forward(session, last_token, i, logits);
        
        // Greedy: pick argmax
        int next_token = 0;
        float max_logit = logits[0];
        for (int j = 1; j < session->model->config.vocab_size; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token = j;
            }
        }
        
        output_tokens[token_count++] = next_token;
        last_token = next_token;
        
        // Simple stopping condition: if token is very small ID (likely end-of-sequence)
        if (next_token < 2) {
            break;
        }
    }
    
    free(logits);
    return token_count;
}

/**
 * Validate expected GEMV dimensions for weight matrix A: [out, in]
 * Returns 0 on success, -1 on mismatch.
 */
static int validate_gemv_dims(const tensor_t *A, int expected_out, int expected_in, const char *name) {
    if (!A) {
        fprintf(stderr, "ERROR: validate_gemv_dims: weight '%s' is NULL\n", name ? name : "(null)");
        return -1;
    }
    if (tensor_ndim(A) != 2) {
        fprintf(stderr, "ERROR: validate_gemv_dims: weight '%s' must be 2D\n", name ? name : "(null)");
        return -1;
    }
    const int *shape = tensor_shape(A);
    if (!shape) {
        fprintf(stderr, "ERROR: validate_gemv_dims: weight '%s' has no shape\n", name ? name : "(null)");
        return -1;
    }
    if (expected_out >= 0 && shape[0] != expected_out) {
        fprintf(stderr, "ERROR: validate_gemv_dims: weight '%s' out dim mismatch: expected %d, got %d\n",
                name ? name : "(null)", expected_out, shape[0]);
        return -1;
    }
    if (expected_in >= 0 && shape[1] != expected_in) {
        fprintf(stderr, "ERROR: validate_gemv_dims: weight '%s' in dim mismatch: expected %d, got %d\n",
                name ? name : "(null)", expected_in, shape[1]);
        return -1;
    }
    return 0;
}

/**
 * @file backend_cpu.h
 * @brief CPU backend session data structure.
 *
 * Private header defining the CPU backend's session data.
 * This structure is opaque to the public inference API.
 */

#ifndef BACKEND_CPU_H
#define BACKEND_CPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct sapphire_context kernel_context_t;
typedef struct kv_cache_t kv_cache_t;

/**
 * CPU backend session data.
 *
 * Contains all CPU-specific state for an inference session.
 * This structure is allocated and stored in inference_session_t.backend_data.
 */
typedef struct {
    /* Memory buffers (aligned for SIMD kernels) */
    float *scratch_buffer;           /**< Reusable scratch for temporary tensors, padded dimensions */
    size_t scratch_size;             /**< Total allocated size of scratch_buffer in bytes */

    float *attn_scores;              /**< Attention weight matrices [max_batch * max_context_len] */
    float *attn_scores_raw;          /**< Optional raw QK diagnostics buffer */

    /* Precomputed RoPE frequencies (two bases for Gemma 3) */
    float *rope_freqs_cos_global;    /**< Global RoPE base (e.g., 1M) cosines [max_context_len * head_dim] */
    float *rope_freqs_sin_global;    /**< Global RoPE base sines */
    float *rope_freqs_cos_local;     /**< Local RoPE base (e.g., 10k) cosines [max_context_len * head_dim] */
    float *rope_freqs_sin_local;     /**< Local RoPE base sines */

    /* SIMD kernel execution context */
    kernel_context_t *gemv_ctx;      /**< Thread pool for parallel matrix-vector operations */

    /* Padded dimensions (multiples of SIMD lanes for safe vector reads) */
    int padded_d_model;              /**< Padded hidden size */
    int padded_d_inner;              /**< Padded query projection dimension */
    int padded_d_kv;                 /**< Padded key/value projection dimension */
    int padded_d_ff;                 /**< Padded feed-forward hidden dimension */

    /* KV cache for all layers */
    kv_cache_t *kv_cache;            /**< Global multi-layer KV cache */

} backend_cpu_session_data_t;

#ifdef __cplusplus
}
#endif

#endif // BACKEND_CPU_H

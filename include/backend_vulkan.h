/**
 * @file backend_vulkan.h
 * @brief Vulkan backend session data structure (P11-02 + P11-03).
 *
 * Private header defining the Vulkan backend's session data.
 * This structure is opaque to the public inference API.
 *
 * Integrates P11-02 buffer management:
 * - Weight buffers (device-local, read-only)
 * - KV cache (device-local, read-write)
 * - GPU scratchpad buffers (ephemeral ping-pong)
 * - Input/Output ring buffers (timeline semaphore sync)
 *
 * Integrates P11-03 compute pipeline abstraction:
 * - Compute pipelines per kernel (embedding, attention, FFN, etc.)
 * - Descriptor sets for buffer bindings
 * - Command buffer recording infrastructure
 */

#ifndef BACKEND_VULKAN_H
#define BACKEND_VULKAN_H

#include "vk_buffers.h"
#include "vk_compute_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * P11-04 Pipeline Enumeration
 * ======================================================================== */

#define PIPELINE_RMSNORM_F32        0
#define PIPELINE_RMSNORM_BF16       1
#define PIPELINE_GEMV_F32           2
#define PIPELINE_GEMV_BF16          3
#define PIPELINE_GEMM_F32           4
#define PIPELINE_GEMM_BF16          5
#define PIPELINE_ROPE_F32           6
#define PIPELINE_ROPE_BF16          7
#define PIPELINE_QK_NORM_F32        8
#define PIPELINE_QK_NORM_BF16       9
#define PIPELINE_ATTENTION_F32      10
#define PIPELINE_ATTENTION_BF16     11

/* ========================================================================
 * Static Descriptor Table (Vulkan 1.0 – No Update-After-Bind)
 *
 * Maps every kernel invocation in every layer to a unique, pre-populated
 * VkDescriptorSet.  All buffer bindings (weights AND activations) are
 * written once during vulkan_prepopulate_descriptors() and then only
 * bound (never updated) during command-buffer recording.
 *
 * Kernel slots per layer:
 *   SDT_SLOT_PRE_ATTN_NORM  (0) – rmsnorm: input_buf, attn_norm_w, norm_buf
 *   SDT_SLOT_Q_PROJ         (1) – gemv: norm_buf, q_proj_w, q_proj
 *   SDT_SLOT_K_PROJ         (2) – gemv: norm_buf, k_proj_w, k_proj
 *   SDT_SLOT_V_PROJ         (3) – gemv: norm_buf, v_proj_w, v_proj
 *   SDT_SLOT_QK_NORM        (4) – qk_norm: q_proj, k_proj, qnorm_w, knorm_w
 *   SDT_SLOT_ROPE           (5) – rope: q_proj, k_proj, cos, sin
 *   SDT_SLOT_ATTENTION      (6) – attention: q_proj, kv_cache, attn_out, scores
 *   SDT_SLOT_O_PROJ         (7) – gemv: attn_out, o_proj_w, norm_buf
 *   SDT_SLOT_POST_ATTN_NORM (8) – rmsnorm: norm_buf, attn_post_w, attn_out
 *   SDT_SLOT_ATTN_RESIDUAL  (9) – vec_add: attn_out, residual, norm_buf
 *   SDT_SLOT_PRE_FFN_NORM  (10) – rmsnorm: norm_buf, ffn_norm_w, ffn_gate
 *   SDT_SLOT_GATE_PROJ     (11) – gemv: ffn_gate, gate_proj_w, ffn_value
 *   SDT_SLOT_UP_PROJ       (12) – gemv: ffn_gate, up_proj_w, attn_out
 *   SDT_SLOT_GELU          (13) – gelu: ffn_value, attn_out  (→ffn_gate)
 *   SDT_SLOT_DOWN_PROJ     (14) – gemv: ffn_gate, down_proj_w, ffn_value
 *   SDT_SLOT_POST_FFN_NORM (15) – rmsnorm: ffn_value, ffn_post_w, ffn_gate
 *   SDT_SLOT_FFN_RESIDUAL  (16) – vec_add: ffn_gate, residual, output_buf
 *                                  (output_buf = hidden[(layer+1)%2], baked at init)
 *
 * Total per layer: SDT_KERNELS_PER_LAYER = 17
 * Total layer sets: 18 × 17 = 306
 * LM-head sets:  SDT_LMHEAD_NORM_SLOT, SDT_LMHEAD_PROJ_SLOT  (+2)
 * Grand total:   308 sets  (well under ~500 pool allocation)
 * ======================================================================== */

#define SDT_SLOT_PRE_ATTN_NORM    0
#define SDT_SLOT_Q_PROJ           1
#define SDT_SLOT_K_PROJ           2
#define SDT_SLOT_V_PROJ           3
#define SDT_SLOT_QK_NORM          4
#define SDT_SLOT_ROPE             5
#define SDT_SLOT_ATTENTION        6
#define SDT_SLOT_O_PROJ           7
#define SDT_SLOT_POST_ATTN_NORM   8
#define SDT_SLOT_ATTN_RESIDUAL    9
#define SDT_SLOT_PRE_FFN_NORM    10
#define SDT_SLOT_GATE_PROJ       11
#define SDT_SLOT_UP_PROJ         12
#define SDT_SLOT_GELU            13
#define SDT_SLOT_DOWN_PROJ       14
#define SDT_SLOT_POST_FFN_NORM   15
#define SDT_SLOT_FFN_RESIDUAL    16
#define SDT_KERNELS_PER_LAYER    17

/* Maximum layers supported by the static table. */
#define SDT_MAX_LAYERS           18

/**
 * Per-kernel descriptor set entry: the pipeline that owns the set and
 * the pre-allocated VkDescriptorSet handle.
 */
typedef struct {
    int              pipeline_idx;   /* Index into bd->pipelines[] */
    VkDescriptorSet  ds;             /* Pre-allocated, pre-populated set */
} sdt_entry_t;

/**
 * Static Descriptor Table: one row per layer, one column per kernel slot.
 *
 * layer_sets[layer][SDT_SLOT_*] gives the descriptor set to bind.
 *
 * Two extra sets handle the lm_head (final RMSNorm + vocab projection).
 */
typedef struct {
    sdt_entry_t layer_sets[SDT_MAX_LAYERS][SDT_KERNELS_PER_LAYER];

    /* LM-head (non-layered) */
    sdt_entry_t lmhead_norm;   /* Final RMSNorm before vocab projection */
    sdt_entry_t lmhead_proj;   /* Vocabulary (lm_head) projection */
    sdt_entry_t lmhead_argmax; /* Argmax over lm_head logits -> selected token ids */

    /* Backing descriptor pool (owns all sets above) */
    VkDescriptorPool pool;

    int num_layers;   /* Actual number of layers (≤ SDT_MAX_LAYERS) */
} VulkanStaticDescriptors;
#define PIPELINE_GELU_F32           12
#define PIPELINE_GELU_BF16          13
#define PIPELINE_VEC_ADD_F32        14
#define PIPELINE_ARGMAX_F32         15
#define PIPELINE_LMHEAD_DECODE_F32  16
#define NUM_PIPELINES               17

/**
 * Vulkan backend session data (P11-02 + P11-03 integration).
 *
 * Contains all Vulkan GPU-specific state for an inference session.
 * This structure is allocated and stored in inference_session_t.backend_data.
 *
 * Phase 11-02 Buffer Management:
 * - weight_buffers: Device-local storage for model weights
 * - kv_cache: Device-local KV cache with position tracking
 * - scratchpad: Ephemeral activation buffers (ping-pong between layers)
 * - input_ring: CPU→GPU token input ring buffer (timeline semaphore sync)
 * - output_ring: GPU→CPU logits output ring buffer (timeline semaphore sync)
 *
 * Phase 11-03 Compute Pipeline Abstraction:
 * - pipelines: Pre-created compute pipelines (embedding, attention, FFN, etc.)
 * - cmd_buffer: Command buffer for recording full forward pass
 */

typedef struct {
    /* Vulkan context handles (from vk_backend_context_t) */
    VkDevice device;
    VkPhysicalDevice phys_dev;
    VmaAllocator vma_allocator;
    VkQueue compute_queue;
    VkCommandPool cmd_pool;
    uint32_t queue_family_idx;

    /* P11-02 Buffer Management */
    vk_buffer_t *weight_buffers;        /* Array of 11 shared weight buffers (all layers packed per type) */
    size_t num_weight_buffers;          /* Number of weight buffers (always 11 for Gemma 3) */
    size_t *weight_layer_strides;       /* Byte stride per layer for each weight type [11] */
    
    vk_kv_cache_t kv_cache;             /* KV cache with position tracking */
    vk_gpu_scratchpad_t scratchpad;     /* Ephemeral activation buffers */
    vk_buffer_t attn_scores;            /* [num_heads × max_seq_len] attention score scratch */
    
    /* RoPE precomputed cos/sin cache (static, uploaded once during init).
     * Gemma3 uses two separate RoPE bases:
     *   Global layers (every 6th): rope_theta = 1,000,000
     *   Local  layers (all others): rope_local_base_freq = 10,000
     * Layout: [max_seq_len × head_dim] floats (stride = head_dim per position). */
    vk_buffer_t rope_cos_cache;         /* Global attention: theta = 1,000,000 */
    vk_buffer_t rope_sin_cache;         /* Global attention: theta = 1,000,000 */
    vk_buffer_t rope_cos_cache_local;   /* Local  attention: theta = 10,000     */
    vk_buffer_t rope_sin_cache_local;   /* Local  attention: theta = 10,000     */
    
    /* Final layer weights (non-layered, shared across model) */
    vk_buffer_t final_norm_weight;      /* [hidden_size] - final RMSNorm weight */
    vk_buffer_t embedding_weight;       /* [vocab_size × hidden_size] - for lm_head projection */
    vk_buffer_t lm_head_logits;         /* [vocab_size] - output buffer for lm_head */
    vk_buffer_t selected_token_ids;     /* [max_batch] int32 selected ids from GPU argmax */
    void *selected_token_ids_mapped;    /* Persistent map of selected_token_ids */
    
    vk_ring_buffer_t input_ring;        /* CPU→GPU token input ring */
    vk_ring_buffer_t output_ring;       /* GPU→CPU logits output ring */

    /* P11-03 Compute Pipelines */
    vk_compute_pipeline_t *pipelines;   /* Array of compute pipelines */
    size_t num_pipelines;               /* Number of pipelines */
    
    VkCommandBuffer cmd_buffer;         /* Command buffer for forward pass */
    
    /* Persistent Transfer Resources (Fix for GPU context loss / resource thrashing) */
    VkCommandBuffer transfer_cmd;       /* Dedicated command buffer for embeddings upload */
    vk_buffer_t embedding_staging;      /* Persistent staging buffer (host-visible) */
    void *embedding_staging_mapped;     /* Optional persistent map of embedding_staging */
    vk_buffer_t download_staging;       /* Persistent readback staging buffer (host-visible) */
    void *download_staging_mapped;      /* Optional persistent map of download_staging */
    VkFence transfer_fence;             /* Fence for transfer synchronization */
    VkFence compute_fence;              /* Fence for forward-pass compute completion */
    VkSemaphore transfer_to_compute_sem;/* Binary semaphore: transfer -> compute dependency */
    
    /* Synchronization (timeline semaphores from ring buffers) */
    uint64_t frame_counter;             /* Current frame counter for timeline sync */

    /* Static Descriptor Table (Vulkan 1.0 – no update-after-bind) */
    VulkanStaticDescriptors sdt;        /* Pre-populated descriptor sets for all kernels */

    /* Optional GPU timestamp profiling (SAPPHIRE_VK_PROFILE=1) */
    VkQueryPool timing_query_pool;
    uint32_t timing_query_capacity;
    uint32_t timing_query_last_count;
    uint32_t timing_kernel_first_query;
    uint32_t timing_kernel_query_count;
    float timestamp_period_ns;
    int timing_enabled;
    int timing_kernel_layer;
} backend_vulkan_session_data_t;

#ifdef __cplusplus
}
#endif

#endif // BACKEND_VULKAN_H

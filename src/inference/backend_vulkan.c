/**
 * @file backend_vulkan.c
 * @brief Vulkan backend implementation (P11-02 + P11-03 integrated).
 *
 * Integrates Vulkan device initialization with the inference backend abstraction.
 * Handles GPU device setup, P11-02 buffer management, P11-03 compute pipelines,
 * and graceful session lifecycle management.
 *
 * P11-02 Buffer Management:
 * - Weight buffers (device-local, loaded via staging)
 * - KV cache (device-local, position tracked)
 * - GPU scratchpad (ephemeral activation buffers)
 * - Input/Output ring buffers (timeline semaphore sync)
 *
 * P11-03 Compute Pipeline:
 * - Compute pipelines per kernel (embedding, attention, FFN, etc.)
 * - Descriptor set management (buffer bindings)
 * - Command buffer recording (full forward pass)
 * - Timeline semaphore synchronization (CPU↔GPU)
 */

#include "../../include/backend.h"
#include "../../include/inference.h"
#include "../../include/vulkan_backend.h"
#include "../../include/backend_vulkan.h"
#include "../../include/vk_buffers.h"
#include "../../include/vk_compute_pipeline.h"
#include "../../include/model_spec.h"
#include "../../include/gemma3_270m_config.h"
#include "../../include/llm_model.h"
#include "../../include/rope.h"
#include "../../include/tensor.h"
#include "../../include/kernels.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include "../../include/log.h"

/* Alignment helper macro */
#define ALIGN_UP(value, alignment) (((value) + (alignment) - 1) & ~((alignment) - 1))

/* Forward declarations of external transformer functions */
extern void sapphire_embed_lookup_batch(inference_session_t* session, const int* token_ids, int batch_size, float* dest);

/* ========================================================================
 * GPU Debug Probe (SAPPHIRE_DEBUG_GPU=1)
 *
 * Binary-searches the 18-layer pipeline by downloading intermediate tensors
 * at three checkpoints and comparing to CPU reference values:
 *
 *   CP1 = Embedding       (hidden[0] after upload, before compute)
 *   CP2 = Layer-0 output  (hidden[1] after single-layer isolated pass)
 *   CP3 = Logits          (top-5 argmax + value spread)
 *
 * Usage: SAPPHIRE_DEBUG_GPU=1 ./out/sapphire -m ... -n 1
 * ======================================================================== */

static int g_debug_gpu = -1;   /* -1 = uninitialised, 0 = off, 1 = on */

static int debug_gpu_enabled(void) {
    if (g_debug_gpu == -1) {
        const char *v = getenv("SAPPHIRE_DEBUG_GPU");
        g_debug_gpu = (v && v[0] == '1') ? 1 : 0;
    }
    return g_debug_gpu;
}

static int vk_timing_enabled(void) {
    const char *v = getenv("SAPPHIRE_VK_PROFILE");
    return (v && v[0] == '1') ? 1 : 0;
}

static int vk_kernel_timing_layer(void) {
    const char *v = getenv("SAPPHIRE_VK_PROFILE_LAYER");
    if (!v || v[0] == '\0') {
        return -1;
    }
    char *end = NULL;
    long parsed = strtol(v, &end, 10);
    if (end == v || *end != '\0' || parsed < 0 || parsed > 1024) {
        return -1;
    }
    return (int)parsed;
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

/** Compute float vector statistics (min, max, mean, L2 norm, NaN/Inf count). */
static void vec_stats(const float *v, int n, const char *tag) {
    if (!v || n <= 0) return;
    float vmin = v[0], vmax = v[0], sum = 0.0f, sumsq = 0.0f;
    int n_bad = 0;
    for (int i = 0; i < n; i++) {
        float x = v[i];
        if (x < vmin) vmin = x;
        if (x > vmax) vmax = x;
        sum  += x;
        sumsq += x * x;
        if (x != x || x == __builtin_inff() || x == -__builtin_inff()) n_bad++;
    }
    float mean = sum / (float)n;
    float l2   = __builtin_sqrtf(sumsq);
    LOG_INFO("[DBG] %s  n=%d  min=%.5f  max=%.5f  mean=%.5f  L2=%.3f  nan/inf=%d",
             tag, n, vmin, vmax, mean, l2, n_bad);
    LOG_INFO("[DBG] %s  [0]=%.6f  [1]=%.6f  [2]=%.6f  [3]=%.6f  [4]=%.6f",
             tag, v[0], v[1], v[2], v[3], v[4]);
}

/** Download `count` floats from a GPU buffer and call vec_stats(). */
static void probe_buf(backend_vulkan_session_data_t *bd, vk_buffer_t *buf,
                      int count, const char *tag) {
    if (!buf || !buf->buffer || count <= 0) {
        LOG_WARN("[DBG] %s: buffer is NULL", tag);
        return;
    }
    vk_transfer_ctx_t tctx = {
        .device = bd->device, .phys_dev = bd->phys_dev,
        .cmd_pool = bd->cmd_pool, .queue = bd->compute_queue
    };
    float *tmp = malloc((size_t)count * sizeof(float));
    if (!tmp) { LOG_ERROR("[DBG] probe_buf: malloc failed"); return; }

    vkQueueWaitIdle(bd->compute_queue);
    int rc = vk_buffer_download(&tctx, buf, (size_t)count * sizeof(float), tmp);
    if (rc != 0) { LOG_ERROR("[DBG] probe_buf: download failed for %s", tag); free(tmp); return; }
    vec_stats(tmp, count, tag);
    free(tmp);
}

/** Print top-5 argmax from logit vector and report min/max spread. */
static void probe_logits(const float *logits, int vocab_size, const char *tag) {
    if (!logits || vocab_size <= 0) return;
    /* Simple top-5 scan */
    int    top_idx[5] = {0};
    float  top_val[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
    float  lmin = logits[0], lmax = logits[0];
    double sum = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        float x = logits[i];
        if (x < lmin) lmin = x;
        if (x > lmax) lmax = x;
        sum += x;
        /* Insert into top-5 */
        for (int k = 0; k < 5; k++) {
            if (x > top_val[k]) {
                for (int j = 4; j > k; j--) { top_val[j] = top_val[j-1]; top_idx[j] = top_idx[j-1]; }
                top_val[k] = x; top_idx[k] = i;
                break;
            }
        }
    }
    float span = lmax - lmin;
    LOG_INFO("[DBG] %s  span=%.4f  min=%.4f  max=%.4f  mean=%.4f",
             tag, span, lmin, lmax, (float)(sum / vocab_size));
    LOG_INFO("[DBG] %s  TOP-5: tok=%d(%.3f) tok=%d(%.3f) tok=%d(%.3f) tok=%d(%.3f) tok=%d(%.3f)",
             tag, top_idx[0], top_val[0], top_idx[1], top_val[1],
                  top_idx[2], top_val[2], top_idx[3], top_val[3],
                  top_idx[4], top_val[4]);
}

/**
 * Debug helper: Download and dump buffer contents for verification.
 */
static void debug_dump_buffer(backend_vulkan_session_data_t *bd, vk_buffer_t *buf, const char *name, int count) {
    if (!buf || !buf->buffer || count <= 0) return;
    
    vk_transfer_ctx_t tctx = {
        .device = bd->device,
        .phys_dev = bd->phys_dev,
        .cmd_pool = bd->cmd_pool,
        .queue = bd->compute_queue
    };
    
    float *host_data = malloc(count * sizeof(float));
    if (!host_data) {
        LOG_ERROR("Failed to allocate debug buffer");
        return;
    }
    
    vkQueueWaitIdle(bd->compute_queue);  /* Force sync */
    
    int rc = vk_buffer_download(&tctx, buf, count * sizeof(float), host_data);
    if (rc != 0) {
        LOG_ERROR("Debug buffer download failed for %s", name);
        free(host_data);
        return;
    }
    
    LOG_DEBUG("DEBUG DUMP %s: [0]=%.6f [1]=%.6f [2]=%.6f [3]=%.6f [4]=%.6f [...]",
              name, host_data[0], host_data[1], host_data[2], host_data[3], host_data[4]);
    free(host_data);
}

/* ========================================================================
 * Weight Buffer Index Convention
 * ======================================================================== */

/* Weight buffer index convention: 13 weights per transformer layer (Gemma3 with post-norms) */
#define VK_WEIGHTS_PER_LAYER    13
#define VK_WGT_ATTN_NORM       0
#define VK_WGT_Q_PROJ          1
#define VK_WGT_K_PROJ          2
#define VK_WGT_V_PROJ          3
#define VK_WGT_QK_NORM_Q       4
#define VK_WGT_QK_NORM_K       5
#define VK_WGT_O_PROJ          6
#define VK_WGT_ATTN_NORM_POST  7  /* Gemma3 post-attention norm */
#define VK_WGT_FFN_NORM        8
#define VK_WGT_GATE_PROJ       9
#define VK_WGT_UP_PROJ         10
#define VK_WGT_DOWN_PROJ       11
#define VK_WGT_FFN_NORM_POST   12  /* Gemma3 post-FFN norm */

/**
 * Upload data to an existing device-local buffer via staging.
 *
 * @param ctx         Transfer context (device, queue, command pool)
 * @param host_data   Source data in host memory
 * @param data_size   Size in bytes
 * @param dst_buffer  Target device-local buffer (must already exist)
 *
 * @return 0 on success, -1 on error (errors logged internally)
 */
static int upload_to_device_buffer(
    const vk_transfer_ctx_t *ctx,
    const void *host_data,
    size_t data_size,
    vk_buffer_t *dst_buffer
) {
    if (!ctx || !host_data || !dst_buffer || !dst_buffer->buffer) {
        LOG_ERROR("Invalid parameters for buffer upload");
        return -1;
    }

    if (data_size == 0 || data_size > dst_buffer->size) {
        LOG_ERROR("Invalid data size %zu for buffer size %zu", data_size, dst_buffer->size);
        return -1;
    }

    /* Create temporary staging buffer (host-visible) */
    vk_buffer_t staging = {0};
    int rc = vk_buffer_create(
        ctx->device, ctx->phys_dev, data_size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging
    );
    if (rc != 0) {
        LOG_ERROR("Failed to create staging buffer");
        return -1;
    }

    /* Stage data to host-visible buffer */
    rc = vk_buffer_stage_data(ctx->device, &staging, host_data, data_size, 0);
    if (rc != 0) {
        LOG_ERROR("Failed to stage data");
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Allocate temporary command buffer for copy */
    VkCommandBufferAllocateInfo cmd_alloc = {0};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = ctx->cmd_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;

    VkCommandBuffer cmd_buf;
    VkResult vr = vkAllocateCommandBuffers(ctx->device, &cmd_alloc, &cmd_buf);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to allocate command buffer: %d", vr);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    VkCommandBufferBeginInfo begin_info = {0};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vr = vkBeginCommandBuffer(cmd_buf, &begin_info);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to begin command buffer: %d", vr);
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd_buf);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Record copy command */
    VkBufferCopy copy_region = {0};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = data_size;
    vkCmdCopyBuffer(cmd_buf, staging.buffer, dst_buffer->buffer, 1, &copy_region);

    vr = vkEndCommandBuffer(cmd_buf);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to end command buffer: %d", vr);
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd_buf);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Create fence for proper GPU synchronization (prevents RADV context loss) */
    VkFenceCreateInfo fence_info = {0};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = 0;  /* Create unsignaled */

    VkFence fence;
    vr = vkCreateFence(ctx->device, &fence_info, NULL, &fence);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to create fence: %d", vr);
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd_buf);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Submit with fence (ensures GPU signals completion) */
    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;

    vr = vkQueueSubmit(ctx->queue, 1, &submit_info, fence);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to submit copy command: %d", vr);
        vkDestroyFence(ctx->device, fence, NULL);
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd_buf);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Wait for fence (GPU explicitly signals when transfer is complete) */
    vr = vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to wait for fence: %d", vr);
        vkDestroyFence(ctx->device, fence, NULL);
        vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd_buf);
        vk_buffer_destroy(ctx->device, &staging);
        return -1;
    }

    /* Cleanup (safe now that GPU has signaled completion via fence) */
    vkDestroyFence(ctx->device, fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd_buf);
    vk_buffer_destroy(ctx->device, &staging);

    return 0;
}

/**
 * Upload a single tensor to GPU device-local memory via staging buffer.
 *
 * @param ctx      Transfer context (device, queue, command pool)
 * @param tensor   Source tensor with host data
 * @param out_buf  Output device-local buffer
 *
 * @return 0 on success, -1 on error (errors logged internally)
 */
static int upload_tensor_to_gpu(
    const vk_transfer_ctx_t *ctx,
    const tensor_t *tensor,
    vk_buffer_t *out_buf
) {
    if (!tensor) {
        LOG_ERROR("Cannot upload NULL tensor to GPU");
        return -1;
    }

    const void *data = tensor_data(tensor);
    size_t size = tensor_nbytes(tensor);

    if (!data || size == 0) {
        LOG_ERROR("Tensor has no data or zero size");
        return -1;
    }

    return vk_weight_buffer_load(ctx, data, size, out_buf);
}

/* =====================================================================
 * LAYER DISPATCH HELPERS & CONSTANTS
 *
 * These are used during session init (descriptor pre-population) and
 * forward pass (command buffer recording).
 * ===================================================================== */

/* Projection descriptor set slots (7 per layer in GEMV/GEMM pipeline) */
#define PROJ_SLOT_Q      0
#define PROJ_SLOT_K      1
#define PROJ_SLOT_V      2
#define PROJ_SLOT_O      3
#define PROJ_SLOT_GATE   4
#define PROJ_SLOT_UP     5
#define PROJ_SLOT_DOWN   6

/* RMSNorm descriptor set slots (2 per layer) */
#define NORM_SLOT_ATTN        0
#define NORM_SLOT_FFN         1
#define NORM_SLOT_ATTN_POST   2
#define NORM_SLOT_FFN_POST    3

/** Layer dispatch context (reduces parameter passing in helper functions). */
typedef struct {
    VkCommandBuffer cmd_buf;
    backend_vulkan_session_data_t *bd;
    const gemma3_270m_config_t *cfg;
    int layer;
    int batch_size;
    int seq_pos;
} layer_dispatch_ctx_t;

/* -----------------------------------------------------------------------
 * Forward declarations for static functions defined later in this file
 * ----------------------------------------------------------------------- */
static int  vulkan_prepopulate_descriptors(backend_vulkan_session_data_t *bd,
                                           const gemma3_270m_config_t   *cfg);
static void write_kv_to_cache(const layer_dispatch_ctx_t *ctx,
                               uint32_t num_kv, uint32_t hdim,
                               uint32_t max_pos);

/** Ceiling division for workgroup count computation. */
static inline uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

/** Get shared weight buffer for a component type (all layers packed). */
static inline vk_buffer_t* layer_weight_buffer(backend_vulkan_session_data_t *bd, int comp) {
    if (!bd->weight_buffers || comp < 0 || comp >= (int)bd->num_weight_buffers) return NULL;
    return &bd->weight_buffers[comp];
}

/** Calculate byte offset for a layer within a packed weight buffer. */
static inline size_t layer_weight_offset(backend_vulkan_session_data_t *bd, int comp, int layer) {
    if (!bd->weight_layer_strides || comp < 0 || comp >= VK_WEIGHTS_PER_LAYER) return 0;
    return (size_t)layer * bd->weight_layer_strides[comp];
}

/** Select F32 or BF16 pipeline variant. */
static inline int select_pipeline(int f32_idx, int bf16_idx, int is_bf16) {
    return is_bf16 ? bf16_idx : f32_idx;
}

/**
 * Build base push constants for a layer dispatch.
 * Caller overrides stride_0/stride_1/reserved for shader-specific semantics.
 */
/**
 * Build base push constants for a layer dispatch.
 * seq_pos  = start_pos + batch_size - 1  (last absolute position in batch)
 * start_pos = start_pos                  (first absolute position in batch)
 * Caller overrides stride_0/stride_1 for shader-specific semantics.
 */
static vk_kernel_push_constants_t build_push_constants(
    const gemma3_270m_config_t *cfg, int layer_idx, int batch_size, int start_pos) {
    vk_kernel_push_constants_t pc = {0};
    pc.layer_idx   = (uint32_t)layer_idx;
    pc.batch_size  = (uint32_t)batch_size;
    pc.seq_pos     = (uint32_t)(start_pos + batch_size - 1);  /* last position */
    pc.start_pos   = (uint32_t)start_pos;                      /* first position */
    pc.max_seq_len = (uint32_t)cfg->max_position_embeddings;
    pc.d_model     = (uint32_t)cfg->hidden_size;
    pc.num_heads   = (uint32_t)cfg->num_attention_heads;
    pc.head_dim    = (uint32_t)cfg->head_dim;
    pc.d_ff        = (uint32_t)cfg->intermediate_size;
    return pc;
}

/**
 * Vulkan backend: Initialize session (P11-02 + P11-03).
 *
 * 1. Initializes Vulkan device context
 * 2. Allocates P11-02 buffers (weights, KV cache, scratchpad, ring buffers)
 * 3. Creates P11-03 compute pipelines (stub for now, requires shaders)
 * 4. Allocates command buffer for forward pass recording
 *
 * On failure, returns -1 and caller falls back to CPU backend.
 *
 * @param session         Inference session (backend_data initially NULL)
 * @param spec            Model specification
 * @param max_context_len Maximum sequence length
 *
 * @return 0 on success, -1 if Vulkan unavailable (CPU fallback expected)
 */

/* -----------------------------------------------------------------------
 * Internal cleanup helper – torn-down a partially-initialised backend_data
 * struct.  All VkHandle and pointer fields are zero-initialised (calloc),
 * so every destroy call here is safe even if that resource was never created.
 * ----------------------------------------------------------------------- */
static void destroy_backend_data(backend_vulkan_session_data_t *bd) {
    if (!bd) return;
    VkDevice dev = bd->device;
    if (!dev) { free(bd); return; }
    if (bd->timing_query_pool != VK_NULL_HANDLE)
        vkDestroyQueryPool(dev, bd->timing_query_pool, NULL);
    if (bd->embedding_staging_mapped)
        vkUnmapMemory(dev, bd->embedding_staging.memory);
    if (bd->transfer_cmd)
        vkFreeCommandBuffers(dev, bd->cmd_pool, 1, &bd->transfer_cmd);
    if (bd->transfer_fence)   vkDestroyFence(dev, bd->transfer_fence, NULL);
    if (bd->compute_fence)    vkDestroyFence(dev, bd->compute_fence, NULL);
    if (bd->transfer_to_compute_sem)
        vkDestroySemaphore(dev, bd->transfer_to_compute_sem, NULL);
    vk_buffer_destroy(dev, &bd->embedding_staging);
    if (bd->cmd_buffer)
        vkFreeCommandBuffers(dev, bd->cmd_pool, 1, &bd->cmd_buffer);
    if (bd->pipelines) {
        for (size_t i = 0; i < bd->num_pipelines; i++)
            vk_pipeline_destroy(dev, &bd->pipelines[i]);
        free(bd->pipelines);
    }
    vk_ring_buffer_destroy(dev, &bd->output_ring);
    vk_ring_buffer_destroy(dev, &bd->input_ring);
    vk_scratchpad_destroy(dev, &bd->scratchpad);
    vk_buffer_destroy(dev, &bd->attn_scores);
    vk_buffer_destroy(dev, &bd->rope_cos_cache);
    vk_buffer_destroy(dev, &bd->rope_sin_cache);
    vk_buffer_destroy(dev, &bd->rope_cos_cache_local);
    vk_buffer_destroy(dev, &bd->rope_sin_cache_local);
    vk_buffer_destroy(dev, &bd->final_norm_weight);
    vk_buffer_destroy(dev, &bd->embedding_weight);
    vk_buffer_destroy(dev, &bd->selected_token_ids);
    vk_buffer_destroy(dev, &bd->lm_head_logits);
    vk_kv_cache_destroy(dev, &bd->kv_cache);
    if (bd->weight_buffers) {
        for (size_t i = 0; i < bd->num_weight_buffers; i++)
            vk_buffer_destroy(dev, &bd->weight_buffers[i]);
        free(bd->weight_buffers);
    }
    free(bd->weight_layer_strides);
    free(bd);
}

/* Allocate KV cache, scratchpad, attn scores, ring buffers, weight buffer arrays. */
static int init_vk_gpu_buffers(backend_vulkan_session_data_t *bd,
                                const gemma3_270m_config_t *cfg,
                                int max_context_len) {
    vk_kv_cache_cfg_t kv_cfg = {
        .num_layers   = cfg->num_hidden_layers,
        .num_kv_heads = cfg->num_key_value_heads,
        .max_seq_len  = max_context_len,
        .head_dim     = cfg->head_dim
    };
    if (vk_kv_cache_create(bd->device, bd->phys_dev, &kv_cfg, &bd->kv_cache) != 0) {
        LOG_ERROR("Failed to create KV cache"); return -1;
    }
    vk_scratchpad_cfg_t sc = {
        .max_batch_size = max_context_len,
        .d_model        = cfg->hidden_size,
        .d_ff           = cfg->intermediate_size,
        .num_kv_heads   = cfg->num_key_value_heads,
        .head_dim       = cfg->head_dim
    };
    if (vk_scratchpad_create(bd->device, bd->phys_dev, &sc, &bd->scratchpad) != 0) {
        LOG_ERROR("Failed to create GPU scratchpad"); return -1;
    }
    size_t scores_sz = (size_t)cfg->num_attention_heads * (size_t)max_context_len * sizeof(float);
    if (vk_buffer_create(bd->device, bd->phys_dev, scores_sz,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &bd->attn_scores) != 0) {
        LOG_ERROR("Failed to create attention scores buffer"); return -1;
    }
    if (vk_ring_buffer_create(bd->device, bd->phys_dev, 3,
                              256 * sizeof(int), &bd->input_ring) != 0) {
        LOG_ERROR("Failed to create input ring buffer"); return -1;
    }
    if (vk_ring_buffer_create(bd->device, bd->phys_dev, 3,
                              (size_t)cfg->vocab_size * sizeof(float),
                              &bd->output_ring) != 0) {
        LOG_ERROR("Failed to create output ring buffer"); return -1;
    }
    bd->num_weight_buffers  = VK_WEIGHTS_PER_LAYER;
    bd->weight_buffers      = calloc(VK_WEIGHTS_PER_LAYER, sizeof(vk_buffer_t));
    bd->weight_layer_strides = calloc(VK_WEIGHTS_PER_LAYER, sizeof(size_t));
    if (!bd->weight_buffers || !bd->weight_layer_strides) {
        LOG_ERROR("Failed to alloc weight buffer/stride arrays");
        free(bd->weight_buffers);
        free(bd->weight_layer_strides);
        bd->weight_buffers       = NULL;
        bd->weight_layer_strides = NULL;
        return -1;
    }
    return 0;
}

/* Allocate transfer cmd buffer, fences, semaphore, staging buffer, forward cmd buffer. */
static int init_vk_transfer_resources(backend_vulkan_session_data_t *bd) {
    const size_t STAGING_SZ = 256 * 1024 * 1024;
    VkCommandBufferAllocateInfo xa = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = bd->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    if (vkAllocateCommandBuffers(bd->device, &xa, &bd->transfer_cmd) != VK_SUCCESS) {
        LOG_ERROR("Failed to alloc transfer cmd buffer"); return -1;
    }
    VkFenceCreateInfo fi = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    if (vkCreateFence(bd->device, &fi, NULL, &bd->transfer_fence) != VK_SUCCESS) {
        LOG_ERROR("Failed to create transfer fence"); return -1;
    }
    if (vkCreateFence(bd->device, &fi, NULL, &bd->compute_fence) != VK_SUCCESS) {
        LOG_ERROR("Failed to create compute fence"); return -1;
    }
    VkSemaphoreCreateInfo si = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    if (vkCreateSemaphore(bd->device, &si, NULL,
                          &bd->transfer_to_compute_sem) != VK_SUCCESS) {
        LOG_ERROR("Failed to create transfer->compute semaphore"); return -1;
    }
    if (vk_buffer_create(bd->device, bd->phys_dev, STAGING_SZ,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &bd->embedding_staging) != 0) {
        LOG_ERROR("Failed to create staging buffer (%zu MB)", STAGING_SZ >> 20); return -1;
    }
    VkCommandBufferAllocateInfo ca = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = bd->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    if (vkAllocateCommandBuffers(bd->device, &ca, &bd->cmd_buffer) != VK_SUCCESS) {
        LOG_ERROR("Failed to alloc forward pass cmd buffer"); return -1;
    }
    LOG_DEBUG("Created transfer resources: staging=%zu MB", STAGING_SZ >> 20);
    return 0;
}

/* Return the weight tensor pointer for a layer given a component index. */
static const tensor_t *get_layer_weight_tensor(const model_layer_weights_t *lw, int comp) {
    switch (comp) {
        case VK_WGT_ATTN_NORM:      return lw->norm_attn_weight;
        case VK_WGT_Q_PROJ:         return lw->q_proj_weight;
        case VK_WGT_K_PROJ:         return lw->k_proj_weight;
        case VK_WGT_V_PROJ:         return lw->v_proj_weight;
        case VK_WGT_QK_NORM_Q:      return lw->q_norm_weight;
        case VK_WGT_QK_NORM_K:      return lw->k_norm_weight;
        case VK_WGT_O_PROJ:         return lw->out_proj_weight;
        case VK_WGT_ATTN_NORM_POST: return lw->norm_attn_post_weight;
        case VK_WGT_FFN_NORM:       return lw->norm_ffn_weight;
        case VK_WGT_GATE_PROJ:      return lw->gate_proj_weight;
        case VK_WGT_UP_PROJ:        return lw->up_proj_weight;
        case VK_WGT_DOWN_PROJ:      return lw->down_proj_weight;
        case VK_WGT_FFN_NORM_POST:  return lw->norm_ffn_post_weight;
        default:                    return NULL;
    }
}

/* Context passed to pack_weight_layer_data for logging (reduces param count). */
typedef struct { const gemma3_270m_config_t *cfg; const char *name; int layer; } weight_log_ctx_t;

/*
 * Convert and pack one layer's weight tensor into layer_dst (F32, pre-allocated).
 * Handles BF16→F32 conversion, QK_NORM_Q broadcast, alignment padding, and logging.
 */
static void pack_weight_layer_data(float *layer_dst, const tensor_t *t,
                                   size_t layer_stride, int comp,
                                   const weight_log_ctx_t *ctx) {
    const gemma3_270m_config_t *cfg = ctx->cfg;
    int layer = ctx->layer;
    if (!t) { memset(layer_dst, 0, layer_stride); return; }
    size_t num_src;
    const float *src_f32 = NULL;
    if (tensor_dtype(t) == DTYPE_F32) {
        num_src = tensor_nbytes(t) / sizeof(float);
        src_f32 = (const float *)tensor_data(t);
    } else {
        num_src = tensor_nbytes(t) / sizeof(uint16_t);
        bf16_to_f32_vec(layer_dst, (const uint16_t *)tensor_data(t), (int)num_src);
        src_f32 = layer_dst;
    }
    size_t payload_bytes;
    if (comp == VK_WGT_QK_NORM_Q && num_src == (size_t)cfg->head_dim) {
        for (int h = 0; h < cfg->num_attention_heads; h++)
            memcpy(layer_dst + h * cfg->head_dim, src_f32,
                   (size_t)cfg->head_dim * sizeof(float));
        payload_bytes = (size_t)cfg->num_attention_heads * cfg->head_dim * sizeof(float);
    } else {
        if (src_f32 != layer_dst) memcpy(layer_dst, src_f32, num_src * sizeof(float));
        payload_bytes = num_src * sizeof(float);
    }
    /* Zero alignment padding */
    if (layer_stride > payload_bytes) {
        float *pad = layer_dst + payload_bytes / sizeof(float);
        size_t npad = (layer_stride - payload_bytes) / sizeof(float);
        for (size_t pi = 0; pi < npad; ++pi) pad[pi] = 0.0f;
    }
    if (layer == 0) {
        const char *name = ctx->name;
        LOG_DEBUG("Upload %s L0: nelems=%zu [0]=%.6f [1]=%.6f [2]=%.6f",
                  name, payload_bytes / sizeof(float),
                  layer_dst[0], layer_dst[1], layer_dst[2]);
    }
}

/*
 * Upload packed float data to a device-local buffer in staging-buffer chunks.
 * Reuses bd->embedding_staging, bd->transfer_cmd, bd->transfer_fence.
 */
static int chunk_upload_to_device(backend_vulkan_session_data_t *bd,
                                  const float *packed_data,
                                  size_t total_size, vk_buffer_t *dst_buf) {
    const size_t STAGINGSZ = bd->embedding_staging.size;
    size_t remaining = total_size, uploaded = 0;
    VkCommandBufferBeginInfo beg = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VkSubmitInfo sub = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &bd->transfer_cmd
    };
    while (remaining > 0) {
        size_t chunk = (remaining > STAGINGSZ) ? STAGINGSZ : remaining;
        vkWaitForFences(bd->device, 1, &bd->transfer_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(bd->device, 1, &bd->transfer_fence);
        void *mapped = NULL;
        if (vkMapMemory(bd->device, bd->embedding_staging.memory,
                        0, chunk, 0, &mapped) != VK_SUCCESS) {
            LOG_ERROR("Failed to map staging buffer for chunk upload"); return -1;
        }
        memcpy(mapped, (const void *)(packed_data + uploaded / sizeof(float)), chunk);
        vkUnmapMemory(bd->device, bd->embedding_staging.memory);
        vkResetCommandBuffer(bd->transfer_cmd, 0);
        vkBeginCommandBuffer(bd->transfer_cmd, &beg);
        VkBufferCopy cp = { .srcOffset = 0, .dstOffset = uploaded, .size = chunk };
        vkCmdCopyBuffer(bd->transfer_cmd, bd->embedding_staging.buffer,
                        dst_buf->buffer, 1, &cp);
        vkEndCommandBuffer(bd->transfer_cmd);
        if (vkQueueSubmit(bd->compute_queue, 1, &sub, bd->transfer_fence) != VK_SUCCESS) {
            LOG_ERROR("Failed to submit upload chunk"); return -1;
        }
        vkWaitForFences(bd->device, 1, &bd->transfer_fence, VK_TRUE, UINT64_MAX);
        uploaded  += chunk;
        remaining -= chunk;
    }
    return 0;
}

/* Upload all layers of one weight type: find first tensor, allocate device buffer,
 * pack BF16→F32, upload in chunks. Returns 0 on success, -1 on error. */
static int upload_one_weight_type(backend_vulkan_session_data_t *bd,
                                  const llm_model_t *model,
                                  const gemma3_270m_config_t *cfg,
                                  int comp, const char *name,
                                  size_t offset_alignment) {
    int nl = cfg->num_hidden_layers;
    const tensor_t *first = NULL;
    size_t layer_stride = 0;
    for (int l = 0; l < nl; l++) {
        const tensor_t *t = get_layer_weight_tensor(&model->layers[l], comp);
        if (!t) continue;
        first = t;
        size_t ne = (tensor_dtype(t) == DTYPE_F32)
            ? tensor_nbytes(t) / sizeof(float)
            : tensor_nbytes(t) / sizeof(uint16_t);
        if (comp == VK_WGT_QK_NORM_Q && ne == (size_t)cfg->head_dim)
            ne = (size_t)cfg->num_attention_heads * cfg->head_dim;
        layer_stride = ALIGN_UP(ne * sizeof(float), offset_alignment);
        break;
    }
    if (!first) { LOG_DEBUG("Weight %s: no valid tensors found", name); return -1; }
    size_t total = layer_stride * nl;
    if (vk_buffer_create(bd->device, bd->phys_dev, total,
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &bd->weight_buffers[comp]) != 0) {
        LOG_ERROR("Failed to create device buffer for %s (%zu B)", name, total); return -1;
    }
    float *packed = (float *)malloc(total);
    if (!packed) { LOG_ERROR("Failed to alloc pack buffer for %s", name); return -1; }
    size_t sf = layer_stride / sizeof(float);
    for (int l = 0; l < nl; l++) {
        const tensor_t *t = get_layer_weight_tensor(&model->layers[l], comp);
        const weight_log_ctx_t _wlc = { .cfg = cfg, .name = name, .layer = l };
        pack_weight_layer_data(packed + (size_t)l * sf, t, layer_stride, comp, &_wlc);
    }
    int rc = chunk_upload_to_device(bd, packed, total, &bd->weight_buffers[comp]);
    free(packed);
    if (rc != 0) { LOG_ERROR("Failed to upload %s weight data", name); return -1; }
    bd->weight_layer_strides[comp] = layer_stride;
    LOG_DEBUG("Uploaded %s: %d layers × %zu B = %zu total", name, nl, layer_stride, total);
    return 0;
}

/* Upload all 13 per-layer weight types to the GPU. */
static int upload_vk_layer_weights(backend_vulkan_session_data_t *bd,
                                   const llm_model_t *model,
                                   const gemma3_270m_config_t *cfg,
                                   size_t offset_alignment) {
    static const struct { int comp; const char *name; } wt[] = {
        {VK_WGT_ATTN_NORM,      "attn_norm"},
        {VK_WGT_Q_PROJ,         "q_proj"},
        {VK_WGT_K_PROJ,         "k_proj"},
        {VK_WGT_V_PROJ,         "v_proj"},
        {VK_WGT_QK_NORM_Q,      "q_norm"},
        {VK_WGT_QK_NORM_K,      "k_norm"},
        {VK_WGT_O_PROJ,         "o_proj"},
        {VK_WGT_ATTN_NORM_POST, "attn_norm_post"},
        {VK_WGT_FFN_NORM,       "ffn_norm"},
        {VK_WGT_GATE_PROJ,      "gate_proj"},
        {VK_WGT_UP_PROJ,        "up_proj"},
        {VK_WGT_DOWN_PROJ,      "down_proj"},
        {VK_WGT_FFN_NORM_POST,  "ffn_norm_post"},
    };
    int fail = 0;
    for (size_t w = 0; w < sizeof(wt)/sizeof(wt[0]); w++) {
        if (upload_one_weight_type(bd, model, cfg, wt[w].comp, wt[w].name,
                                   offset_alignment) != 0)
            fail++;
    }
    if (fail > 0) { LOG_ERROR("%d weight upload(s) failed", fail); return -1; }
    if (vkQueueWaitIdle(bd->compute_queue) != VK_SUCCESS) {
        LOG_ERROR("vkQueueWaitIdle after weight uploads failed"); return -1;
    }
    LOG_INFO("Uploaded %d packed weight buffers to GPU", VK_WEIGHTS_PER_LAYER);
    return 0;
}

/* Upload one cos+sin RoPE buffer pair via staging. */
static int upload_rope_pair(backend_vulkan_session_data_t *bd,
                            const float *cos_h, const float *sin_h, size_t sz,
                            vk_buffer_t *cos_dst, vk_buffer_t *sin_dst) {
    VkCommandBufferBeginInfo beg = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VkSubmitInfo sub = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &bd->transfer_cmd
    };
    VkBufferCopy cp = { .srcOffset = 0, .dstOffset = 0, .size = sz };
    const float *bufs[2] = { cos_h, sin_h };
    vk_buffer_t *dsts[2]  = { cos_dst, sin_dst };
    for (int b = 0; b < 2; b++) {
        void *mapped = NULL;
        vkWaitForFences(bd->device, 1, &bd->transfer_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(bd->device, 1, &bd->transfer_fence);
        vkMapMemory(bd->device, bd->embedding_staging.memory, 0, sz, 0, &mapped);
        memcpy(mapped, bufs[b], sz);
        vkUnmapMemory(bd->device, bd->embedding_staging.memory);
        vkResetCommandBuffer(bd->transfer_cmd, 0);
        vkBeginCommandBuffer(bd->transfer_cmd, &beg);
        vkCmdCopyBuffer(bd->transfer_cmd, bd->embedding_staging.buffer,
                        dsts[b]->buffer, 1, &cp);
        vkEndCommandBuffer(bd->transfer_cmd);
        vkResetFences(bd->device, 1, &bd->transfer_fence);
        vkQueueSubmit(bd->compute_queue, 1, &sub, bd->transfer_fence);
        vkWaitForFences(bd->device, 1, &bd->transfer_fence, VK_TRUE, UINT64_MAX);
    }
    return 0;
}

/* Pre-compute and upload global+local RoPE cos/sin caches. */
static int upload_vk_rope_caches(backend_vulkan_session_data_t *bd,
                                 const gemma3_270m_config_t *cfg) {
    size_t sz = (size_t)bd->kv_cache.max_seq_len * (size_t)cfg->head_dim * sizeof(float);
    float *ch = malloc(sz), *sh = malloc(sz);
    if (!ch || !sh) {
        LOG_ERROR("Failed to alloc RoPE cache host buffers");
        free(ch); free(sh); return -1;
    }
    /* Global RoPE */
    if (rope_precompute_freqs(ch, sh, cfg->head_dim,
                              bd->kv_cache.max_seq_len, cfg->rope_theta) != 0 ||
        vk_buffer_create(bd->device, bd->phys_dev, sz,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &bd->rope_cos_cache) != 0 ||
        vk_buffer_create(bd->device, bd->phys_dev, sz,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &bd->rope_sin_cache) != 0) {
        LOG_ERROR("Failed to prepare global RoPE buffers");
        free(ch); free(sh); return -1;
    }
    upload_rope_pair(bd, ch, sh, sz, &bd->rope_cos_cache, &bd->rope_sin_cache);
    LOG_INFO("Uploaded global RoPE (theta=%.0f): %zu B", (double)cfg->rope_theta, sz);
    /* Local RoPE */
    if (rope_precompute_freqs(ch, sh, cfg->head_dim,
                              bd->kv_cache.max_seq_len, cfg->rope_local_base_freq) != 0 ||
        vk_buffer_create(bd->device, bd->phys_dev, sz,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &bd->rope_cos_cache_local) != 0 ||
        vk_buffer_create(bd->device, bd->phys_dev, sz,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &bd->rope_sin_cache_local) != 0) {
        LOG_ERROR("Failed to prepare local RoPE buffers");
        free(ch); free(sh); return -1;
    }
    upload_rope_pair(bd, ch, sh, sz, &bd->rope_cos_cache_local, &bd->rope_sin_cache_local);
    LOG_INFO("Uploaded local RoPE (theta=%.0f): %zu B", (double)cfg->rope_local_base_freq, sz);
    free(ch); free(sh);
    return 0;
}

/*
 * Convert tensor to F32 (BF16 → F32 or memcpy if already F32),
 * create a device-local buffer, and upload via chunked staging.
 */
static int upload_single_weight_f32(backend_vulkan_session_data_t *bd,
                                    const tensor_t *tensor,
                                    size_t f32_bytes, vk_buffer_t *dst_buf) {
    float *f32 = malloc(f32_bytes);
    if (!f32) { LOG_ERROR("Failed to alloc F32 conversion buffer"); return -1; }
    if (tensor_dtype(tensor) == DTYPE_BF16) {
        bf16_to_f32_vec(f32, (const uint16_t *)tensor_data(tensor),
                        (int)(f32_bytes / sizeof(float)));
    } else {
        memcpy(f32, tensor_data(tensor), f32_bytes);
    }
    if (vk_buffer_create(bd->device, bd->phys_dev, f32_bytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dst_buf) != 0) {
        LOG_ERROR("Failed to create device buffer (%zu B)", f32_bytes);
        free(f32); return -1;
    }
    int rc = chunk_upload_to_device(bd, f32, f32_bytes, dst_buf);
    free(f32);
    return rc;
}

/* Upload final_norm_weight and embedding_weight to the GPU. */
static int upload_vk_final_weights(backend_vulkan_session_data_t *bd,
                                   const llm_model_t *model,
                                   const gemma3_270m_config_t *cfg) {
    size_t norm_sz = (size_t)cfg->hidden_size * sizeof(float);
    if (upload_single_weight_f32(bd, model->norm_final_weight,
                                 norm_sz, &bd->final_norm_weight) != 0) {
        LOG_ERROR("Failed to upload final_norm_weight"); return -1;
    }
    size_t emb_sz = (size_t)cfg->vocab_size * (size_t)cfg->hidden_size * sizeof(float);
    if (upload_single_weight_f32(bd, model->embedding_weight,
                                 emb_sz, &bd->embedding_weight) != 0) {
        LOG_ERROR("Failed to upload embedding_weight"); return -1;
    }
    LOG_INFO("Uploaded final weights: norm=%zu B, embedding=%zu B", norm_sz, emb_sz);
    return 0;
}

/* Create all shader pipelines with pre-allocated descriptor sets (SDT). */
static int create_vk_compute_pipelines(backend_vulkan_session_data_t *bd,
                                       const gemma3_270m_config_t *cfg) {
    bd->pipelines = calloc(NUM_PIPELINES, sizeof(vk_compute_pipeline_t));
    if (!bd->pipelines) { LOG_ERROR("Failed to alloc pipeline array"); return -1; }
    bd->num_pipelines = NUM_PIPELINES;

    static const char *spv[NUM_PIPELINES] = {
        [PIPELINE_RMSNORM_F32]    = "src/kernels/backends/vulkan/shaders/rmsnorm_f32.comp.spv",
        [PIPELINE_RMSNORM_BF16]   = "src/kernels/backends/vulkan/shaders/rmsnorm_bf16.comp.spv",
        [PIPELINE_GEMV_F32]       = "src/kernels/backends/vulkan/shaders/gemv_f32.comp.spv",
        [PIPELINE_GEMV_BF16]      = "src/kernels/backends/vulkan/shaders/gemv_bf16.comp.spv",
        [PIPELINE_GEMM_F32]       = "src/kernels/backends/vulkan/shaders/gemm_f32.comp.spv",
        [PIPELINE_GEMM_BF16]      = "src/kernels/backends/vulkan/shaders/gemm_bf16.comp.spv",
        [PIPELINE_ROPE_F32]       = "src/kernels/backends/vulkan/shaders/rope_apply_f32.comp.spv",
        [PIPELINE_ROPE_BF16]      = "src/kernels/backends/vulkan/shaders/rope_apply_bf16.comp.spv",
        [PIPELINE_QK_NORM_F32]    = "src/kernels/backends/vulkan/shaders/qk_norm_f32.comp.spv",
        [PIPELINE_QK_NORM_BF16]   = "src/kernels/backends/vulkan/shaders/qk_norm_bf16.comp.spv",
        [PIPELINE_ATTENTION_F32]  = "src/kernels/backends/vulkan/shaders/attention_gqa_f32.comp.spv",
        [PIPELINE_ATTENTION_BF16] = "src/kernels/backends/vulkan/shaders/attention_gqa_bf16.comp.spv",
        [PIPELINE_GELU_F32]       = "src/kernels/backends/vulkan/shaders/gelu_f32.comp.spv",
        [PIPELINE_GELU_BF16]      = "src/kernels/backends/vulkan/shaders/gelu_bf16.comp.spv",
        [PIPELINE_VEC_ADD_F32]    = "src/kernels/backends/vulkan/shaders/vec_add_f32.comp.spv",
        [PIPELINE_ARGMAX_F32]     = "src/kernels/backends/vulkan/shaders/argmax_f32.comp.spv",
    };
    static const vk_desc_layout_type_t lt[NUM_PIPELINES] = {
        [PIPELINE_RMSNORM_F32] = VK_DESC_LAYOUT_PROJECTION,
        [PIPELINE_RMSNORM_BF16] = VK_DESC_LAYOUT_PROJECTION,
        [PIPELINE_GEMV_F32] = VK_DESC_LAYOUT_PROJECTION,
        [PIPELINE_GEMV_BF16] = VK_DESC_LAYOUT_PROJECTION,
        [PIPELINE_GEMM_F32] = VK_DESC_LAYOUT_PROJECTION,
        [PIPELINE_GEMM_BF16] = VK_DESC_LAYOUT_PROJECTION,
        [PIPELINE_ROPE_F32] = VK_DESC_LAYOUT_ATTENTION,
        [PIPELINE_ROPE_BF16] = VK_DESC_LAYOUT_ATTENTION,
        [PIPELINE_QK_NORM_F32] = VK_DESC_LAYOUT_ATTENTION,
        [PIPELINE_QK_NORM_BF16] = VK_DESC_LAYOUT_ATTENTION,
        [PIPELINE_ATTENTION_F32] = VK_DESC_LAYOUT_ATTENTION,
        [PIPELINE_ATTENTION_BF16] = VK_DESC_LAYOUT_ATTENTION,
        [PIPELINE_GELU_F32] = VK_DESC_LAYOUT_WEIGHT_ONLY,
        [PIPELINE_GELU_BF16] = VK_DESC_LAYOUT_WEIGHT_ONLY,
        [PIPELINE_VEC_ADD_F32] = VK_DESC_LAYOUT_PROJECTION,
        [PIPELINE_ARGMAX_F32] = VK_DESC_LAYOUT_CUSTOM,
    };
    static const uint32_t spl[NUM_PIPELINES] = {
        [PIPELINE_RMSNORM_F32] = 4, [PIPELINE_RMSNORM_BF16] = 4,
        [PIPELINE_GEMV_F32] = 7,    [PIPELINE_GEMV_BF16] = 7,
        [PIPELINE_GEMM_F32] = 0,    [PIPELINE_GEMM_BF16] = 0,
        [PIPELINE_ROPE_F32] = 1,    [PIPELINE_ROPE_BF16] = 1,
        [PIPELINE_QK_NORM_F32] = 1, [PIPELINE_QK_NORM_BF16] = 1,
        [PIPELINE_ATTENTION_F32] = 1, [PIPELINE_ATTENTION_BF16] = 1,
        [PIPELINE_GELU_F32] = 1,    [PIPELINE_GELU_BF16] = 1,
        [PIPELINE_VEC_ADD_F32] = 2, [PIPELINE_ARGMAX_F32] = 0,
    };
    static const vk_desc_binding_t argmax_b[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
    };
    int fail = 0;
    for (int i = 0; i < NUM_PIPELINES; i++) {
        vk_pipeline_config_t pc = {0};
        pc.shader_path        = spv[i];
        pc.entry_point        = "main";
        pc.layout_type        = lt[i];
        pc.push_constant_size = sizeof(vk_kernel_push_constants_t);
        if (i == PIPELINE_ARGMAX_F32) {
            pc.custom_bindings = argmax_b; pc.num_custom_bindings = 2;
        }
        pc.num_desc_sets = spl[i] * (uint32_t)cfg->num_hidden_layers;
        if (i == PIPELINE_RMSNORM_F32 || i == PIPELINE_RMSNORM_BF16) pc.num_desc_sets++;
        if (i == PIPELINE_GEMV_F32    || i == PIPELINE_GEMV_BF16)    pc.num_desc_sets++;
        if (i == PIPELINE_ARGMAX_F32) pc.num_desc_sets = 1;
        if (pc.num_desc_sets == 0)    pc.num_desc_sets = 1;
        if (vk_pipeline_create(bd->device, &pc, &bd->pipelines[i]) != 0) fail++;
    }
    if (fail == NUM_PIPELINES) LOG_WARN("All pipelines failed – run 'make shaders'");
    else if (fail > 0)         LOG_WARN("%d/%d pipelines failed", fail, NUM_PIPELINES);
    else                       LOG_INFO("All %d compute pipelines created", NUM_PIPELINES);
    return 0;
}

/* Alloc lm_head logits buffer and selected token id buffer. */
static int alloc_vk_inference_bufs(backend_vulkan_session_data_t *bd,
                                   const gemma3_270m_config_t *cfg,
                                   int max_context_len) {
    size_t logits_sz = (size_t)cfg->vocab_size * sizeof(float);
    if (vk_buffer_create(bd->device, bd->phys_dev, logits_sz,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                         &bd->lm_head_logits) != 0) {
        LOG_ERROR("Failed to alloc lm_head logits (%zu B)", logits_sz); return -1;
    }
    size_t sel_sz = (size_t)max_context_len * sizeof(int32_t);
    if (vk_buffer_create(bd->device, bd->phys_dev, sel_sz,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         &bd->selected_token_ids) != 0) {
        LOG_ERROR("Failed to alloc selected token ids (%zu B)", sel_sz); return -1;
    }
    LOG_INFO("Alloc'd inference buffers: logits=%zu B, sel_ids=%zu B", logits_sz, sel_sz);
    return 0;
}

/* Set up GPU timestamp query pool and persistently mapped staging buffer. */
static int setup_vk_timing_and_staging(backend_vulkan_session_data_t *bd,
                                       const gemma3_270m_config_t *cfg) {
    bd->timing_enabled            = vk_timing_enabled();
    bd->timing_query_pool         = VK_NULL_HANDLE;
    bd->timing_query_last_count   = 0;
    bd->timing_query_capacity     = 0;
    bd->timing_kernel_first_query = 0;
    bd->timing_kernel_query_count = 0;
    bd->timestamp_period_ns       = 1.0f;
    bd->timing_kernel_layer       = vk_kernel_timing_layer();

    if (bd->timing_enabled) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(bd->phys_dev, &props);
        bd->timestamp_period_ns = props.limits.timestampPeriod;
        uint32_t cap = (uint32_t)(2 * (cfg->num_hidden_layers + 1));
        if (bd->timing_kernel_layer >= 0) cap += 64;
        VkQueryPoolCreateInfo qp = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = cap
        };
        if (vkCreateQueryPool(bd->device, &qp, NULL,
                              &bd->timing_query_pool) == VK_SUCCESS) {
            bd->timing_query_capacity = cap;
            LOG_DEBUG("GPU timing enabled: %u queries, %.3f ns/tick",
                      cap, bd->timestamp_period_ns);
        } else {
            LOG_WARN("Failed to create timestamp query pool - timing disabled");
            bd->timing_enabled = 0;
        }
    }

    VkResult mr = vkMapMemory(bd->device, bd->embedding_staging.memory,
                              0, bd->embedding_staging.size, 0,
                              &bd->embedding_staging_mapped);
    if (mr != VK_SUCCESS || !bd->embedding_staging_mapped) {
        LOG_ERROR("Failed to create persistent staging mapping: %d", (int)mr);
        return -1;
    }
    bd->frame_counter = 0;
    return 0;
}

static int vulkan_session_init(inference_session_t* session, const model_spec_t* spec, int max_context_len) {
    if (!session || !spec) {
        LOG_ERROR("session or spec is NULL");
        return -1;
    }

    vk_backend_context_t *vk_ctx = NULL;
    int rc = vk_backend_init(&vk_ctx);
    if (rc != 0 || !vk_ctx) {
        LOG_WARN("Vulkan backend initialization failed, will fall back to CPU backend");
        return -1;
    }

    backend_vulkan_session_data_t *bd = calloc(1, sizeof(backend_vulkan_session_data_t));
    if (!bd) {
        LOG_ERROR("Failed to allocate Vulkan backend session data");
        vk_backend_shutdown(vk_ctx);
        return -1;
    }

    bd->device          = vk_backend_get_device(vk_ctx);
    bd->phys_dev        = vk_backend_get_physical_device(vk_ctx);
    bd->compute_queue   = vk_backend_get_compute_queue(vk_ctx);
    bd->cmd_pool        = vk_backend_get_command_pool(vk_ctx);
    bd->queue_family_idx = vk_backend_get_compute_queue_family_idx(vk_ctx);

    if (!bd->device || !bd->phys_dev || !bd->compute_queue || !bd->cmd_pool) {
        LOG_ERROR("Failed to extract Vulkan handles from context");
        free(bd);
        vk_backend_shutdown(vk_ctx);
        return -1;
    }

    const gemma3_270m_config_t *cfg = (const gemma3_270m_config_t *)spec->variant_config;
    if (!cfg) {
        LOG_ERROR("Model config is NULL in vulkan_session_init");
        free(bd);
        vk_backend_shutdown(vk_ctx);
        return -1;
    }

    const llm_model_t *model = (const llm_model_t *)spec->llm_model;
    if (!model || !model->layers) {
        LOG_WARN("Model weights not loaded; Vulkan backend requires weights in llm_model");
        free(bd);
        vk_backend_shutdown(vk_ctx);
        return -1;
    }

    size_t offset_alignment = vk_backend_get_min_storage_buffer_offset_alignment(vk_ctx);
    LOG_DEBUG("Storage buffer offset alignment: %zu bytes", offset_alignment);

    if (init_vk_gpu_buffers(bd, cfg, max_context_len)             != 0) goto fail;
    if (init_vk_transfer_resources(bd)                            != 0) goto fail;
    if (upload_vk_layer_weights(bd, model, cfg, offset_alignment) != 0) goto fail;
    if (upload_vk_rope_caches(bd, cfg)                            != 0) goto fail;
    if (upload_vk_final_weights(bd, model, cfg)                   != 0) goto fail;
    if (create_vk_compute_pipelines(bd, cfg)                      != 0) goto fail;
    if (alloc_vk_inference_bufs(bd, cfg, max_context_len)         != 0) goto fail;
    if (vulkan_prepopulate_descriptors(bd, cfg)                   != 0) goto fail;
    if (setup_vk_timing_and_staging(bd, cfg)                      != 0) goto fail;

    session->backend_data = (void *)bd;

    LOG_DEBUG("Vulkan config: hid=%d ff=%d vocab=%d layers=%d hdim=%d",
              cfg->hidden_size, cfg->intermediate_size, cfg->vocab_size,
              cfg->num_hidden_layers, cfg->head_dim);
    LOG_INFO("Vulkan backend session initialized (P11-02 buffers, P11-04 pipelines)");
    (void)vk_ctx;  /* vk_ctx deferred cleanup; refactor in later phase */
    return 0;

fail:
    destroy_backend_data(bd);
    vk_backend_shutdown(vk_ctx);
    return -1;
}

/**
 * Vulkan backend: Destroy session (P11-02 + P11-03 cleanup).
 *
 * Cleans up all allocated resources in reverse order:
 * 1. Command buffer
 * 2. Compute pipelines (if any)
 * 3. Ring buffers
 * 4. GPU scratchpad
 * 5. KV cache
 * 6. Weight buffers
 * 7. Vulkan context
 *
 * @param session Inference session with Vulkan backend data
 */
static void vulkan_session_destroy(inference_session_t* session) {
    if (!session || !session->backend_data) {
        return;
    }

    backend_vulkan_session_data_t *backend_data = (backend_vulkan_session_data_t *)session->backend_data;

    if (backend_data->timing_query_pool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(backend_data->device, backend_data->timing_query_pool, NULL);
        backend_data->timing_query_pool = VK_NULL_HANDLE;
    }

    /* Free persistent transfer resources */
    if (backend_data->embedding_staging_mapped) {
        vkUnmapMemory(backend_data->device, backend_data->embedding_staging.memory);
        backend_data->embedding_staging_mapped = NULL;
    }
    if (backend_data->transfer_cmd) {
        vkFreeCommandBuffers(backend_data->device, backend_data->cmd_pool, 1, &backend_data->transfer_cmd);
    }
    if (backend_data->transfer_fence) {
        vkDestroyFence(backend_data->device, backend_data->transfer_fence, NULL);
    }
    if (backend_data->compute_fence) {
        vkDestroyFence(backend_data->device, backend_data->compute_fence, NULL);
    }
    if (backend_data->transfer_to_compute_sem) {
        vkDestroySemaphore(backend_data->device, backend_data->transfer_to_compute_sem, NULL);
    }
    vk_buffer_destroy(backend_data->device, &backend_data->embedding_staging);

    /* Free command buffer */
    if (backend_data->cmd_buffer) {
        vkFreeCommandBuffers(backend_data->device, backend_data->cmd_pool, 1, &backend_data->cmd_buffer);
    }

    /* Destroy compute pipelines */
    if (backend_data->pipelines) {
        for (size_t i = 0; i < backend_data->num_pipelines; i++) {
            vk_pipeline_destroy(backend_data->device, &backend_data->pipelines[i]);
        }
        free(backend_data->pipelines);
    }

    /* Destroy ring buffers */
    vk_ring_buffer_destroy(backend_data->device, &backend_data->output_ring);
    vk_ring_buffer_destroy(backend_data->device, &backend_data->input_ring);

    /* Destroy scratchpad */
    vk_scratchpad_destroy(backend_data->device, &backend_data->scratchpad);

    /* Destroy attention scores buffer */
    vk_buffer_destroy(backend_data->device, &backend_data->attn_scores);
    
    /* Destroy RoPE cos/sin cache buffers (global and local) */
    vk_buffer_destroy(backend_data->device, &backend_data->rope_cos_cache);
    vk_buffer_destroy(backend_data->device, &backend_data->rope_sin_cache);
    vk_buffer_destroy(backend_data->device, &backend_data->rope_cos_cache_local);
    vk_buffer_destroy(backend_data->device, &backend_data->rope_sin_cache_local);
    
    /* Destroy final layer weight buffers */
    vk_buffer_destroy(backend_data->device, &backend_data->final_norm_weight);
    vk_buffer_destroy(backend_data->device, &backend_data->embedding_weight);
    vk_buffer_destroy(backend_data->device, &backend_data->selected_token_ids);
    vk_buffer_destroy(backend_data->device, &backend_data->lm_head_logits);

    /* Destroy KV cache */
    vk_kv_cache_destroy(backend_data->device, &backend_data->kv_cache);

    /* Destroy weight buffers */
    if (backend_data->weight_buffers) {
        for (size_t i = 0; i < backend_data->num_weight_buffers; i++) {
            vk_buffer_destroy(backend_data->device, &backend_data->weight_buffers[i]);
        }
        free(backend_data->weight_buffers);
    }
    
    /* Free weight stride array */
    if (backend_data->weight_layer_strides) {
        free(backend_data->weight_layer_strides);
    }

    /* Free backend data */
    free(backend_data);
    session->backend_data = NULL;

    /* NOTE: vk_backend_context_t cleanup deferred (leaked in init); 
     * proper refactoring needed to store context pointer in backend_data */
    LOG_INFO("Vulkan backend session destroyed (P11-02 buffers freed)");
}

/* =====================================================================
 * P11-05: Vulkan Forward Batch Orchestration
 *
 * Records and submits GPU command buffers for the full transformer
 * forward pass. Each layer records 17 compute dispatches mapped to
 * pre-populated descriptor sets in VulkanStaticDescriptors.
 * ===================================================================== */

/* ------------------------------------------------------------------
 * SDT allocation helper: allocate one VkDescriptorSet from a pipeline's
 * pre-allocated pool and return it.  We use a per-pipeline cursor.
 * ------------------------------------------------------------------ */
static uint32_t g_sdt_cursor[NUM_PIPELINES];  /* allocation cursors, reset before prepopulate */

static VkDescriptorSet sdt_alloc_ds(backend_vulkan_session_data_t *bd, int pipeline_idx) {
    vk_compute_pipeline_t *p = &bd->pipelines[pipeline_idx];
    uint32_t idx = g_sdt_cursor[pipeline_idx]++;
    if (idx >= p->num_desc_sets) {
        LOG_ERROR("SDT: descriptor set overflow for pipeline %d (cursor=%u, max=%u)",
                  pipeline_idx, idx, p->num_desc_sets);
        return VK_NULL_HANDLE;
    }
    return p->desc_sets[idx];
}

/* Shorthand: write one STORAGE_BUFFER binding into a descriptor set. */
static void sdt_write_buf(VkDevice device, VkDescriptorSet ds, uint32_t binding,
                           VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
    VkDescriptorBufferInfo buf_info = { .buffer = buffer, .offset = offset, .range = range };
    VkWriteDescriptorSet w = {
        .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet           = ds,
        .dstBinding       = binding,
        .dstArrayElement  = 0,
        .descriptorCount  = 1,
        .descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo      = &buf_info
    };
    vkUpdateDescriptorSets(device, 1, &w, 0, NULL);
}

/**
 * vulkan_prepopulate_descriptors() – Populate the Static Descriptor Table.
 *
 * Called once during session init, after all GPU buffers are allocated and
 * pipelines are created.  Allocates one unique VkDescriptorSet per kernel
 * per layer (and 2 more for lm_head) and writes ALL buffer bindings,
 * including activation (scratchpad) buffers.  Since scratchpad VkBuffer
 * handles never change (only data changes), this is safe under Vulkan 1.0.
 *
 * No vkUpdateDescriptorSets calls are made during command recording.
 *
 * @param bd   Backend session data (pipelines, scratchpad, weights, caches)
 * @param cfg  Model configuration
 * @return 0 on success, -1 on error (logged)
 */
/*
 * Populate all SDT descriptor slots for a single transformer layer.
 * Called from vulkan_prepopulate_descriptors() once per layer.
 */
/* Setup SDT slots: PRE_ATTN_NORM, Q_PROJ, K_PROJ, V_PROJ, QK_NORM */
static void setup_qkv_sdt_slots(VkDevice dev, backend_vulkan_session_data_t *bd,
    vk_gpu_scratchpad_t *sp, int L, sdt_entry_t *S)
{
    const int PI_NORM  = PIPELINE_RMSNORM_F32;
    const int PI_GEMV  = PIPELINE_GEMV_F32;
    const int PI_QKNRM = PIPELINE_QK_NORM_F32;
    VkBuffer   wgt_attn_norm   = layer_weight_buffer(bd, VK_WGT_ATTN_NORM)->buffer;
    VkDeviceSize off_attn_norm = layer_weight_offset(bd, VK_WGT_ATTN_NORM, L);
    VkBuffer   wgt_q           = layer_weight_buffer(bd, VK_WGT_Q_PROJ)->buffer;
    VkDeviceSize off_q         = layer_weight_offset(bd, VK_WGT_Q_PROJ, L);
    VkBuffer   wgt_k           = layer_weight_buffer(bd, VK_WGT_K_PROJ)->buffer;
    VkDeviceSize off_k         = layer_weight_offset(bd, VK_WGT_K_PROJ, L);
    VkBuffer   wgt_v           = layer_weight_buffer(bd, VK_WGT_V_PROJ)->buffer;
    VkDeviceSize off_v         = layer_weight_offset(bd, VK_WGT_V_PROJ, L);
    VkBuffer   wgt_qnorm       = layer_weight_buffer(bd, VK_WGT_QK_NORM_Q)->buffer;
    VkDeviceSize off_qnorm     = layer_weight_offset(bd, VK_WGT_QK_NORM_Q, L);
    VkBuffer   wgt_knorm       = layer_weight_buffer(bd, VK_WGT_QK_NORM_K)->buffer;
    VkDeviceSize off_knorm     = layer_weight_offset(bd, VK_WGT_QK_NORM_K, L);
    VkBuffer input_buf = sp->layer_hidden[L % 2].buffer;
    S[SDT_SLOT_PRE_ATTN_NORM].pipeline_idx = PI_NORM;
    S[SDT_SLOT_PRE_ATTN_NORM].ds = sdt_alloc_ds(bd, PI_NORM);
    sdt_write_buf(dev, S[SDT_SLOT_PRE_ATTN_NORM].ds, 0, input_buf, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_PRE_ATTN_NORM].ds, 1, wgt_attn_norm, off_attn_norm, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_PRE_ATTN_NORM].ds, 2, sp->norm_buf.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_Q_PROJ].pipeline_idx = PI_GEMV;
    S[SDT_SLOT_Q_PROJ].ds = sdt_alloc_ds(bd, PI_GEMV);
    sdt_write_buf(dev, S[SDT_SLOT_Q_PROJ].ds, 0, sp->norm_buf.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_Q_PROJ].ds, 1, wgt_q, off_q, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_Q_PROJ].ds, 2, sp->q_proj.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_K_PROJ].pipeline_idx = PI_GEMV;
    S[SDT_SLOT_K_PROJ].ds = sdt_alloc_ds(bd, PI_GEMV);
    sdt_write_buf(dev, S[SDT_SLOT_K_PROJ].ds, 0, sp->norm_buf.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_K_PROJ].ds, 1, wgt_k, off_k, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_K_PROJ].ds, 2, sp->k_proj.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_V_PROJ].pipeline_idx = PI_GEMV;
    S[SDT_SLOT_V_PROJ].ds = sdt_alloc_ds(bd, PI_GEMV);
    sdt_write_buf(dev, S[SDT_SLOT_V_PROJ].ds, 0, sp->norm_buf.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_V_PROJ].ds, 1, wgt_v, off_v, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_V_PROJ].ds, 2, sp->v_proj.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_QK_NORM].pipeline_idx = PI_QKNRM;
    S[SDT_SLOT_QK_NORM].ds = sdt_alloc_ds(bd, PI_QKNRM);
    sdt_write_buf(dev, S[SDT_SLOT_QK_NORM].ds, 0, sp->q_proj.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_QK_NORM].ds, 1, sp->k_proj.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_QK_NORM].ds, 2, wgt_qnorm, off_qnorm, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_QK_NORM].ds, 3, wgt_knorm, off_knorm, VK_WHOLE_SIZE);
}

/* Setup SDT slots: ROPE, ATTENTION, O_PROJ, POST_ATTN_NORM, ATTN_RESIDUAL */
static void setup_attn_sdt_slots(VkDevice dev, backend_vulkan_session_data_t *bd,
    vk_gpu_scratchpad_t *sp, int L, sdt_entry_t *S)
{
    const int PI_GEMV  = PIPELINE_GEMV_F32;
    const int PI_ROPE  = PIPELINE_ROPE_F32;
    const int PI_ATTN  = PIPELINE_ATTENTION_F32;
    const int PI_NORM  = PIPELINE_RMSNORM_F32;
    const int PI_ADD   = PIPELINE_VEC_ADD_F32;
    VkBuffer   wgt_o         = layer_weight_buffer(bd, VK_WGT_O_PROJ)->buffer;
    VkDeviceSize off_o       = layer_weight_offset(bd, VK_WGT_O_PROJ, L);
    VkBuffer   wgt_attn_post   = layer_weight_buffer(bd, VK_WGT_ATTN_NORM_POST)->buffer;
    VkDeviceSize off_attn_post = layer_weight_offset(bd, VK_WGT_ATTN_NORM_POST, L);
    bool is_global_rope = ((L + 1) % 6 == 0);
    VkBuffer cos_buf = is_global_rope ? bd->rope_cos_cache.buffer : bd->rope_cos_cache_local.buffer;
    VkBuffer sin_buf = is_global_rope ? bd->rope_sin_cache.buffer : bd->rope_sin_cache_local.buffer;
    S[SDT_SLOT_ROPE].pipeline_idx = PI_ROPE;
    S[SDT_SLOT_ROPE].ds = sdt_alloc_ds(bd, PI_ROPE);
    sdt_write_buf(dev, S[SDT_SLOT_ROPE].ds, 0, sp->q_proj.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_ROPE].ds, 1, sp->k_proj.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_ROPE].ds, 2, cos_buf, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_ROPE].ds, 3, sin_buf, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_ATTENTION].pipeline_idx = PI_ATTN;
    S[SDT_SLOT_ATTENTION].ds = sdt_alloc_ds(bd, PI_ATTN);
    sdt_write_buf(dev, S[SDT_SLOT_ATTENTION].ds, 0, sp->q_proj.buffer,       0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_ATTENTION].ds, 1, bd->kv_cache.kv_data.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_ATTENTION].ds, 2, sp->attn_output.buffer,   0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_ATTENTION].ds, 3, bd->attn_scores.buffer,   0, VK_WHOLE_SIZE);
    S[SDT_SLOT_O_PROJ].pipeline_idx = PI_GEMV;
    S[SDT_SLOT_O_PROJ].ds = sdt_alloc_ds(bd, PI_GEMV);
    sdt_write_buf(dev, S[SDT_SLOT_O_PROJ].ds, 0, sp->attn_output.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_O_PROJ].ds, 1, wgt_o, off_o, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_O_PROJ].ds, 2, sp->norm_buf.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_POST_ATTN_NORM].pipeline_idx = PI_NORM;
    S[SDT_SLOT_POST_ATTN_NORM].ds = sdt_alloc_ds(bd, PI_NORM);
    sdt_write_buf(dev, S[SDT_SLOT_POST_ATTN_NORM].ds, 0, sp->norm_buf.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_POST_ATTN_NORM].ds, 1, wgt_attn_post, off_attn_post, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_POST_ATTN_NORM].ds, 2, sp->attn_output.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_ATTN_RESIDUAL].pipeline_idx = PI_ADD;
    S[SDT_SLOT_ATTN_RESIDUAL].ds = sdt_alloc_ds(bd, PI_ADD);
    sdt_write_buf(dev, S[SDT_SLOT_ATTN_RESIDUAL].ds, 0, sp->attn_output.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_ATTN_RESIDUAL].ds, 1, sp->residual.buffer,    0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_ATTN_RESIDUAL].ds, 2, sp->residual.buffer,    0, VK_WHOLE_SIZE);
}

/* Setup SDT slots: PRE_FFN_NORM, GATE_PROJ, UP_PROJ, GELU */
static void setup_ffn_in_sdt_slots(VkDevice dev, backend_vulkan_session_data_t *bd,
    vk_gpu_scratchpad_t *sp, int L, sdt_entry_t *S)
{
    const int PI_GEMV  = PIPELINE_GEMV_F32;
    const int PI_NORM  = PIPELINE_RMSNORM_F32;
    const int PI_GELU  = PIPELINE_GELU_F32;
    VkBuffer   wgt_ffn_norm  = layer_weight_buffer(bd, VK_WGT_FFN_NORM)->buffer;
    VkDeviceSize off_ffn_norm = layer_weight_offset(bd, VK_WGT_FFN_NORM, L);
    VkBuffer   wgt_gate      = layer_weight_buffer(bd, VK_WGT_GATE_PROJ)->buffer;
    VkDeviceSize off_gate    = layer_weight_offset(bd, VK_WGT_GATE_PROJ, L);
    VkBuffer   wgt_up        = layer_weight_buffer(bd, VK_WGT_UP_PROJ)->buffer;
    VkDeviceSize off_up      = layer_weight_offset(bd, VK_WGT_UP_PROJ, L);
    S[SDT_SLOT_PRE_FFN_NORM].pipeline_idx = PI_NORM;
    S[SDT_SLOT_PRE_FFN_NORM].ds = sdt_alloc_ds(bd, PI_NORM);
    sdt_write_buf(dev, S[SDT_SLOT_PRE_FFN_NORM].ds, 0, sp->residual.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_PRE_FFN_NORM].ds, 1, wgt_ffn_norm, off_ffn_norm, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_PRE_FFN_NORM].ds, 2, sp->ffn_gate.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_GATE_PROJ].pipeline_idx = PI_GEMV;
    S[SDT_SLOT_GATE_PROJ].ds = sdt_alloc_ds(bd, PI_GEMV);
    sdt_write_buf(dev, S[SDT_SLOT_GATE_PROJ].ds, 0, sp->ffn_gate.buffer,  0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_GATE_PROJ].ds, 1, wgt_gate, off_gate,   VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_GATE_PROJ].ds, 2, sp->ffn_value.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_UP_PROJ].pipeline_idx = PI_GEMV;
    S[SDT_SLOT_UP_PROJ].ds = sdt_alloc_ds(bd, PI_GEMV);
    sdt_write_buf(dev, S[SDT_SLOT_UP_PROJ].ds, 0, sp->ffn_gate.buffer,    0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_UP_PROJ].ds, 1, wgt_up, off_up,         VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_UP_PROJ].ds, 2, sp->attn_output.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_GELU].pipeline_idx = PI_GELU;
    S[SDT_SLOT_GELU].ds = sdt_alloc_ds(bd, PI_GELU);
    sdt_write_buf(dev, S[SDT_SLOT_GELU].ds, 0, sp->ffn_value.buffer,   0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_GELU].ds, 1, sp->attn_output.buffer, 0, VK_WHOLE_SIZE);
}

/* Setup SDT slots: DOWN_PROJ, POST_FFN_NORM, FFN_RESIDUAL */
static void setup_ffn_out_sdt_slots(VkDevice dev, backend_vulkan_session_data_t *bd,
    vk_gpu_scratchpad_t *sp, int L, sdt_entry_t *S)
{
    const int PI_GEMV  = PIPELINE_GEMV_F32;
    const int PI_NORM  = PIPELINE_RMSNORM_F32;
    const int PI_ADD   = PIPELINE_VEC_ADD_F32;
    VkBuffer   wgt_down     = layer_weight_buffer(bd, VK_WGT_DOWN_PROJ)->buffer;
    VkDeviceSize off_down   = layer_weight_offset(bd, VK_WGT_DOWN_PROJ, L);
    VkBuffer   wgt_ffn_post = layer_weight_buffer(bd, VK_WGT_FFN_NORM_POST)->buffer;
    VkDeviceSize off_ffn_post = layer_weight_offset(bd, VK_WGT_FFN_NORM_POST, L);
    VkBuffer output_buf = sp->layer_hidden[(L + 1) % 2].buffer;
    S[SDT_SLOT_DOWN_PROJ].pipeline_idx = PI_GEMV;
    S[SDT_SLOT_DOWN_PROJ].ds = sdt_alloc_ds(bd, PI_GEMV);
    sdt_write_buf(dev, S[SDT_SLOT_DOWN_PROJ].ds, 0, sp->ffn_value.buffer,   0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_DOWN_PROJ].ds, 1, wgt_down, off_down,     VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_DOWN_PROJ].ds, 2, sp->attn_output.buffer, 0, VK_WHOLE_SIZE);
    S[SDT_SLOT_POST_FFN_NORM].pipeline_idx = PI_NORM;
    S[SDT_SLOT_POST_FFN_NORM].ds = sdt_alloc_ds(bd, PI_NORM);
    sdt_write_buf(dev, S[SDT_SLOT_POST_FFN_NORM].ds, 0, sp->attn_output.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_POST_FFN_NORM].ds, 1, wgt_ffn_post, off_ffn_post, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_POST_FFN_NORM].ds, 2, sp->ffn_gate.buffer,    0, VK_WHOLE_SIZE);
    S[SDT_SLOT_FFN_RESIDUAL].pipeline_idx = PI_ADD;
    S[SDT_SLOT_FFN_RESIDUAL].ds = sdt_alloc_ds(bd, PI_ADD);
    sdt_write_buf(dev, S[SDT_SLOT_FFN_RESIDUAL].ds, 0, sp->ffn_gate.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_FFN_RESIDUAL].ds, 1, sp->residual.buffer, 0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, S[SDT_SLOT_FFN_RESIDUAL].ds, 2, output_buf,          0, VK_WHOLE_SIZE);
}

static void populate_layer_descriptor_slots(
    backend_vulkan_session_data_t *bd,
    const gemma3_270m_config_t *cfg,
    vk_gpu_scratchpad_t *sp,
    int L,
    sdt_entry_t *S)
{
    VkDevice dev = bd->device;
    (void)cfg;
    setup_qkv_sdt_slots    (dev, bd, sp, L, S);
    setup_attn_sdt_slots   (dev, bd, sp, L, S);
    setup_ffn_in_sdt_slots (dev, bd, sp, L, S);
    setup_ffn_out_sdt_slots(dev, bd, sp, L, S);
}

static int vulkan_prepopulate_descriptors(backend_vulkan_session_data_t *bd,
                                          const gemma3_270m_config_t *cfg) {
    int num_layers = cfg->num_hidden_layers;

    if (num_layers > SDT_MAX_LAYERS) {
        LOG_ERROR("num_layers=%d exceeds SDT_MAX_LAYERS=%d", num_layers, SDT_MAX_LAYERS);
        return -1;
    }

    /* Reset allocation cursors */
    memset(g_sdt_cursor, 0, sizeof(g_sdt_cursor));
    vk_gpu_scratchpad_t *sp = &bd->scratchpad;
    bd->sdt.num_layers = num_layers;
    VkDevice dev = bd->device;
    const int PI_NORM = PIPELINE_RMSNORM_F32;
    const int PI_GEMV = PIPELINE_GEMV_F32;

    for (int L = 0; L < num_layers; L++) {
        populate_layer_descriptor_slots(bd, cfg, sp, L, bd->sdt.layer_sets[L]);
    }

    /* -- LM-Head: Final RMSNorm + Vocabulary Projection ------------------- */
    /* After 18 layers, final output is in hidden[(18)%2] = hidden[0] */
    VkBuffer final_hidden = sp->layer_hidden[num_layers % 2].buffer;

    bd->sdt.lmhead_norm.pipeline_idx = PI_NORM;
    bd->sdt.lmhead_norm.ds = sdt_alloc_ds(bd, PI_NORM);
    sdt_write_buf(dev, bd->sdt.lmhead_norm.ds, 0, final_hidden,                     0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, bd->sdt.lmhead_norm.ds, 1, bd->final_norm_weight.buffer,     0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, bd->sdt.lmhead_norm.ds, 2, sp->norm_buf.buffer,              0, VK_WHOLE_SIZE);

    bd->sdt.lmhead_proj.pipeline_idx = PI_GEMV;
    bd->sdt.lmhead_proj.ds = sdt_alloc_ds(bd, PI_GEMV);
    sdt_write_buf(dev, bd->sdt.lmhead_proj.ds, 0, sp->norm_buf.buffer,              0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, bd->sdt.lmhead_proj.ds, 1, bd->embedding_weight.buffer,      0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, bd->sdt.lmhead_proj.ds, 2, bd->lm_head_logits.buffer,        0, VK_WHOLE_SIZE);

    bd->sdt.lmhead_argmax.pipeline_idx = PIPELINE_ARGMAX_F32;
    bd->sdt.lmhead_argmax.ds = sdt_alloc_ds(bd, PIPELINE_ARGMAX_F32);
    sdt_write_buf(dev, bd->sdt.lmhead_argmax.ds, 0, bd->lm_head_logits.buffer,      0, VK_WHOLE_SIZE);
    sdt_write_buf(dev, bd->sdt.lmhead_argmax.ds, 1, bd->selected_token_ids.buffer,  0, VK_WHOLE_SIZE);

    LOG_INFO("SDT pre-populated: %d layers × %d kernel slots + 3 lm_head = %d total descriptor sets",
             num_layers, SDT_KERNELS_PER_LAYER,
             num_layers * SDT_KERNELS_PER_LAYER + 3);
    return 0;
}

/**
 * Bind a descriptor set from the Static Descriptor Table to the command buffer.
 *
 * This replaces the old bind_desc() NO-OP.  The descriptor set was fully
 * written during vulkan_prepopulate_descriptors(); here we only bind it.
 *
 * @param cmd_buf  Active command buffer (must be in recording state)
 * @param entry    SDT entry (pipeline index + pre-populated VkDescriptorSet)
 * @param bd       Backend session data (pipelines array)
 */
static inline void sdt_bind(VkCommandBuffer cmd_buf,
                             const sdt_entry_t *entry,
                             const backend_vulkan_session_data_t *bd) {
    const vk_compute_pipeline_t *p = &bd->pipelines[entry->pipeline_idx];
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            p->pipeline_layout, 0, 1, &entry->ds, 0, NULL);
}

/**
 * record_transformer_layer() – Record all kernel dispatches for one layer.
 *
 * Uses pre-populated SDT entries for every kernel.  No dynamic descriptor
 * updates occur during recording.  Both residual saves use vkCmdCopyBuffer
 * which is safe during compute recording.
 *
 * Ping-pong:
 *   Even layer L: input = hidden[0], output = hidden[1]
 *   Odd  layer L: input = hidden[1], output = hidden[0]
 * (Both hidden buffer VkHandles are baked into the SDT at init time.)
 *
 * @return 0 on success, -1 on error
 */
/* -----------------------------------------------------------------------
 * File-scope barrier/timing helpers extracted from record_transformer_layer.
 * ----------------------------------------------------------------------- */

static inline void kts_maybe(VkCommandBuffer cmd_buf, backend_vulkan_session_data_t *bd,
    int ena, uint32_t *idx_io, uint32_t cap)
{
    if (ena && idx_io && (*idx_io) < cap)
        vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            bd->timing_query_pool, (*idx_io)++);
}

static inline void barrier_buf(VkCommandBuffer cmd_buf,
    VkPipelineStageFlags src, VkPipelineStageFlags dst, VkBuffer buf)
{
    VkBufferMemoryBarrier bb = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buf, .offset = 0, .size = VK_WHOLE_SIZE
    };
    vkCmdPipelineBarrier(cmd_buf, src, dst, 0, 0, NULL, 1, &bb, 0, NULL);
}

static inline void barrier_bufs2(VkCommandBuffer cmd_buf,
    VkPipelineStageFlags src, VkPipelineStageFlags dst, VkBuffer b0, VkBuffer b1)
{
    VkBufferMemoryBarrier bb[2] = {
        { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = b0, .offset = 0, .size = VK_WHOLE_SIZE },
        { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
          .buffer = b1, .offset = 0, .size = VK_WHOLE_SIZE }
    };
    vkCmdPipelineBarrier(cmd_buf, src, dst, 0, 0, NULL, 2, bb, 0, NULL);
}

/** Packs proj_dispatch mode parameters (reduces param count below max). */
typedef struct { int proj_pipeline; int use_gemm; int batch_size; } proj_mode_t;

static inline void proj_dispatch(VkCommandBuffer cmd_buf, backend_vulkan_session_data_t *bd,
    const sdt_entry_t *e, const proj_mode_t *mode,
    uint32_t out_elems, const vk_kernel_push_constants_t *pc)
{
    int proj_pipeline = mode->proj_pipeline;
    int use_gemm      = mode->use_gemm;
    int batch_size    = mode->batch_size;
    const vk_compute_pipeline_t *pipe = &bd->pipelines[proj_pipeline];
    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
    vkCmdBindDescriptorSets(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe->pipeline_layout, 0, 1, &e->ds, 0, NULL);
    if (use_gemm)
        vk_pipeline_dispatch(cmd_buf, pipe, pc, ceil_div(out_elems, 16),
                             ceil_div((uint32_t)batch_size, 16), 1);
    else
        vk_pipeline_dispatch(cmd_buf, pipe, pc, out_elems, 1, 1);
}

/** Parameters struct for record_transformer_layer (reduces param count). */
typedef struct {
    int batch_size;
    int start_pos;
    int enable_kernel_timing;
    uint32_t *query_idx_io;
    uint32_t query_capacity;
} record_layer_params_t;

/*
 * record_qkv_attn_phase: save residual, pre-attn norm, Q/K/V proj, QK-norm
 * Part of record_transformer_layer (split for complexity compliance).
 */
static void record_qkv_attn_phase(
    VkCommandBuffer cmd_buf, backend_vulkan_session_data_t *bd,
    const gemma3_270m_config_t *cfg, int layer_idx,
    const record_layer_params_t *p, vk_kernel_push_constants_t *pc)
{
    vk_gpu_scratchpad_t *sp = &bd->scratchpad;
    const sdt_entry_t   *S  = bd->sdt.layer_sets[layer_idx];
    uint32_t d_model = (uint32_t)cfg->hidden_size;
    uint32_t d_inner = (uint32_t)(cfg->num_attention_heads * cfg->head_dim);
    uint32_t d_kv    = (uint32_t)(cfg->num_key_value_heads * cfg->head_dim);
    uint32_t num_q   = (uint32_t)cfg->num_attention_heads;
    uint32_t num_kv  = (uint32_t)cfg->num_key_value_heads;
    int batch_size   = p->batch_size;
    int ekt          = p->enable_kernel_timing;
    uint32_t *qidx   = p->query_idx_io;
    uint32_t  qcap   = p->query_capacity;
    bool use_gemm    = (batch_size > 1);
    int proj_pipeline = use_gemm ? PIPELINE_GEMM_F32 : PIPELINE_GEMV_F32;
    const proj_mode_t _pm = { proj_pipeline, (int)use_gemm, batch_size };
    size_t hidden_bytes = (size_t)batch_size * d_model * sizeof(float);

    VkBufferMemoryBarrier xfer_to_comp = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = sp->residual.buffer, .offset = 0, .size = VK_WHOLE_SIZE
    };

    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 0 start */

    VkBuffer input_vkbuf = sp->layer_hidden[layer_idx % 2].buffer;
    VkBufferCopy save_residual = { .srcOffset = 0, .dstOffset = 0, .size = hidden_bytes };
    vkCmdCopyBuffer(cmd_buf, input_vkbuf, sp->residual.buffer, 1, &save_residual);
    vkCmdPipelineBarrier(cmd_buf,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &xfer_to_comp, 0, NULL);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 1 save residual */

    pc->stride_0 = d_model;
    sdt_bind(cmd_buf, &S[SDT_SLOT_PRE_ATTN_NORM], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_PRE_ATTN_NORM].pipeline_idx],
                         pc, (uint32_t)batch_size, 1, 1);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->norm_buf.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 2 pre-attn norm */

    pc->stride_0 = d_model; pc->stride_1 = use_gemm ? d_inner : 0;
    pc->num_heads = d_inner;
    proj_dispatch(cmd_buf, bd, &S[SDT_SLOT_Q_PROJ], &_pm, d_inner, pc);
    pc->num_heads = d_kv; pc->stride_1 = use_gemm ? d_kv : 0;
    proj_dispatch(cmd_buf, bd, &S[SDT_SLOT_K_PROJ], &_pm, d_kv, pc);
    pc->num_heads = d_kv;
    proj_dispatch(cmd_buf, bd, &S[SDT_SLOT_V_PROJ], &_pm, d_kv, pc);
    barrier_bufs2(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->q_proj.buffer, sp->k_proj.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 3 qkv projections */

    pc->num_heads = num_q; pc->stride_0 = num_kv;
    sdt_bind(cmd_buf, &S[SDT_SLOT_QK_NORM], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_QK_NORM].pipeline_idx],
                         pc, num_q + num_kv, (uint32_t)batch_size, 1);
    barrier_bufs2(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->q_proj.buffer, sp->k_proj.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 4 qk norm */
}

/*
 * record_rope_attn_phase: RoPE, KV-cache write, attention, O-proj,
 * post-attn norm, and attention residual add.
 */
static void record_rope_attn_phase(
    VkCommandBuffer cmd_buf, backend_vulkan_session_data_t *bd,
    const gemma3_270m_config_t *cfg, int layer_idx,
    const record_layer_params_t *p, vk_kernel_push_constants_t *pc)
{
    vk_gpu_scratchpad_t *sp = &bd->scratchpad;
    const sdt_entry_t   *S  = bd->sdt.layer_sets[layer_idx];
    uint32_t d_model  = (uint32_t)cfg->hidden_size;
    uint32_t d_inner  = (uint32_t)(cfg->num_attention_heads * cfg->head_dim);
    uint32_t num_q    = (uint32_t)cfg->num_attention_heads;
    uint32_t num_kv   = (uint32_t)cfg->num_key_value_heads;
    uint32_t hdim     = (uint32_t)cfg->head_dim;
    int batch_size    = p->batch_size;
    int start_pos     = p->start_pos;
    int ekt           = p->enable_kernel_timing;
    uint32_t *qidx    = p->query_idx_io;
    uint32_t  qcap    = p->query_capacity;
    bool use_gemm     = (batch_size > 1);
    int proj_pipeline = use_gemm ? PIPELINE_GEMM_F32 : PIPELINE_GEMV_F32;
    const proj_mode_t _pm = { proj_pipeline, (int)use_gemm, batch_size };

    pc->num_heads = num_q; pc->stride_0 = num_kv;
    sdt_bind(cmd_buf, &S[SDT_SLOT_ROPE], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_ROPE].pipeline_idx],
                         pc,
                         ceil_div((num_q > num_kv ? num_q : num_kv), 16),
                         ceil_div(hdim / 2, 16),
                         (uint32_t)batch_size);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 5 rope */

    {
        layer_dispatch_ctx_t ctx = {
            .cmd_buf    = cmd_buf, .bd = bd, .cfg = cfg,
            .layer      = layer_idx, .batch_size = batch_size,
            .seq_pos    = start_pos + batch_size - 1
        };
        write_kv_to_cache(&ctx, num_kv, hdim, (uint32_t)bd->kv_cache.max_seq_len);
    }
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 6 kv write */

    pc->num_heads = num_q; pc->stride_0 = num_kv;
    pc->stride_1  = (uint32_t)cfg->sliding_window;
    pc->reserved[0] = (uint32_t)(cfg->layer_types_mask & 0xFFFFFFFF);
    sdt_bind(cmd_buf, &S[SDT_SLOT_ATTENTION], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_ATTENTION].pipeline_idx],
                         pc, num_q, (uint32_t)batch_size, 1);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->attn_output.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 7 attention */

    pc->d_model = d_inner; pc->stride_0 = d_inner; pc->stride_1 = use_gemm ? d_model : 0;
    pc->num_heads = d_model;
    proj_dispatch(cmd_buf, bd, &S[SDT_SLOT_O_PROJ], &_pm, d_model, pc);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->norm_buf.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 8 o-proj */

    pc->d_model = d_model; pc->stride_0 = d_model;
    sdt_bind(cmd_buf, &S[SDT_SLOT_POST_ATTN_NORM], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_POST_ATTN_NORM].pipeline_idx],
                         pc, (uint32_t)batch_size, 1, 1);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->attn_output.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 9 post-attn norm */

    pc->stride_0 = d_model * (uint32_t)batch_size;
    sdt_bind(cmd_buf, &S[SDT_SLOT_ATTN_RESIDUAL], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_ATTN_RESIDUAL].pipeline_idx],
                         pc, ceil_div(pc->stride_0, 256), 1, 1);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->residual.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 10 attn residual */
}

/*
 * record_ffn_phase: pre-FFN norm, gate/up proj, GeGLU, down proj,
 * post-FFN norm, and FFN residual add.
 */
static void record_ffn_phase(
    VkCommandBuffer cmd_buf, backend_vulkan_session_data_t *bd,
    const gemma3_270m_config_t *cfg, int layer_idx,
    const record_layer_params_t *p, vk_kernel_push_constants_t *pc)
{
    vk_gpu_scratchpad_t *sp = &bd->scratchpad;
    const sdt_entry_t   *S  = bd->sdt.layer_sets[layer_idx];
    uint32_t d_model  = (uint32_t)cfg->hidden_size;
    uint32_t d_ff     = (uint32_t)cfg->intermediate_size;
    int batch_size    = p->batch_size;
    int ekt           = p->enable_kernel_timing;
    uint32_t *qidx    = p->query_idx_io;
    uint32_t  qcap    = p->query_capacity;
    bool use_gemm     = (batch_size > 1);
    int proj_pipeline = use_gemm ? PIPELINE_GEMM_F32 : PIPELINE_GEMV_F32;
    const proj_mode_t _pm = { proj_pipeline, (int)use_gemm, batch_size };

    pc->stride_0 = d_model; pc->d_model = d_model;
    sdt_bind(cmd_buf, &S[SDT_SLOT_PRE_FFN_NORM], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_PRE_FFN_NORM].pipeline_idx],
                         pc, (uint32_t)batch_size, 1, 1);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->ffn_gate.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 11 pre-ffn norm */

    pc->d_model = d_model; pc->num_heads = d_ff;
    pc->stride_0 = d_model; pc->stride_1 = use_gemm ? d_ff : 0;
    proj_dispatch(cmd_buf, bd, &S[SDT_SLOT_GATE_PROJ], &_pm, d_ff, pc);
    pc->num_heads = d_ff;
    proj_dispatch(cmd_buf, bd, &S[SDT_SLOT_UP_PROJ], &_pm, d_ff, pc);
    barrier_bufs2(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->ffn_value.buffer, sp->attn_output.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 12 gate/up proj */

    pc->stride_0 = 0; pc->stride_1 = 0;
    sdt_bind(cmd_buf, &S[SDT_SLOT_GELU], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_GELU].pipeline_idx],
                         pc, ceil_div((uint32_t)batch_size * d_ff, 256), 1, 1);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->ffn_value.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 13 geglu */

    pc->d_model = d_ff; pc->num_heads = d_model;
    pc->stride_0 = d_ff; pc->stride_1 = use_gemm ? d_model : 0;
    proj_dispatch(cmd_buf, bd, &S[SDT_SLOT_DOWN_PROJ], &_pm, d_model, pc);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->attn_output.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 14 down proj */

    pc->d_model = d_model; pc->stride_0 = d_model;
    sdt_bind(cmd_buf, &S[SDT_SLOT_POST_FFN_NORM], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_POST_FFN_NORM].pipeline_idx],
                         pc, (uint32_t)batch_size, 1, 1);
    barrier_buf(cmd_buf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, sp->ffn_gate.buffer);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 15 post-ffn norm */

    pc->stride_0 = d_model * (uint32_t)batch_size;
    sdt_bind(cmd_buf, &S[SDT_SLOT_FFN_RESIDUAL], bd);
    vk_pipeline_dispatch(cmd_buf, &bd->pipelines[S[SDT_SLOT_FFN_RESIDUAL].pipeline_idx],
                         pc, ceil_div(pc->stride_0, 256), 1, 1);
    kts_maybe(cmd_buf, bd, ekt, qidx, qcap); /* stage 16 ffn residual/end */
}

static int record_transformer_layer(
    VkCommandBuffer cmd_buf,
    backend_vulkan_session_data_t *bd,
    const gemma3_270m_config_t *cfg,
    int layer_idx,
    const record_layer_params_t *p)
{
    int batch_size           = p->batch_size;
    int start_pos            = p->start_pos;
    if (!cmd_buf || !bd || !cfg) {
        LOG_ERROR("Invalid arguments to record_transformer_layer");
        return -1;
    }

    if (layer_idx < 0 || layer_idx >= bd->sdt.num_layers) {
        LOG_ERROR("layer_idx=%d out of SDT range [0,%d)", layer_idx, bd->sdt.num_layers);
        return -1;
    }

    vk_kernel_push_constants_t pc = build_push_constants(cfg, layer_idx, batch_size, start_pos);
    /* CRITICAL FIX: pc.max_seq_len must match the KV cache buffer stride, not
     * cfg->max_position_embeddings. The KV cache is allocated for max_context_len
     * positions; using max_position_embeddings (32768) instead of max_context_len
     * (2048) offsets V reads 16× past the written position, returning zeros and
     * making attention a no-op. */
    pc.max_seq_len = (uint32_t)bd->kv_cache.max_seq_len;

    /* Dispatch attention + FFN via extracted sub-functions */
    record_qkv_attn_phase(cmd_buf, bd, cfg, layer_idx, p, &pc);
    record_rope_attn_phase(cmd_buf, bd, cfg, layer_idx, p, &pc);
    record_ffn_phase(cmd_buf, bd, cfg, layer_idx, p, &pc);

    return 0;
}


/**
 * vulkan_record_forward_pass() – Record the complete 18-layer transformer.
 *
 * Builds one VkCommandBuffer containing:
 *   1. Embedding upload fence-wait + transfer copy (done before this call)
 *   2. 18 × record_transformer_layer() using SDT (ping-pong hidden state)
 *   3. LM-head: final RMSNorm + vocab projection (reads from correct buffer)
 *
 * Does NOT submit.  Caller calls vkQueueSubmit + vkQueueWaitIdle.
 *
 * @param bd         Backend session data
 * @param cfg        Model configuration
 * @param batch_size Tokens in current batch (1 for decode)
 * @param start_pos  KV cache write position for this step
 * @param emit_lmhead If non-zero, record the lm_head dispatches
 * @return 0 on success, -1 on error
 */
/** Options struct for vulkan_record_forward_pass (reduces param count). */
typedef struct {
    int batch_size;
    int start_pos;
    int emit_lmhead;
    int emit_argmax;
    int copy_embeddings_from_staging;
} vk_fwd_opts_t;

/**
 * Record the LM-head block (final RMSNorm + vocab projection + optional argmax).
 * Extracted to reduce NLOC and token count of vulkan_record_forward_pass.
 */
static void record_lmhead_block(
    backend_vulkan_session_data_t *bd,
    const gemma3_270m_config_t    *cfg,
    const vk_fwd_opts_t           *opts,
    int                            timing_active,
    uint32_t                      *query_idx)
{
    if (timing_active && (*query_idx + 1) < bd->timing_query_capacity)
        vkCmdWriteTimestamp(bd->cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            bd->timing_query_pool, (*query_idx)++);

    int num_layers = cfg->num_hidden_layers;
    VkBufferMemoryBarrier last_layer_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = bd->scratchpad.layer_hidden[num_layers % 2].buffer,
        .offset = 0, .size = VK_WHOLE_SIZE
    };
    vkCmdPipelineBarrier(bd->cmd_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &last_layer_barrier, 0, NULL);

    vk_kernel_push_constants_t pc = {0};
    pc.batch_size = 1;
    pc.d_model    = (uint32_t)cfg->hidden_size;
    pc.stride_0   = (uint32_t)cfg->hidden_size;

    sdt_bind(bd->cmd_buffer, &bd->sdt.lmhead_norm, bd);
    vk_pipeline_dispatch(bd->cmd_buffer,
                         &bd->pipelines[bd->sdt.lmhead_norm.pipeline_idx], &pc, 1, 1, 1);

    VkBufferMemoryBarrier norm_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = bd->scratchpad.norm_buf.buffer, .offset = 0, .size = VK_WHOLE_SIZE
    };
    vkCmdPipelineBarrier(bd->cmd_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &norm_barrier, 0, NULL);

    /* Vocab Projection via tiled GEMM (GEMV would exceed maxComputeWorkGroupCount) */
    pc.stride_0  = (uint32_t)cfg->hidden_size;
    pc.stride_1  = 0;
    pc.num_heads = (uint32_t)cfg->vocab_size;
    {
        const vk_compute_pipeline_t *gemm_pipe = &bd->pipelines[PIPELINE_GEMM_F32];
        vkCmdBindPipeline(bd->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gemm_pipe->pipeline);
        vkCmdBindDescriptorSets(bd->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                gemm_pipe->pipeline_layout, 0, 1, &bd->sdt.lmhead_proj.ds, 0, NULL);
        vk_pipeline_dispatch(bd->cmd_buffer, gemm_pipe, &pc,
                             ceil_div((uint32_t)cfg->vocab_size, 16),
                             ceil_div((uint32_t)pc.batch_size, 16), 1);
    }

    if (opts->emit_argmax) {
        VkBufferMemoryBarrier proj_to_argmax = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = bd->lm_head_logits.buffer, .offset = 0, .size = VK_WHOLE_SIZE
        };
        vkCmdPipelineBarrier(bd->cmd_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 1, &proj_to_argmax, 0, NULL);

        pc.stride_0 = (uint32_t)cfg->vocab_size;
        pc.stride_1 = (uint32_t)cfg->vocab_size;
        sdt_bind(bd->cmd_buffer, &bd->sdt.lmhead_argmax, bd);
        vk_pipeline_dispatch(bd->cmd_buffer,
                            &bd->pipelines[bd->sdt.lmhead_argmax.pipeline_idx],
                            &pc, ceil_div((uint32_t)opts->batch_size, 256), 1, 1);

        VkBufferMemoryBarrier argmax_to_host = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = bd->selected_token_ids.buffer, .offset = 0, .size = VK_WHOLE_SIZE
        };
        vkCmdPipelineBarrier(bd->cmd_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, NULL, 1, &argmax_to_host, 0, NULL);
    } else {
        VkBufferMemoryBarrier proj_to_xfer = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = bd->lm_head_logits.buffer, .offset = 0, .size = VK_WHOLE_SIZE
        };
        vkCmdPipelineBarrier(bd->cmd_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, NULL, 1, &proj_to_xfer, 0, NULL);
    }

    if (timing_active && (*query_idx) < bd->timing_query_capacity)
        vkCmdWriteTimestamp(bd->cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            bd->timing_query_pool, (*query_idx)++);
}

static int vulkan_record_forward_pass(
    backend_vulkan_session_data_t *bd,
    const gemma3_270m_config_t    *cfg,
    const vk_fwd_opts_t           *opts)
{
    int batch_size  = opts->batch_size;
    int start_pos   = opts->start_pos;
    int emit_lmhead = opts->emit_lmhead;
    int num_layers  = cfg->num_hidden_layers;
    uint32_t query_idx = 0;
    int timing_active = (bd->timing_enabled && bd->timing_query_pool != VK_NULL_HANDLE);
    bd->timing_kernel_first_query = 0;
    bd->timing_kernel_query_count = 0;

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VkResult vr = vkBeginCommandBuffer(bd->cmd_buffer, &begin_info);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to begin command buffer: %d", vr);
        return -1;
    }

    if (timing_active && bd->timing_query_capacity > 0) {
        vkCmdResetQueryPool(bd->cmd_buffer, bd->timing_query_pool, 0, bd->timing_query_capacity);
    }

    if (opts->copy_embeddings_from_staging) {
        VkBufferCopy copy_region = {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = (VkDeviceSize)((size_t)batch_size * cfg->hidden_size * sizeof(float))
        };
        vkCmdCopyBuffer(bd->cmd_buffer,
                        bd->embedding_staging.buffer,
                        bd->scratchpad.layer_hidden[0].buffer,
                        1,
                        &copy_region);

        VkBufferMemoryBarrier transfer_to_compute = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = bd->scratchpad.layer_hidden[0].buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        };
        vkCmdPipelineBarrier(bd->cmd_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,
                             0, NULL,
                             1, &transfer_to_compute,
                             0, NULL);
    }

    /* ----------------------------------------------------------------
     * 18-layer transformer loop (single-pass, one command buffer)
     * ---------------------------------------------------------------- */
    for (int L = 0; L < num_layers; L++) {
        if (timing_active && (query_idx + 1) < bd->timing_query_capacity) {
            vkCmdWriteTimestamp(bd->cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                bd->timing_query_pool, query_idx++);
        }


        int enable_kernel_timing = (timing_active
                        && batch_size == 1
                        && bd->timing_kernel_layer == L
                        && bd->timing_kernel_query_count == 0);
        uint32_t kernel_first = query_idx;

        record_layer_params_t _rlp = {
            .batch_size = batch_size, .start_pos = start_pos,
            .enable_kernel_timing = enable_kernel_timing,
            .query_idx_io = &query_idx, .query_capacity = bd->timing_query_capacity
        };
        int rc = record_transformer_layer(bd->cmd_buffer, bd, cfg, L, &_rlp);
        if (rc != 0) {
            LOG_ERROR("Failed to record layer %d commands", L);
            vkEndCommandBuffer(bd->cmd_buffer);
            return -1;
        }

        if (enable_kernel_timing) {
            bd->timing_kernel_first_query = kernel_first;
            bd->timing_kernel_query_count =
                query_idx - kernel_first;
        }

        /* Layer boundary dependency: layer output hidden[(L+1)%2] is input of layer L+1. */
        if (L + 1 < num_layers) {
            VkBufferMemoryBarrier layer_boundary = {
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = bd->scratchpad.layer_hidden[(L + 1) % 2].buffer,
                .offset = 0,
                .size = VK_WHOLE_SIZE
            };
            vkCmdPipelineBarrier(bd->cmd_buffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, NULL, 1, &layer_boundary, 0, NULL);
        }

        if (timing_active && query_idx < bd->timing_query_capacity) {
            vkCmdWriteTimestamp(bd->cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                bd->timing_query_pool, query_idx++);
        }
    }

    /* ----------------------------------------------------------------
     * LM-Head: Final RMSNorm + Vocabulary Projection
     *
     * After 18 layers the final hidden state lives in:
     *   hidden[num_layers % 2]  (= hidden[0] for 18 layers)
     * This buffer handle was baked into sdt.lmhead_norm at init time.
     * ---------------------------------------------------------------- */
    if (emit_lmhead) {
        record_lmhead_block(bd, cfg, opts, timing_active, &query_idx);
    }

    bd->timing_query_last_count = timing_active ? query_idx : 0;

    vr = vkEndCommandBuffer(bd->cmd_buffer);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to end command buffer: %d", vr);
        return -1;
    }

    LOG_DEBUG("Recorded forward pass: %d layers, batch=%d, start_pos=%d, lmhead=%d",
              num_layers, batch_size, start_pos, emit_lmhead);
    return 0;
}

/**
 * Write K and V projections to KV cache at current sequence position.
 */
static void write_kv_to_cache(
    const layer_dispatch_ctx_t *ctx,
    uint32_t num_kv, uint32_t hdim, uint32_t max_pos) {
    vk_gpu_scratchpad_t *sp = &ctx->bd->scratchpad;
    size_t kv_head_elems = (size_t)num_kv * hdim;
    size_t layer_stride  = (size_t)max_pos * kv_head_elems * 2;
    
    /* FIX #2: Prefill offset - write to range [seq_pos - batch_size + 1, seq_pos] */
    int start_seq_pos = ctx->seq_pos - ctx->batch_size + 1;
    size_t k_dst = ((size_t)ctx->layer * layer_stride +
                    (size_t)start_seq_pos * kv_head_elems) * sizeof(float);
    /* CRITICAL FIX: V offset must include start_seq_pos, not just layer offset! */
    size_t v_dst = ((size_t)ctx->layer * layer_stride + (size_t)max_pos * kv_head_elems +
                    (size_t)start_seq_pos * kv_head_elems) * sizeof(float);
    /* FIX #2: Copy entire batch (all tokens in prefill), not just last one */
    size_t copy_bytes = kv_head_elems * sizeof(float) * ctx->batch_size;
    
    /* DEBUG: Log offset calculation to diagnose out-of-bounds errors */
    size_t kv_buffer_size = ctx->bd->kv_cache.kv_data.size;
    LOG_DEBUG("KV cache write: layer=%d seq_pos=%d batch=%d | k_dst=%zu v_dst=%zu copy=%zu | buffer_size=%zu",
              ctx->layer, ctx->seq_pos, ctx->batch_size, k_dst, v_dst, copy_bytes, kv_buffer_size);
    if (k_dst + copy_bytes > kv_buffer_size || v_dst + copy_bytes > kv_buffer_size) {
        LOG_ERROR("KV cache OUT OF BOUNDS: k_end=%zu v_end=%zu > buffer_size=%zu",
                  k_dst + copy_bytes, v_dst + copy_bytes, kv_buffer_size);
    }

    /* FIX #1: Wait for RoPE compute to finish writing k_proj/v_proj before copy (prevents GPU hang) */
    VkBufferMemoryBarrier rope_to_copy[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = sp->k_proj.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        },
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = sp->v_proj.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        }
    };
    vkCmdPipelineBarrier(ctx->cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  /* Wait for RoPE */
        VK_PIPELINE_STAGE_TRANSFER_BIT,        /* Block Copy */
        0, 0, NULL, 2, rope_to_copy, 0, NULL);

    VkBufferCopy k_region = {.srcOffset = 0, .dstOffset = k_dst, .size = copy_bytes};
    VkBufferCopy v_region = {.srcOffset = 0, .dstOffset = v_dst, .size = copy_bytes};
    vkCmdCopyBuffer(ctx->cmd_buf, sp->k_proj.buffer,
                    ctx->bd->kv_cache.kv_data.buffer, 1, &k_region);
    vkCmdCopyBuffer(ctx->cmd_buf, sp->v_proj.buffer,
                    ctx->bd->kv_cache.kv_data.buffer, 1, &v_region);

    /* Ensure copy completes before next compute shader reads KV cache */
    VkBufferMemoryBarrier xfer = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = ctx->bd->kv_cache.kv_data.buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE
    };
    vkCmdPipelineBarrier(ctx->cmd_buf, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &xfer, 0, NULL);
}

/**
 * Vulkan backend: Execute forward batch (P11-05 full GPU pipeline).
 *
 * Records a single command buffer with all transformer layer dispatches,
 * submits to GPU queue, and waits for completion (synchronous).
 *
 * @param session    Inference session with Vulkan backend
 * @param token_ids  Input token IDs (length = batch_size)
 * @param start_pos  Current sequence position (for RoPE and KV cache)
 * @param batch_size Number of tokens (1 for decoding, >1 for prefill)
 * @param logits     Output logits buffer [batch_size × vocab_size]
 * @return 0 on success, -1 on error (triggers CPU fallback)
 */
/**
 * GPU debug checkpoints: CP1 (embedding), CP2 (layer-0 isolation), CP2.5 (rolling probe).
 * Only active when debug_gpu_enabled() && start_pos == 0.
 */
/* Helper: reset cmd buf, record copy embedding→hidden[0] with barrier, end+submit+wait */
static void cp25_restore_embedding(backend_vulkan_session_data_t *bd,
    const VkCommandBufferBeginInfo *begin,
    const VkBufferCopy *region, const VkMemoryBarrier *bmr)
{
    vkResetCommandBuffer(bd->cmd_buffer, 0);
    vkBeginCommandBuffer(bd->cmd_buffer, begin);
    vkCmdCopyBuffer(bd->cmd_buffer, bd->embedding_staging.buffer,
                    bd->scratchpad.layer_hidden[0].buffer, 1, region);
    vkCmdPipelineBarrier(bd->cmd_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, bmr, 0, NULL, 0, NULL);
    vkEndCommandBuffer(bd->cmd_buffer);
    vk_pipeline_submit_timeline(bd->compute_queue, bd->cmd_buffer, NULL, 0, NULL, 0);
    vkQueueWaitIdle(bd->compute_queue);
}

static void run_debug_checkpoints(
    backend_vulkan_session_data_t *bd,
    inference_session_t *session,
    const gemma3_270m_config_t *cfg,
    const int *token_ids,
    int start_pos,
    int num_layers)
{
    if (!debug_gpu_enabled() || start_pos != 0) return;

    int probe_n  = cfg->hidden_size < 640 ? cfg->hidden_size : 640;

    /* CP1: Embedding integrity check */
    float *cpu_embed = malloc((size_t)cfg->hidden_size * sizeof(float));
    if (cpu_embed) {
        sapphire_embed_lookup_batch(session, token_ids, 1, cpu_embed);
        vec_stats(cpu_embed, probe_n, "CP1_CPU_embed[tok0]");
        free(cpu_embed);
    }
    probe_buf(bd, &bd->scratchpad.layer_hidden[0], probe_n, "CP1_GPU_hidden0");
    LOG_INFO("[DBG] CP1: if CPU/GPU min/max/L2 differ -> staging upload or BF16->F32 conversion is wrong");

    /* CP2: Layer-0 isolation pass */
    VkResult vr_cp2 = vkResetCommandBuffer(bd->cmd_buffer, 0);
    if (vr_cp2 == VK_SUCCESS) {
        VkCommandBufferBeginInfo dbg_begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        vkBeginCommandBuffer(bd->cmd_buffer, &dbg_begin);
        { record_layer_params_t _cp2p = {1,0,0,NULL,0};
          record_transformer_layer(bd->cmd_buffer, bd, cfg, 0, &_cp2p); }
        vkEndCommandBuffer(bd->cmd_buffer);
        vk_pipeline_submit_timeline(bd->compute_queue, bd->cmd_buffer, NULL, 0, NULL, 0);
        vkQueueWaitIdle(bd->compute_queue);
        probe_buf(bd, &bd->scratchpad.norm_buf,        probe_n, "CP2_norm_buf");
        probe_buf(bd, &bd->scratchpad.q_proj,          probe_n, "CP2_q_proj");
        probe_buf(bd, &bd->scratchpad.attn_output,     cfg->hidden_size, "CP2_attn_output");
        probe_buf(bd, &bd->scratchpad.residual,        cfg->hidden_size, "CP2_residual_pre_ffn");
        probe_buf(bd, &bd->scratchpad.ffn_gate,        cfg->hidden_size, "CP2_ffn_gate");
        probe_buf(bd, &bd->scratchpad.ffn_value,       cfg->hidden_size, "CP2_ffn_value_gelu");
        probe_buf(bd, &bd->scratchpad.layer_hidden[1], cfg->hidden_size, "CP2_layer0_output");
        LOG_INFO("[DBG] CP2: norm_buf L2≈0 → RMSNorm broken | q_proj≈norm_buf → GEMV broken");
    }

    /* CP2.5: Per-layer rolling probe */
    const char *dbg_layers_env = getenv("SAPPHIRE_DEBUG_LAYERS");
    fprintf(stderr, "[CP2.5_DIAG] SAPPHIRE_DEBUG_LAYERS=%s\n",
            dbg_layers_env ? dbg_layers_env : "NULL");
    if (!dbg_layers_env || dbg_layers_env[0] != '1') return;

    LOG_INFO("[DBG] CP2.5: per-layer rolling probe");
    VkCommandBufferBeginInfo lp_begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    size_t emb_bytes = (size_t)cfg->hidden_size * sizeof(float);
    VkBufferCopy emb_region = { .srcOffset = 0, .dstOffset = 0, .size = emb_bytes };
    VkMemoryBarrier emb_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
    };
    for (int N = 1; N <= num_layers; N++) {
        cp25_restore_embedding(bd, &lp_begin, &emb_region, &emb_barrier);
        for (int l = 0; l < N; l++) {
            record_layer_params_t _cp25p = {1,0,0,NULL,0};
            record_transformer_layer(bd->cmd_buffer, bd, cfg, l, &_cp25p);
        }
        vkEndCommandBuffer(bd->cmd_buffer);
        vk_pipeline_submit_timeline(bd->compute_queue, bd->cmd_buffer, NULL, 0, NULL, 0);
        vkQueueWaitIdle(bd->compute_queue);
        char tag[32];
        snprintf(tag, sizeof(tag), "CP25_L%02d_out", N - 1);
        probe_buf(bd, &bd->scratchpad.layer_hidden[N % 2], probe_n, tag);
    }
    /* Restore hidden[0] to original embedding */
    cp25_restore_embedding(bd, &lp_begin, &emb_region, &emb_barrier);
    LOG_INFO("[DBG] CP2.5: done. hidden[0] restored.");
}

/**
 * Process and log GPU timing query results for the forward pass.
 */
static void process_forward_timing(backend_vulkan_session_data_t *bd, int num_layers)
{
    if (!bd->timing_enabled || bd->timing_query_pool == VK_NULL_HANDLE ||
        bd->timing_query_last_count < 2) return;

    uint64_t ts[256] = {0};
    uint32_t qcount = bd->timing_query_last_count;
    if (qcount > 256u) qcount = 256u;

    VkResult qres = vkGetQueryPoolResults(bd->device, bd->timing_query_pool,
        0, qcount, sizeof(uint64_t) * qcount, ts, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (qres != VK_SUCCESS) { LOG_WARN("vkGetQueryPoolResults failed: %d", (int)qres); return; }

    if (bd->timing_kernel_query_count > 1 && qcount > 1) {
        double total_ms = (double)(ts[qcount - 1] - ts[0]) * (double)bd->timestamp_period_ns / 1000000.0;
        LOG_DEBUG("VK GPU timing summary: total=%.3f ms (kernel-profile mode)", total_ms);
        return;
    }
    double total_ms = 0.0;
    int pair_idx = 0;
    for (int layer = 0; layer < num_layers && (pair_idx + 1) < (int)qcount; layer++) {
        uint64_t t0 = ts[pair_idx++], t1 = ts[pair_idx++];
        double layer_ms = (double)(t1 - t0) * (double)bd->timestamp_period_ns / 1000000.0;
        total_ms += layer_ms;
        LOG_DEBUG("VK layer %02d GPU time: %.3f ms", layer, layer_ms);
    }
    if ((pair_idx + 1) < (int)qcount) {
        uint64_t t0 = ts[pair_idx++], t1 = ts[pair_idx++];
        total_ms += (double)(t1 - t0) * (double)bd->timestamp_period_ns / 1000000.0;
        LOG_DEBUG("VK lm_head GPU time: %.3f ms", (double)(t1 - t0) * (double)bd->timestamp_period_ns / 1000000.0);
    }
    LOG_DEBUG("VK GPU timing summary: total=%.3f ms (layers=%d)", total_ms, num_layers);
}

/**
 * Compute embeddings for the current batch and submit a GPU transfer to
 * hidden[0]. Returns 0 on success, -1 on any Vulkan failure.
 * The transfer is submitted WITHOUT a CPU wait; the caller relies on
 * queue-ordered execution with the subsequent compute submission.
 */
static int submit_embedding_transfer(
    backend_vulkan_session_data_t *bd,
    inference_session_t *session,
    const gemma3_270m_config_t *cfg,
    const int *token_ids, int batch_size)
{
    size_t embed_size = (size_t)batch_size * cfg->hidden_size * sizeof(float);
    if (!bd->embedding_staging_mapped) {
        LOG_ERROR("Persistent staging mapping is NULL");
        return -1;
    }
    sapphire_embed_lookup_batch(session, token_ids, batch_size, (float*)bd->embedding_staging_mapped);

    VkResult vr = vkResetCommandBuffer(bd->transfer_cmd, 0);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to reset transfer command buffer: %d", (int)vr);
        return -1;
    }
    VkCommandBufferBeginInfo begin_xfer = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vr = vkBeginCommandBuffer(bd->transfer_cmd, &begin_xfer);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to begin transfer command buffer: %d", (int)vr);
        return -1;
    }
    VkBufferCopy copy_region = { .srcOffset = 0, .dstOffset = 0, .size = embed_size };
    vkCmdCopyBuffer(bd->transfer_cmd, bd->embedding_staging.buffer,
                    bd->scratchpad.layer_hidden[0].buffer, 1, &copy_region);
    VkBufferMemoryBarrier xfer_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = bd->scratchpad.layer_hidden[0].buffer,
        .offset = 0, .size = VK_WHOLE_SIZE
    };
    vkCmdPipelineBarrier(bd->transfer_cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 1, &xfer_barrier, 0, NULL);
    vr = vkEndCommandBuffer(bd->transfer_cmd);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to end transfer command buffer: %d", (int)vr);
        return -1;
    }
    VkSubmitInfo submit_xfer = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &bd->transfer_cmd
    };
    vr = vkQueueSubmit(bd->compute_queue, 1, &submit_xfer, VK_NULL_HANDLE);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to submit transfer command: %d", (int)vr);
        return -1;
    }
    return 0;
}

/* Submit the recorded command buffer to the GPU and optionally wait. */
static int execute_gpu_forward(backend_vulkan_session_data_t *bd,
                               int wait_for_fence,
                               double *out_record_ms,
                               double *out_wait_ms,
                               double t_stage_ms)
{
    bd->frame_counter++;
    VkResult vr = vkResetFences(bd->device, 1, &bd->compute_fence);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to reset compute fence: %d", (int)vr);
        return -1;
    }

    VkSubmitInfo si = {0};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &bd->cmd_buffer;

    VkResult vr_submit = vkQueueSubmit(bd->compute_queue, 1, &si, bd->compute_fence);
    if (vr_submit != VK_SUCCESS) {
        LOG_ERROR("Failed to submit command buffer to GPU queue: %d", (int)vr_submit);
        return -1;
    }
    *out_record_ms = monotonic_ms() - t_stage_ms;

    if (wait_for_fence) {
        double t_wait_start = monotonic_ms();
        vr = vkWaitForFences(bd->device, 1, &bd->compute_fence, VK_TRUE, UINT64_MAX);
        if (vr != VK_SUCCESS) {
            LOG_ERROR("GPU execution fence wait failed: %d", (int)vr);
            return -1;
        }
        *out_wait_ms = monotonic_ms() - t_wait_start;
    }
    return 0;
}

/* Download logits from GPU lm_head buffer and run optional debug probes. */
static int download_forward_logits(backend_vulkan_session_data_t *bd,
                                   const gemma3_270m_config_t    *cfg,
                                   float                         *logits,
                                   double                        *out_download_ms)
{
    double t0 = monotonic_ms();
    vk_transfer_ctx_t ctx = {
        .device   = bd->device,
        .phys_dev = bd->phys_dev,
        .cmd_pool = bd->cmd_pool,
        .queue    = bd->compute_queue
    };
    size_t logits_size = (size_t)cfg->vocab_size * sizeof(float);
    if (vk_buffer_download(&ctx, &bd->lm_head_logits, logits_size, logits) != 0) {
        LOG_ERROR("Failed to download logits from GPU");
        return -1;
    }
    *out_download_ms = monotonic_ms() - t0;

    LOG_DEBUG("GPU logits sample: [0]=%.6f, [1]=%.6f, [2]=%.6f, [100]=%.6f, [1000]=%.6f",
              logits[0], logits[1], logits[2], logits[100], logits[1000]);

    if (debug_gpu_enabled()) {
        probe_logits(logits, cfg->vocab_size, "CP3_logits");
        probe_buf(bd, &bd->scratchpad.norm_buf, 64, "CP3_lmhead_norm_out");
        LOG_INFO("[DBG] CP3: span<1 → final hidden near-zero (residual collapse) | span>100 → activation explosion");
    }
    return 0;
}

static int vulkan_forward_batch(inference_session_t* session, const int* token_ids,
                                int start_pos, int batch_size, float* logits) {
    if (!session || !session->backend_data) {
        LOG_ERROR("session or backend_data is NULL");
        return -1;
    }

    backend_vulkan_session_data_t *bd =
        (backend_vulkan_session_data_t *)session->backend_data;
    const gemma3_270m_config_t *cfg =
        (const gemma3_270m_config_t *)session->model_spec->variant_config;
    if (!cfg) {
        LOG_ERROR("Model config is NULL in vulkan_forward_batch");
        return -1;
    }

    /* Verify GPU resources are ready */
    if (!bd->weight_buffers || bd->num_weight_buffers == 0) {
        LOG_ERROR("Weight buffers not loaded on GPU; falling back to CPU");
        return -1;
    }

    for (int i = 0; i < NUM_PIPELINES; i++) {
        if (!bd->pipelines[i].pipeline) {
            LOG_ERROR("Pipeline %d not created; cannot execute GPU forward pass", i);
            return -1;
        }
    }

    int num_layers = cfg->num_hidden_layers;
    double t_forward_start_ms = monotonic_ms();
    double t_stage_ms = t_forward_start_ms;
    double transfer_submit_ms = 0.0;
    double record_submit_ms = 0.0;
    double compute_wait_ms = 0.0;
    double download_ms = 0.0;

    /* ========================================================================
     * Optimized Embedding Upload (Zero Allocations - Reuse Persistent Resources)
     * ======================================================================== */

    if (submit_embedding_transfer(bd, session, cfg, token_ids, batch_size) != 0)
        return -1;
    transfer_submit_ms = monotonic_ms() - t_stage_ms;
    /* Do not block CPU here.
     * Transfer and compute are submitted to the same queue; Vulkan guarantees
     * execution order, so the forward-pass compute submission recorded below
     * will naturally execute after this copy. We only wait on transfer_fence
     * at the start of the next batch before reusing transfer resources. */

    run_debug_checkpoints(bd, session, cfg, token_ids, start_pos, num_layers);
    /* Reset command buffer before recording the new forward pass */
    VkResult vr = vkResetCommandBuffer(bd->cmd_buffer, 0);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to reset command buffer: %d", (int)vr);
        return -1;
    }

    /* ========================================================================
     * SINGLE-PASS TRANSFORMER RECORDING + LM HEAD
     *
     * vulkan_record_forward_pass() records the complete 18-layer transformer
     * into one VkCommandBuffer using the pre-populated Static Descriptor Table.
     * No vkUpdateDescriptorSets calls occur during recording.
     * ======================================================================== */

    vk_fwd_opts_t _fwd_opts = {
        .batch_size = batch_size, .start_pos = start_pos,
        .emit_lmhead = (logits != NULL), .emit_argmax = 0,
        .copy_embeddings_from_staging = 0
    };
    int rc_rec = vulkan_record_forward_pass(bd, cfg, &_fwd_opts);
    if (rc_rec != 0) {
        LOG_ERROR("Failed to record forward pass command buffer");
        return -1;
    }

    /* ========================================================================
     * GPU Execution: Submit & Wait
     * ======================================================================== */


    /* GPU Execution: Submit & wait via execute_gpu_forward().           */
    /* If logits==NULL we wait here; otherwise the download fence syncs. */
    if (execute_gpu_forward(bd, (logits == NULL), &record_submit_ms,
                            &compute_wait_ms,
                            t_stage_ms + transfer_submit_ms) != 0)
        return -1;


    process_forward_timing(bd, num_layers);

    /* Update KV cache position tracking */
    vk_kv_cache_set_seq_pos(&bd->kv_cache, start_pos + batch_size);

    /* ========================================================================
     * Download Logits from GPU
     * ======================================================================== */

    if (logits) {
        double t_download_start_ms = monotonic_ms();
        /* Download logits from GPU to CPU */
        vk_transfer_ctx_t transfer_ctx = {
            .device   = bd->device,
            .phys_dev = bd->phys_dev,
            .cmd_pool = bd->cmd_pool,
            .queue    = bd->compute_queue
        };

        size_t logits_size = (size_t)cfg->vocab_size * sizeof(float);
        int rc_download = vk_buffer_download(&transfer_ctx, &bd->lm_head_logits,
                             logits_size, logits);
        if (rc_download != 0) {
            LOG_ERROR("Failed to download logits from GPU");
            return -1;
        }
        download_ms = monotonic_ms() - t_download_start_ms;

        LOG_DEBUG("GPU logits sample: [0]=%.6f, [1]=%.6f, [2]=%.6f, [100]=%.6f, [1000]=%.6f",
                  logits[0], logits[1], logits[2], logits[100], logits[1000]);

        /* ================================================================
         * DEBUG CP3 — Logit quality analysis.
         * Checks:
         *   span (max-min): healthy model has span > 10. Span < 1 → hidden
         *     state is near-zero; span ≈ 0 → uniform garbage.
         *   top-5 tokens: with a sensible prompt they should be coherent
         *     English subwords, not random multilingual fragments.
         * ================================================================ */
        if (debug_gpu_enabled()) {
            probe_logits(logits, cfg->vocab_size, "CP3_logits");
            probe_buf(bd, &bd->scratchpad.norm_buf, 64, "CP3_lmhead_norm_out");
            LOG_INFO("[DBG] CP3: span<1 → final hidden near-zero (residual collapse) | span>100 → activation explosion");
        }
    }

    LOG_DEBUG("Vulkan forward batch: %d layers, batch=%d, pos=%d (SDT single-pass)",
              num_layers, batch_size, start_pos);
    if (bd->timing_enabled) {
        LOG_INFO("VK stage timing (host): transfer_submit=%.3f ms, record_submit=%.3f ms, compute_wait=%.3f ms, download(wait+copy)=%.3f ms, total=%.3f ms",
                 transfer_submit_ms,
                 record_submit_ms,
                 compute_wait_ms,
                 download_ms,
                 monotonic_ms() - t_forward_start_ms);
    }

    return 0;
}

/**
 * Vulkan backend: Execute forward batch and return selected token IDs.
 *
 * Uses LM-head + GPU argmax and downloads only int32 token IDs.
 */
/**
 * Process and log GPU timing query results for vulkan_forward_select_batch.
 */
static void process_select_timing(
    backend_vulkan_session_data_t *bd,
    const gemma3_270m_config_t *cfg)
{
    if (!bd->timing_enabled || bd->timing_query_pool == VK_NULL_HANDLE ||
        bd->timing_query_last_count < 2) return;

    uint64_t ts[256] = {0};
    uint32_t qcount = bd->timing_query_last_count;
    if (qcount > 256u) qcount = 256u;

    VkResult qres = vkGetQueryPoolResults(bd->device, bd->timing_query_pool,
        0, qcount, sizeof(uint64_t) * qcount, ts, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (qres != VK_SUCCESS) {
        LOG_WARN("vkGetQueryPoolResults failed in select path: %d", (int)qres);
        return;
    }

    if (bd->timing_kernel_query_count > 1 && qcount > 1) {
        static const char *knames[] = {
            "save_residual", "pre_attn_norm", "qkv_projections", "qk_norm", "rope",
            "kv_write", "attention", "o_proj", "post_attn_norm", "attn_residual_add",
            "pre_ffn_norm", "gate_up_proj", "geglu", "down_proj", "post_ffn_norm", "ffn_residual_add"
        };
        double total_ms = (double)(ts[qcount - 1] - ts[0]) * (double)bd->timestamp_period_ns / 1000000.0;
        uint32_t first = bd->timing_kernel_first_query;
        uint32_t end   = first + bd->timing_kernel_query_count;
        if (end > qcount) end = qcount;
        uint32_t intervals = (end > first) ? (end - first - 1u) : 0u;
        for (uint32_t i = 0; i < intervals && first < qcount; i++) {
            uint64_t t0 = ts[first + i], t1 = ts[first + i + 1u];
            double stage_ms = (double)(t1 - t0) * (double)bd->timestamp_period_ns / 1000000.0;
            const char *sn = (i < (sizeof(knames)/sizeof(knames[0]))) ? knames[i] : "stage_extra";
            LOG_INFO("VK layer %02d kernel %-18s: %.3f ms", bd->timing_kernel_layer, sn, stage_ms);
        }
        LOG_INFO("VK select GPU timing summary: total=%.3f ms", total_ms);
        return;
    }

    double total_ms = 0.0;
    int pair_idx = 0;
    for (int layer = 0; layer < cfg->num_hidden_layers && (pair_idx + 1) < (int)qcount; layer++) {
        uint64_t t0 = ts[pair_idx++], t1 = ts[pair_idx++];
        double layer_ms = (double)(t1 - t0) * (double)bd->timestamp_period_ns / 1000000.0;
        total_ms += layer_ms;
        LOG_DEBUG("VK select layer %02d GPU time: %.3f ms", layer, layer_ms);
    }
    if ((pair_idx + 1) < (int)qcount) {
        uint64_t t0 = ts[pair_idx++], t1 = ts[pair_idx++];
        total_ms += (double)(t1 - t0) * (double)bd->timestamp_period_ns / 1000000.0;
        LOG_DEBUG("VK select lm_head+argmax GPU time: %.3f ms",
                  (double)(ts[pair_idx-1] - ts[pair_idx-2]) * (double)bd->timestamp_period_ns / 1000000.0);
    }
    LOG_INFO("VK select GPU timing summary: total=%.3f ms", total_ms);
}

static int vulkan_forward_select_batch(inference_session_t* session, const int* token_ids,
                                       int start_pos, int batch_size, int* selected_ids) {
    if (!session || !session->backend_data || !selected_ids) {
        LOG_ERROR("session/backend_data/selected_ids is NULL");
        return -1;
    }

    backend_vulkan_session_data_t *bd =
        (backend_vulkan_session_data_t *)session->backend_data;
    const gemma3_270m_config_t *cfg =
        (const gemma3_270m_config_t *)session->model_spec->variant_config;
    if (!cfg) {
        LOG_ERROR("Model config is NULL in vulkan_forward_select_batch");
        return -1;
    }

    if (!bd->weight_buffers || bd->num_weight_buffers == 0) {
        LOG_ERROR("Weight buffers not loaded on GPU; cannot select on GPU");
        return -1;
    }
    if (!bd->pipelines || !bd->pipelines[PIPELINE_ARGMAX_F32].pipeline) {
        LOG_ERROR("Argmax pipeline not available");
        return -1;
    }

    double t_start_ms = monotonic_ms();
    double transfer_submit_ms = 0.0;
    double record_submit_ms = 0.0;
    double compute_wait_ms = 0.0;
    double download_ms = 0.0;

    VkResult vr = VK_SUCCESS;
    if (!bd->embedding_staging_mapped) {
        LOG_ERROR("Persistent staging mapping is NULL");
        return -1;
    }
    sapphire_embed_lookup_batch(session, token_ids, batch_size, (float*)bd->embedding_staging_mapped);

    transfer_submit_ms = 0.0;

    vr = vkResetCommandBuffer(bd->cmd_buffer, 0);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to reset command buffer: %d", (int)vr);
        return -1;
    }

    vk_fwd_opts_t _sel_opts = {
        .batch_size = batch_size, .start_pos = start_pos,
        .emit_lmhead = 1, .emit_argmax = 1,
        .copy_embeddings_from_staging = 1
    };
    int rc_rec = vulkan_record_forward_pass(bd, cfg, &_sel_opts);
    if (rc_rec != 0) {
        LOG_ERROR("Failed to record forward+argmax command buffer");
        return -1;
    }

    bd->frame_counter++;
    vr = vkResetFences(bd->device, 1, &bd->compute_fence);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to reset compute fence: %d", (int)vr);
        return -1;
    }

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &bd->cmd_buffer;

    vr = vkQueueSubmit(bd->compute_queue, 1, &submit_info, bd->compute_fence);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed to submit command buffer to GPU queue: %d", (int)vr);
        return -1;
    }
    record_submit_ms = monotonic_ms() - t_start_ms - transfer_submit_ms;

    double t_wait_start_ms = monotonic_ms();
    vr = vkWaitForFences(bd->device, 1, &bd->compute_fence, VK_TRUE, UINT64_MAX);
    if (vr != VK_SUCCESS) {
        LOG_ERROR("Failed waiting for compute fence in select path: %d", (int)vr);
        return -1;
    }
    compute_wait_ms = monotonic_ms() - t_wait_start_ms;

    process_select_timing(bd, cfg);

    double t_download_start_ms = monotonic_ms();
    void *mapped_ids = NULL;
    vr = vkMapMemory(bd->device,
                     bd->selected_token_ids.memory,
                     0,
                     (VkDeviceSize)((size_t)batch_size * sizeof(int32_t)),
                     0,
                     &mapped_ids);
    if (vr != VK_SUCCESS || !mapped_ids) {
        LOG_ERROR("Failed to map selected token id buffer: %d", (int)vr);
        return -1;
    }
    memcpy(selected_ids, mapped_ids, (size_t)batch_size * sizeof(int32_t));
    vkUnmapMemory(bd->device, bd->selected_token_ids.memory);

    download_ms = monotonic_ms() - t_download_start_ms;

    vk_kv_cache_set_seq_pos(&bd->kv_cache, start_pos + batch_size);
    if (bd->timing_enabled) {
        LOG_INFO("VK select timing (host): transfer_submit=%.3f ms, record_submit=%.3f ms, compute_wait=%.3f ms, readback(ids)=%.3f ms, total=%.3f ms",
                 transfer_submit_ms,
                 record_submit_ms,
             compute_wait_ms,
                 download_ms,
                 monotonic_ms() - t_start_ms);
    }
    return 0;
}

/**
 * Vulkan backend: Reset session for new sequence.
 *
 * Resets KV cache position and frame counter.
 *
 * @param session Inference session
 */
static void vulkan_reset(inference_session_t* session) {
    if (!session || !session->backend_data) {
        return;
    }

    backend_vulkan_session_data_t *backend_data = (backend_vulkan_session_data_t *)session->backend_data;

    /* Reset KV cache position for new sequence */
    vk_kv_cache_reset(&backend_data->kv_cache);

    /* Reset frame counter for timeline semaphore sync */
    backend_data->frame_counter = 0;

    LOG_DEBUG("Vulkan backend reset for new sequence");
}

/**
 * Vulkan backend implementation (P11-02 + P11-03 integrated).
 */
sapphire_backend_t backend_vulkan_impl = {
    .type = SAPPHIRE_BACKEND_TYPE_VULKAN,
    .name = "vulkan",
    .session_init = vulkan_session_init,
    .session_destroy = vulkan_session_destroy,
    .forward_batch = vulkan_forward_batch,
    .forward_select_batch = vulkan_forward_select_batch,
    .reset = vulkan_reset,
};

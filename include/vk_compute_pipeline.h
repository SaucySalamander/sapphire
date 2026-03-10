/**
 * @file vk_compute_pipeline.h
 * @brief Vulkan compute pipeline abstraction (P11-03).
 *
 * Standardizes shader module loading, descriptor set layout creation,
 * and compute pipeline creation for all GPU kernels.
 *
 * Integrates with P11-02 buffer types:
 * - Weight buffers (device-local, read-only)
 * - KV cache (device-local, read-write)
 * - GPU scratchpad (device-local, ephemeral ping-pong)
 * - Input/Output ring buffers (timeline semaphore sync)
 */

#ifndef VK_COMPUTE_PIPELINE_H
#define VK_COMPUTE_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

/* Include Vulkan headers */
#include <vulkan/vulkan.h>

/* ========================================================================
 * Push Constants (Dynamic Per-Dispatch Parameters)
 * ======================================================================== */

/**
 * Standard push constants for all GPU kernels.
 * 
 * Carries per-dispatch parameters to avoid descriptor updates.
 * Size: 64 bytes (well within 128-byte minimum guaranteed limit).
 */
typedef struct {
    uint32_t layer_idx;        /* Current layer index (0 to num_layers-1) */
    uint32_t batch_size;       /* Number of tokens in batch (1 for decoding) */
    uint32_t seq_pos;          /* Last absolute sequence position (start_pos + batch_size - 1) */
    uint32_t max_seq_len;      /* Maximum sequence length */
    uint32_t d_model;          /* Model hidden dimension */
    uint32_t num_heads;        /* Number of attention heads */
    uint32_t head_dim;         /* Dimension per attention head */
    uint32_t d_ff;             /* FFN intermediate dimension */
    uint32_t stride_0;         /* Tensor stride dimension 0 */
    uint32_t stride_1;         /* Tensor stride dimension 1 */
    uint32_t start_pos;        /* First absolute position of this batch (for RoPE/causal mask) */
    uint32_t reserved[5];      /* Padding for future use */
} vk_kernel_push_constants_t;

/* ========================================================================
 * Descriptor Set Layout Presets (P11-02 Buffer Type Support)
 * ======================================================================== */

/**
 * Descriptor set layout type for standardized binding patterns.
 */
typedef enum {
    VK_DESC_LAYOUT_WEIGHT_ONLY = 0,    /* 1-2 read-only storage buffers */
    VK_DESC_LAYOUT_PROJECTION,         /* 1 input + 1 weight + 1 output */
    VK_DESC_LAYOUT_ATTENTION,          /* Q/K/V + cached K/V + output */
    VK_DESC_LAYOUT_RING_IO,            /* 1 ring buffer slot */
    VK_DESC_LAYOUT_CUSTOM              /* User-defined bindings */
} vk_desc_layout_type_t;

/**
 * Descriptor binding specification.
 */
typedef struct {
    uint32_t binding;                  /* Binding index in shader */
    VkDescriptorType type;             /* Storage buffer, uniform buffer, etc. */
    VkShaderStageFlags stage_flags;    /* Compute shader only (typically) */
} vk_desc_binding_t;

/* ========================================================================
 * Compute Pipeline State
 * ======================================================================== */


/**
 * Compute pipeline container (opaque to users, managed by abstraction).
 */
typedef struct {
    VkShaderModule shader_module;
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    
    /* Pre-allocated descriptor pool for this pipeline */
    VkDescriptorPool desc_pool;
    VkDescriptorSet *desc_sets;        /* Array of allocated descriptor sets */
    uint32_t num_desc_sets;            /* Number of pre-allocated sets */
    
    /* Push constant range */
    uint32_t push_constant_size;
} vk_compute_pipeline_t;

/* ========================================================================
 * Pipeline Creation Configuration
 * ======================================================================== */

/**
 * Configuration for pipeline creation.
 */
typedef struct {
    const char *shader_path;           /* Path to SPIR-V .spv file */
    const char *entry_point;           /* Shader entry point (default: "main") */
    
    /* Descriptor set layout (use preset or custom) */
    vk_desc_layout_type_t layout_type;
    const vk_desc_binding_t *custom_bindings; /* If layout_type == CUSTOM */
    uint32_t num_custom_bindings;
    
    /* Push constants */
    uint32_t push_constant_size;       /* Size of push constants (0 if none) */
    
    /* Descriptor set pre-allocation */
    uint32_t num_desc_sets;            /* Number of descriptor sets to pre-allocate */
} vk_pipeline_config_t;

/* ========================================================================
 * Pipeline Management Functions
 * ======================================================================== */

/**
 * Create compute pipeline from SPIR-V shader.
 *
 * Loads shader, creates descriptor set layout, pipeline layout, and pipeline.
 * Pre-allocates descriptor sets to avoid hot-loop allocation.
 *
 * @param device          Vulkan logical device
 * @param config          Pipeline configuration
 * @param out_pipeline    Output pipeline container
 *
 * @return 0 on success, -1 on error (logged)
 */
int vk_pipeline_create(
    VkDevice device,
    const vk_pipeline_config_t *config,
    vk_compute_pipeline_t *out_pipeline
);

/**
 * Destroy compute pipeline and free all resources.
 *
 * @param device    Vulkan logical device
 * @param pipeline  Pipeline to destroy
 */
void vk_pipeline_destroy(VkDevice device, vk_compute_pipeline_t *pipeline);

/**
 * Bind pipeline and descriptor sets to command buffer.
 *
 * @param cmd_buf       Command buffer
 * @param pipeline      Pipeline to bind
 * @param desc_set_idx  Descriptor set index (if multiple pre-allocated)
 */
void vk_pipeline_bind(
    VkCommandBuffer cmd_buf,
    const vk_compute_pipeline_t *pipeline,
    uint32_t desc_set_idx
);

/**
 * Record compute dispatch with push constants.
 *
 * @param cmd_buf           Command buffer
 * @param pipeline          Pipeline (must be bound via vk_pipeline_bind)
 * @param push_constants    Push constant data (can be NULL if size = 0)
 * @param group_count_x     Workgroup count X
 * @param group_count_y     Workgroup count Y
 * @param group_count_z     Workgroup count Z
 */
void vk_pipeline_dispatch(
    VkCommandBuffer cmd_buf,
    const vk_compute_pipeline_t *pipeline,
    const void *push_constants,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z
);

/**
 * Update descriptor set with buffer bindings.
 *
 * Writes storage buffer bindings to a descriptor set.
 *
 * @param device        Vulkan logical device
 * @param desc_set      Descriptor set to update
 * @param binding       Binding index
 * @param buffer        Vulkan buffer handle
 * @param offset        Offset in buffer
 * @param range         Range to bind (VK_WHOLE_SIZE for entire buffer)
 */
void vk_pipeline_update_descriptor_buffer(
    VkDevice device,
    VkDescriptorSet desc_set,
    uint32_t binding,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkDeviceSize range
);

/**
 * Submit command buffer with timeline semaphore synchronization.
 *
 * Supports P11-02 ring buffer timeline semaphore waits/signals.
 *
 * @param queue                 Vulkan queue
 * @param cmd_buf               Command buffer to submit
 * @param wait_semaphore        Timeline semaphore to wait on (NULL if none)
 * @param wait_value            Timeline value to wait for
 * @param signal_semaphore      Timeline semaphore to signal (NULL if none)
 * @param signal_value          Timeline value to signal
 *
 * @return 0 on success, -1 on error (logged)
 */
int vk_pipeline_submit_timeline(
    VkQueue queue,
    VkCommandBuffer cmd_buf,
    VkSemaphore wait_semaphore,
    uint64_t wait_value,
    VkSemaphore signal_semaphore,
    uint64_t signal_value
);

#endif /* VK_COMPUTE_PIPELINE_H */

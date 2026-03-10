/**
 * @file vk_compute_pipeline.c
 * @brief Vulkan compute pipeline abstraction implementation (P11-03).
 *
 * Implements shader loading, descriptor layout creation, pipeline creation,
 * and command buffer recording helpers.
 */

#include "../../../../include/vk_compute_pipeline.h"
#include "../../../../include/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ========================================================================
 * Shader Module Loading
 * ======================================================================== */

/**
 * Load SPIR-V shader from file.
 *
 * @param shader_path   Path to .spv file
 * @param out_code      Output: Allocated SPIR-V code buffer (caller must free)
 * @param out_size      Output: Size of SPIR-V code in bytes
 *
 * @return 0 on success, -1 on error (logged)
 */
static int load_spirv_shader(const char *shader_path, uint32_t **out_code, size_t *out_size) {
    if (!shader_path || !out_code || !out_size) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return -1;
    }

    FILE *f = fopen(shader_path, "rb");
    if (!f) {
        LOG_ERROR("Failed to open shader file: %s", shader_path);
        return -1;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0 || file_size % 4 != 0) {
        LOG_ERROR("Invalid SPIR-V file size: %ld (must be multiple of 4)", file_size);
        fclose(f);
        return -1;
    }

    /* Allocate buffer */
    uint32_t *code = malloc((size_t)file_size);
    if (!code) {
        LOG_ERROR("Failed to allocate %ld bytes for shader code", file_size);
        fclose(f);
        return -1;
    }

    /* Read shader code */
    size_t read_count = fread(code, 1, (size_t)file_size, f);
    fclose(f);

    if (read_count != (size_t)file_size) {
        LOG_ERROR("Failed to read shader file (read %zu, expected %ld)", read_count, file_size);
        free(code);
        return -1;
    }

    *out_code = code;
    *out_size = (size_t)file_size;

    LOG_DEBUG("Loaded SPIR-V shader: %s (%zu bytes)", shader_path, (size_t)file_size);
    return 0;
}

/**
 * Create shader module from SPIR-V code.
 *
 * @param device        Vulkan device
 * @param code          SPIR-V code buffer
 * @param code_size     Size of code in bytes
 * @param out_module    Output shader module
 *
 * @return 0 on success, -1 on error (logged)
 */
static int create_shader_module(VkDevice device, const uint32_t *code, size_t code_size,
                                VkShaderModule *out_module) {
    if (!device || !code || !out_module) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return -1;
    }

    VkShaderModuleCreateInfo create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = code_size;
    create_info.pCode = code;

    VkResult res = vkCreateShaderModule(device, &create_info, NULL, out_module);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateShaderModule failed: %d", res);
        return -1;
    }

    LOG_DEBUG("Created shader module (%zu bytes)", code_size);
    return 0;
}

/* ========================================================================
 * Descriptor Set Layout Presets
 * ======================================================================== */

/**
 * Get preset descriptor bindings for common patterns.
 *
 * @param layout_type       Preset type (WEIGHT_ONLY, PROJECTION, etc.)
 * @param out_bindings      Output: Pointer to static binding array
 * @param out_count         Output: Number of bindings
 *
 * @return 0 on success, -1 if preset not found
 */
static int get_preset_bindings(vk_desc_layout_type_t layout_type,
                               const vk_desc_binding_t **out_bindings,
                               uint32_t *out_count) {
    /* Preset binding definitions (static, read-only) */
    static const vk_desc_binding_t weight_only_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}
    };

    static const vk_desc_binding_t projection_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}, /* Input */
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}, /* Weight */
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}  /* Output */
    };

    static const vk_desc_binding_t attention_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}, /* Q */
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}, /* K */
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}, /* V */
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}, /* KV cache */
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}  /* Output */
    };

    static const vk_desc_binding_t ring_io_bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}  /* Ring slot */
    };

    switch (layout_type) {
        case VK_DESC_LAYOUT_WEIGHT_ONLY:
            *out_bindings = weight_only_bindings;
            *out_count = 2;
            return 0;
        case VK_DESC_LAYOUT_PROJECTION:
            *out_bindings = projection_bindings;
            *out_count = 3;
            return 0;
        case VK_DESC_LAYOUT_ATTENTION:
            *out_bindings = attention_bindings;
            *out_count = 5;
            return 0;
        case VK_DESC_LAYOUT_RING_IO:
            *out_bindings = ring_io_bindings;
            *out_count = 1;
            return 0;
        default:
            LOG_ERROR("Unknown preset layout type: %d", layout_type);
            return -1;
    }
}

/**
 * Create descriptor set layout from bindings.
 *
 * @param device        Vulkan device
 * @param bindings      Binding specifications
 * @param num_bindings  Number of bindings
 * @param out_layout    Output descriptor set layout
 *
 * @return 0 on success, -1 on error (logged)
 */
static int create_descriptor_layout(VkDevice device, const vk_desc_binding_t *bindings,
                                    uint32_t num_bindings, VkDescriptorSetLayout *out_layout) {
    if (!device || !bindings || !out_layout) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return -1;
    }

    if (num_bindings == 0) {
        LOG_ERROR("num_bindings cannot be 0");
        return -1;
    }

    /* Convert to Vulkan binding structures */
    VkDescriptorSetLayoutBinding *vk_bindings = malloc(num_bindings * sizeof(VkDescriptorSetLayoutBinding));
    if (!vk_bindings) {
        LOG_ERROR("Failed to allocate binding array (%u bindings)", num_bindings);
        return -1;
    }

    for (uint32_t i = 0; i < num_bindings; i++) {
        vk_bindings[i].binding = bindings[i].binding;
        vk_bindings[i].descriptorType = bindings[i].type;
        vk_bindings[i].descriptorCount = 1;
        vk_bindings[i].stageFlags = bindings[i].stage_flags;
        vk_bindings[i].pImmutableSamplers = NULL;
    }

    VkDescriptorSetLayoutCreateInfo layout_info = {0};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = num_bindings;
    layout_info.pBindings = vk_bindings;

    VkResult res = vkCreateDescriptorSetLayout(device, &layout_info, NULL, out_layout);
    free(vk_bindings);

    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateDescriptorSetLayout failed: %d", res);
        return -1;
    }

    LOG_DEBUG("Created descriptor set layout (%u bindings)", num_bindings);
    return 0;
}

/* ========================================================================
 * Pipeline Creation
 * ======================================================================== */

/**
 * Allocate descriptor pool and descriptor sets for a pipeline.
 *
 * On failure, only resources allocated within this call are cleaned up;
 * the caller must handle cleanup of shader_module / layouts / pipeline.
 *
 * @return 0 on success, -1 on error (logged)
 */
static int alloc_pipeline_descriptor_sets(VkDevice device,
                                          vk_compute_pipeline_t *out_pipeline,
                                          uint32_t num_bindings,
                                          uint32_t num_desc_sets) {
    VkDescriptorPoolSize pool_size = {0};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = num_bindings * num_desc_sets;

    VkDescriptorPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = num_desc_sets;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    VkResult res = vkCreateDescriptorPool(device, &pool_info, NULL, &out_pipeline->desc_pool);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateDescriptorPool failed: %d", res);
        return -1;
    }

    out_pipeline->desc_sets = calloc(num_desc_sets, sizeof(VkDescriptorSet));
    if (!out_pipeline->desc_sets) {
        LOG_ERROR("Failed to allocate descriptor set array (%u sets)", num_desc_sets);
        vkDestroyDescriptorPool(device, out_pipeline->desc_pool, NULL);
        out_pipeline->desc_pool = VK_NULL_HANDLE;
        return -1;
    }

    VkDescriptorSetLayout *layouts = malloc(num_desc_sets * sizeof(VkDescriptorSetLayout));
    if (!layouts) {
        LOG_ERROR("Failed to allocate layout array (%u sets)", num_desc_sets);
        free(out_pipeline->desc_sets);
        out_pipeline->desc_sets = NULL;
        vkDestroyDescriptorPool(device, out_pipeline->desc_pool, NULL);
        out_pipeline->desc_pool = VK_NULL_HANDLE;
        return -1;
    }

    for (uint32_t i = 0; i < num_desc_sets; i++) {
        layouts[i] = out_pipeline->desc_layout;
    }

    VkDescriptorSetAllocateInfo alloc_info = {0};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = out_pipeline->desc_pool;
    alloc_info.descriptorSetCount = num_desc_sets;
    alloc_info.pSetLayouts = layouts;

    res = vkAllocateDescriptorSets(device, &alloc_info, out_pipeline->desc_sets);
    free(layouts);

    if (res != VK_SUCCESS) {
        LOG_ERROR("vkAllocateDescriptorSets failed: %d", res);
        free(out_pipeline->desc_sets);
        out_pipeline->desc_sets = NULL;
        vkDestroyDescriptorPool(device, out_pipeline->desc_pool, NULL);
        out_pipeline->desc_pool = VK_NULL_HANDLE;
        return -1;
    }

    out_pipeline->num_desc_sets = num_desc_sets;
    return 0;
}

int vk_pipeline_create(VkDevice device, const vk_pipeline_config_t *config,
                       vk_compute_pipeline_t *out_pipeline) {
    if (!device || !config || !out_pipeline) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return -1;
    }

    if (!config->shader_path) {
        LOG_ERROR("shader_path is NULL");
        return -1;
    }

    /* Zero-initialize output */
    memset(out_pipeline, 0, sizeof(vk_compute_pipeline_t));

    /* Load SPIR-V shader */
    uint32_t *spirv_code = NULL;
    size_t spirv_size = 0;
    if (load_spirv_shader(config->shader_path, &spirv_code, &spirv_size) != 0) {
        return -1;
    }

    /* Create shader module */
    if (create_shader_module(device, spirv_code, spirv_size, &out_pipeline->shader_module) != 0) {
        free(spirv_code);
        return -1;
    }
    free(spirv_code);

    /* Determine descriptor bindings */
    const vk_desc_binding_t *bindings = NULL;
    uint32_t num_bindings = 0;

    if (config->layout_type == VK_DESC_LAYOUT_CUSTOM) {
        if (!config->custom_bindings || config->num_custom_bindings == 0) {
            LOG_ERROR("Custom layout requires custom_bindings and num_custom_bindings");
            vkDestroyShaderModule(device, out_pipeline->shader_module, NULL);
            return -1;
        }
        bindings = config->custom_bindings;
        num_bindings = config->num_custom_bindings;
    } else {
        if (get_preset_bindings(config->layout_type, &bindings, &num_bindings) != 0) {
            vkDestroyShaderModule(device, out_pipeline->shader_module, NULL);
            return -1;
        }
    }

    /* Create descriptor set layout */
    if (create_descriptor_layout(device, bindings, num_bindings, &out_pipeline->desc_layout) != 0) {
        vkDestroyShaderModule(device, out_pipeline->shader_module, NULL);
        return -1;
    }

    /* Create pipeline layout with push constants */
    VkPushConstantRange push_constant_range = {0};
    VkPipelineLayoutCreateInfo pipeline_layout_info = {0};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &out_pipeline->desc_layout;

    if (config->push_constant_size > 0) {
        push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant_range.offset = 0;
        push_constant_range.size = config->push_constant_size;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant_range;
    }

    VkResult res = vkCreatePipelineLayout(device, &pipeline_layout_info, NULL, &out_pipeline->pipeline_layout);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreatePipelineLayout failed: %d", res);
        vkDestroyDescriptorSetLayout(device, out_pipeline->desc_layout, NULL);
        vkDestroyShaderModule(device, out_pipeline->shader_module, NULL);
        return -1;
    }

    /* Create compute pipeline */
    VkComputePipelineCreateInfo pipeline_info = {0};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = out_pipeline->shader_module;
    pipeline_info.stage.pName = config->entry_point ? config->entry_point : "main";
    pipeline_info.layout = out_pipeline->pipeline_layout;

    res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &out_pipeline->pipeline);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkCreateComputePipelines failed: %d", res);
        vkDestroyPipelineLayout(device, out_pipeline->pipeline_layout, NULL);
        vkDestroyDescriptorSetLayout(device, out_pipeline->desc_layout, NULL);
        vkDestroyShaderModule(device, out_pipeline->shader_module, NULL);
        return -1;
    }

    /* Pre-allocate descriptor pool and sets */
    if (config->num_desc_sets > 0) {
        int rc = alloc_pipeline_descriptor_sets(device, out_pipeline,
                                                num_bindings, config->num_desc_sets);
        if (rc != 0) {
            vkDestroyPipeline(device, out_pipeline->pipeline, NULL);
            vkDestroyPipelineLayout(device, out_pipeline->pipeline_layout, NULL);
            vkDestroyDescriptorSetLayout(device, out_pipeline->desc_layout, NULL);
            vkDestroyShaderModule(device, out_pipeline->shader_module, NULL);
            return -1;
        }
    }

    out_pipeline->push_constant_size = config->push_constant_size;

    LOG_INFO("Created compute pipeline: shader=%s, desc_sets=%u, push_const_size=%u",
             config->shader_path, config->num_desc_sets, config->push_constant_size);
    return 0;
}

void vk_pipeline_destroy(VkDevice device, vk_compute_pipeline_t *pipeline) {
    if (!device || !pipeline) {
        return;
    }

    if (pipeline->desc_sets) {
        free(pipeline->desc_sets);
    }

    if (pipeline->desc_pool) {
        vkDestroyDescriptorPool(device, pipeline->desc_pool, NULL);
    }

    if (pipeline->pipeline) {
        vkDestroyPipeline(device, pipeline->pipeline, NULL);
    }

    if (pipeline->pipeline_layout) {
        vkDestroyPipelineLayout(device, pipeline->pipeline_layout, NULL);
    }

    if (pipeline->desc_layout) {
        vkDestroyDescriptorSetLayout(device, pipeline->desc_layout, NULL);
    }

    if (pipeline->shader_module) {
        vkDestroyShaderModule(device, pipeline->shader_module, NULL);
    }

    memset(pipeline, 0, sizeof(vk_compute_pipeline_t));
    LOG_DEBUG("Destroyed compute pipeline");
}

/* ========================================================================
 * Command Buffer Recording Helpers
 * ======================================================================== */

void vk_pipeline_bind(VkCommandBuffer cmd_buf, const vk_compute_pipeline_t *pipeline,
                     uint32_t desc_set_idx) {
    if (!cmd_buf || !pipeline) {
        LOG_ERROR("cmd_buf or pipeline is NULL");
        return;
    }

    vkCmdBindPipeline(cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);

    if (pipeline->desc_sets && desc_set_idx < pipeline->num_desc_sets) {
        vkCmdBindDescriptorSets(
            cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline->pipeline_layout,
            0, 1, &pipeline->desc_sets[desc_set_idx],
            0, NULL
        );
    }
}

void vk_pipeline_dispatch(VkCommandBuffer cmd_buf, const vk_compute_pipeline_t *pipeline,
                         const void *push_constants, uint32_t group_count_x,
                         uint32_t group_count_y, uint32_t group_count_z) {
    if (!cmd_buf || !pipeline) {
        LOG_ERROR("cmd_buf or pipeline is NULL");
        return;
    }

    if (push_constants && pipeline->push_constant_size > 0) {
        vkCmdPushConstants(
            cmd_buf,
            pipeline->pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            pipeline->push_constant_size,
            push_constants
        );
    }

    vkCmdDispatch(cmd_buf, group_count_x, group_count_y, group_count_z);
}

void vk_pipeline_update_descriptor_buffer(VkDevice device, VkDescriptorSet desc_set,
                                         uint32_t binding, VkBuffer buffer,
                                         VkDeviceSize offset, VkDeviceSize range) {
    if (!device || !desc_set || !buffer) {
        LOG_ERROR("Invalid parameters (NULL pointer)");
        return;
    }

    VkDescriptorBufferInfo buffer_info = {0};
    buffer_info.buffer = buffer;
    buffer_info.offset = offset;
    buffer_info.range = range;

    VkWriteDescriptorSet write = {0};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = desc_set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);
}

/* ========================================================================
 * Timeline Semaphore Queue Submit
 * ======================================================================== */

int vk_pipeline_submit_timeline(VkQueue queue, VkCommandBuffer cmd_buf,
                                VkSemaphore wait_semaphore, uint64_t wait_value,
                                VkSemaphore signal_semaphore, uint64_t signal_value) {
    if (!queue || !cmd_buf) {
        LOG_ERROR("queue or cmd_buf is NULL");
        return -1;
    }

    VkTimelineSemaphoreSubmitInfo timeline_info = {0};
    timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;

    uint64_t wait_values[1] = {wait_value};
    uint64_t signal_values[1] = {signal_value};

    uint32_t wait_count = 0;
    uint32_t signal_count = 0;

    if (wait_semaphore) {
        timeline_info.waitSemaphoreValueCount = 1;
        timeline_info.pWaitSemaphoreValues = wait_values;
        wait_count = 1;
    }

    if (signal_semaphore) {
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = signal_values;
        signal_count = 1;
    }

    VkPipelineStageFlags wait_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    VkSubmitInfo submit_info = {0};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = &timeline_info;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd_buf;

    if (wait_count > 0) {
        submit_info.waitSemaphoreCount = wait_count;
        submit_info.pWaitSemaphores = &wait_semaphore;
        submit_info.pWaitDstStageMask = &wait_stages;
    }

    if (signal_count > 0) {
        submit_info.signalSemaphoreCount = signal_count;
        submit_info.pSignalSemaphores = &signal_semaphore;
    }

    VkResult res = vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        LOG_ERROR("vkQueueSubmit failed: %d", res);
        return -1;
    }

    LOG_DEBUG("Submitted command buffer with timeline semaphores (wait=%lu, signal=%lu)",
              wait_value, signal_value);
    return 0;
}

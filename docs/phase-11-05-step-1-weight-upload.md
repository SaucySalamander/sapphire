# Phase 11-05 Step 1: Upload Final Layer Weights to GPU

**Status:** Not Started  
**Dependencies:** `docs/vulkan-core-principles.md`, `docs/phase-11-05-gpu-lm-head-overview.md`

## Objective

Upload two non-layered weight buffers to GPU:
1. `final_norm_weight` - Final RMSNorm weight (256 floats)
2. `embedding_weight` - Vocabulary embedding matrix (262,144 × 256, BF16 format)

## Data Structures

### Source Tensors (CPU)
```c
// From llm_model_t (already loaded by gemma3_loader.c)
llm_model_t *model = (llm_model_t *)spec->llm_model;
tensor_t *final_norm = model->norm_final_weight;   // [256] BF16
tensor_t *embedding = model->embedding_weight;     // [262144, 256] BF16
```

### Destination Buffers (GPU)
```c
// In backend_vulkan.h backend_vulkan_session_data_t:
vk_buffer_t final_norm_weight;      /* [hidden_size] F32 */
vk_buffer_t embedding_weight;       /* [vocab_size × hidden_size] F32 */
```

**Format Conversion:** BF16 → F32 on CPU before upload (vocabulary projection shader expects F32)

## Implementation Location

**File:** `src/inference/backend_vulkan.c`  
**Function:** `vulkan_session_init()` (after RoPE cache upload, before pipeline creation)  
**Line Range:** After line ~970 (after RoPE cache upload completes)

## Upload Process (Using Persistent Transfer Resources)

### Step 1.1: Convert BF16 → F32 (CPU)
```c
extern void bf16_to_f32_vec(float *dst, const uint16_t *src, size_t n);

size_t final_norm_f32_size = config->hidden_size * sizeof(float);
size_t embedding_f32_size = config->vocab_size * config->hidden_size * sizeof(float);

float *final_norm_f32 = malloc(final_norm_f32_size);
float *embedding_f32 = malloc(embedding_f32_size);

const uint16_t *final_norm_bf16 = (const uint16_t *)tensor_data(model->norm_final_weight);
const uint16_t *embedding_bf16 = (const uint16_t *)tensor_data(model->embedding_weight);

bf16_to_f32_vec(final_norm_f32, final_norm_bf16, config->hidden_size);
bf16_to_f32_vec(embedding_f32, embedding_bf16, config->vocab_size * config->hidden_size);
```

### Step 1.2: Create GPU Buffers
```c
rc = vk_buffer_create(
    backend_data->device, backend_data->phys_dev,
    final_norm_f32_size,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    &backend_data->final_norm_weight
);
// Check rc, cleanup on failure

rc = vk_buffer_create(
    backend_data->device, backend_data->phys_dev,
    embedding_f32_size,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    &backend_data->embedding_weight
);
// Check rc, cleanup on failure
```

### Step 1.3: Upload Final Norm (Single Chunk)
```c
vkWaitForFences(backend_data->device, 1, &backend_data->transfer_fence, VK_TRUE, UINT64_MAX);
vkResetFences(backend_data->device, 1, &backend_data->transfer_fence);

void *mapped = NULL;
vkMapMemory(backend_data->device, backend_data->embedding_staging.memory,
            0, final_norm_f32_size, 0, &mapped);
memcpy(mapped, final_norm_f32, final_norm_f32_size);
vkUnmapMemory(backend_data->device, backend_data->embedding_staging.memory);

VkCommandBufferBeginInfo begin = {
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
};
vkBeginCommandBuffer(backend_data->transfer_cmd, &begin);

VkBufferCopy copy = {.srcOffset = 0, .dstOffset = 0, .size = final_norm_f32_size};
vkCmdCopyBuffer(backend_data->transfer_cmd,
                backend_data->embedding_staging.buffer,
                backend_data->final_norm_weight.buffer,
                1, &copy);
vkEndCommandBuffer(backend_data->transfer_cmd);

VkSubmitInfo submit = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &backend_data->transfer_cmd
};
vkQueueSubmit(backend_data->compute_queue, 1, &submit, backend_data->transfer_fence);
vkWaitForFences(backend_data->device, 1, &backend_data->transfer_fence, VK_TRUE, UINT64_MAX);
```

### Step 1.4: Upload Embedding (Chunked - 256MB at a time)
```c
const size_t STAGING_BUFFER_SIZE = 256 * 1024 * 1024;
size_t bytes_remaining = embedding_f32_size;
size_t bytes_uploaded = 0;

while (bytes_remaining > 0) {
    size_t chunk = (bytes_remaining > STAGING_BUFFER_SIZE) ? STAGING_BUFFER_SIZE : bytes_remaining;
    
    vkWaitForFences(backend_data->device, 1, &backend_data->transfer_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(backend_data->device, 1, &backend_data->transfer_fence);
    
    // Map, copy chunk, record transfer, submit, wait (same pattern as weight uploads)
    // ... (see backend_vulkan.c weight upload for full pattern)
    
    bytes_uploaded += chunk;
    bytes_remaining -= chunk;
}

free(final_norm_f32);
free(embedding_f32);

LOG_INFO("Uploaded final layer weights: norm=%zu bytes, embedding=%zu bytes",
         final_norm_f32_size, embedding_f32_size);
```

## Cleanup (vulkan_session_destroy)

**File:** `src/inference/backend_vulkan.c`  
**Function:** `vulkan_session_destroy()`  
**Location:** After RoPE cache cleanup, before KV cache destroy

```c
/* Destroy RoPE cos/sin cache buffers */
vk_buffer_destroy(backend_data->device, &backend_data->rope_cos_cache);
vk_buffer_destroy(backend_data->device, &backend_data->rope_sin_cache);

/* Destroy final layer weight buffers */
vk_buffer_destroy(backend_data->device, &backend_data->final_norm_weight);
vk_buffer_destroy(backend_data->device, &backend_data->embedding_weight);

/* Destroy KV cache */
vk_kv_cache_destroy(backend_data->device, &backend_data->kv_cache);
```

## Error Handling

All failures must:
1. Log descriptive error with `LOG_ERROR()`
2. Free allocated host buffers (`final_norm_f32`, `embedding_f32`)
3. Destroy partially created GPU buffers
4. Return -1 (triggers fallback to CPU backend)

## Validation

**Build Test:**
```bash
make bin
# Should compile without errors
```

**Initialization Test:**
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 1 2>&1 | grep "Uploaded final"
# Expected: "Uploaded final layer weights: norm=1024 bytes, embedding=268435456 bytes"
```

## Next Step

After successful upload, proceed to:  
**`docs/phase-11-05-step-2-gpu-dispatch.md`** - Implement GPU lm_head dispatch

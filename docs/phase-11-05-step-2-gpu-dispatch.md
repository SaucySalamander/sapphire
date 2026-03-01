# Phase 11-05 Step 2: Implement GPU LM Head Dispatch

**Status:** Not Started  
**Dependencies:** Step 1 (weight upload) must be completed first

## Objective

Create `dispatch_lm_head()` helper function that:
1. Runs final RMSNorm on GPU (256-dim hidden state)
2. Projects normalized state to vocabulary logits (GEMV: 262K × 256)
3. Outputs logits to GPU scratchpad buffer (for CPU download)

## Input/Output Specification

### Inputs
- **Hidden State:** `layer_hidden[pong]` scratchpad buffer (last layer output, GPU)
- **Final Norm Weight:** `backend_data->final_norm_weight` (pre-uploaded, GPU)
- **Embedding Weight:** `backend_data->embedding_weight` (pre-uploaded, GPU)
- **Batch Size:** Always 1 for lm_head (only process last token)

### Output
- **Logits Buffer:** `scratchpad.norm_buf` (temporary reuse, 262K floats)
  - **Alternative:** Allocate dedicated `logits_buf` if norm_buf too small

## Pipeline Selection

### RMSNorm Pipeline
- Use existing `PIPELINE_RMSNORM_F32` pipeline
- **Descriptor Set:** NOT layer-indexed (this is non-layered)
  - Need 1 descriptor set for lm_head RMSNorm (separate from layer norms)
  - **Index:** Use set index `num_layers * 2` (after all layer norm sets)

### GEMV Pipeline
- Use existing `PIPELINE_GEMV_F32` pipeline (batch_size=1)
- **Descriptor Set:** NOT layer-indexed
  - Need 1 descriptor set for lm_head projection
  - **Index:** Use set index `num_layers * 7` (after all layer projection sets)

## Descriptor Set Pre-Population

**Location:** `vulkan_session_init()` after layer descriptor pre-population

```c
/* ========================================================================
 * LM Head Descriptor Pre-Population (Final Layer, Non-Layered)
 * ======================================================================== */

const int pi_norm_f32 = PIPELINE_RMSNORM_F32;
const int pi_gemv_f32 = PIPELINE_GEMV_F32;

/* Final RMSNorm descriptor set (after all layer norm sets) */
uint32_t lmhead_norm_ds = config->num_hidden_layers * 2;
VkDescriptorBufferInfo lmhead_norm_info = {
    .buffer = backend_data->final_norm_weight.buffer,
    .offset = 0,
    .range = VK_WHOLE_SIZE
};
VkWriteDescriptorSet lmhead_norm_write = {
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = backend_data->pipelines[pi_norm_f32].desc_sets[lmhead_norm_ds],
    .dstBinding = 1,  /* Weight binding */
    .dstArrayElement = 0,
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .pBufferInfo = &lmhead_norm_info
};
vkUpdateDescriptorSets(backend_data->device, 1, &lmhead_norm_write, 0, NULL);

/* Final projection descriptor set (after all layer projection sets) */
uint32_t lmhead_proj_ds = config->num_hidden_layers * 7;
VkDescriptorBufferInfo lmhead_proj_info = {
    .buffer = backend_data->embedding_weight.buffer,
    .offset = 0,
    .range = VK_WHOLE_SIZE
};
VkWriteDescriptorSet lmhead_proj_write = {
    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    .dstSet = backend_data->pipelines[pi_gemv_f32].desc_sets[lmhead_proj_ds],
    .dstBinding = 1,  /* Weight binding */
    .dstArrayElement = 0,
    .descriptorCount = 1,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .pBufferInfo = &lmhead_proj_info
};
vkUpdateDescriptorSets(backend_data->device, 1, &lmhead_proj_write, 0, NULL);

LOG_INFO("Pre-populated lm_head descriptor sets (norm + projection)");
```

**Update descriptor set count log:**
```c
LOG_INFO("Pre-populated %d layers × (...) + 2 lm_head = %d descriptor sets",
         config->num_hidden_layers, config->num_hidden_layers * (2+7+7+1+1) + 2);
```

## Descriptor Set Allocation Update

**Location:** `vulkan_session_init()` pipeline creation loop

```c
/* Descriptor sets per layer for each pipeline type:
 * - Projections (GEMV/GEMM): 7 per layer + 1 for lm_head
 * - RMSNorm: 2 per layer + 1 for lm_head
 * - Others: 1 per layer
 */
static const uint32_t desc_sets_per_layer[NUM_PIPELINES] = {
    [PIPELINE_RMSNORM_F32]     = 2,  // Will allocate: num_layers * 2 + 1
    [PIPELINE_RMSNORM_BF16]    = 2,
    [PIPELINE_GEMV_F32]        = 7,  // Will allocate: num_layers * 7 + 1
    [PIPELINE_GEMV_BF16]       = 7,
    [PIPELINE_GEMM_F32]        = 7,
    [PIPELINE_GEMM_BF16]       = 7,
    // ... rest unchanged
};

// Update allocation line:
cfg.num_desc_sets = desc_sets_per_layer[i] * config->num_hidden_layers;
if (i == PIPELINE_RMSNORM_F32 || i == PIPELINE_RMSNORM_BF16) {
    cfg.num_desc_sets += 1;  // +1 for lm_head norm
}
if (i == PIPELINE_GEMV_F32 || i == PIPELINE_GEMV_BF16) {
    cfg.num_desc_sets += 1;  // +1 for lm_head projection
}
```

## Dispatch Helper Function

**Location:** `src/inference/backend_vulkan.c` (near other dispatch helpers)

```c
/**
 * Dispatch GPU lm_head: final RMSNorm + vocabulary projection.
 * 
 * @param ctx         Layer dispatch context (reused, layer field ignored)
 * @param final_hidden Final layer output buffer (GPU)
 * @param logits_out  Output logits buffer (GPU, must fit vocab_size floats)
 */
static void dispatch_lm_head(
    const layer_dispatch_ctx_t *ctx,
    const vk_buffer_t *final_hidden,
    vk_buffer_t *logits_out
) {
    vk_gpu_scratchpad_t *sp = &ctx->bd->scratchpad;
    const gemma3_270m_config_t *cfg = ctx->cfg;
    int bf16 = 0;
    
    int pi_norm = select_pipeline(PIPELINE_RMSNORM_F32, PIPELINE_RMSNORM_BF16, bf16);
    int pi_proj = select_pipeline(PIPELINE_GEMV_F32, PIPELINE_GEMV_BF16, bf16);
    
    vk_kernel_push_constants_t pc = {0};
    pc.batch_size = 1;  /* Always process single token for lm_head */
    pc.d_model = (uint32_t)cfg->hidden_size;
    pc.stride_0 = (uint32_t)cfg->hidden_size;
    
    /* [1] Final RMSNorm: final_hidden → norm_buf */
    uint32_t lmhead_norm_ds = cfg->num_hidden_layers * 2;
    bind_desc(ctx->bd, &ctx->bd->pipelines[pi_norm], lmhead_norm_ds, 0, final_hidden, 0);
    bind_desc(ctx->bd, &ctx->bd->pipelines[pi_norm], lmhead_norm_ds, 1,
              &ctx->bd->final_norm_weight, 0);
    bind_desc(ctx->bd, &ctx->bd->pipelines[pi_norm], lmhead_norm_ds, 2, &sp->norm_buf, 0);
    record_compute_dispatch(ctx->cmd_buf, &ctx->bd->pipelines[pi_norm],
                            lmhead_norm_ds, &pc, 1, 1);
    
    /* [2] Vocabulary Projection: norm_buf → logits_out (GEMV: 262K × 256) */
    uint32_t lmhead_proj_ds = cfg->num_hidden_layers * 7;
    pc.stride_0 = (uint32_t)cfg->hidden_size;
    pc.stride_1 = 0;
    bind_desc(ctx->bd, &ctx->bd->pipelines[pi_proj], lmhead_proj_ds, 0, &sp->norm_buf, 0);
    bind_desc(ctx->bd, &ctx->bd->pipelines[pi_proj], lmhead_proj_ds, 1,
              &ctx->bd->embedding_weight, 0);
    bind_desc(ctx->bd, &ctx->bd->pipelines[pi_proj], lmhead_proj_ds, 2, logits_out, 0);
    
    /* Workgroup calculation: vocab_size rows, 1 column */
    uint32_t vocab_workgroups = ceil_div((uint32_t)cfg->vocab_size, 256);  /* Typical workgroup size */
    record_compute_dispatch(ctx->cmd_buf, &ctx->bd->pipelines[pi_proj],
                            lmhead_proj_ds, &pc, vocab_workgroups, 1);
}
```

## Logits Buffer Strategy

### Option A: Reuse Scratchpad (Requires Size Check)
```c
// Check if norm_buf is large enough for logits
size_t logits_size = config->vocab_size * sizeof(float);
size_t norm_buf_size = config->hidden_size * sizeof(float);
if (logits_size > norm_buf_size) {
    // Need dedicated logits buffer (vocab_size > hidden_size)
}
```

### Option B: Dedicated Logits Buffer (Recommended)
Add to `backend_vulkan_session_data_t`:
```c
vk_buffer_t lm_head_logits;  /* [vocab_size] - output buffer for lm_head */
```

Allocate in `vulkan_session_init()`:
```c
size_t logits_size = config->vocab_size * sizeof(float);
rc = vk_buffer_create(backend_data->device, backend_data->phys_dev, logits_size,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    &backend_data->lm_head_logits);
```

## Testing

**Unit Test (Descriptor Pre-Population):**
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 1 2>&1 | grep "Pre-populated"
# Expected: "Pre-populated 18 layers × (...) + 2 lm_head = 326 descriptor sets"
```

**Integration Test (Inference):**
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 5 2>&1 | grep -E "(kernel_gemv|ERROR)"
# Expected: No "kernel_gemv null pointer" errors
```

## Next Step

After dispatch implementation:  
**`docs/phase-11-05-step-3-integration.md`** - Replace CPU lm_head call with GPU version

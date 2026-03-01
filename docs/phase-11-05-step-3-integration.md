# Phase 11-05 Step 3: Integrate GPU LM Head into Forward Pass

**Status:** Not Started  
**Dependencies:** Steps 1 and 2 must be completed first

## Objective

Replace CPU `lm_head()` call with GPU `dispatch_lm_head()` in forward batch pipeline:
1. Run GPU lm_head dispatch (norm + projection)
2. Download logits from GPU to CPU
3. Return logits for sampling

## Current CPU Implementation (TO REPLACE)

**File:** `src/inference/backend_vulkan.c`  
**Function:** `vulkan_forward_batch()`  
**Lines:** ~1935-1965

```c
/* Download final hidden state from GPU */
extern int vk_buffer_download(...);
vk_buffer_download(&transfer_ctx, final_hidden_buf, download_size, cpu_hidden_full);

/* Extract last token's hidden state */
const float *last_hidden = cpu_hidden_full + (batch_size - 1) * cfg->hidden_size;

/* Compute final RMSNorm + LM head projection on CPU */
lm_head(session, last_hidden, logits);  // <--- REMOVE THIS

free(cpu_hidden_full);
```

**Problem:** Downloads 256 floats, uploads to CPU GEMV, computes 262K logits on CPU

## New GPU Implementation

### Step 3.1: Record LM Head Dispatch

**Location:** After layer loop completes in `vulkan_forward_batch()`

```c
/* ========================================================================
 * GPU LM Head: Final RMSNorm + Vocabulary Projection
 * ======================================================================== */

/* Identify which scratchpad buffer holds final layer output */
int final_pong = (num_layers - 1) % 2;  /* Last layer's output buffer */
vk_buffer_t *final_hidden_buf = &bd->scratchpad.layer_hidden[final_pong];

/* Dispatch GPU lm_head (final norm + vocab projection) */
layer_dispatch_ctx_t lmhead_ctx = {
    .cmd_buf = bd->cmd_buffer,
    .bd = bd,
    .cfg = cfg,
    .layer = 0,  /* Not used (lm_head is non-layered) */
    .batch_size = 1,  /* Always process last token only */
    .seq_pos = start_pos + batch_size - 1  /* Position of last token */
};

dispatch_lm_head(&lmhead_ctx, final_hidden_buf, &bd->lm_head_logits);

/* Insert memory barrier: compute write → transfer read */
VkMemoryBarrier lmhead_barrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
};
vkCmdPipelineBarrier(bd->cmd_buffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    0, 1, &lmhead_barrier, 0, NULL, 0, NULL);
```

### Step 3.2: Submit Command Buffer

**Location:** After lm_head dispatch, before logits download

```c
/* End command buffer recording */
vr = vkEndCommandBuffer(bd->cmd_buffer);
if (vr != VK_SUCCESS) {
    LOG_ERROR("Failed to end command buffer: %d", vr);
    return -1;
}

/* Submit to GPU (non-blocking) */
VkSubmitInfo submit_info = {
    .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    .commandBufferCount = 1,
    .pCommandBuffers = &bd->cmd_buffer
};

vkResetFences(bd->device, 1, &bd->transfer_fence);
vr = vkQueueSubmit(bd->compute_queue, 1, &submit_info, bd->transfer_fence);
if (vr != VK_SUCCESS) {
    LOG_ERROR("Failed to submit command buffer: %d", vr);
    return -1;
}
```

### Step 3.3: Download Logits from GPU

**Location:** After command buffer submission

```c
/* Wait for GPU computation to complete */
vr = vkWaitForFences(bd->device, 1, &bd->transfer_fence, VK_TRUE, UINT64_MAX);
if (vr != VK_SUCCESS) {
    LOG_ERROR("Failed to wait for GPU completion: %d", vr);
    return -1;
}

/* Download logits from GPU to CPU */
vk_transfer_ctx_t transfer_ctx = {
    .device = bd->device,
    .phys_dev = bd->phys_dev,
    .cmd_pool = bd->cmd_pool,
    .queue = bd->compute_queue
};

size_t logits_size = (size_t)cfg->vocab_size * sizeof(float);
extern int vk_buffer_download(const vk_transfer_ctx_t *ctx, vk_buffer_t *device_buf,
                               size_t size, void *out_host_data);

int rc_download = vk_buffer_download(&transfer_ctx, &bd->lm_head_logits,
                                     logits_size, logits);
if (rc_download != 0) {
    LOG_ERROR("Failed to download logits from GPU");
    return -1;
}

LOG_DEBUG("Vulkan forward batch: %d layers, batch=%d, pos=%d (fully GPU-accelerated)",
          num_layers, batch_size, start_pos);

return 0;
```

## Code Removal

**Delete these lines from `vulkan_forward_batch()`:**
```c
// REMOVE: CPU hidden state download
size_t download_size = (size_t)batch_size * cfg->hidden_size * sizeof(float);
float *cpu_hidden_full = (float *)malloc(download_size);
// ... download code ...

// REMOVE: CPU lm_head call
extern void lm_head(inference_session_t* session, const float* hidden, float* logits);
lm_head(session, last_hidden, logits);

// REMOVE: CPU buffer cleanup
free(cpu_hidden_full);
```

**Remove external declarations:**
```c
// DELETE from top of backend_vulkan.c:
extern void lm_head(inference_session_t* session, const float* hidden, float* logits);
extern void sapphire_embed_lookup_batch(...);  // Keep this one
```

## Command Buffer Recording Structure

**Updated flow:**
1. Begin command buffer
2. Upload embeddings (transfer cmd, separate)
3. Record 18 transformer layers (all in main cmd_buffer)
4. Record GPU lm_head (final norm + projection)
5. Insert compute→transfer barrier
6. End command buffer
7. Submit to GPU
8. Wait for completion
9. Download logits

## Error Handling

All GPU failures must:
1. Log descriptive error
2. Return -1 immediately (no CPU fallback)
3. Let inference session cleanup handle resource deallocation

**Rationale:** Production inference engine should fail fast on GPU errors, not silently fall back to CPU.

## Testing Strategy

### Standard Build Command

**ALWAYS use this command to build:**
```bash
make clean bin shaders
```

### Standard Validation Commands

**CPU Baseline (MUST WORK FIRST):**
```bash
SAPPHIRE_BACKEND=cpu ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "Write me a poem about the sea." -n 1000
```

**Vulkan Target (MUST MATCH CPU OUTPUT):**
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "Write me a poem about the sea." -n 1000
```

### Test 1: Smoke Test (Single Token)
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 1
```
**Expected:** No errors, produces single  token output

### Test 2: Short Generation (20 Tokens)
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "Write a haiku" -n 20
```
**Expected:** Coherent 3-line haiku output

### Test 3: Validation Layer Check
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 5 2>&1 | grep -E "VALIDATION|ERROR"
```
**Expected:** Zero validation warnings, zero errors

### Test 4: No CPU Fallback
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 5 2>&1 | grep "kernel_gemv"
```
**Expected:** No output (kernel_gemv is CPU-only, should not be called)

### Test 5: Performance Baseline
```bash
time SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "Write me a poem about the sea." -n 100
```
**Expected:** Faster than CPU backend for same prompt

## Success Criteria

- [x] No `kernel_gemv null pointer` errors
- [x] No validation layer warnings
- [x] Produces coherent text output
- [x] All compute happens on GPU (no CPU fallback logs)
- [x] Logits downloaded only once per token
- [x] Forward pass completes without crashes

## Performance Metrics

**Expected improvements over CPU:**
- Per-token latency: ~30-50ms (GPU) vs ~100-200ms (CPU)
- Throughput: ~20-30 tokens/sec (GPU) vs ~5-10 tokens/sec (CPU)
- GPU utilization: ~80-95% (was ~60% with CPU lm_head)

## Cleanup and Documentation

### Update Copilot Instructions
Add to `.github/copilot-instructions.md` under Vulkan Backend Architecture Rules:
```markdown
## LM Head (Final Layer)
- Final RMSNorm and vocabulary projection run entirely on GPU
- Logits buffer: Dedicated GPU buffer, downloaded once per token
- Descriptor sets: lm_head uses non-layered sets (after all layer sets)
  - RMSNorm: ds = num_layers * 2
  - GEMV: ds = num_layers * 7
```

### Update Phase 11 Status
Mark Phase 11-05 as **COMPLETED** in `vision/phase-11/` documentation.

## Next Phase

**Phase 11-06:** Timeline Semaphore Optimization
- Replace `vkQueueWaitIdle` with timeline semaphore sync
- Implement ring buffer pipelining for embeddings/logits
- Target: Eliminate all host-side blocking waits in hot path

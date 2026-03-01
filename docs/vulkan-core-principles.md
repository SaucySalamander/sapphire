# Vulkan Backend Core Principles

**Document Purpose:** Overarching architectural principles for Vulkan backend development. Pass this document to ALL AI sessions working on Vulkan backend features.

## Production-Grade GPU-First Design

### Principle 1: No Hybrid CPU/GPU Bottlenecks
- **Rule:** ALL compute operations MUST execute on GPU when using GPU backend
- **Rationale:** Hybrid CPU/GPU introduces PCIe transfer overhead and underutilizes GPU
- **Allowed CPU operations:** I/O, tokenization, sampling logic (post-logits)
- **Forbidden:** Downloading intermediate activations, running compute on CPU, uploading results

### Principle 2: Static Graph Architecture
- **All weight buffers pre-populated during init** via `vkUpdateDescriptorSets`
- **No descriptor updates during inference** (forward pass)
- **Activation buffers bound during command recording** using `bind_desc()`
- **Benefit:** Eliminates UPDATE_AFTER_BIND validation errors, reduces driver overhead

### Principle 3: Pipeline Selection and Descriptor Indexing
- **GEMV vs GEMM:** Select at dispatch time based on `batch_size`
  - `batch_size == 1` → GEMV pipeline
  - `batch_size > 1` → GEMM pipeline
- **Pre-populate BOTH pipelines:** Must write descriptor sets for all pipelines that could be selected
- **Descriptor set indexing formulas:**
  - RMSNorm: `ds = layer * 2 + slot` (slot: 0=pre-attn, 1=pre-FFN)
  - GEMV/GEMM: `ds = layer * 7 + proj_slot` (Q=0,K=1,V=2,O=3,Gate=4,Up=5,Down=6)
  - QK-Norm: `ds = layer`
  - RoPE: `ds = layer`
  - Attention: `ds = layer`
  - GELU: `ds = layer`

### Principle 4: Memory Layout and Alignment
- **Weight packing:** All layers' weights for a given type (e.g., Q projection) packed into single buffer
- **Offset alignment:** All `VkDescriptorBufferInfo.offset` values MUST be aligned to `min_storage_buffer_offset_alignment`
- **Use ALIGN_UP macro:** `ALIGN_UP(size, alignment)` for all buffer offsets
- **KV cache sizing:** Use `kv_cache.max_seq_len` (NOT `max_position_embeddings`)

### Principle 5: Transfer Resource Management
- **Persistent resources:** Reuse `transfer_cmd`, `embedding_staging`, `transfer_fence`
- **Staging buffer size:** 256MB (PCIe 3.0 BAR compatibility)
- **Weight upload chunking:** Large weights uploaded in 256MB chunks
- **Fence synchronization:** ALWAYS wait for fence before reusing transfer resources

### Principle 6: Validation and Error Handling
- **VK_LAYER_KHRONOS_validation ENABLED** in release builds (not just debug)
- **Log ALL validation warnings** via debug messenger
- **Zero tolerance for errors:** Fix validation warnings immediately, don't ship with them
- **GPU-First verification:** If a feature CAN run on GPU, it MUST run on GPU

## Current Architecture Status (Phase 11)

### Completed ✓
- [x] Static Graph Architecture (180 descriptor sets pre-populated)
- [x] Weight buffer packing (11 types × 18 layers)
- [x] RoPE cos/sin cache precomputation and upload
- [x] GEMV and GEMM descriptor set pre-population
- [x] ds=0 bug fixed (QK-Norm, RoPE use `ds = ctx->layer`)
- [x] All UPDATE_AFTER_BIND errors eliminated

### In Progress 🔄
- [ ] GPU lm_head (final RMSNorm + vocab projection)
  - Need: Upload `final_norm_weight` and `embedding_weight` to GPU
  - Need: GPU dispatch helper for lm_head
  - Need: Replace CPU lm_head call with GPU version

### Known Issues ⚠️
- `lm_head()` currently uses CPU `tensor_gemv_with_ctx` (hybrid bottleneck)
- `vkQueueWaitIdle` in hot path (should use timeline semaphores)
- No per-layer descriptor set tracking struct

## Design Patterns

### Buffer Binding Pattern
```c
bind_desc(ctx->bd, pipeline, ds, binding_slot, buffer, offset);
```
- **Weight buffers:** Skipped automatically via buffer-identity check
- **Activation buffers:** Bound during command recording
- **Pre-populated buffers:** Never re-bound (RoPE cos/sin, weight bindings)

### Pipeline Dispatch Pattern
```c
int pi = select_pipeline(PIPELINE_X_F32, PIPELINE_X_BF16, bf16);
vk_kernel_push_constants_t pc = build_push_constants(cfg, layer, batch_size, seq_pos);
pc.stride_0 = custom_value;  // Shader-specific
record_compute_dispatch(cmd_buf, &bd->pipelines[pi], ds, &pc, workgroup_x, workgroup_y);
```

### Weight Upload Pattern (Chunked)
```c
size_t bytes_remaining = total_size;
size_t bytes_uploaded = 0;
while (bytes_remaining > 0) {
    size_t chunk = min(bytes_remaining, STAGING_BUFFER_SIZE);
    vkWaitForFences(device, 1, &transfer_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &transfer_fence);
    // Map staging, copy chunk, record transfer cmd, submit with fence
    bytes_uploaded += chunk;
    bytes_remaining -= chunk;
}
```

## Common Pitfalls (DO NOT)
- ❌ Use `ds = 0` for pipelines with 1 set per layer → use `ds = ctx->layer`
- ❌ Call `vkUpdateDescriptorSets` after binding descriptor set to command buffer
- ❌ Create new VkBuffer for every tensor → use packed super buffers with offsets
- ❌ Use `max_position_embeddings` for KV cache sizing → use `kv_cache.max_seq_len`
- ❌ Skip offset alignment → validation errors and potential GPU crashes
- ❌ Download intermediate activations for CPU processing → full GPU pipeline required

## Testing and Validation Checklist
- [ ] No validation warnings in release build
- [ ] No `kernel_gemv null pointer` or similar CPU fallback errors
- [ ] Logits produced entirely on GPU (only final download)
- [ ] Inference produces coherent text output
- [ ] Timeline semaphore sync (no `vkQueueWaitIdle` in hot path)
- [ ] All descriptor sets pre-populated (log confirms 324 sets for 18 layers)

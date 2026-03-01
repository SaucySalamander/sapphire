# Phase 11-05 Quick Start Guide

**When starting a new AI session to work on Phase 11-05, read documents in this order:**

## 1. Core Principles (ALWAYS READ FIRST)
📄 **`docs/vulkan-core-principles.md`**
- GPU-first design principles
- Static Graph Architecture rules
- Common pitfalls and patterns
- **Critical for all Vulkan work**

## 2. Phase Overview
📄 **`docs/phase-11-05-gpu-lm-head-overview.md`**
- Problem statement
- Architecture decision rationale
- Success criteria
- Performance expectations

## 3. Implementation Steps (Execute in Order)

### Step 1: Weight Upload ✅
📄 **`docs/phase-11-05-step-1-weight-upload.md`**
- Upload `final_norm_weight` and `embedding_weight` to GPU
- BF16 → F32 conversion
- Chunked transfer for large embedding matrix
- Cleanup in session destroy

**Status Check:**
```bash
make bin && SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 1 2>&1 | grep "Uploaded final"
```
Expected: `Uploaded final layer weights: norm=1024 bytes, embedding=268435456 bytes`

### Step 2: GPU Dispatch ✅
📄 **`docs/phase-11-05-step-2-gpu-dispatch.md`**
- Implement `dispatch_lm_head()` helper
- Pre-populate lm_head descriptor sets
- Update descriptor set allocation counts

**Status Check:**
```bash
make bin && SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 1 2>&1 | grep "Pre-populated"
```
Expected: `Pre-populated 18 layers × (...) + 2 lm_head = 326 descriptor sets`

### Step 3: Integration ⏳
📄 **`docs/phase-11-05-step-3-integration.md`**
- Replace CPU `lm_head()` call with GPU version
- Download logits from GPU
- Remove CPU fallback code

**Status Check:**
```bash
make bin && SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "Write a haiku" -n 20 2>&1 | grep -c "kernel_gemv"
```
Expected: `0` (no CPU fallback)

## Current Status

**Phase 11-04 Completed:**
- ✅ Descriptor pre-population (180 sets → 324 sets with GEMM)
- ✅ RoPE cos/sin cache
- ✅ All UPDATE_AFTER_BIND errors eliminated
- ✅ Validation warnings at zero

**Phase 11-05 In Progress:**
- ⏳ GPU lm_head implementation
- Current blocker: `kernel_gemv null pointer` (CPU fallback in lm_head)

## File Locations

**Header:** `include/backend_vulkan.h`
```c
typedef struct {
    // ... existing fields ...
    vk_buffer_t rope_cos_cache;
    vk_buffer_t rope_sin_cache;
    vk_buffer_t final_norm_weight;     // ADD THIS
    vk_buffer_t embedding_weight;      // ADD THIS
    vk_buffer_t lm_head_logits;        // ADD THIS
    // ...
} backend_vulkan_session_data_t;
```

**Implementation:** `src/inference/backend_vulkan.c`
- Weight upload: After line ~970 (after RoPE cache)
- Dispatch helper: Near line ~1440 (with other dispatch helpers)
- Integration: Replace lines ~1935-1965 (CPU lm_head call)

## Validation Commands

**Build (ALWAYS USE THIS):**
```bash
make clean bin shaders
```

**Single token test:**
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 1
```

**Validation check:**
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "hi" -n 5 2>&1 | grep -E "(VALIDATION|ERROR|kernel_gemv)"
```
Expected: No output (all errors eliminated)

**Short generation test:**
```bash
SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "Write a haiku" -n 20
```
Expected: Coherent 3-line haiku

## Common Issues and Solutions

### Issue: "kernel_gemv null pointer"
**Cause:** CPU lm_head still being called  
**Fix:** Complete Step 3 (integration) to replace with GPU version

### Issue: Validation errors after adding lm_head
**Cause:** Descriptor sets not pre-populated or wrong indexing  
**Fix:** Check Step 2 pre-population code, verify ds calculations

### Issue: Compilation errors about missing fields
**Cause:** Forgot to update backend_vulkan.h struct  
**Fix:** Add `final_norm_weight`, `embedding_weight`, `lm_head_logits` fields

### Issue: Inference produces garbage output
**Cause:** BF16 → F32 conversion skipped, or wrong buffer bound  
**Fix:** Verify `bf16_to_f32_vec()` called before upload

## Success Metrics

✅ **Phase 11-05 Complete When:**
1. No `kernel_gemv` errors
2. Zero validation warnings
3. Coherent text generation (20+ token test)
4. All compute on GPU (confirmed via no CPU kernel calls)
5. Logits downloaded exactly once per generated token

## Next Phase After Completion

**Phase 11-06:** Timeline Semaphore Optimization
- Replace `vkQueueWaitIdle` with timeline semaphores
- Implement pipelined ring buffer sync
- Target: Zero blocking waits in hot path

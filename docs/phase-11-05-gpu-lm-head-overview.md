# Phase 11-05: GPU LM Head Implementation

**Status:** In Progress  
**Goal:** Eliminate CPU fallback in `lm_head()` by implementing final RMSNorm + vocabulary projection on GPU

## Problem Statement

Currently, the Vulkan backend runs all 18 transformer layers on GPU, but then:
1. Downloads final hidden state to CPU (256 floats × batch_size)
2. Runs CPU `lm_head()` which calls `tensor_gemv_with_ctx()` (CPU GEMV)
3. Vocabulary projection: 256-dim → 262,144 vocab tokens (CPU bottleneck)

**Result:** Hybrid CPU/GPU pipeline violates production-grade design principles.

## Architecture Decision

**Chosen Approach:** Full GPU lm_head
- Final RMSNorm on GPU
- Vocabulary projection (GEMV: 262K × 256) on GPU
- Download ONLY final logits to CPU for sampling

**Why not CPU fallback:**
- Production-grade inference engine must maximize GPU utilization
- Vocab projection is compute-bound, benefits massively from GPU
- Logits must be on CPU for sampling anyway (minimal transfer)
- CPU GEMV requires initializing `session->gemv_ctx` (more complexity)

## Implementation Steps

### Step 1: Upload Final Layer Weights to GPU
**File:** `docs/phase-11-05-step-1-weight-upload.md`
- Upload `model->norm_final_weight` (256 floats)
- Upload `model->embedding_weight` (262,144 × 256 matrix, BF16 → F32 on GPU)
- Store in `backend_data->final_norm_weight` and `backend_data->embedding_weight`
- Add cleanup in `vulkan_session_destroy()`

### Step 2: Implement GPU LM Head Dispatch
**File:** `docs/phase-11-05-step-2-gpu-dispatch.md`
- Create `dispatch_lm_head()` helper function
- Use existing RMSNorm pipeline + GEMV pipeline
- Input: final hidden state in scratchpad (GPU)
- Output: logits in scratchpad (GPU)

### Step 3: Integrate GPU LM Head into Forward Pass
**File:** `docs/phase-11-05-step-3-integration.md`
- Replace CPU `lm_head()` call with GPU `dispatch_lm_head()`
- Download logits from GPU to CPU staging buffer
- Update validation test to ensure no CPU fallback errors

## Success Criteria

- [ ] No `kernel_gemv null pointer` errors
- [ ] All compute happens on GPU (validated via absence of CPU kernel calls)
- [ ] Inference produces coherent text output
- [ ] Validation warnings remain at zero
- [ ] Logits downloaded only once after GPU compute completes

## Performance Impact

**Expected improvement:**
- Eliminates 2 PCIe transfers per token (hidden state download + upload for GEMV)
- Vocabulary projection benefits from GPU parallelism (262K SIMD lanes)
- Reduces CPU load (no thread pool spinning for GEMV)

**Estimated speedup:** 5-10% per-token latency reduction (depends on PCIe bandwidth)

## Dependencies

- [x] RMSNorm pipeline (PIPELINE_RMSNORM_F32)
- [x] GEMV pipeline (PIPELINE_GEMV_F32)
- [x] Persistent staging buffer for logits download
- [x] Transfer fence for synchronization
- [x] Static Graph Architecture for weight pre-population

## Related Documents

- `docs/vulkan-core-principles.md` - Architectural principles (ALWAYS READ FIRST)
- `vision/phase-11/` - Phase 11 development plan
- `.github/copilot-instructions.md` - Project-wide coding standards

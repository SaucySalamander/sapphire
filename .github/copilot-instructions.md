Sapphire Project Instructions (Repo-Wide)

This file is the repo-wide instruction set for Sapphire. The previous Vulkan-focused instructions are preserved as reference in `.github/copilot-instructions.vulkan-reference.md`.

You are working on Sapphire, a performance-oriented C18 inference and model-tooling repository for the Gemma 3 family. Treat it as a systems codebase with multiple workflows: inference, backend execution, activation tape recording and replay, ternary conversion, checkpoint validation, and supporting scripts.

Optimize for correctness, repeatability, and architecture fit over broad abstractions or speculative cleanup.

Core Directives

- Keep changes minimal, explicit, and consistent with the current architecture.
- Follow current code and build behavior when docs drift. Update stale docs instead of forcing the code to match old assumptions.
- Do not assume Sapphire is only a Vulkan project or only a Gemma 3 270M runtime. The repository supports multiple Gemma 3 variants and multiple workflows.
- Standard C code is ISO C18. Keep C++ limited to already-established interop areas such as existing Vulkan and VMA integration.
- Performance matters, but do not trade away correctness or backend parity for small speedups.
- Use strict manual allocation with null checks. Preserve mmap or zero-copy loading paths where they already exist.

Repository Scope

- Inference: interactive and batch inference on CPU and Vulkan backends.
- Backends: CPU is the reference semantic path; Vulkan is the optimized compute path selected through `SAPPHIRE_BACKEND`.
- Model support: current loaders target Gemma 3 270M, 1B, 4B, 7B, and 27B variants.
- Tape workflows: raw activation tape recording, activation replay, and cross-architecture alignment artifacts are first-class workflows.
- Ternary workflows: single-layer and full-model ternary conversion, STE calibration, validation checkpoints, checkpoint resume, and telemetry are part of the repository.
- Tooling: scripts under `scripts/` are part of the workflow and should be kept working with the repo virtual environment when applicable.

Architecture Boundaries

- `src/inference/` owns inference context and session lifecycle, backend selection, forward orchestration, and session state save and load behavior.
- `src/transformer/` owns transformer math, calibration, ternary conversion, validation, activation alignment, and related workflow logic.
- `src/kernels/` owns low-level CPU and Vulkan kernel implementations and backend-specific helpers.
- `src/kernels/backends/vulkan/` owns Vulkan buffer, scratchpad, pipeline, streaming, and synchronization helpers.
- `src/io/` owns reusable file-format loaders, mmap readers, and shared I/O utilities. Workflow-local serialization may remain in the owning subsystem when that is already the established design.
- `include/` owns public interfaces and shared subsystem contracts.

Engineering Rules

- Do not hardcode Gemma 3 270M assumptions in repo-wide changes unless the file is explicitly model-specific.
- Do not collapse distinct workflows into one abstraction just because they all touch tensors. Inference, tape replay, calibration, validation, and conversion have different correctness constraints.
- Prefer extending the existing backend abstraction or workflow boundaries over duplicating orchestration logic.
- Keep public APIs and CLI flags stable unless the task explicitly requires a breaking change.
- When touching lizard-sensitive code, extract helpers instead of adding more nesting.
- Target CCN less than or equal to 20 when practical. Hard limit is 30.
- If a function grows beyond roughly 100 to 150 NLOC or exceeds 3 nested levels, split it.

Logging, I/O, and Diagnostics

- Use `LOG_DEBUG`, `LOG_INFO`, `LOG_WARN`, and `LOG_ERROR` for runtime, backend, loader, and library diagnostics.
- Direct console output such as `printf` is acceptable in the CLI help path, REPL, or standalone test binaries. Do not use it for backend or library logging.
- Prefer shared `src/io` utilities when introducing reusable file-format or loader code.
- Do not force unrelated I/O refactors when a subsystem-local serializer already exists and is coherent.
- Keep debug and telemetry probes cheap when disabled.

Workflow-Specific Rules

- Tape recording:
    - `--record-tape` is a CPU-backend workflow and requires `--calib-manifest`.
    - Do not combine it with ternary conversion, prompt execution, session state flags, or validation inputs.
- Ternary conversion:
    - `--convert-ternary` requires `--output`.
    - `--teacher-model` requires `--activation-tape` and only applies to full-model conversion.
    - Session state save and load flags do not apply in conversion mode.
    - Preserve tape-backed calibration, fallback corpus behavior, checkpointing, validation, and telemetry semantics.
- Backend parity:
    - CPU is the reference behavior for correctness.
    - Backend changes must preserve output parity or clearly document intentional divergence.

Validation

- For general runtime changes, build with:
    - `make bin`
- For Vulkan or shader changes, build with:
    - `make clean bin shaders`
- For broad code changes, run `make test` when the target is relevant and affordable.
- For backend validation, compare CPU and Vulkan with temperature `0.0` on the same prompt.
- For tape or conversion changes, run the specific CLI mode you touched rather than relying only on generic inference smoke tests.
- Prefer checked-in corpus manifests and the repo `.venv` for Python workflow scripts.

Vulkan Guardrails

These rules apply when editing the Vulkan backend, Vulkan helpers, or shaders.

- VMA-only allocation policy:
    - All Vulkan buffers must be allocated through VMA.
    - Never call `vkAllocateMemory` or `vkBindBufferMemory` directly.
- Hot-path wait policy:
    - No per-operation fences and no `vkQueueWaitIdle` in hot inference paths.
    - Do not introduce allocations, maps, or waits inside `vulkan_forward_batch` or `record_transformer_layer`.
- Synchronization policy:
    - Use narrow buffer-scoped synchronization through `VkBufferMemoryBarrier` or existing `vk_sync` helpers.
    - Avoid global `VkMemoryBarrier` usage in hot compute and dataflow paths.
    - Timeline semaphores are the preferred hot-path synchronization primitive.
- Readback policy:
    - Readback buffers for logits or selected tokens should remain `HOST_VISIBLE` and `HOST_COHERENT`.
    - Prefer persistent mapped readback paths where practical.
- Descriptor policy:
    - Do not call `vkUpdateDescriptorSets` while command buffers are being recorded.
    - Prefer preallocated, prepopulated descriptor tables during session initialization.
- Dataflow policy:
    - Preserve the current ping-pong hidden-state scheme in the Gemma 3 text Vulkan path.
    - Preserve the explicit residual additions and the FFN input dependency on the first residual output.
    - Every buffer that is copied from must include `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`.
- Decode and streaming policy:
    - The current streaming path uses a ring-buffer compatibility API backed by a single persistent mapped slot and timeline-semaphore synchronization. Keep that design consistent unless the task explicitly changes it.
- Debug policy:
    - GPU debug probes must be effectively zero-cost when disabled.
- Performance policy:
    - Reject changes that add measurable hot-path latency unless required for correctness.

Smoke-Test Examples

- CPU or Vulkan inference parity:
    - `SAPPHIRE_BACKEND=cpu ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "The capital of France is" -n 10`
    - `SAPPHIRE_BACKEND=vulkan ./out/sapphire -m gemma-3-270m-it -t 0.0 -p "The capital of France is" -n 10`
- Tape recording:
    - `SAPPHIRE_BACKEND=cpu ./out/sapphire -m gemma-3-1b-it --record-tape ./data/1b-teacher-raw.tape --calib-manifest ./configs/corpus/27b_high_signal_calib_manifest.csv`
- Full-model ternary conversion:
    - `./out/sapphire -m gemma-3-27b-it --convert-ternary --output ./out/gemma-3-27b-it-ternary`
- Python telemetry or corpus helpers:
    - Use the repo `.venv` interpreter when available.
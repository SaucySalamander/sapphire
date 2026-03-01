/**
 * @file backend.h
 * @brief Hardware backend abstraction for inference execution.
 *
 * Defines a virtual table interface for different hardware backends (CPU, Vulkan, etc.)
 * allowing seamless switching between execution strategies while maintaining a unified
 * inference API.
 */

#ifndef BACKEND_H
#define BACKEND_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct inference_session_t inference_session_t;
typedef struct model_spec model_spec_t;  /* Forward declare, defined in model_spec.h */

/**
 * Backend type enumeration.
 * Used to identify which hardware backend is in use.
 */
typedef enum {
    SAPPHIRE_BACKEND_TYPE_CPU,       /**< CPU backend (AVX2/AVX512 SIMD kernels) */
    SAPPHIRE_BACKEND_TYPE_VULKAN,    /**< Vulkan compute shader backend */
} sapphire_backend_type_t;

/**
 * Backend virtual table.
 *
 * Each backend implementation provides a static instance of this structure
 * with function pointers for initialization, execution, and cleanup.
 *
 * Performance considerations:
 * - CPU backend: Layer-by-layer execution with per-layer synchronization
 * - Vulkan backend: Entire forward pass as single GPU command pipeline
 */
typedef struct sapphire_backend_t {
    sapphire_backend_type_t type;    /**< Backend type identifier */
    const char* name;                /**< Human-readable backend name (e.g., "cpu", "vulkan") */

    /**
     * Initialize backend session data.
     *
     * Called once during inference_session_create(). This hook performs all
     * backend-specific initialization that must happen before the first forward pass.
     *
     * CPU implementation:
     *   - Allocates scratch buffers with SIMD-friendly alignment
     *   - Allocates attention score matrices
     *   - Creates KV cache for all layers
     *   - Precomputes RoPE frequencies (global and local bases)
     *   - Creates kernel execution context and launches worker threads
     *
     * Vulkan implementation:
     *   - Detects GPU device and creates VkDevice handle
     *   - Allocates GPU memory for KV cache and scratch buffers
     *   - Uploads model weights to GPU
     *   - Compiles compute shaders
     *   - Creates descriptor sets and pipeline layouts
     *   - Allocates command buffers and synchronization primitives
     *
     * @param session         Inference session (partially initialized, backend_data unset)
     * @param spec            Model specification with variant_config
     * @param max_context_len Maximum sequence length for KV cache and RoPE frequencies
     *
     * @return 0 on success, -1 on error (all errors logged internally via LOG_ERROR)
     */
    int (*session_init)(inference_session_t* session, const model_spec_t* spec, int max_context_len);

    /**
     * Cleanup backend session data.
     *
     * Called from destroy_inference_session(). Must free all backend-specific resources.
     *
     * CPU implementation: Frees scratch buffers, KV cache, RoPE frequencies, worker threads
     * Vulkan implementation: Frees GPU memory, destroys device, closes handles
     *
     * @param session Inference session with backend initialized
     */
    void (*session_destroy)(inference_session_t* session);

    /**
     * Execute full forward pass for a batch of tokens.
     *
     * This is the primary execution entry point. The granularity differs per backend:
     *
     * CPU backend:
     *   - Iterates through transformer layers one at a time (layer-by-layer)
     *   - Calls sapphire_transformer_layer_batch() for each layer
     *   - Computes attention with SIMD kernels for efficient prefill
     *   - Returns after final RMSNorm and LM head computation
     *
     * Vulkan backend:
     *   - Records entire transformer graph as single GPU command buffer
     *   - Submits all layers at once to GPU compute queue
     *   - Waits for GPU completion (synchronizes CPU-GPU)
     *   - Copies final logits from GPU memory back to system RAM
     *
     * @param session      Inference session (initialized and ready for execution)
     * @param token_ids    Array of input token IDs, length = batch_size
     * @param start_pos    Position of first token in sequence (for KV cache indexing)
     * @param batch_size   Number of tokens in batch (typically 1 for autoregressive, up to 32 for prefill)
     * @param logits       Output buffer for final logits [batch_size * vocab_size floats].
     *                     If NULL, computation still proceeds but logits not returned.
     *
     * @return 0 on success, -1 on error (all errors logged internally)
     */
    int (*forward_batch)(inference_session_t* session, const int* token_ids,
                         int start_pos, int batch_size, float* logits);

    /**
     * Optional fast path: run forward pass and return selected token ids.
     *
     * Intended for backends that can perform on-device selection (e.g. GPU argmax)
     * and return compact ids instead of full logits.
     *
     * If NULL, caller should fall back to forward_batch()+CPU sampling.
     */
    int (*forward_select_batch)(inference_session_t* session, const int* token_ids,
                                int start_pos, int batch_size, int* selected_ids);

    /**
     * Reset session state for a new sequence.
     *
     * Clears KV caches and any other sequence-specific state so the session
     * can be reused for a new prompt/generation sequence without reinitializing.
     *
     * @param session Inference session with backend initialized
     */
    void (*reset)(inference_session_t* session);

} sapphire_backend_t;

/**
 * Get backend implementation by type.
 *
 * Returns a pointer to a statically allocated backend implementation.
 * Do not free the returned pointer.
 *
 * @param type Backend type to retrieve (CPU, Vulkan, etc.)
 * @return Pointer to backend implementation, or NULL if type is unsupported
 */
sapphire_backend_t* backend_get(sapphire_backend_type_t type);

/**
 * Detect backend from environment or defaults to CPU.
 *
 * Checks the SAPPHIRE_BACKEND environment variable:
 *   - "cpu"     → SAPPHIRE_BACKEND_TYPE_CPU (also if env var not set)
 *   - "vulkan"  → SAPPHIRE_BACKEND_TYPE_VULKAN
 *   - other     → SAPPHIRE_BACKEND_TYPE_CPU (silently defaults)
 *
 * This function is called by create_inference_context() to select the backend
 * unless the backend type is explicitly specified.
 *
 * @return Detected backend type
 */
sapphire_backend_type_t backend_detect(void);

#ifdef __cplusplus
}
#endif

#endif // BACKEND_H

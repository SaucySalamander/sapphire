/**
 * @file tensor_mapper.h
 * @brief Generic tensor mapping interface for multiple model formats.
 *
 * This header defines the plugin architecture for loading models from different
 * tensor formats (Safetensors, GGML, etc.) into the unified Sapphire llm_model_t.
 *
 * Design:
 *   - sapphire_load_model(path) - auto-detects format from file extension
 *   - sapphire_load_safetensors(path) - explicitly load Safetensors format
 *   - sapphire_load_ggml(path) - explicitly load GGML format
 *
 * All three functions populate a single llm_model_t structure with the same
 * semantics and validation, enabling format-agnostic model usage.
 */

#ifndef SAPPHIRE_TENSOR_MAPPER_H
#define SAPPHIRE_TENSOR_MAPPER_H

#include "ggml_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum sapphire_load_error_t
 * Error codes returned by model loading functions.
 */
typedef enum {
    SAPPHIRE_OK = 0,                           /**< Success */
    SAPPHIRE_ERR_LOAD_GENERIC = -1,            /**< Generic load error */
    SAPPHIRE_ERR_LOAD_UNSUPPORTED_FORMAT = -2, /**< Unknown file format */
    SAPPHIRE_ERR_LOAD_MISSING_TENSOR = -3,     /**< Required tensor not found */
    SAPPHIRE_ERR_LOAD_SHAPE_MISMATCH = -4,     /**< Tensor shape mismatch */
} sapphire_load_error_t;

/**
 * @typedef tensor_mapper_plugin_t
 * Plugin interface for format-specific tensor loading.
 *
 * Each format (Safetensors, GGML) implements these function pointers to:
 * 1. Extract model config from file format metadata
 * 2. Map individual tensors to model fields
 * 3. Validate all required tensors are present
 * 4. Clean up format-specific resources
 */
typedef struct tensor_mapper_plugin {
    /**
     * Extract model configuration from format metadata.
     *
     * @param handle Format-specific file handle
     * @param config Output: populated model_config_t
     * @return 0 on success, -1 if config cannot be extracted
     */
    int (*extract_config)(void *handle, model_config_t *config);

    /**
     * Map a single tensor from file to model field.
     *
     * @param handle Format-specific file handle
     * @param meta Format-specific tensor metadata
     * @param model Destination model structure (partially filled)
     * @param error_msg Output buffer for error messages
     * @param max_error_len Size of error_msg buffer
     * @return 0 on success, -1 if mapping failed
     */
    int (*map_tensor)(void *handle, void *meta, llm_model_t *model,
                     char *error_msg, int max_error_len);

    /**
     * Validate that all required tensors were loaded.
     *
     * @param handle Format-specific file handle
     * @param model Model structure (after all tensors mapped)
     * @param error_msg Output buffer for validation error details
     * @param max_error_len Size of error_msg buffer
     * @return 0 if all required tensors present, -1 otherwise
     */
    int (*validate_tensors)(void *handle, llm_model_t *model,
                           char *error_msg, int max_error_len);

    /**
     * Clean up format-specific resources.
     *
     * @param handle Format-specific file handle to close/cleanup
     */
    void (*close)(void *handle);
} tensor_mapper_plugin_t;

/**
 * Auto-detect file format and load model.
 *
 * Examines file extension to determine format:
 *   .safetensors → Safetensors format
 *   .gguf → GGML format
 *
 * Populates out_model with all required tensors and configuration.
 * Caller is responsible for freeing out_model via llm_model_destroy().
 *
 * @param path File path (extension determines format)
 * @param out_model Output: populated model_t structure
 * @param error_message Optional output buffer for error messages
 * @param max_error_len Size of error_message buffer (e.g., 256)
 *
 * @return SAPPHIRE_OK on success
 * @return SAPPHIRE_ERR_LOAD_UNSUPPORTED_FORMAT if extension not recognized
 * @return SAPPHIRE_ERR_LOAD_* on other failures
 */
int sapphire_load_model(const char *path, llm_model_t *out_model,
                        char *error_message, int max_error_len);

/**
 * Load model from Safetensors format file.
 *
 * Implements the full loading pipeline:
 *
 * 1. Opens .safetensors file (mmapped for zero-copy)
 * 2. Extracts model_config_t from tensor shapes
 * 3. Allocates llm_model_t and layer arrays
 * 4. Maps each Safetensors tensor to model field using gemma3_270m_map.h
 * 5. Validates all required tensors are present
 * 6. Handles weight tying (e.g., shared embedding/lm_head)
 * 7. Closes file (keeps mmap alive for tensor references)
 *
 * @param safetensors_path Path to .safetensors file
 * @param out_model Output: populated model_t (caller must free)
 * @param error_message Optional output buffer for detailed error text
 * @param max_error_len Size of error_message buffer
 *
 * @return SAPPHIRE_OK on success
 * @return SAPPHIRE_ERR_LOAD_MISSING_TENSOR if required tensor not found
 * @return SAPPHIRE_ERR_LOAD_GENERIC on other failures
 *
 * @note Safetensors tensors are mmapped and remain valid after file is closed.
 * @note Caller must invoke llm_model_destroy() to free allocated layers.
 */
int sapphire_load_safetensors(const char *safetensors_path, llm_model_t *out_model,
                              char *error_message, int max_error_len);

/**
 * Load model from GGML format file.
 *
 * Similar to sapphire_load_safetensors() but for GGML quantized format:
 *
 * 1. Opens .gguf file and parses GGML format
 * 2. Extracts config from tensor metadata
 * 3. Maps GGML tensor names to model fields
 * 4. Supports quantization (Q4_0, Q8_0) transparently
 * 5. Validates all required tensors present
 *
 * @param ggml_path Path to .gguf file
 * @param out_model Output: populated model_t (caller must free)
 * @param error_message Optional output buffer for error details
 * @param max_error_len Size of error_message buffer
 *
 * @return SAPPHIRE_OK on success
 * @return SAPPHIRE_ERR_LOAD_MISSING_TENSOR if required tensor not found
 * @return SAPPHIRE_ERR_LOAD_GENERIC on other failures
 *
 * @note Caller must invoke llm_model_destroy() to free allocated layers.
 */
int sapphire_load_ggml(const char *ggml_path, llm_model_t *out_model,
                       char *error_message, int max_error_len);

/**
 * Utility function to validate tensor shapes match expected dimensions.
 *
 * @param actual Array of actual shape dimensions
 * @param ndim_actual Number of dimensions in actual shape
 * @param expected Array of expected shape dimensions
 * @param ndim_expected Number of dimensions in expected shape
 *
 * @return 0 if shapes match, -1 if they differ
 */
int validate_tensor_shape(const uint32_t *actual, int ndim_actual,
                          const uint32_t *expected, int ndim_expected);

#ifdef __cplusplus
}
#endif

#endif /* SAPPHIRE_TENSOR_MAPPER_H */

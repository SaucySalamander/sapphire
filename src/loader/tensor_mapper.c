/**
 * @file tensor_mapper.c
 * @brief Generic tensor mapper dispatcher for multiple format support.
 *
 * Routes model loading to format-specific implementations based on file extension.
 * Implements the tensor_mapper_plugin_t interface pattern.
 * 
 * Supported formats:
 *   - Safetensors (.safetensors): via sapphire_load_safetensors()
 *   - GGML (.gguf): via sapphire_load_ggml()
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/tensor_mapper.h"

/**
 * Detect file format from path extension.
 *
 * Returns:
 *   0 for Safetensors (.safetensors)
 *   1 for GGML (.gguf)
 *  -1 for unknown format
 */
static int detect_format(const char *path) {
    if (!path) return -1;

    const char *ext = strrchr(path, '.');
    if (!ext) return -1;

    if (strcmp(ext, ".safetensors") == 0) {
        return 0;  // Safetensors
    } else if (strcmp(ext, ".gguf") == 0) {
        return 1;  // GGML/GGUF
    }

    return -1;  // Unknown
}

/**
 * Auto-detect format and load model.
 *
 * Examines the file extension to determine which format plugin to use.
 * Supported formats: .safetensors, .gguf
 *
 * Returns: SAPPHIRE_OK on success, error code otherwise
 */
int sapphire_load_model(const char *path, llm_model_t *out_model,
                        char *error_message, int max_error_len) {
    if (!path || !out_model) {
        if (error_message && max_error_len > 0) {
            snprintf(error_message, max_error_len, "Invalid arguments (path or model NULL)");
        }
        return SAPPHIRE_ERR_LOAD_GENERIC;
    }

    int format = detect_format(path);

    switch (format) {
        case 0:  // Safetensors
            printf("Detected Safetensors format, loading...\n");
            return sapphire_load_safetensors(path, out_model, error_message, max_error_len);

        case 1:  // GGML
            printf("Detected GGML format, loading...\n");
            return sapphire_load_ggml(path, out_model, error_message, max_error_len);

        default:
            if (error_message && max_error_len > 0) {
                snprintf(error_message, max_error_len,
                         "Unknown file format. Supported: .safetensors, .gguf");
            }
            return SAPPHIRE_ERR_LOAD_UNSUPPORTED_FORMAT;
    }
}

/**
 * Validate tensor shape matches expected dimensions.
 */
int validate_tensor_shape(const uint32_t *actual, int ndim_actual,
                          const uint32_t *expected, int ndim_expected) {
    if (ndim_actual != ndim_expected) return -1;
    for (int i = 0; i < ndim_actual; i++) {
        if (actual[i] != expected[i]) return -1;
    }
    return 0;
}

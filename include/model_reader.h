/**
 * @file model_reader.h
 * @brief Unified model loading and specification retrieval.
 */

#ifndef MODEL_READER_H
#define MODEL_READER_H

#include "llm_model.h"
#include "model_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// MODEL SPECIFICATION REGISTRY
// ============================================================================

/**
 * @brief Retrieve a model specification by name.
 * @param model_name Model identifier (e.g., "gemma3-270m-it").
 * @return Pointer to static model_spec_t, or NULL if unknown.
 */
model_spec_t* get_model_spec(const char *model_name);

// ============================================================================
// UNIFIED MODEL LOADER
// ============================================================================

/**
 * @brief Load a model from a directory using a spec.
 *
 * @param model_dir Path to the model directory.
 * @param model_spec Specification to use for mapping and validation.
 * @return Allocated model state, or NULL on failure.
 */
llm_model_t* load_model(const char *model_dir, const model_spec_t *model_spec);

#ifdef __cplusplus
}
#endif

#endif // MODEL_READER_H

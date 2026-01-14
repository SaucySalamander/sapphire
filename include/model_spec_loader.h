/**
 * @file model_spec_loader.h
 * @brief Model specification loader interface
 *
 * Provides functions to load and retrieve model specifications by name.
 */

#ifndef MODEL_SPEC_LOADER_H
#define MODEL_SPEC_LOADER_H

#include "model_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the model specification for a given model name.
 *
 * Returns a pointer to the static model specification structure for the
 * requested model. Currently supports "gemma3-270m-it".
 *
 * @param model_name Name of the model (e.g., "gemma3-270m-it")
 *
 * @return Pointer to model_spec_t on success, NULL if model not found
 *
 * @note The returned pointer points to a static structure and should not be freed.
 */
model_spec_t* get_model_spec(const char *model_name);

#ifdef __cplusplus
}
#endif

#endif // MODEL_SPEC_LOADER_H

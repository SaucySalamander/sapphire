/*
 * @file model_spec_loader.c
 * @brief Generic model spec file parsers and helpers
 */

#include <stdio.h>
#include <string.h>
#include "model_spec.h"
#include "gemma3_270m_spec.h"

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
model_spec_t* get_model_spec(const char *model_name) {
    if (!model_name) return NULL;
    
    if (strcmp(model_name, "gemma3-270m-it") == 0) {
        return &GEMMA3_270M_SPEC;
    }
    
    fprintf(stderr, "ERROR: Unknown model: %s\n", model_name);
    return NULL;
}
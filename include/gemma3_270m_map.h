/**
 * @file gemma3_270m_map.h
 * @brief Tensor name mapping table for Gemma 3 270M model.
 *
 * Maps Hugging Face Safetensors tensor names to Sapphire internal structure pointers.
 * This is a declaration layer that references the implementation in src/loader/gemma3_map.c.
 */

#ifndef GEMMA3_270M_MAP_H
#define GEMMA3_270M_MAP_H

#include <stddef.h>
#include "model_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The static mapping table for Gemma-3 270M.
 * Terminated by an entry with NULL src_key.
 * Implementation in src/loader/gemma3_map.c.
 */
extern const tensor_map_entry_t GEMMA3_270M_TENSOR_MAP[];

/**
 * Returns the number of entries in the mapping table (excluding sentinel).
 */
size_t get_gemma3_270m_tensor_map_size(void);

/**
 * Static constant for the number of entries.
 * Note: Matches count in src/loader/gemma3_map.c (excluding sentinel).
 */
#define GEMMA3_270M_TENSOR_MAP_SIZE 237

#ifdef __cplusplus
}
#endif

#endif // GEMMA3_270M_MAP_H

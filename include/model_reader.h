#ifndef MODEL_READER_H
#define MODEL_READER_H

#include "llm_model.h"
#include "model_spec.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load a model by inspecting a model directory for known model files.
 *
 * Attempts to locate and load a model from `model_dir`. The function will
 * look for common model files (e.g. `model.safetensors`, `model.gguf`,
 * `model.bin`) and delegate to the appropriate loader. On success it returns
 * an allocated `llm_model_t*` which the caller must free via
 * `llm_model_destroy()`.
 *
 * @param model_dir Path to the directory containing model files.
 * @return Pointer to an allocated `llm_model_t` on success, or NULL on failure.
 * @note This is a public declaration for `load_model` which selects and
 *       loads a model from the provided directory.
 */
llm_model_t* load_model(const char *model_dir, const model_spec_t *model_spec);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_READER_H */

/*
 * @file ternary_hessian_oracle.h
 * @brief Native Vulkan-side Hessian sidecar recording workflow.
 */

#ifndef TERNARY_HESSIAN_ORACLE_H
#define TERNARY_HESSIAN_ORACLE_H

#include <stddef.h>
#include <stdint.h>

#include "activation_tape.h"
#include "calibration_corpus.h"
#include "gemma3_config.h"
#include "inference.h"
#include "transformer.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t ternary_hessian_oracle_capture_bytes(const gemma3_270m_config_t *cfg);

typedef struct {
    const char *output_path;
    const char *teacher_model_name;
    const gemma3_270m_config_t *config;
    const tape_manifest_entry_t *manifest;
    uint32_t entry_count;
    uint32_t tape_crc32;
    uint32_t sample_count;
    const float *diagonal_capture_buffer;
} ternary_hessian_oracle_write_request_t;

int ternary_hessian_oracle_capture_view(const gemma3_270m_config_t *cfg,
                                        int layer_idx,
                                        capture_target_t target,
                                        size_t *out_offset_bytes,
                                        uint32_t *out_vector_dim);

int ternary_hessian_oracle_write_sidecar(const ternary_hessian_oracle_write_request_t *request);

int ternary_record_hessian_sidecar_vulkan(const char *output_path,
                                          inference_context_t *ctx,
                                          const calibration_corpus_t *corpus,
                                          const activation_tape_t *tape,
                                          float hessian_proxy_strength,
                                          float hessian_proxy_floor);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_HESSIAN_ORACLE_H */
/**
 * @file ternary_validation.h
 * @brief Full-model validation checkpoints for ternary conversion.
 */

#ifndef TERNARY_VALIDATION_H
#define TERNARY_VALIDATION_H

#include <stdint.h>

#include "inference.h"
#include "ternary_calibration.h"
#include "ternary_telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int validate_every_n;
    int student_down_proj_input_rmsnorm;
    const char *output_dir;
    const char *telemetry_path;
    const char *const *sample_texts;
    int sample_count;
} ternary_validation_config_t;

typedef struct tensor_t tensor_t;
typedef struct ternary_layer_payload_t ternary_layer_payload_t;

typedef struct ternary_validation_patch ternary_validation_patch_t;

typedef struct {
    char tensor_name[256];
    tensor_t **slot;
    tensor_t *original_tensor;
    tensor_t *proxy_tensor;
    uint32_t crc32;
} ternary_validation_patch_record_t;

typedef struct ternary_validation_state {
    ternary_validation_config_t config;
    inference_context_t *ctx;
    float *baseline_logits;
    float *current_logits;
    float *baseline_probs;
    float *current_probs;
    float *baseline_mean_nll;
    int *baseline_top1;
    ternary_validation_patch_t *patches;
    int patch_count;
    int patch_capacity;
    uint32_t last_crc32;
    char last_tensor_name[256];
    int last_reported_count;
    int telemetry_enabled;
    ternary_telemetry_writer_t telemetry_writer;
} ternary_validation_state_t;

int ternary_validation_init(ternary_validation_state_t *state,
                            const ternary_validation_config_t *config,
                            inference_context_t *ctx);

int ternary_validation_apply_proxy(ternary_validation_state_t *state,
                                   const char *tensor_name,
                                   const ternary_calibration_result_t *result,
                                   uint32_t crc32,
                                   int converted_count);

int ternary_validation_capture_proxy_record(inference_context_t *ctx,
                                            const char *tensor_name,
                                            const ternary_calibration_result_t *result,
                                            ternary_validation_patch_record_t *out_record);

int ternary_validation_apply_proxy_from_payload(ternary_validation_state_t *state,
                                                const char *tensor_name,
                                                const ternary_layer_payload_t *payload,
                                                uint32_t crc32,
                                                int converted_count);

int ternary_validation_adopt_patch_records(ternary_validation_state_t *state,
                                           const ternary_validation_patch_record_t *records,
                                           int record_count);

int ternary_validation_finish(ternary_validation_state_t *state,
                              int converted_count);

void ternary_validation_destroy(ternary_validation_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_VALIDATION_H */
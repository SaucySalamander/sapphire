/**
 * @file ternary_validation.h
 * @brief Full-model validation checkpoints for ternary conversion.
 */

#ifndef TERNARY_VALIDATION_H
#define TERNARY_VALIDATION_H

#include <stdint.h>

#include "inference.h"
#include "ternary_calibration.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int validate_every_n;
    const char *output_dir;
    const char *const *sample_texts;
    int sample_count;
} ternary_validation_config_t;

typedef struct ternary_validation_patch ternary_validation_patch_t;

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
} ternary_validation_state_t;

int ternary_validation_init(ternary_validation_state_t *state,
                            const ternary_validation_config_t *config,
                            inference_context_t *ctx);

int ternary_validation_apply_proxy(ternary_validation_state_t *state,
                                   const char *tensor_name,
                                   const ternary_calibration_result_t *result,
                                   uint32_t crc32,
                                   int converted_count);

int ternary_validation_finish(ternary_validation_state_t *state,
                              int converted_count);

void ternary_validation_destroy(ternary_validation_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_VALIDATION_H */
/*
 * @file ternary_telemetry.h
 * @brief JSONL telemetry for ternary student update experiments.
 */

#ifndef TERNARY_TELEMETRY_H
#define TERNARY_TELEMETRY_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TERNARY_TELEMETRY_PATH_MAX 1024u

typedef struct {
    uint32_t layer_idx;
    uint32_t resume_step_idx;
    uint32_t step_idx;
    uint32_t tape_hash;
    uint32_t student_checkpoint_hash;
    float mse_loss;
    float grad_norm;
    float raw_grad_norm;
    float clipped_grad_norm;
    float clip_scale;
    float latent_saturation;
    float p_neg1;
    float p_zero;
    float p_pos1;
    float gamma_scale;
    float gamma_scale_min;
    float gamma_scale_max;
    float gamma_floor_fraction;
    float hessian_proxy_mean;
    float hessian_proxy_max;
    uint32_t hessian_proxy_source;
    float effective_learning_rate;
    float effective_hessian_scale;
    float io_ms;
    float compute_ms;
    uint32_t config_hash;
} ternary_telemetry_t;

typedef struct {
    int converted_count;
    const char *tensor_name;
    uint32_t crc32;
    float baseline_mean_nll;
    float current_mean_nll;
    float mean_kl;
    float max_kl;
    float top1_agreement;
    int sample_count;
} ternary_validation_telemetry_t;

typedef struct {
    FILE *stream;
    char path[TERNARY_TELEMETRY_PATH_MAX];
} ternary_telemetry_writer_t;

int ternary_telemetry_writer_init(ternary_telemetry_writer_t *writer,
                                  const char *telemetry_path);

void ternary_telemetry_writer_close(ternary_telemetry_writer_t *writer);

int telemetry_dump_step(ternary_telemetry_writer_t *writer,
                        const ternary_telemetry_t *telemetry);

int telemetry_dump_validation_checkpoint(ternary_telemetry_writer_t *writer,
                                         const ternary_validation_telemetry_t *telemetry);

void ternary_telemetry_print_pass_stdout(const ternary_telemetry_t *telemetry);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_TELEMETRY_H */
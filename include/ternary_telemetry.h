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
    uint32_t step_idx;
    float mse_loss;
    float grad_norm;
    float p_neg1;
    float p_zero;
    float p_pos1;
    float gamma_scale;
    float io_ms;
    float compute_ms;
    uint32_t config_hash;
} ternary_telemetry_t;

typedef struct {
    FILE *stream;
    char path[TERNARY_TELEMETRY_PATH_MAX];
} ternary_telemetry_writer_t;

int ternary_telemetry_writer_init(ternary_telemetry_writer_t *writer,
                                  const char *telemetry_path);

void ternary_telemetry_writer_close(ternary_telemetry_writer_t *writer);

int telemetry_dump_step(ternary_telemetry_writer_t *writer,
                        const ternary_telemetry_t *telemetry);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_TELEMETRY_H */
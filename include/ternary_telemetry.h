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
#define TERNARY_SPATIAL_TELEMETRY_HISTOGRAM_BINS 129u

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
    float hessian_proxy_active_max;
    uint32_t hessian_proxy_source;
    float effective_learning_rate;
    float effective_hessian_scale;
    float io_ms;
    float compute_ms;
    uint32_t config_hash;
    /* Mixed-precision anchor mode telemetry (Prompt 06) */
    uint32_t use_anchor_mode;
    uint32_t anchor_count;
    uint32_t anchor_budget_ppm;
    uint32_t anchor_saliency_mode;
    float anchor_saliency_cutoff;
    float anchor_value_rms;
    float bulk_gamma_mean;              /* Gamma computed over ternary bulk only */
    float anchor_contribution_norm;     /* L2 norm of anchor contribution */
    float bulk_contribution_norm;       /* L2 norm of ternary bulk contribution */
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
    const char *tensor_name;
    uint32_t config_hash;
    uint32_t layer_idx;
    uint32_t resume_step_idx;
    uint32_t step_idx;
    uint32_t tape_hash;
    uint32_t student_checkpoint_hash;
    uint32_t rows;
    uint32_t cols;
    uint32_t scale_group_size;
    uint32_t groups_per_row;
    uint32_t row_bucket_size;
    uint32_t row_bucket_count;
    uint32_t hessian_proxy_source;
    uint32_t use_anchor_mode;
    uint32_t anchor_count;
    uint32_t histogram_bin_count;
    float histogram_min;
    float histogram_max;
    float effective_learning_rate;
    float effective_hessian_scale;
    float hessian_proxy_cap;
} ternary_spatial_telemetry_meta_t;

typedef struct {
    uint32_t config_hash;
    uint32_t layer_idx;
    uint32_t step_idx;
    uint32_t row_bucket_idx;
    uint32_t group_idx;
    uint32_t row_start;
    uint32_t row_end;
    uint32_t col_start;
    uint32_t col_end;
    float gamma_mean;
    float gamma_min;
    float gamma_max;
    float hessian_group_mean;
    float hessian_group_max;
    float block_weight_mse;
    float block_hessian_error;
    float p_zero_fraction;
    float anchor_fraction;
} ternary_spatial_telemetry_block_t;

typedef struct {
    const char *tensor_name;
    uint32_t config_hash;
    uint32_t layer_idx;
    uint32_t step_idx;
    uint32_t histogram_bin_count;
    float histogram_min;
    float histogram_max;
    const uint32_t *teacher_counts;
    const uint32_t *student_counts;
    const uint32_t *student_bulk_counts;
} ternary_spatial_telemetry_histogram_t;

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

int telemetry_dump_spatial_snapshot_meta(ternary_telemetry_writer_t *writer,
                                         const ternary_spatial_telemetry_meta_t *telemetry);

int telemetry_dump_spatial_snapshot_block(ternary_telemetry_writer_t *writer,
                                          const ternary_spatial_telemetry_block_t *telemetry);

int telemetry_dump_spatial_snapshot_histogram(ternary_telemetry_writer_t *writer,
                                              const ternary_spatial_telemetry_histogram_t *telemetry);

void ternary_telemetry_print_pass_stdout(const ternary_telemetry_t *telemetry);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_TELEMETRY_H */
/**
 * @file ternary_calibration.h
 * @brief CPU-side STE calibration shell for ternary conversion.
 */

#ifndef TERNARY_CALIBRATION_H
#define TERNARY_CALIBRATION_H

#include <stddef.h>
#include <stdint.h>

#include "ternary_hessian_sidecar.h"
#include "ternary_hessian_proxy.h"
#include "ternary_telemetry.h"

/* Forward declaration for anchor types */
typedef struct ternary_anchor_entry_t ternary_anchor_entry_t;

typedef struct sapphire_tokenizer_t sapphire_tokenizer_t;
typedef struct model_spec model_spec_t;
typedef struct inference_session_t inference_session_t;
typedef struct activation_tape_t activation_tape_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int ste_steps;
    float learning_rate;
    float zero_threshold;
    float momentum;
    float regularization_strength;
    float non_collapse_weight;
    float zero_occupancy_floor;
    float clip_value;
    int calibration_samples;
    float kl_weight;
    float kl_temperature;
    int kl_update_interval;
    int kl_sample_count;
    int early_stop_patience;
    float early_stop_min_delta;
    float early_stop_divergence_ratio;
    int student_down_proj_input_rmsnorm;
    int simulate_activation_a8;
    int use_hessian_proxy;
    float hessian_proxy_strength;
    float hessian_proxy_floor;
    float max_grad_norm;
    float adam_beta2;
    float adam_epsilon;
    int telemetry_interval;
    const char *telemetry_path;
    const char *spatial_telemetry_path;
    int spatial_telemetry_row_bucket_size;
    const char *hessian_sidecar_path;
    uint32_t hessian_sidecar_crc32;
    ternary_telemetry_t *telemetry;
    /* Anchor mode configuration (Prompt 03) */
    int use_anchor_mode;              /* Enable hybrid ternary+anchor */
    uint32_t anchor_budget_ppm;       /* Anchor budget in parts-per-million */
    int anchor_saliency_mode;         /* 0=none, 1=weight*hessian, 2=hessian, 3=weight */
    float anchor_learning_rate_mult;  /* LR multiplier for anchor values vs bulk */
    const ternary_anchor_entry_t *protected_anchor_entries; /* Internal STE freeze mask */
    const uint32_t *protected_anchor_row_offsets;           /* Internal STE freeze mask */
    uint32_t protected_anchor_count;                        /* Internal STE freeze mask */
    const uint16_t *telemetry_reference_weights;           /* Internal telemetry-only teacher reference */
} transformer_ste_config_t;

typedef struct {
    const char *const *sample_texts;
    int sample_count;
    sapphire_tokenizer_t *tokenizer;
    const model_spec_t *model_spec;
    inference_session_t *session;
    const char *tensor_name;
} ternary_calibration_corpus_t;

typedef struct {
    const activation_tape_t *tape;
    const char *tensor_name;
    ternary_hessian_proxy_cache_t *proxy_cache;
} ternary_activation_tape_context_t;

typedef struct {
    const ternary_calibration_corpus_t *corpus;
    const ternary_activation_tape_context_t *tape_context;
    const ternary_hessian_sidecar_t *sidecar;
} ternary_calibration_source_t;

typedef struct {
    float *latent_weights;
    int8_t *ternary_weights;
    uint8_t *packed_weights;
    float *scales;
    size_t scale_count;
    size_t weight_count;
    size_t packed_weight_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t scale_group_size;
    /* Anchor mode outputs (NULL if not in anchor mode) */
    ternary_anchor_entry_t *anchor_entries;  /* Sorted by (row, col) */
    uint32_t *anchor_row_offsets;            /* CSR-like row offsets (rows+1) */
    uint32_t anchor_count;
    float anchor_saliency_cutoff;
    float anchor_value_rms;
    float bulk_gamma_mean;                   /* Mean gamma for ternary bulk only */
} ternary_calibration_result_t;

int transformer_calibrate_layer_ste_with_tape(const uint16_t *bf16_weights,
                                              uint32_t rows,
                                              uint32_t cols,
                                              const transformer_ste_config_t *config,
                                              const ternary_calibration_source_t *source,
                                              ternary_calibration_result_t *out_result);

int transformer_calibrate_layer_ste(const uint16_t *bf16_weights,
                                    uint32_t rows,
                                    uint32_t cols,
                                    const transformer_ste_config_t *config,
                                    const ternary_calibration_corpus_t *corpus,
                                    ternary_calibration_result_t *out_result);

/**
 * @brief Calibrate a layer in hybrid anchor mode.
 *
 * Protected anchor coordinates bypass ternary quantization and are stored as BF16.
 * The ternary bulk's gamma is computed excluding anchor positions.
 *
 * @param bf16_weights  Original BF16 weights [rows x cols]
 * @param rows          Number of rows
 * @param cols          Number of columns
 * @param config        STE configuration (must have use_anchor_mode=1)
 * @param source        Calibration source (tape, sidecar, etc.)
 * @param out_result    Output calibration result with anchors
 * @return 0 on success, -1 on error
 */
int transformer_calibrate_layer_ste_hybrid(const uint16_t *bf16_weights,
                                           uint32_t rows,
                                           uint32_t cols,
                                           const transformer_ste_config_t *config,
                                           const ternary_calibration_source_t *source,
                                           ternary_calibration_result_t *out_result);

void transformer_free_ternary_calibration_result(ternary_calibration_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_CALIBRATION_H */
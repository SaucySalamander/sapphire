/**
 * @file ternary_conversion.h
 * @brief CLI-facing entry point for ternary model conversion.
 */

#ifndef TERNARY_CONVERSION_H
#define TERNARY_CONVERSION_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *model_name;
    const char *output_path;
    const char *layer_name;
    const char *activation_tape_path;
    const char *hessian_sidecar_path;
    const char *teacher_model_name;
    const char *structural_map_path;
    const char *calibration_corpus_path;
    const char *calibration_corpus_manifest_path;
    const char *validation_corpus_path;
    const char *validation_corpus_manifest_path;
    int context_len;
    int calibration_sample_limit;
    int validation_sample_limit;
    int checkpoint_every_n_layers;
    int validate_every_n;
    int ste_steps;
    int progressive_calib;
    float kl_weight;
    int disable_hessian_proxy;
    float hessian_proxy_strength;
    float hessian_proxy_floor;
    float max_grad_norm;
} ternary_conversion_config_t;

int transformer_run_ternary_conversion(const ternary_conversion_config_t *config);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_CONVERSION_H */
/**
 * @file ternary_calibration.h
 * @brief CPU-side STE calibration shell for ternary conversion.
 */

#ifndef TERNARY_CALIBRATION_H
#define TERNARY_CALIBRATION_H

#include <stddef.h>
#include <stdint.h>

typedef struct sapphire_tokenizer_t sapphire_tokenizer_t;
typedef struct model_spec model_spec_t;
typedef struct inference_session_t inference_session_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int ste_steps;
    float learning_rate;
    float zero_threshold;
    float momentum;
    float regularization_strength;
    float clip_value;
    int calibration_samples;
    float kl_weight;
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
    float *latent_weights;
    int8_t *ternary_weights;
    uint8_t *packed_weights;
    float *scales;
    size_t weight_count;
    size_t packed_weight_bytes;
    uint32_t rows;
    uint32_t cols;
} ternary_calibration_result_t;

int transformer_calibrate_layer_ste(const uint16_t *bf16_weights,
                                    uint32_t rows,
                                    uint32_t cols,
                                    const transformer_ste_config_t *config,
                                    const ternary_calibration_corpus_t *corpus,
                                    ternary_calibration_result_t *out_result);

void transformer_free_ternary_calibration_result(ternary_calibration_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_CALIBRATION_H */
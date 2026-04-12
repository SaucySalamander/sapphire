/**
 * @file ternary_calibration.c
 * @brief Prompt 2 STE calibration shell for ternary conversion.
 */

#include "ternary_calibration.h"

#include "log.h"
#include "activation_tape.h"
#include "llm_model.h"
#include "inference.h"
#include "model_spec.h"
#include "ternary_io.h"
#include "ternary_hessian_proxy.h"
#include "ternary_telemetry.h"
#include "tokenizer.h"
#include "transformer.h"
#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TERNARY_SYMBOL_ZERO     0u
#define TERNARY_SYMBOL_POSITIVE 1u
#define TERNARY_SYMBOL_NEGATIVE 2u
#define TERNARY_PACK_WIDTH      4u
#define STE_PROGRESSIVE_EARLY_LAYER_COUNT        6u
#define STE_PROGRESSIVE_WARMUP_STEPS             12u
#define STE_PROGRESSIVE_MIN_GAMMA_SCALE          0.10f
#define STE_PROGRESSIVE_LR_SCALE                 2.50f
#define STE_PROGRESSIVE_HESSIAN_SCALE            0.35f
#define STE_PROGRESSIVE_REGULARIZATION_SCALE     0.50f
#define STE_PROGRESSIVE_NON_COLLAPSE_SCALE       3.00f
#define STE_EXTERNAL_PROXY_CAP                   12.0f
#define STE_EARLY_LAYER_PROXY_CAP                6.0f
#define STE_GAMMA_FLOOR_EPSILON                  1e-5f
#define STE_A8_QUANT_MAX                         127.0f

static const char *g_calibration_docs[] = {
    "A careful systems design balances latency, memory locality, and correctness under constrained hardware budgets.",
    "Educational prose with precise explanations improves retention when each paragraph introduces one concept at a time.",
    "Static descriptor tables reduce runtime churn and make command recording deterministic for high-throughput inference.",
    "for (int i = 0; i < n; ++i) { acc += weights[i] * input[i]; }",
    "VkBufferMemoryBarrier protects only the buffers that actually participate in a read-after-write dependency.",
    "Given three switches and one lamp, determine the state transitions that isolate which switch controls the lamp.",
    "If every valid path must satisfy two independent constraints, enumerate the cases and eliminate contradictions.",
    "A C18 codebase benefits from explicit ownership, manual allocation checks, and small helper functions with narrow scope."
};

#define CALIB_DOC_COUNT ((int)(sizeof(g_calibration_docs) / sizeof(g_calibration_docs[0])))

static float bf16_to_f32_scalar(uint16_t value) {
    union {
        uint32_t u32;
        float f32;
    } bits;
    bits.u32 = ((uint32_t)value) << 16;
    return bits.f32;
}

static transformer_ste_config_t default_ste_config(void) {
    transformer_ste_config_t config;
    config.ste_steps = 3;
    config.learning_rate = 0.03f;
    config.zero_threshold = 0.05f;
    config.momentum = 0.85f;
    config.regularization_strength = 0.01f;
    config.non_collapse_weight = 0.02f;
    config.zero_occupancy_floor = 0.75f;
    config.clip_value = 1.0f;
    config.calibration_samples = 4;
    config.kl_weight = 0.05f;
    config.kl_temperature = 1.0f;
    config.simulate_activation_a8 = 0;
    config.use_hessian_proxy = 1;
    config.hessian_proxy_strength = 1.0f;
    config.hessian_proxy_floor = 0.05f;
    config.max_grad_norm = 1.0f;
    config.adam_beta2 = 0.95f;
    config.adam_epsilon = 1e-8f;
    config.telemetry_interval = 10;
    config.telemetry_path = "./out/ternary_telemetry.jsonl";
    config.hessian_sidecar_path = NULL;
    config.hessian_sidecar_crc32 = 0u;
    config.telemetry = NULL;
    return config;
}

static void normalize_ste_config(transformer_ste_config_t *config) {
    if (!config) {
        return;
    }

    if (config->ste_steps <= 0) config->ste_steps = 1;
    if (config->learning_rate <= 0.0f) config->learning_rate = 0.05f;
    if (config->zero_threshold < 0.0f) config->zero_threshold = 0.05f;
    if (config->momentum < 0.0f || config->momentum >= 1.0f) config->momentum = 0.85f;
    if (config->regularization_strength < 0.0f) config->regularization_strength = 0.01f;
    if (config->non_collapse_weight < 0.0f) config->non_collapse_weight = 0.02f;
    if (config->zero_occupancy_floor < 0.0f || config->zero_occupancy_floor >= 1.0f) {
        config->zero_occupancy_floor = 0.75f;
    }
    if (config->clip_value <= 0.0f) config->clip_value = 1.0f;
    if (config->calibration_samples <= 0) config->calibration_samples = 4;
    if (config->calibration_samples > 16) config->calibration_samples = 16;
    if (config->kl_weight < 0.0f) config->kl_weight = 0.0f;
    if (config->kl_temperature <= 0.0f) config->kl_temperature = 1.0f;
    config->simulate_activation_a8 = config->simulate_activation_a8 ? 1 : 0;
    config->use_hessian_proxy = config->use_hessian_proxy ? 1 : 0;
    if (config->hessian_proxy_strength < 0.0f) config->hessian_proxy_strength = 1.0f;
    if (config->hessian_proxy_floor < 0.0f) config->hessian_proxy_floor = 0.05f;
    if (config->max_grad_norm <= 0.0f) config->max_grad_norm = 1.0f;
    if (config->adam_beta2 < 0.0f || config->adam_beta2 >= 1.0f) config->adam_beta2 = 0.95f;
    if (config->adam_epsilon <= 0.0f) config->adam_epsilon = 1e-8f;
    if (config->telemetry_interval <= 0) config->telemetry_interval = 10;
}

static size_t ternary_packed_bytes(uint32_t rows, uint32_t cols) {
    size_t packed_cols = ((size_t)cols + (TERNARY_PACK_WIDTH - 1u)) / TERNARY_PACK_WIDTH;
    return (size_t)rows * packed_cols;
}

static uint8_t ternary_symbol_from_value(int8_t value) {
    if (value > 0) return TERNARY_SYMBOL_POSITIVE;
    if (value < 0) return TERNARY_SYMBOL_NEGATIVE;
    return TERNARY_SYMBOL_ZERO;
}

static uint32_t mix_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

static int telemetry_try_parse_layer_index(const char *tensor_name,
                                           uint32_t *out_layer_index)
{
    const char *cursor = NULL;

    if (!tensor_name || !out_layer_index) {
        return 0;
    }

    *out_layer_index = 0u;
    cursor = strstr(tensor_name, ".layers.");
    if (cursor) {
        char *end = NULL;
        unsigned long parsed = strtoul(cursor + 8, &end, 10);
        if (end != cursor + 8 && parsed <= 0xFFFFFFFFul) {
            *out_layer_index = (uint32_t)parsed;
            return 1;
        }
    }

    cursor = strstr(tensor_name, "blk.");
    if (cursor) {
        char *end = NULL;
        unsigned long parsed = strtoul(cursor + 4, &end, 10);
        if (end != cursor + 4 && parsed <= 0xFFFFFFFFul) {
            *out_layer_index = (uint32_t)parsed;
            return 1;
        }
    }

    return 0;
}

typedef struct {
    float mean;
    float min;
    float max;
    float floor_fraction;
} ternary_scale_stats_t;

typedef struct {
    int has_layer_index;
    uint32_t layer_index;
    int early_layer_warmup;
    uint32_t warmup_steps;
    float layer_step_multiplier;
    float hessian_scale;
    float regularization_scale;
    float non_collapse_scale;
    float min_gamma_scale;
    float hessian_proxy_cap;
} ste_layer_schedule_t;

typedef struct {
    float learning_rate;
    float regularization_strength;
    float non_collapse_weight;
    float hessian_scale;
    float hessian_proxy_cap;
} ste_step_schedule_t;

static void ste_layer_schedule_reset(ste_layer_schedule_t *schedule)
{
    if (!schedule) {
        return;
    }

    memset(schedule, 0, sizeof(*schedule));
    schedule->layer_step_multiplier = 1.0f;
    schedule->hessian_scale = 1.0f;
    schedule->regularization_scale = 1.0f;
    schedule->non_collapse_scale = 1.0f;
}

static void ste_build_layer_schedule(const char *tensor_name,
                                     int ste_steps,
                                     ste_layer_schedule_t *out_schedule)
{
    uint32_t layer_index = 0u;

    if (!out_schedule) {
        return;
    }

    ste_layer_schedule_reset(out_schedule);
    if (!telemetry_try_parse_layer_index(tensor_name, &layer_index)) {
        return;
    }

    out_schedule->has_layer_index = 1;
    out_schedule->layer_index = layer_index;
    if (layer_index >= STE_PROGRESSIVE_EARLY_LAYER_COUNT) {
        return;
    }

    out_schedule->early_layer_warmup = 1;
    out_schedule->warmup_steps = (uint32_t)ste_steps;
    if (out_schedule->warmup_steps > STE_PROGRESSIVE_WARMUP_STEPS) {
        out_schedule->warmup_steps = STE_PROGRESSIVE_WARMUP_STEPS;
    }
    out_schedule->layer_step_multiplier = STE_PROGRESSIVE_LR_SCALE;
    out_schedule->hessian_scale = STE_PROGRESSIVE_HESSIAN_SCALE;
    out_schedule->regularization_scale = STE_PROGRESSIVE_REGULARIZATION_SCALE;
    out_schedule->non_collapse_scale = STE_PROGRESSIVE_NON_COLLAPSE_SCALE;
    out_schedule->min_gamma_scale = STE_PROGRESSIVE_MIN_GAMMA_SCALE;
    out_schedule->hessian_proxy_cap = STE_EARLY_LAYER_PROXY_CAP;
}

static float ste_compute_teacher_abs_mean(const float *calibration_vectors,
                                          int sample_count,
                                          uint32_t cols)
{
    double abs_sum = 0.0;
    size_t value_count = 0u;

    if (!calibration_vectors || sample_count <= 0 || cols == 0u) {
        return 0.0f;
    }

    value_count = (size_t)sample_count * cols;
    for (size_t value_idx = 0u; value_idx < value_count; ++value_idx) {
        abs_sum += fabs((double)calibration_vectors[value_idx]);
    }

    if (value_count == 0u) {
        return 0.0f;
    }

    return (float)(abs_sum / (double)value_count);
}

static void telemetry_compute_scale_stats(const float *scales,
                                          const float *scale_floor,
                                          uint32_t rows,
                                          ternary_scale_stats_t *out_stats)
{
    float mean = 0.0f;
    float min_scale = 0.0f;
    float max_scale = 0.0f;
    uint32_t floor_hits = 0u;

    if (!out_stats) {
        return;
    }

    memset(out_stats, 0, sizeof(*out_stats));
    if (!scales || rows == 0u) {
        return;
    }

    min_scale = scales[0];
    max_scale = scales[0];
    for (uint32_t row = 0; row < rows; ++row) {
        float scale = scales[row];

        mean += scale;
        if (scale < min_scale) {
            min_scale = scale;
        }
        if (scale > max_scale) {
            max_scale = scale;
        }
        if (scale_floor && scale <= scale_floor[row] + STE_GAMMA_FLOOR_EPSILON) {
            ++floor_hits;
        }
    }

    out_stats->mean = mean / (float)rows;
    out_stats->min = min_scale;
    out_stats->max = max_scale;
    out_stats->floor_fraction = scale_floor
        ? ((float)floor_hits / (float)rows)
        : 0.0f;
}

static uint32_t telemetry_compute_config_hash(const transformer_ste_config_t *config,
                                             const char *tensor_name,
                                             uint32_t rows,
                                             uint32_t cols)
{
    uint32_t crc32 = 0u;

    if (!config) {
        return 0u;
    }

    crc32 = io_crc32_update(crc32, &config->ste_steps, sizeof(config->ste_steps));
    crc32 = io_crc32_update(crc32, &config->learning_rate, sizeof(config->learning_rate));
    crc32 = io_crc32_update(crc32, &config->zero_threshold, sizeof(config->zero_threshold));
    crc32 = io_crc32_update(crc32, &config->momentum, sizeof(config->momentum));
    crc32 = io_crc32_update(crc32, &config->regularization_strength, sizeof(config->regularization_strength));
    crc32 = io_crc32_update(crc32, &config->non_collapse_weight, sizeof(config->non_collapse_weight));
    crc32 = io_crc32_update(crc32, &config->zero_occupancy_floor, sizeof(config->zero_occupancy_floor));
    crc32 = io_crc32_update(crc32, &config->clip_value, sizeof(config->clip_value));
    crc32 = io_crc32_update(crc32, &config->calibration_samples, sizeof(config->calibration_samples));
    crc32 = io_crc32_update(crc32, &config->kl_weight, sizeof(config->kl_weight));
    crc32 = io_crc32_update(crc32, &config->kl_temperature, sizeof(config->kl_temperature));
    crc32 = io_crc32_update(crc32, &config->simulate_activation_a8, sizeof(config->simulate_activation_a8));
    crc32 = io_crc32_update(crc32, &config->use_hessian_proxy, sizeof(config->use_hessian_proxy));
    crc32 = io_crc32_update(crc32, &config->hessian_proxy_strength, sizeof(config->hessian_proxy_strength));
    crc32 = io_crc32_update(crc32, &config->hessian_proxy_floor, sizeof(config->hessian_proxy_floor));
    crc32 = io_crc32_update(crc32, &config->max_grad_norm, sizeof(config->max_grad_norm));
    crc32 = io_crc32_update(crc32, &config->adam_beta2, sizeof(config->adam_beta2));
    crc32 = io_crc32_update(crc32, &config->adam_epsilon, sizeof(config->adam_epsilon));
    crc32 = io_crc32_update(crc32, &config->telemetry_interval, sizeof(config->telemetry_interval));
    crc32 = io_crc32_update(crc32, config->hessian_sidecar_path, config->hessian_sidecar_path ? strlen(config->hessian_sidecar_path) + 1u : 0u);
    crc32 = io_crc32_update(crc32, &config->hessian_sidecar_crc32, sizeof(config->hessian_sidecar_crc32));
    crc32 = io_crc32_update(crc32, tensor_name, tensor_name ? strlen(tensor_name) + 1u : 0u);
    crc32 = io_crc32_update(crc32, &rows, sizeof(rows));
    crc32 = io_crc32_update(crc32, &cols, sizeof(cols));
    return crc32;
}

static float telemetry_diff_ms(const struct timespec *start, const struct timespec *end)
{
    time_t sec = end->tv_sec - start->tv_sec;
    long nsec = end->tv_nsec - start->tv_nsec;

    return (float)sec * 1000.0f + (float)nsec / 1000000.0f;
}

typedef struct {
    float p_neg1;
    float p_zero;
    float p_pos1;
} ternary_distribution_stats_t;

static void telemetry_compute_packed_distribution(const uint8_t *packed_weights,
                                                  size_t packed_weight_bytes,
                                                  uint32_t rows,
                                                  uint32_t cols,
                                                  ternary_distribution_stats_t *out_stats)
{
    size_t zero_count = 0u;
    size_t pos_count = 0u;
    size_t neg_count = 0u;
    size_t total_count = (size_t)rows * cols;

    if (!packed_weights || packed_weight_bytes == 0u || total_count == 0u) {
        if (out_stats) {
            out_stats->p_neg1 = 0.0f;
            out_stats->p_zero = 0.0f;
            out_stats->p_pos1 = 0.0f;
        }
        return;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        size_t row_base = (size_t)r * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            size_t packed_idx = ((size_t)r * ((cols + (TERNARY_PACK_WIDTH - 1u)) / TERNARY_PACK_WIDTH)) + (size_t)c / TERNARY_PACK_WIDTH;
            uint32_t lane = c % TERNARY_PACK_WIDTH;
            uint8_t symbol = (uint8_t)((packed_weights[packed_idx] >> (lane * 2u)) & 0x3u);

            (void)row_base;
            if (symbol == TERNARY_SYMBOL_ZERO) {
                ++zero_count;
            } else if (symbol == TERNARY_SYMBOL_POSITIVE) {
                ++pos_count;
            } else if (symbol == TERNARY_SYMBOL_NEGATIVE) {
                ++neg_count;
            }
        }
    }

    if (out_stats) {
        out_stats->p_neg1 = (float)neg_count / (float)total_count;
        out_stats->p_zero = (float)zero_count / (float)total_count;
        out_stats->p_pos1 = (float)pos_count / (float)total_count;
    }
}

static void seed_latent_weights(ternary_calibration_result_t *result,
                                const uint16_t *bf16_weights,
                                size_t weight_count,
                                float clip_value)
{
    for (size_t i = 0; i < weight_count; ++i) {
        uint16_t raw_value = 0;
        memcpy(&raw_value,
               (const unsigned char *)bf16_weights + i * sizeof(uint16_t),
               sizeof(raw_value));
        float value = bf16_to_f32_scalar(raw_value);

        if (value > clip_value) {
            value = clip_value;
        }
        if (value < -clip_value) {
            value = -clip_value;
        }

        result->latent_weights[i] = value;
    }
}

typedef struct {
    int enabled;
    uint32_t interval;
    ternary_telemetry_writer_t writer;
    ternary_telemetry_t telemetry;
} ste_telemetry_runtime_t;

typedef struct {
    uint32_t step_idx;
    uint32_t total_steps;
    float mse_loss;
    float raw_grad_norm;
    float clipped_grad_norm;
    float clip_scale;
    float latent_saturation;
    float compute_ms;
    float effective_learning_rate;
    float effective_hessian_scale;
} ste_telemetry_step_t;

static int ste_telemetry_runtime_init(ste_telemetry_runtime_t *runtime,
                                      const transformer_ste_config_t *config,
                                      const char *tensor_name,
                                      uint32_t rows,
                                      uint32_t cols)
{
    if (!runtime || !config) {
        return 0;
    }

    memset(runtime, 0, sizeof(*runtime));
    if (!config->telemetry_path || config->telemetry_path[0] == '\0' || config->telemetry_interval <= 0) {
        return 0;
    }

    runtime->interval = (uint32_t)config->telemetry_interval;
    runtime->telemetry = config->telemetry ? *config->telemetry : (ternary_telemetry_t){0};
    if (runtime->telemetry.config_hash == 0u) {
        runtime->telemetry.config_hash = telemetry_compute_config_hash(config, tensor_name, rows, cols);
    }
    {
        uint32_t parsed_layer_index = 0u;

        if (telemetry_try_parse_layer_index(tensor_name, &parsed_layer_index)) {
            runtime->telemetry.layer_idx = parsed_layer_index;
        }
    }

    if (ternary_telemetry_writer_init(&runtime->writer, config->telemetry_path) != 0) {
        LOG_WARN("telemetry: disabled for %s", tensor_name ? tensor_name : "<unknown>");
        memset(runtime, 0, sizeof(*runtime));
        return 0;
    }

    runtime->enabled = 1;
    return 1;
}

static void ste_telemetry_runtime_close(ste_telemetry_runtime_t *runtime)
{
    if (!runtime) {
        return;
    }

    if (runtime->enabled) {
        ternary_telemetry_writer_close(&runtime->writer);
    }
    memset(runtime, 0, sizeof(*runtime));
}

static int ste_emit_telemetry_step(ste_telemetry_runtime_t *runtime,
                                   const ternary_calibration_result_t *result,
                                   const float *scale_floor,
                                   const ste_telemetry_step_t *step)
{
    ternary_telemetry_t telemetry;
    ternary_scale_stats_t scale_stats;

    if (!runtime || !runtime->enabled || !result || !step) {
        return 0;
    }

    telemetry = runtime->telemetry;
    telemetry.step_idx = step->step_idx;
    telemetry.mse_loss = step->mse_loss;
    telemetry.grad_norm = step->raw_grad_norm;
    telemetry.raw_grad_norm = step->raw_grad_norm;
    telemetry.clipped_grad_norm = step->clipped_grad_norm;
    telemetry.clip_scale = step->clip_scale;
    telemetry.latent_saturation = step->latent_saturation;
    telemetry.compute_ms = step->compute_ms;
    telemetry_compute_scale_stats(result->scales, scale_floor, result->rows, &scale_stats);
    telemetry.gamma_scale = scale_stats.mean;
    telemetry.gamma_scale_min = scale_stats.min;
    telemetry.gamma_scale_max = scale_stats.max;
    telemetry.gamma_floor_fraction = scale_stats.floor_fraction;
    {
        ternary_distribution_stats_t distribution_stats;

        telemetry_compute_packed_distribution(result->packed_weights,
                                              result->packed_weight_bytes,
                                              result->rows,
                                              result->cols,
                                              &distribution_stats);
        telemetry.p_neg1 = distribution_stats.p_neg1;
        telemetry.p_zero = distribution_stats.p_zero;
        telemetry.p_pos1 = distribution_stats.p_pos1;
    }
    telemetry.effective_learning_rate = step->effective_learning_rate;
    telemetry.effective_hessian_scale = step->effective_hessian_scale;
    ternary_telemetry_print_pass_stdout(&telemetry);

    if (step->step_idx + 1u < step->total_steps && runtime->interval > 0u && ((step->step_idx + 1u) % runtime->interval) != 0u) {
        return 0;
    }

    return telemetry_dump_step(&runtime->writer, &telemetry);
}

static void build_feature_hashed_vector(const int *tokens,
                                        int token_count,
                                        uint32_t cols,
                                        float *out_vector) {
    float norm = 0.0f;

    memset(out_vector, 0, (size_t)cols * sizeof(float));
    if (!tokens || token_count <= 0 || cols == 0) {
        return;
    }

    for (int t = 0; t < token_count; ++t) {
        uint32_t base = mix_u32((uint32_t)tokens[t] ^ ((uint32_t)(t + 1) * 2654435761u));
        for (uint32_t proj = 0; proj < 3u; ++proj) {
            uint32_t state = mix_u32(base ^ ((proj + 1u) * 2246822519u));
            uint32_t idx = state % cols;
            float sign = (state & 1u) ? 1.0f : -1.0f;
            out_vector[idx] += sign / sqrtf((float)(proj + 1u));
        }
        if (t > 0) {
            uint32_t pair_state = mix_u32((uint32_t)tokens[t - 1] * 1315423911u ^ (uint32_t)tokens[t]);
            uint32_t pair_idx = pair_state % cols;
            out_vector[pair_idx] += (pair_state & 2u) ? 0.5f : -0.5f;
        }
    }

    for (uint32_t c = 0; c < cols; ++c) {
        norm += out_vector[c] * out_vector[c];
    }
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / sqrtf(norm);
        for (uint32_t c = 0; c < cols; ++c) {
            out_vector[c] *= inv_norm;
        }
    }
}

static int build_tokenized_prompt_vector(const ternary_calibration_corpus_t *corpus,
                                         const char *text,
                                         uint32_t cols,
                                         float *out_vector) {
    int token_count = 0;
    int *tokens = NULL;
    const int max_tokens = 1024;

    if (!corpus || !corpus->tokenizer || !corpus->model_spec || !text || !out_vector) {
        return -1;
    }

    tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) {
        LOG_ERROR("build_tokenized_prompt_vector: token buffer allocation failed");
        return -1;
    }

    token_count = build_gemma3_prompt(corpus->model_spec, text, tokens, max_tokens);
    if (token_count <= 0) {
        token_count = tokenize(corpus->tokenizer, text, tokens, max_tokens);
    }
    if (token_count <= 0) {
        LOG_WARN("build_tokenized_prompt_vector: tokenization failed for a calibration sample");
        free(tokens);
        return -1;
    }

    build_feature_hashed_vector(tokens, token_count, cols, out_vector);
    free(tokens);
    return 0;
}

static float calibration_value_from_text(const char *text, uint32_t sample_idx, uint32_t dim_idx) {
    uint32_t state = 2166136261u ^ (sample_idx * 16777619u) ^ (dim_idx * 374761393u);
    size_t len = 0;
    float acc = 0.0f;

    if (!text) {
        return 0.0f;
    }

    len = strlen(text);
    if (len == 0) {
        return 0.0f;
    }

    for (size_t k = 0; k < 4; ++k) {
        unsigned char ch = (unsigned char)text[(dim_idx + sample_idx * 13u + (uint32_t)k * 17u) % len];
        state = mix_u32(state ^ (uint32_t)ch ^ ((uint32_t)k << 24));
        acc += ((float)(int)(ch % 31u) - 15.0f) / 15.0f;
    }

    acc += ((float)(state & 1023u) / 511.5f) - 1.0f;
    return acc * 0.5f;
}

static float* build_fallback_vectors(uint32_t cols, int sample_count) {
    float *vectors = NULL;

    if (cols == 0 || sample_count <= 0) {
        return NULL;
    }

    vectors = (float *)malloc((size_t)sample_count * cols * sizeof(float));
    if (!vectors) {
        LOG_ERROR("build_calibration_vectors: allocation failed");
        return NULL;
    }

    for (int s = 0; s < sample_count; ++s) {
        const char *doc = g_calibration_docs[s % CALIB_DOC_COUNT];
        float norm = 0.0f;
        size_t base = (size_t)s * cols;

        for (uint32_t c = 0; c < cols; ++c) {
            float value = calibration_value_from_text(doc, (uint32_t)s, c);
            vectors[base + c] = value;
            norm += value * value;
        }

        if (norm > 1e-12f) {
            float inv_norm = 1.0f / sqrtf(norm);
            for (uint32_t c = 0; c < cols; ++c) {
                vectors[base + c] *= inv_norm;
            }
        }
    }

    return vectors;
}

static float *build_tape_calibration_vectors(const ternary_activation_tape_context_t *tape_context,
                                             uint32_t cols,
                                             int requested_sample_count,
                                             int *out_sample_count) {
    const activation_tape_t *tape = NULL;
    const char *tensor_name = NULL;
    uint32_t tape_cols = 0u;
    int tape_samples = 0;
    int sample_count = requested_sample_count;
    float *vectors = NULL;

    if (!tape_context || !tape_context->tape || !tape_context->tensor_name || requested_sample_count <= 0) {
        return NULL;
    }

    tape = tape_context->tape;
    tensor_name = tape_context->tensor_name;
    tape_cols = activation_tape_vector_dim(tape, tensor_name);
    if (tape_cols == 0u) {
        LOG_ERROR("build_calibration_vectors: activation tape does not contain tensor %s", tensor_name);
        return NULL;
    }
    if (tape_cols != cols) {
        LOG_ERROR("build_calibration_vectors: tape dimension mismatch for %s (tape=%u expected=%u)",
                  tensor_name,
                  tape_cols,
                  cols);
        return NULL;
    }

    tape_samples = activation_tape_sample_count(tape);
    if (tape_samples <= 0) {
        LOG_ERROR("build_calibration_vectors: activation tape has no samples for %s", tensor_name);
        return NULL;
    }
    if (sample_count > tape_samples) {
        LOG_WARN("build_calibration_vectors: clamping sample count for %s from %d to %d",
                 tensor_name,
                 sample_count,
                 tape_samples);
        sample_count = tape_samples;
    }

    vectors = (float *)calloc((size_t)sample_count * cols, sizeof(float));
    if (!vectors) {
        LOG_ERROR("build_calibration_vectors: tape-backed allocation failed for %s", tensor_name);
        return NULL;
    }

    for (int s = 0; s < sample_count; ++s) {
        if (activation_tape_get_vector(tape,
                                       tensor_name,
                                       s,
                                       vectors + (size_t)s * cols) != 0) {
            LOG_ERROR("build_calibration_vectors: failed to read %s sample %d from activation tape",
                      tensor_name,
                      s);
            free(vectors);
            return NULL;
        }
    }

    if (out_sample_count) {
        *out_sample_count = sample_count;
    }

    LOG_INFO("Built tape-backed calibration vectors: tensor=%s samples=%d",
             tensor_name,
             sample_count);
    return vectors;
}

static float* build_calibration_vectors(uint32_t cols,
                                        int sample_count,
                                        const ternary_calibration_corpus_t *corpus,
                                        const ternary_activation_tape_context_t *tape_context,
                                        int *out_sample_count) {
    float *vectors = NULL;
    int used_activation_replay = 0;

    if (cols == 0 || sample_count <= 0) {
        return NULL;
    }

    if (tape_context && tape_context->tape) {
        vectors = build_tape_calibration_vectors(tape_context, cols, sample_count, out_sample_count);
        if (vectors) {
            return vectors;
        }
        return NULL;
    }

    if (out_sample_count) {
        *out_sample_count = sample_count;
    }

    if (!corpus || !corpus->sample_texts || corpus->sample_count <= 0 ||
        !corpus->tokenizer || !corpus->model_spec) {
        return build_fallback_vectors(cols, sample_count);
    }

    vectors = (float *)malloc((size_t)sample_count * cols * sizeof(float));
    if (!vectors) {
        LOG_ERROR("build_calibration_vectors: allocation failed");
        return NULL;
    }

    for (int s = 0; s < sample_count; ++s) {
        const char *sample_text = corpus->sample_texts[s % corpus->sample_count];
        float *dst = vectors + (size_t)s * cols;
        if (corpus->session && corpus->tensor_name && corpus->tensor_name[0] != '\0') {
            transformer_activation_capture_request_t activation_request;

            memset(&activation_request, 0, sizeof(activation_request));
            activation_request.spec = corpus->model_spec;
            activation_request.text = sample_text;
            activation_request.tensor_name = corpus->tensor_name;
            activation_request.out_vector = dst;
            activation_request.out_dim = cols;

            if (sapphire_collect_tensor_activation(corpus->session,
                                                  corpus->tokenizer,
                                                  &activation_request) == 0) {
                used_activation_replay = 1;
                continue;
            }
        }
        if (build_tokenized_prompt_vector(corpus, sample_text, cols, dst) != 0) {
            free(vectors);
            return build_fallback_vectors(cols, sample_count);
        }
    }

    if (used_activation_replay) {
        LOG_INFO("Built activation-replay calibration vectors: requested=%d available_samples=%d tensor=%s",
                 sample_count,
                 corpus->sample_count,
                 corpus->tensor_name ? corpus->tensor_name : "<unknown>");
    } else {
        LOG_INFO("Built tokenized calibration vectors: requested=%d available_samples=%d",
                 sample_count, corpus->sample_count);
    }
    if (out_sample_count) {
        *out_sample_count = sample_count;
    }
    return vectors;
}

static void quantize_row_ternary(const float *latent_row,
                                 uint32_t cols,
                                 float zero_threshold,
                                 float min_output_scale,
                                 float *out_scale,
                                 int8_t *out_ternary_row) {
    float max_abs = 0.0f;
    float threshold_scale = 0.0f;
    float scale = min_output_scale;
    float threshold = 0.0f;

    for (uint32_t c = 0; c < cols; ++c) {
        float magnitude = fabsf(latent_row[c]);
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }

    if (max_abs > 1e-12f) {
        threshold_scale = max_abs;
    }
    if (scale < threshold_scale) {
        scale = threshold_scale;
    }
    threshold = threshold_scale * zero_threshold;

    for (uint32_t c = 0; c < cols; ++c) {
        float value = latent_row[c];
        if (value > threshold) {
            out_ternary_row[c] = 1;
        } else if (value < -threshold) {
            out_ternary_row[c] = -1;
        } else {
            out_ternary_row[c] = 0;
        }
    }

    *out_scale = scale;
}

static float dot_row(const float *row, const float *vec, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
        acc += row[c] * vec[c];
    }
    return acc;
}

static float dot_row_ternary(const int8_t *row, float scale, const float *vec, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
        acc += ((float)row[c] * scale) * vec[c];
    }
    return acc;
}

static float ternary_regularizer_grad(float value, int8_t ternary_value) {
    if (ternary_value > 0) return value - 1.0f;
    if (ternary_value < 0) return value + 1.0f;
    return value;
}

static float ternary_non_collapse_push(float latent_value) {
    return (latent_value < 0.0f) ? -1.0f : 1.0f;
}

typedef struct {
    uint32_t cols;
    const float *calibration_vectors;
    const float *sample_weights;
    const float *hessian_proxy;
    int sample_count;
    float regularization_strength;
    float non_collapse_weight;
    float zero_occupancy_floor;
    float hessian_scale;
    float hessian_proxy_cap;
    float *mse_sum;
} ste_gradient_accum_context_t;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float clip_value;
    uint32_t step_index;
} ste_optimizer_context_t;

typedef struct {
    const float *latent;
    int8_t *ternary;
    float *scales;
    const float *scale_floor;
    uint32_t rows;
    uint32_t cols;
    float zero_threshold;
} ste_quantize_rows_request_t;

static void quantize_all_rows(const ste_quantize_rows_request_t *request)
{
    if (!request || !request->latent || !request->ternary || !request->scales) {
        return;
    }

    for (uint32_t row = 0; row < request->rows; ++row) {
        size_t row_base = (size_t)row * request->cols;

        quantize_row_ternary(request->latent + row_base,
                             request->cols,
                             request->zero_threshold,
                             request->scale_floor ? request->scale_floor[row] : 0.0f,
                             &request->scales[row],
                             request->ternary + row_base);
    }
}

static void ste_fill_unit_sample_weights(float *sample_weights, int sample_count)
{
    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        sample_weights[sample_idx] = 1.0f;
    }
}

static void quantize_activation_vectors_a8(float *vectors,
                                           int sample_count,
                                           uint32_t cols)
{
    if (!vectors || sample_count <= 0 || cols == 0u) {
        return;
    }

    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        float *row = vectors + (size_t)sample_idx * cols;
        float absmax = 0.0f;
        float scale = 0.0f;

        for (uint32_t col = 0; col < cols; ++col) {
            float magnitude = fabsf(row[col]);
            if (magnitude > absmax) {
                absmax = magnitude;
            }
        }
        if (absmax <= 1e-12f) {
            continue;
        }

        scale = absmax / STE_A8_QUANT_MAX;
        for (uint32_t col = 0; col < cols; ++col) {
            float quantized = roundf(row[col] / scale);

            if (quantized > STE_A8_QUANT_MAX) {
                quantized = STE_A8_QUANT_MAX;
            }
            if (quantized < -STE_A8_QUANT_MAX) {
                quantized = -STE_A8_QUANT_MAX;
            }
            row[col] = quantized * scale;
        }
    }
}

static void ste_accumulate_row_gradients(const float *latent_row,
                                         const int8_t *ternary_row,
                                         float scale,
                                         float *gradient_row,
                                         const ste_gradient_accum_context_t *context)
{
    float sample_diffs[16];
    float weight_sum = 0.0f;
    uint32_t cols = context->cols;
    int sample_count = context->sample_count;
    uint32_t zero_count = 0;
    float non_collapse_scale = 0.0f;

    if (sample_count > (int)(sizeof(sample_diffs) / sizeof(sample_diffs[0]))) {
        sample_count = (int)(sizeof(sample_diffs) / sizeof(sample_diffs[0]));
    }

    for (uint32_t c = 0; c < cols; ++c) {
        if (ternary_row[c] == 0) {
            ++zero_count;
        }
    }
    if (context->non_collapse_weight > 0.0f && cols > 0u) {
        float zero_fraction = (float)zero_count / (float)cols;
        if (zero_fraction > context->zero_occupancy_floor) {
            non_collapse_scale = context->non_collapse_weight *
                                 (zero_fraction - context->zero_occupancy_floor);
        }
    }

    for (int s = 0; s < sample_count; ++s) {
        float w = context->sample_weights ? context->sample_weights[s] : 1.0f;
        if (w < 0.0f) w = 0.0f;
        weight_sum += w;
    }
    if (weight_sum <= 1e-12f) {
        weight_sum = (float)sample_count;
    }

    for (int s = 0; s < sample_count; ++s) {
        const float *vec = context->calibration_vectors + (size_t)s * cols;
        float reference = dot_row(latent_row, vec, cols);
        float proxy = dot_row_ternary(ternary_row, scale, vec, cols);
        sample_diffs[s] = proxy - reference;
    }

    if (context->mse_sum) {
        float row_mse = 0.0f;

        for (int s = 0; s < sample_count; ++s) {
            float w = context->sample_weights ? context->sample_weights[s] : 1.0f;

            if (w < 0.0f) {
                w = 0.0f;
            }
            row_mse += w * sample_diffs[s] * sample_diffs[s];
        }

        *context->mse_sum += row_mse / weight_sum;
    }

    for (uint32_t c = 0; c < cols; ++c) {
        float grad = context->regularization_strength *
                     ternary_regularizer_grad(latent_row[c], ternary_row[c]);
        float curvature_scale = context->hessian_proxy ? context->hessian_proxy[c] : 1.0f;

        if (context->hessian_proxy_cap > 0.0f && curvature_scale > context->hessian_proxy_cap) {
            curvature_scale = context->hessian_proxy_cap;
        }
        curvature_scale *= context->hessian_scale;

        /* Anti-collapse: if a row quantizes to too many zeros, push zeroed
         * weights away from the dead-zone so STE keeps exploring +/-1 states. */
        if (non_collapse_scale > 0.0f && ternary_row[c] == 0) {
            grad -= non_collapse_scale * ternary_non_collapse_push(latent_row[c]);
        }

        for (int s = 0; s < sample_count; ++s) {
            const float *vec = context->calibration_vectors + (size_t)s * cols;
            float w = context->sample_weights ? context->sample_weights[s] : 1.0f;
            grad += curvature_scale * (2.0f * w / weight_sum) * sample_diffs[s] * vec[c];
        }

        gradient_row[c] = grad;
    }
}

static float compute_gradient_norm(const float *gradient, size_t weight_count)
{
    double sumsq = 0.0;

    if (!gradient || weight_count == 0u) {
        return 0.0f;
    }

    for (size_t weight_idx = 0; weight_idx < weight_count; ++weight_idx) {
        double grad = (double)gradient[weight_idx];

        sumsq += grad * grad;
    }

    return (float)sqrt(sumsq);
}

static float clip_tensor_gradients(float *gradient,
                                   size_t weight_count,
                                   float max_grad_norm,
                                   float *out_raw_grad_norm,
                                   float *out_clipped_grad_norm)
{
    float raw_grad_norm = compute_gradient_norm(gradient, weight_count);
    float clip_scale = 1.0f;

    if (out_raw_grad_norm) {
        *out_raw_grad_norm = raw_grad_norm;
    }
    if (raw_grad_norm > 0.0f && max_grad_norm > 0.0f && raw_grad_norm > max_grad_norm) {
        clip_scale = max_grad_norm / raw_grad_norm;
        for (size_t weight_idx = 0; weight_idx < weight_count; ++weight_idx) {
            gradient[weight_idx] *= clip_scale;
        }
    }
    if (out_clipped_grad_norm) {
        *out_clipped_grad_norm = raw_grad_norm * clip_scale;
    }

    return clip_scale;
}

static float apply_adam_updates(float *latent,
                                const float *gradient,
                                float *first_moment,
                                float *second_moment,
                                size_t weight_count,
                                const ste_optimizer_context_t *context)
{
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float beta1_correction = 0.0f;
    float beta2_correction = 0.0f;
    float inv_beta1_correction = 1.0f;
    float inv_beta2_correction = 1.0f;
    float one_minus_beta1 = 0.0f;
    float one_minus_beta2 = 0.0f;
    size_t saturated_count = 0u;

    if (!latent || !gradient || !first_moment || !second_moment || !context || weight_count == 0u) {
        return 0.0f;
    }

    beta1 = context->beta1;
    beta2 = context->beta2;
    beta1_correction = 1.0f - powf(beta1, (float)context->step_index);
    beta2_correction = 1.0f - powf(beta2, (float)context->step_index);
    inv_beta1_correction = (beta1_correction > 1e-12f) ? (1.0f / beta1_correction) : 1.0f;
    inv_beta2_correction = (beta2_correction > 1e-12f) ? (1.0f / beta2_correction) : 1.0f;
    one_minus_beta1 = 1.0f - beta1;
    one_minus_beta2 = 1.0f - beta2;

    for (size_t weight_idx = 0; weight_idx < weight_count; ++weight_idx) {
        float grad = gradient[weight_idx];
        float first = beta1 * first_moment[weight_idx] + one_minus_beta1 * grad;
        float second = beta2 * second_moment[weight_idx] + one_minus_beta2 * grad * grad;
        float first_hat = first * inv_beta1_correction;
        float second_hat = second * inv_beta2_correction;
        float denom = sqrtf(second_hat) + context->epsilon;

        first_moment[weight_idx] = first;
        second_moment[weight_idx] = second;
        latent[weight_idx] -= context->learning_rate * (first_hat / denom);
        if (latent[weight_idx] > context->clip_value) {
            latent[weight_idx] = context->clip_value;
        }
        if (latent[weight_idx] < -context->clip_value) {
            latent[weight_idx] = -context->clip_value;
        }
        if (fabsf(latent[weight_idx]) >= context->clip_value - 1e-6f) {
            ++saturated_count;
        }
    }

    return (float)saturated_count / (float)weight_count;
}

static tensor_t **resolve_tensor_slot(llm_model_t *model,
                                      const model_spec_t *spec,
                                      const char *tensor_name) {
    int layer_idx = -1;
    const char *field_name = NULL;
    char *endptr = NULL;

    if (!model || !spec || !spec->tensor_map || !tensor_name) {
        return NULL;
    }

    for (int i = 0; i < spec->tensor_map_size; ++i) {
        const tensor_map_entry_t *entry = &spec->tensor_map[i];
        if (!entry->hf_name || strcmp(entry->hf_name, tensor_name) != 0) {
            continue;
        }
        field_name = entry->field_name;
        if (entry->internal_key && strncmp(entry->internal_key, "blk.", 4) == 0) {
            layer_idx = (int)strtol(entry->internal_key + 4, &endptr, 10);
            if (endptr == entry->internal_key + 4) {
                layer_idx = -1;
            }
        }
        break;
    }

    if (!field_name) {
        return NULL;
    }
    if (layer_idx < 0) {
        if (strcmp(field_name, "embedding_weight") == 0) return &model->embedding_weight;
        if (strcmp(field_name, "lm_head_weight") == 0 || strcmp(field_name, "lm_head") == 0) return &model->lm_head_weight;
        return NULL;
    }

    if (strcmp(field_name, "q_proj_weight") == 0) return &model->layers[layer_idx].q_proj_weight;
    if (strcmp(field_name, "k_proj_weight") == 0) return &model->layers[layer_idx].k_proj_weight;
    if (strcmp(field_name, "v_proj_weight") == 0) return &model->layers[layer_idx].v_proj_weight;
    if (strcmp(field_name, "out_proj_weight") == 0) return &model->layers[layer_idx].out_proj_weight;
    if (strcmp(field_name, "gate_proj_weight") == 0) return &model->layers[layer_idx].gate_proj_weight;
    if (strcmp(field_name, "up_proj_weight") == 0) return &model->layers[layer_idx].up_proj_weight;
    if (strcmp(field_name, "down_proj_weight") == 0) return &model->layers[layer_idx].down_proj_weight;
    return NULL;
}

static tensor_t *build_proxy_tensor_from_ternary(const int8_t *ternary,
                                                 const float *scales,
                                                 uint32_t rows,
                                                 uint32_t cols) {
    const int shape[2] = { (int)rows, (int)cols };
    tensor_t *proxy = NULL;
    float *data = NULL;

    if (!ternary || !scales || rows == 0 || cols == 0) {
        return NULL;
    }

    proxy = tensor_create(2, shape, DTYPE_F32);
    if (!proxy) {
        return NULL;
    }
    data = tensor_data_f32(proxy);
    if (!data) {
        tensor_release(proxy);
        return NULL;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        float scale = scales[r];
        size_t row_base = (size_t)r * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            data[row_base + c] = (float)ternary[row_base + c] * scale;
        }
    }

    return proxy;
}

static int run_prompt_logits(inference_session_t *session,
                             sapphire_tokenizer_t *tokenizer,
                             const model_spec_t *spec,
                             const char *text,
                             float *out_logits,
                             int vocab_size) {
    int *tokens = NULL;
    int token_count = 0;
    const int max_tokens = 1024;

    if (!session || !tokenizer || !spec || !text || !out_logits || vocab_size <= 0) {
        return -1;
    }

    tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) {
        return -1;
    }

    token_count = build_gemma3_prompt(spec, text, tokens, max_tokens);
    if (token_count <= 0) {
        token_count = tokenize(tokenizer, text, tokens, max_tokens);
    }
    if (token_count <= 0) {
        free(tokens);
        return -1;
    }

    inference_session_reset(session);
    inference_forward_batch(session, tokens, 0, token_count, out_logits);
    free(tokens);
    return 0;
}

static float compute_logits_kl(const float *reference_logits,
                               const float *proxy_logits,
                               float *reference_probs,
                               float *proxy_probs,
                               int vocab_size,
                               float temperature) {
    float kl = 0.0f;
    float effective_temperature = (temperature > 0.0f) ? temperature : 1.0f;
    float inv_temperature = 1.0f / effective_temperature;

    for (int i = 0; i < vocab_size; ++i) {
        reference_probs[i] = reference_logits[i] * inv_temperature;
        proxy_probs[i] = proxy_logits[i] * inv_temperature;
    }
    softmax(reference_probs, vocab_size);
    softmax(proxy_probs, vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        float p = reference_probs[i];
        float q = proxy_probs[i];
        if (p > 1e-12f && q > 1e-12f) {
            kl += p * logf(p / q);
        }
    }

    if (effective_temperature != 1.0f) {
        kl *= effective_temperature * effective_temperature;
    }

    return kl;
}

typedef struct {
    const int8_t *ternary;
    const float *scales;
    uint32_t rows;
    uint32_t cols;
    const transformer_ste_config_t *config;
    const ternary_calibration_corpus_t *corpus;
    float *out_sample_weights;
    int sample_count;
} distillation_weight_request_t;

typedef struct {
    const ternary_calibration_corpus_t *corpus;
    tensor_t **slot;
    tensor_t *original_tensor;
    tensor_t *proxy_tensor;
    const gemma3_270m_config_t *model_config;
    float *reference_logits;
    float *proxy_logits;
    float *reference_probs;
    float *proxy_probs;
    float kl_weight;
    float kl_temperature;
} distillation_runtime_t;

static float compute_sample_distillation_weight(const distillation_runtime_t *runtime,
                                                const char *sample_text) {
    float kl = 0.0f;

    if (!runtime || !sample_text) {
        return 1.0f;
    }

    *runtime->slot = runtime->original_tensor;
    if (run_prompt_logits(runtime->corpus->session,
                          runtime->corpus->tokenizer,
                          runtime->corpus->model_spec,
                          sample_text,
                          runtime->reference_logits,
                          runtime->model_config->vocab_size) != 0) {
        return 1.0f;
    }

    *runtime->slot = runtime->proxy_tensor;
    if (run_prompt_logits(runtime->corpus->session,
                          runtime->corpus->tokenizer,
                          runtime->corpus->model_spec,
                          sample_text,
                          runtime->proxy_logits,
                          runtime->model_config->vocab_size) != 0) {
        *runtime->slot = runtime->original_tensor;
        return 1.0f;
    }
    *runtime->slot = runtime->original_tensor;

    kl = compute_logits_kl(runtime->reference_logits,
                           runtime->proxy_logits,
                           runtime->reference_probs,
                           runtime->proxy_probs,
                           runtime->model_config->vocab_size,
                           runtime->kl_temperature);
    if (kl < 0.0f) kl = 0.0f;
    if (kl > 8.0f) kl = 8.0f;
    return 1.0f + (runtime->kl_weight * kl);
}

static void compute_distillation_sample_weights(const distillation_weight_request_t *request) {
    distillation_runtime_t runtime;
    llm_model_t *model = NULL;
    tensor_t **slot = NULL;
    tensor_t *original_tensor = NULL;
    tensor_t *proxy_tensor = NULL;
    const gemma3_270m_config_t *model_config = NULL;
    float *reference_logits = NULL;
    float *proxy_logits = NULL;
    float *reference_probs = NULL;
    float *proxy_probs = NULL;
    const transformer_ste_config_t *config = request ? request->config : NULL;
    const ternary_calibration_corpus_t *corpus = request ? request->corpus : NULL;
    float *out_sample_weights = request ? request->out_sample_weights : NULL;
    int sample_count = request ? request->sample_count : 0;

    for (int s = 0; s < sample_count; ++s) {
        out_sample_weights[s] = 1.0f;
    }

    if (!config || config->kl_weight <= 0.0f || !corpus || !corpus->session ||
        !corpus->tokenizer || !corpus->model_spec || !corpus->tensor_name ||
        !corpus->sample_texts || corpus->sample_count <= 0 || sample_count <= 0) {
        return;
    }

    model = (llm_model_t *)corpus->model_spec->llm_model;
    model_config = (gemma3_270m_config_t *)corpus->model_spec->variant_config;
    if (!model || !model_config) {
        return;
    }

    slot = resolve_tensor_slot(model, corpus->model_spec, corpus->tensor_name);
    if (!slot || !*slot) {
        return;
    }
    original_tensor = *slot;
    proxy_tensor = build_proxy_tensor_from_ternary(request->ternary,
                                                   request->scales,
                                                   request->rows,
                                                   request->cols);
    if (!proxy_tensor) {
        return;
    }

    reference_logits = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    proxy_logits = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    reference_probs = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    proxy_probs = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    if (!reference_logits || !proxy_logits || !reference_probs || !proxy_probs) {
        goto cleanup;
    }

    memset(&runtime, 0, sizeof(runtime));
    runtime.corpus = corpus;
    runtime.slot = slot;
    runtime.original_tensor = original_tensor;
    runtime.proxy_tensor = proxy_tensor;
    runtime.model_config = model_config;
    runtime.reference_logits = reference_logits;
    runtime.proxy_logits = proxy_logits;
    runtime.reference_probs = reference_probs;
    runtime.proxy_probs = proxy_probs;
    runtime.kl_weight = config->kl_weight;
    runtime.kl_temperature = config->kl_temperature;

    for (int s = 0; s < sample_count; ++s) {
        const char *sample_text = corpus->sample_texts[s % corpus->sample_count];
        out_sample_weights[s] = compute_sample_distillation_weight(&runtime, sample_text);
    }

    {
        float avg_weight = 0.0f;
        for (int s = 0; s < sample_count; ++s) {
            avg_weight += out_sample_weights[s];
        }
        avg_weight /= (float)sample_count;
        LOG_INFO("Computed KL distillation weights: tensor=%s samples=%d temp=%.2f avg_weight=%.4f",
                 corpus->tensor_name,
                 sample_count,
                 (double)config->kl_temperature,
                 (double)avg_weight);
    }

cleanup:
    if (slot) {
        *slot = original_tensor;
    }
    free(reference_logits);
    free(proxy_logits);
    free(reference_probs);
    free(proxy_probs);
    tensor_release(proxy_tensor);
}

typedef struct {
    float *latent;
    float *first_moment;
    float *second_moment;
    float *gradient;
    int8_t *ternary;
    float *scales;
    const float *scale_floor;
    uint32_t rows;
    uint32_t cols;
    const transformer_ste_config_t *config;
    const float *calibration_vectors;
    const ternary_calibration_corpus_t *corpus;
    const float *hessian_proxy;
    ternary_hessian_proxy_stats_t hessian_proxy_stats;
    ternary_hessian_proxy_source_t hessian_proxy_source;
    ste_layer_schedule_t layer_schedule;
} ste_calibration_context_t;

typedef struct {
    float mse_sum;
    float raw_grad_norm;
    float clipped_grad_norm;
    float clip_scale;
    float latent_saturation;
} ste_step_metrics_t;

typedef struct {
    const transformer_ste_config_t *config;
    const ternary_activation_tape_context_t *tape_context;
    const ternary_hessian_sidecar_t *sidecar;
    const float *calibration_vectors;
    int sample_count;
    uint32_t cols;
} hessian_proxy_request_t;

typedef struct {
    float *owned_proxy;
    const float *proxy;
    ternary_hessian_proxy_stats_t stats;
    ternary_hessian_proxy_source_t source;
} hessian_proxy_result_t;

static void ste_build_step_schedule(const ste_calibration_context_t *context,
                                    uint32_t step_index,
                                    ste_step_schedule_t *out_schedule)
{
    if (!out_schedule) {
        return;
    }

    memset(out_schedule, 0, sizeof(*out_schedule));
    if (!context || !context->config) {
        return;
    }

    out_schedule->learning_rate = context->config->learning_rate;
    out_schedule->regularization_strength = context->config->regularization_strength;
    out_schedule->non_collapse_weight = context->config->non_collapse_weight;
    out_schedule->hessian_scale = 1.0f;
    if (context->hessian_proxy_source == TERNARY_HESSIAN_PROXY_SOURCE_EXTERNAL_SIDECAR) {
        out_schedule->hessian_proxy_cap = STE_EXTERNAL_PROXY_CAP;
    }
    out_schedule->learning_rate *= context->layer_schedule.layer_step_multiplier;

    if (!context->layer_schedule.early_layer_warmup ||
        step_index > context->layer_schedule.warmup_steps) {
        return;
    }

    out_schedule->regularization_strength *= context->layer_schedule.regularization_scale;
    out_schedule->non_collapse_weight *= context->layer_schedule.non_collapse_scale;
    out_schedule->hessian_scale = context->layer_schedule.hessian_scale;
    if (out_schedule->hessian_proxy_cap <= 0.0f ||
        out_schedule->hessian_proxy_cap > context->layer_schedule.hessian_proxy_cap) {
        out_schedule->hessian_proxy_cap = context->layer_schedule.hessian_proxy_cap;
    }
}

static ste_step_metrics_t ste_calibrate_with_samples(const ste_calibration_context_t *context,
                                                     uint32_t step_index,
                                                     ste_step_schedule_t *out_step_schedule)
{
    float sample_weights[16];
    ste_gradient_accum_context_t gradient_context;
    ste_optimizer_context_t optimizer_context;
    distillation_weight_request_t distillation_request;
    ste_step_metrics_t metrics;
    ste_step_schedule_t step_schedule;
    size_t weight_count = 0u;
    int sample_count = 0;
    float mse_sum = 0.0f;

    memset(&metrics, 0, sizeof(metrics));
    metrics.clip_scale = 1.0f;

    if (!context || !context->config || !context->gradient || !context->first_moment || !context->second_moment) {
        return metrics;
    }

    sample_count = context->config->calibration_samples;
    if (sample_count > (int)(sizeof(sample_weights) / sizeof(sample_weights[0]))) {
        sample_count = (int)(sizeof(sample_weights) / sizeof(sample_weights[0]));
    }
    weight_count = (size_t)context->rows * context->cols;
    ste_fill_unit_sample_weights(sample_weights, sample_count);
    ste_build_step_schedule(context, step_index, &step_schedule);
    if (out_step_schedule) {
        *out_step_schedule = step_schedule;
    }

    quantize_all_rows(&(ste_quantize_rows_request_t){
        .latent = context->latent,
        .ternary = context->ternary,
        .scales = context->scales,
        .scale_floor = context->scale_floor,
        .rows = context->rows,
        .cols = context->cols,
        .zero_threshold = context->config->zero_threshold
    });

    if (context->config->kl_weight > 0.0f) {
        memset(&distillation_request, 0, sizeof(distillation_request));
        distillation_request.ternary = context->ternary;
        distillation_request.scales = context->scales;
        distillation_request.rows = context->rows;
        distillation_request.cols = context->cols;
        distillation_request.config = context->config;
        distillation_request.corpus = context->corpus;
        distillation_request.out_sample_weights = sample_weights;
        distillation_request.sample_count = sample_count;
        compute_distillation_sample_weights(&distillation_request);
    }

    memset(context->gradient, 0, weight_count * sizeof(float));

    memset(&gradient_context, 0, sizeof(gradient_context));
    gradient_context.cols = context->cols;
    gradient_context.calibration_vectors = context->calibration_vectors;
    gradient_context.sample_weights = sample_weights;
    gradient_context.hessian_proxy = context->hessian_proxy;
    gradient_context.sample_count = sample_count;
    gradient_context.regularization_strength = step_schedule.regularization_strength;
    gradient_context.non_collapse_weight = step_schedule.non_collapse_weight;
    gradient_context.zero_occupancy_floor = context->config->zero_occupancy_floor;
    gradient_context.hessian_scale = step_schedule.hessian_scale;
    gradient_context.hessian_proxy_cap = step_schedule.hessian_proxy_cap;
    gradient_context.mse_sum = &mse_sum;

    for (uint32_t r = 0; r < context->rows; ++r) {
        size_t row_base = (size_t)r * context->cols;
        ste_accumulate_row_gradients(context->latent + row_base,
                                     context->ternary + row_base,
                                     context->scales[r],
                                     context->gradient + row_base,
                                     &gradient_context);
    }

    metrics.clip_scale = clip_tensor_gradients(context->gradient,
                                               weight_count,
                                               context->config->max_grad_norm,
                                               &metrics.raw_grad_norm,
                                               &metrics.clipped_grad_norm);

    memset(&optimizer_context, 0, sizeof(optimizer_context));
    optimizer_context.learning_rate = step_schedule.learning_rate;
    optimizer_context.beta1 = context->config->momentum;
    optimizer_context.beta2 = context->config->adam_beta2;
    optimizer_context.epsilon = context->config->adam_epsilon;
    optimizer_context.clip_value = context->config->clip_value;
    optimizer_context.step_index = step_index;

    metrics.latent_saturation = apply_adam_updates(context->latent,
                                                   context->gradient,
                                                   context->first_moment,
                                                   context->second_moment,
                                                   weight_count,
                                                   &optimizer_context);

    quantize_all_rows(&(ste_quantize_rows_request_t){
        .latent = context->latent,
        .ternary = context->ternary,
        .scales = context->scales,
        .scale_floor = context->scale_floor,
        .rows = context->rows,
        .cols = context->cols,
        .zero_threshold = context->config->zero_threshold
    });

    metrics.mse_sum = mse_sum;
    return metrics;
}

static void hessian_proxy_result_reset(hessian_proxy_result_t *result)
{
    if (!result) {
        return;
    }

    memset(result, 0, sizeof(*result));
    result->source = TERNARY_HESSIAN_PROXY_SOURCE_NONE;
}

static int hessian_proxy_request_is_valid(const hessian_proxy_request_t *request)
{
    return request && request->config && request->config->use_hessian_proxy &&
           request->calibration_vectors && request->sample_count > 0 && request->cols > 0u;
}

static int hessian_proxy_cache_matches(const ternary_hessian_proxy_cache_t *cache,
                                       int tape_entry_idx,
                                       int sample_count,
                                       uint32_t cols)
{
    return cache && cache->valid && cache->diagonal && cache->tape_entry_idx == tape_entry_idx &&
           cache->sample_count == sample_count && cache->cols == cols;
}

static void hessian_proxy_publish_cache_hit(const ternary_hessian_proxy_cache_t *cache,
                                            hessian_proxy_result_t *result)
{
    if (!cache || !result) {
        return;
    }

    result->proxy = cache->diagonal;
    result->stats = cache->stats;
    result->source = TERNARY_HESSIAN_PROXY_SOURCE_ACTIVATION_DIAGONAL;
}

static void hessian_proxy_invalidate_cache(ternary_hessian_proxy_cache_t *cache);

static int hessian_proxy_publish_sidecar_hit(const hessian_proxy_request_t *request,
                                             hessian_proxy_result_t *result)
{
    const ternary_hessian_sidecar_entry_t *entry = NULL;
    const float *diagonal = NULL;

    if (!request || !result || !request->sidecar || !request->tape_context ||
        !request->tape_context->tape || !request->tape_context->tensor_name) {
        return -1;
    }

    if (activation_tape_crc32(request->tape_context->tape) != ternary_hessian_sidecar_tape_crc32(request->sidecar)) {
        LOG_ERROR("hessian proxy: sidecar tape provenance mismatch for %s", request->tape_context->tensor_name);
        return -1;
    }

    entry = ternary_hessian_sidecar_entry(request->sidecar, request->tape_context->tensor_name);
    if (!entry) {
        LOG_ERROR("hessian proxy: sidecar missing tensor %s", request->tape_context->tensor_name);
        return -1;
    }
    if (entry->vector_dim != request->cols) {
        LOG_ERROR("hessian proxy: sidecar dimension mismatch for %s (sidecar=%u expected=%u)",
                  request->tape_context->tensor_name,
                  entry->vector_dim,
                  request->cols);
        return -1;
    }
    if (entry->sample_count < (uint32_t)request->sample_count) {
        LOG_ERROR("hessian proxy: sidecar sample count is too small for %s (sidecar=%u required=%d)",
                  request->tape_context->tensor_name,
                  entry->sample_count,
                  request->sample_count);
        return -1;
    }
    if (entry->sample_count != (uint32_t)request->sample_count) {
        LOG_INFO("hessian proxy: reusing sidecar diagonal for %s with subset calibration (sidecar=%u calibration=%d)",
                 request->tape_context->tensor_name,
                 entry->sample_count,
                 request->sample_count);
    }

    diagonal = ternary_hessian_sidecar_diagonal(request->sidecar, request->tape_context->tensor_name);
    if (!diagonal) {
        LOG_ERROR("hessian proxy: failed to resolve sidecar diagonal for %s", request->tape_context->tensor_name);
        return -1;
    }
    if (ternary_hessian_sidecar_stats(request->sidecar,
                                      request->tape_context->tensor_name,
                                      &result->stats) != 0) {
        LOG_ERROR("hessian proxy: failed to read sidecar stats for %s", request->tape_context->tensor_name);
        return -1;
    }

    result->proxy = diagonal;
    result->source = TERNARY_HESSIAN_PROXY_SOURCE_EXTERNAL_SIDECAR;
    LOG_INFO("Built external Hessian sidecar proxy: tensor=%s samples=%d mean=%.4f max=%.4f",
             request->tape_context->tensor_name,
             request->sample_count,
             (double)result->stats.mean,
             (double)result->stats.max);
    return 0;
}

static int hessian_proxy_acquire_buffer(ternary_hessian_proxy_cache_t *cache,
                                        uint32_t cols,
                                        float **out_proxy_buffer,
                                        float **out_owned_proxy)
{
    if (!out_proxy_buffer || !out_owned_proxy) {
        return -1;
    }

    *out_proxy_buffer = NULL;
    *out_owned_proxy = NULL;

    if (cache) {
        float *resized = (float *)realloc(cache->diagonal, (size_t)cols * sizeof(float));

        if (!resized) {
            return -1;
        }
        cache->diagonal = resized;
        *out_proxy_buffer = cache->diagonal;
        return 0;
    }

    *out_owned_proxy = (float *)malloc((size_t)cols * sizeof(float));
    if (!*out_owned_proxy) {
        return -1;
    }

    *out_proxy_buffer = *out_owned_proxy;
    return 0;
}

static void hessian_proxy_invalidate_cache(ternary_hessian_proxy_cache_t *cache)
{
    if (cache) {
        cache->valid = 0;
    }
}

static void hessian_proxy_store_cache(ternary_hessian_proxy_cache_t *cache,
                                      int tape_entry_idx,
                                      int sample_count,
                                      uint32_t cols,
                                      const ternary_hessian_proxy_stats_t *stats)
{
    if (!cache || !stats) {
        return;
    }

    cache->valid = 1;
    cache->tape_entry_idx = tape_entry_idx;
    cache->sample_count = sample_count;
    cache->cols = cols;
    cache->stats = *stats;
}

static int prepare_hessian_proxy(const hessian_proxy_request_t *request,
                                 hessian_proxy_result_t *result)
{
    ternary_hessian_proxy_build_request_t build_request;
    ternary_hessian_proxy_cache_t *cache = NULL;
    float *proxy_buffer = NULL;
    int tape_entry_idx = -1;
    const char *tensor_name = NULL;

    if (!result) {
        return -1;
    }

    hessian_proxy_result_reset(result);
    if (!hessian_proxy_request_is_valid(request)) {
        return 0;
    }

    if (request->sidecar) {
        if (hessian_proxy_publish_sidecar_hit(request, result) != 0) {
            hessian_proxy_invalidate_cache(request->tape_context ? request->tape_context->proxy_cache : NULL);
            return -1;
        }
        return 0;
    }

    tensor_name = (request->tape_context && request->tape_context->tensor_name)
        ? request->tape_context->tensor_name
        : "<calibration-vectors>";

    if (request->tape_context && request->tape_context->tape && request->tape_context->tensor_name) {
        tape_entry_idx = activation_tape_entry_index(request->tape_context->tape,
                                                     request->tape_context->tensor_name);
        if (tape_entry_idx < 0) {
            LOG_WARN("hessian proxy: failed to resolve tape entry for %s", request->tape_context->tensor_name);
        }
    }

    cache = request->tape_context ? request->tape_context->proxy_cache : NULL;
    if (cache && tape_entry_idx >= 0 && hessian_proxy_cache_matches(cache, tape_entry_idx, request->sample_count, request->cols)) {
        hessian_proxy_publish_cache_hit(cache, result);
        return 0;
    }

    if (hessian_proxy_acquire_buffer(cache, request->cols, &proxy_buffer, &result->owned_proxy) != 0) {
        LOG_WARN("hessian proxy: allocation failed for %s", tensor_name);
        hessian_proxy_invalidate_cache(cache);
        return 0;
    }

    memset(&build_request, 0, sizeof(build_request));
    build_request.calibration_vectors = request->calibration_vectors;
    build_request.sample_count = request->sample_count;
    build_request.cols = request->cols;
    build_request.floor = request->config->hessian_proxy_floor;
    build_request.strength = request->config->hessian_proxy_strength;

    if (ternary_hessian_proxy_build_diagonal(&build_request, proxy_buffer, &result->stats) != 0) {
        if (result->owned_proxy) {
            free(result->owned_proxy);
            result->owned_proxy = NULL;
        }
        hessian_proxy_invalidate_cache(cache);
        return 0;
    }

    if (cache) {
        hessian_proxy_store_cache(cache,
                                  tape_entry_idx,
                                  request->sample_count,
                                  request->cols,
                                  &result->stats);
    }

    result->proxy = proxy_buffer;
    result->source = TERNARY_HESSIAN_PROXY_SOURCE_ACTIVATION_DIAGONAL;

    LOG_INFO("Built activation-diagonal Hessian proxy: tensor=%s entry=%d samples=%d mean=%.4f max=%.4f",
             tensor_name,
             tape_entry_idx,
             request->sample_count,
             (double)result->stats.mean,
             (double)result->stats.max);

    return 0;
}

static int pack_ternary_2bit(const int8_t *ternary,
                             uint32_t rows,
                             uint32_t cols,
                             uint8_t *out_packed,
                             size_t packed_bytes) {
    size_t expected_bytes = ternary_packed_bytes(rows, cols);
    size_t out_idx = 0;

    if (!ternary || !out_packed || packed_bytes != expected_bytes) {
        LOG_ERROR("pack_ternary_2bit: invalid arguments");
        return -1;
    }

    memset(out_packed, 0, packed_bytes);
    for (uint32_t r = 0; r < rows; ++r) {
        size_t row_base = (size_t)r * cols;
        for (uint32_t c = 0; c < cols; c += TERNARY_PACK_WIDTH) {
            uint8_t packed = 0;
            for (uint32_t lane = 0; lane < TERNARY_PACK_WIDTH; ++lane) {
                uint32_t idx = c + lane;
                uint8_t symbol = 0;
                if (idx < cols) {
                    symbol = ternary_symbol_from_value(ternary[row_base + idx]);
                }
                packed |= (uint8_t)(symbol << (lane * 2u));
            }
            out_packed[out_idx++] = packed;
        }
    }

    return 0;
}

static int run_ste_calibration_steps(const ste_calibration_context_t *context,
                                     ternary_calibration_result_t *out_result,
                                     ste_telemetry_runtime_t *telemetry_runtime)
{
    size_t packed_bytes = 0u;

    if (!context || !out_result) {
        return -1;
    }

    packed_bytes = ternary_packed_bytes(context->rows, context->cols);
    for (int step = 0; step < context->config->ste_steps; ++step) {
        ste_step_metrics_t step_metrics;
        ste_step_schedule_t step_schedule;
        struct timespec step_start;
        struct timespec step_end;
        ste_telemetry_step_t telemetry_step;

        memset(&step_schedule, 0, sizeof(step_schedule));

        if (telemetry_runtime && telemetry_runtime->enabled) {
            (void)clock_gettime(CLOCK_MONOTONIC, &step_start);
        }

        step_metrics = ste_calibrate_with_samples(context,
                              (uint32_t)(step + 1),
                              &step_schedule);

        if (telemetry_runtime && telemetry_runtime->enabled) {
            (void)clock_gettime(CLOCK_MONOTONIC, &step_end);
            memset(&telemetry_step, 0, sizeof(telemetry_step));
            telemetry_step.step_idx = (uint32_t)step;
            telemetry_step.total_steps = (uint32_t)context->config->ste_steps;
            telemetry_step.mse_loss = (context->rows > 0u) ? (step_metrics.mse_sum / (float)context->rows) : 0.0f;
            telemetry_step.raw_grad_norm = step_metrics.raw_grad_norm;
            telemetry_step.clipped_grad_norm = step_metrics.clipped_grad_norm;
            telemetry_step.clip_scale = step_metrics.clip_scale;
            telemetry_step.latent_saturation = step_metrics.latent_saturation;
            telemetry_step.compute_ms = telemetry_diff_ms(&step_start, &step_end);
            telemetry_step.effective_learning_rate = step_schedule.learning_rate;
            telemetry_step.effective_hessian_scale = step_schedule.hessian_scale;

            if (pack_ternary_2bit(out_result->ternary_weights,
                                  context->rows,
                                  context->cols,
                                  out_result->packed_weights,
                                  packed_bytes) == 0) {
                if (ste_emit_telemetry_step(telemetry_runtime,
                                            out_result,
                                            context->scale_floor,
                                            &telemetry_step) != 0) {
                    LOG_WARN("telemetry: failed to dump step %d", step);
                }
            } else {
                LOG_WARN("telemetry: failed to pack step snapshot");
            }
        }
    }

    return 0;
}

typedef struct {
    float *first_moment;
    float *second_moment;
    float *gradient;
    float *calibration_vectors;
    float *scale_floor;
    float teacher_abs_mean;
    int actual_sample_count;
    hessian_proxy_result_t hessian_proxy;
    ste_layer_schedule_t layer_schedule;
} ste_calibration_workspace_t;

typedef struct {
    const transformer_ste_config_t *config;
    const ternary_calibration_corpus_t *corpus;
    const ternary_activation_tape_context_t *tape_context;
    const ternary_hessian_sidecar_t *sidecar;
    const uint16_t *bf16_weights;
    size_t weight_count;
    uint32_t rows;
    uint32_t cols;
    const ste_layer_schedule_t *layer_schedule;
} ste_calibration_workspace_request_t;

typedef struct {
    ternary_calibration_result_t *result;
    uint32_t rows;
    uint32_t cols;
    const transformer_ste_config_t *config;
    const ternary_calibration_corpus_t *corpus;
    const ste_calibration_workspace_t *workspace;
} ste_calibration_context_request_t;

typedef struct {
    const transformer_ste_config_t *config;
    const char *tensor_name;
    uint32_t rows;
    uint32_t cols;
    const ternary_activation_tape_context_t *tape_context;
    const ste_calibration_workspace_t *workspace;
} ste_telemetry_request_t;

typedef struct {
    const uint16_t *bf16_weights;
    const float *calibration_vectors;
    int sample_count;
    int use_teacher_abs_mean;
    uint32_t rows;
    uint32_t cols;
    const ste_layer_schedule_t *layer_schedule;
    float *out_scale_floor;
    float *out_teacher_abs_mean;
} ste_scale_floor_request_t;

static const char *resolve_calibration_tensor_name(const ternary_calibration_corpus_t *corpus,
                                                   const ternary_activation_tape_context_t *tape_context);

static void ste_calibration_workspace_reset(ste_calibration_workspace_t *workspace)
{
    if (!workspace) {
        return;
    }

    memset(workspace, 0, sizeof(*workspace));
}

static void ste_calibration_workspace_release(ste_calibration_workspace_t *workspace)
{
    if (!workspace) {
        return;
    }

    free(workspace->first_moment);
    free(workspace->second_moment);
    free(workspace->gradient);
    free(workspace->calibration_vectors);
    free(workspace->scale_floor);
    free(workspace->hessian_proxy.owned_proxy);
    ste_calibration_workspace_reset(workspace);
}

static int ste_prepare_output_buffers(ternary_calibration_result_t *out_result,
                                      size_t weight_count,
                                      uint32_t rows,
                                      uint32_t cols)
{
    size_t packed_bytes = ternary_packed_bytes(rows, cols);

    if (!out_result) {
        return -1;
    }

    memset(out_result, 0, sizeof(*out_result));
    out_result->latent_weights = (float *)malloc(weight_count * sizeof(float));
    out_result->ternary_weights = (int8_t *)malloc(weight_count * sizeof(int8_t));
    out_result->packed_weights = (uint8_t *)malloc(packed_bytes);
    out_result->scales = (float *)malloc((size_t)rows * sizeof(float));
    if (!out_result->latent_weights || !out_result->ternary_weights ||
        !out_result->packed_weights || !out_result->scales) {
        transformer_free_ternary_calibration_result(out_result);
        return -1;
    }

    out_result->weight_count = weight_count;
    out_result->packed_weight_bytes = packed_bytes;
    out_result->rows = rows;
    out_result->cols = cols;
    return 0;
}

static int ste_build_scale_floor(const ste_scale_floor_request_t *request)
{
    float teacher_abs_mean = 0.0f;

    if (!request || !request->out_scale_floor || request->rows == 0u || request->cols == 0u) {
        return -1;
    }

    memset(request->out_scale_floor, 0, (size_t)request->rows * sizeof(float));
    if (request->out_teacher_abs_mean) {
        *request->out_teacher_abs_mean = 0.0f;
    }
    if (!request->bf16_weights || !request->layer_schedule || !request->layer_schedule->early_layer_warmup) {
        return 0;
    }

    if (request->use_teacher_abs_mean) {
        teacher_abs_mean = ste_compute_teacher_abs_mean(request->calibration_vectors,
                                                        request->sample_count,
                                                        request->cols);
        if (request->out_teacher_abs_mean) {
            *request->out_teacher_abs_mean = teacher_abs_mean;
        }
    }

    for (uint32_t row = 0; row < request->rows; ++row) {
        float scale_floor = teacher_abs_mean;

        if (scale_floor <= 1e-12f) {
            double abs_mean = 0.0;
            size_t row_base = (size_t)row * request->cols;

            for (uint32_t col = 0; col < request->cols; ++col) {
                abs_mean += fabs((double)bf16_to_f32_scalar(request->bf16_weights[row_base + col]));
            }

            abs_mean /= (double)request->cols;
            scale_floor = (float)abs_mean;
        }

        if (scale_floor < request->layer_schedule->min_gamma_scale) {
            scale_floor = request->layer_schedule->min_gamma_scale;
        }
        request->out_scale_floor[row] = scale_floor;
    }

    return 0;
}

static int ste_prepare_workspace(const ste_calibration_workspace_request_t *request,
                                 ste_calibration_workspace_t *workspace)
{
    hessian_proxy_request_t proxy_request;

    if (!request || !request->config || !workspace) {
        return -1;
    }

    ste_calibration_workspace_reset(workspace);
    workspace->first_moment = (float *)calloc(request->weight_count, sizeof(float));
    workspace->second_moment = (float *)calloc(request->weight_count, sizeof(float));
    workspace->gradient = (float *)malloc(request->weight_count * sizeof(float));
    workspace->actual_sample_count = request->config->calibration_samples;
    if (request->layer_schedule) {
        workspace->layer_schedule = *request->layer_schedule;
    } else {
        ste_layer_schedule_reset(&workspace->layer_schedule);
    }
    workspace->calibration_vectors = build_calibration_vectors(request->cols,
                                                               request->config->calibration_samples,
                                                               request->corpus,
                                                               request->tape_context,
                                                               &workspace->actual_sample_count);
    if (workspace->layer_schedule.early_layer_warmup) {
        workspace->scale_floor = (float *)malloc((size_t)request->rows * sizeof(float));
    }
    if (!workspace->first_moment || !workspace->second_moment || !workspace->gradient ||
        !workspace->calibration_vectors ||
        (workspace->layer_schedule.early_layer_warmup && !workspace->scale_floor)) {
        ste_calibration_workspace_release(workspace);
        return -1;
    }
    if (workspace->scale_floor &&
        ste_build_scale_floor(&(ste_scale_floor_request_t){
            .bf16_weights = request->bf16_weights,
            .calibration_vectors = workspace->calibration_vectors,
            .sample_count = workspace->actual_sample_count,
            .use_teacher_abs_mean = request->tape_context && request->tape_context->tape,
            .rows = request->rows,
            .cols = request->cols,
            .layer_schedule = &workspace->layer_schedule,
            .out_scale_floor = workspace->scale_floor,
            .out_teacher_abs_mean = &workspace->teacher_abs_mean
        }) != 0) {
        ste_calibration_workspace_release(workspace);
        return -1;
    }
    if (workspace->teacher_abs_mean > 0.0f && request->tape_context && request->tape_context->tensor_name) {
        LOG_INFO("Early-layer gamma warm start: tensor=%s teacher_abs_mean=%.4f gamma_floor=%.4f",
                 request->tape_context->tensor_name,
                 (double)workspace->teacher_abs_mean,
                 (double)workspace->layer_schedule.min_gamma_scale);
    }

    memset(&proxy_request, 0, sizeof(proxy_request));
    proxy_request.config = request->config;
    proxy_request.tape_context = request->tape_context;
    proxy_request.sidecar = request->sidecar;
    proxy_request.calibration_vectors = workspace->calibration_vectors;
    proxy_request.sample_count = workspace->actual_sample_count;
    proxy_request.cols = request->cols;
    if (prepare_hessian_proxy(&proxy_request, &workspace->hessian_proxy) != 0) {
        ste_calibration_workspace_release(workspace);
        return -1;
    }
    if (request->config->simulate_activation_a8) {
        quantize_activation_vectors_a8(workspace->calibration_vectors,
                                       workspace->actual_sample_count,
                                       request->cols);
        LOG_INFO("Activation A8 simulation enabled: tensor=%s samples=%d cols=%u",
                 resolve_calibration_tensor_name(request->corpus, request->tape_context),
                 workspace->actual_sample_count,
                 request->cols);
    }
    return 0;
}

static const char *resolve_calibration_tensor_name(const ternary_calibration_corpus_t *corpus,
                                                   const ternary_activation_tape_context_t *tape_context)
{
    if (corpus && corpus->tensor_name && corpus->tensor_name[0] != '\0') {
        return corpus->tensor_name;
    }

    return tape_context ? tape_context->tensor_name : NULL;
}

static void ste_init_calibration_context(ste_calibration_context_t *context,
                                         const ste_calibration_context_request_t *request)
{
    if (!context || !request || !request->result || !request->config || !request->workspace) {
        return;
    }

    memset(context, 0, sizeof(*context));
    context->latent = request->result->latent_weights;
    context->first_moment = request->workspace->first_moment;
    context->second_moment = request->workspace->second_moment;
    context->gradient = request->workspace->gradient;
    context->ternary = request->result->ternary_weights;
    context->scales = request->result->scales;
    context->scale_floor = request->workspace->scale_floor;
    context->rows = request->rows;
    context->cols = request->cols;
    context->config = request->config;
    context->calibration_vectors = request->workspace->calibration_vectors;
    context->corpus = request->corpus;
    context->hessian_proxy = request->workspace->hessian_proxy.proxy;
    context->hessian_proxy_stats = request->workspace->hessian_proxy.stats;
    context->hessian_proxy_source = request->workspace->hessian_proxy.source;
    context->layer_schedule = request->workspace->layer_schedule;
}

static int ste_prepare_telemetry_runtime(ste_telemetry_runtime_t *runtime,
                                         const ste_telemetry_request_t *request)
{
    int telemetry_active = 0;

    if (!runtime || !request || !request->config) {
        return 0;
    }

    memset(runtime, 0, sizeof(*runtime));
    telemetry_active = ste_telemetry_runtime_init(runtime,
                                                  request->config,
                                                  request->tensor_name,
                                                  request->rows,
                                                  request->cols);
    if (telemetry_active && request->tensor_name) {
        uint32_t parsed_layer_index = 0u;

        if (telemetry_try_parse_layer_index(request->tensor_name, &parsed_layer_index)) {
            runtime->telemetry.layer_idx = parsed_layer_index;
        }
    }
    if (telemetry_active && runtime->telemetry.tape_hash == 0u && request->tape_context && request->tape_context->tape) {
        runtime->telemetry.tape_hash = activation_tape_crc32(request->tape_context->tape);
    }
    if (telemetry_active && request->workspace) {
        runtime->telemetry.hessian_proxy_mean = request->workspace->hessian_proxy.stats.mean;
        runtime->telemetry.hessian_proxy_max = request->workspace->hessian_proxy.stats.max;
        runtime->telemetry.hessian_proxy_source = (uint32_t)request->workspace->hessian_proxy.source;
    }

    return telemetry_active;
}

void transformer_free_ternary_calibration_result(ternary_calibration_result_t *result) {
    if (!result) {
        return;
    }
    free(result->latent_weights);
    free(result->ternary_weights);
    free(result->packed_weights);
    free(result->scales);
    memset(result, 0, sizeof(*result));
}

int transformer_calibrate_layer_ste_with_tape(const uint16_t *bf16_weights,
                                              uint32_t rows,
                                              uint32_t cols,
                                              const transformer_ste_config_t *config,
                                              const ternary_calibration_source_t *source,
                                              ternary_calibration_result_t *out_result) {
    ste_calibration_workspace_request_t workspace_request;
    ste_calibration_context_request_t context_request;
    ste_telemetry_request_t telemetry_request;
    transformer_ste_config_t effective_config;
    ste_calibration_context_t calibration_context;
    ste_calibration_workspace_t workspace;
    ste_telemetry_runtime_t telemetry_runtime;
    ste_layer_schedule_t layer_schedule;
    int status = -1;
    size_t weight_count = 0;
    const ternary_calibration_corpus_t *corpus = source ? source->corpus : NULL;
    const ternary_activation_tape_context_t *tape_context = source ? source->tape_context : NULL;
    const char *tensor_name = NULL;

    if (!bf16_weights || !out_result || rows == 0 || cols == 0) {
        LOG_ERROR("transformer_calibrate_layer_ste: invalid arguments");
        return -1;
    }

    effective_config = config ? *config : default_ste_config();
    normalize_ste_config(&effective_config);
    ste_calibration_workspace_reset(&workspace);
    memset(&telemetry_runtime, 0, sizeof(telemetry_runtime));

    weight_count = (size_t)rows * cols;
    if (ste_prepare_output_buffers(out_result, weight_count, rows, cols) != 0) {
        LOG_ERROR("transformer_calibrate_layer_ste: allocation failed");
        return -1;
    }

    tensor_name = resolve_calibration_tensor_name(corpus, tape_context);
    ste_build_layer_schedule(tensor_name, effective_config.ste_steps, &layer_schedule);

    memset(&workspace_request, 0, sizeof(workspace_request));
    workspace_request.config = &effective_config;
    workspace_request.corpus = corpus;
    workspace_request.tape_context = tape_context;
    workspace_request.sidecar = source ? source->sidecar : NULL;
    workspace_request.bf16_weights = bf16_weights;
    workspace_request.weight_count = weight_count;
    workspace_request.rows = rows;
    workspace_request.cols = cols;
    workspace_request.layer_schedule = &layer_schedule;
    if (ste_prepare_workspace(&workspace_request, &workspace) != 0) {
        LOG_ERROR("transformer_calibrate_layer_ste: calibration buffer allocation failed");
        goto cleanup;
    }

    effective_config.calibration_samples = workspace.actual_sample_count;

    seed_latent_weights(out_result, bf16_weights, weight_count, effective_config.clip_value);

    memset(&context_request, 0, sizeof(context_request));
    context_request.result = out_result;
    context_request.rows = rows;
    context_request.cols = cols;
    context_request.config = &effective_config;
    context_request.corpus = corpus;
    context_request.workspace = &workspace;
    ste_init_calibration_context(&calibration_context, &context_request);

    memset(&telemetry_request, 0, sizeof(telemetry_request));
    telemetry_request.config = &effective_config;
    telemetry_request.tensor_name = tensor_name;
    telemetry_request.rows = rows;
    telemetry_request.cols = cols;
    telemetry_request.tape_context = tape_context;
    telemetry_request.workspace = &workspace;
    (void)ste_prepare_telemetry_runtime(&telemetry_runtime, &telemetry_request);

    if (run_ste_calibration_steps(&calibration_context, out_result, &telemetry_runtime) != 0) {
        goto cleanup;
    }

    quantize_all_rows(&(ste_quantize_rows_request_t){
        .latent = out_result->latent_weights,
        .ternary = out_result->ternary_weights,
        .scales = out_result->scales,
        .scale_floor = workspace.scale_floor,
        .rows = rows,
        .cols = cols,
        .zero_threshold = effective_config.zero_threshold
    });

    if (pack_ternary_2bit(out_result->ternary_weights, rows, cols,
                          out_result->packed_weights,
                          out_result->packed_weight_bytes) != 0) {
        goto cleanup;
    }

    status = 0;

cleanup:
    ste_telemetry_runtime_close(&telemetry_runtime);
    if (status != 0) {
        ste_calibration_workspace_release(&workspace);
        transformer_free_ternary_calibration_result(out_result);
        return -1;
    }

    LOG_INFO("Calibrated ternary layer: rows=%u cols=%u steps=%d samples=%d proxy=%u grad_clip=%.4f kl_weight=%.4f kl_temp=%.2f a8=%u non_collapse=%.4f floor=%.2f gamma_floor=%.3f warmup=%u packed=%zuB",
             rows,
             cols,
             effective_config.ste_steps,
             effective_config.calibration_samples,
             (unsigned)workspace.hessian_proxy.source,
             (double)effective_config.max_grad_norm,
             (double)effective_config.kl_weight,
             (double)effective_config.kl_temperature,
             (unsigned)effective_config.simulate_activation_a8,
             (double)effective_config.non_collapse_weight,
             (double)effective_config.zero_occupancy_floor,
             (double)workspace.layer_schedule.min_gamma_scale,
             workspace.layer_schedule.warmup_steps,
             out_result->packed_weight_bytes);
    ste_calibration_workspace_release(&workspace);
    return 0;
}

int transformer_calibrate_layer_ste(const uint16_t *bf16_weights,
                                    uint32_t rows,
                                    uint32_t cols,
                                    const transformer_ste_config_t *config,
                                    const ternary_calibration_corpus_t *corpus,
                                    ternary_calibration_result_t *out_result) {
    ternary_calibration_source_t source;

    memset(&source, 0, sizeof(source));
    source.corpus = corpus;
    return transformer_calibrate_layer_ste_with_tape(bf16_weights,
                                                     rows,
                                                     cols,
                                                     config,
                                                     &source,
                                                     out_result);
}

/**
 * @file ternary_conversion.c
 * @brief Prompt 2 conversion entry-point shell.
 */

#include "ternary_conversion.h"

#include "calibration_corpus.h"
#include "activation_alignment.h"
#include "activation_tape.h"
#include "file_reader.h"
#include "gemma3_config.h"
#include "inference.h"
#include "kernels.h"
#include "log.h"
#include "model_reader.h"
#include "model_spec.h"
#include "ternary_checkpoint.h"
#include "ternary_anchor.h"
#include "ternary_calibration.h"
#include "ternary_io.h"
#include "ternary_hessian_sidecar.h"
#include "tracy_profile.h"
#include "ternary_validation.h"
#include "transformer.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

/* Number of tensor entries per transformer layer in the activation tape.
 * Matches ALIGNMENT_WEIGHT_SPECS in expand_tape.py. */
#define TAPE_TENSORS_PER_LAYER 7
#define PROGRESSIVE_CALIB_STAGE_COUNT 5u
#define PROGRESSIVE_CALIB_STAGE1_LAST_LAYER 5u
#define PROGRESSIVE_CALIB_STAGE2_HOLD_LAST_LAYER 10u
#define PROGRESSIVE_CALIB_STAGE2_RAMP_FIRST_LAYER 11u
#define PROGRESSIVE_CALIB_STAGE2_LAST_LAYER 15u
#define PROGRESSIVE_CALIB_STAGE2_TARGET_KL_WEIGHT 0.20f
#define PROGRESSIVE_CALIB_STAGE3_TARGET_KL_WEIGHT 0.20f
#define PROGRESSIVE_CALIB_STAGE2_KL_TEMPERATURE 2.0f
#define PROGRESSIVE_CALIB_STAGE3_BLOCK1_LAST_LAYER 9u
#define PROGRESSIVE_CALIB_STAGE3_BLOCK2_LAST_LAYER 19u
#define PROGRESSIVE_CALIB_STAGE3_BLOCK3_LAST_LAYER 27u
#define PROGRESSIVE_CALIB_STAGE3_LR_SCALE 0.50f
#define PROGRESSIVE_RESIDUAL_CORRECTION_MAX_SAMPLES 2
#define PROGRESSIVE_RESIDUAL_CORRECTION_MIN_VARIANCE_RATIO 0.85f
#define PROGRESSIVE_RESIDUAL_CORRECTION_MAX_SCALE 1.10f
#define PROGRESSIVE_CALIB_SCHEDULE_VERSION 4u
#define CONVERSION_CONFIG_OVERRIDE_FILENAME "sapphire_config_overrides.json"
#define CONVERSION_CONFIG_OVERRIDE_TMP_FILENAME "sapphire_config_overrides.json.tmp"
#define FULL_MODEL_ANCHOR_POLICY_VERSION 1u
#define FULL_MODEL_ANCHOR_POLICY_NAME "ffn_down_proj_only"
#define FULL_MODEL_ANCHOR_POLICY_PATTERN "*.mlp.down_proj.weight"

typedef enum {
    PROGRESSIVE_CALIB_STAGE_DISABLED = 0,
    PROGRESSIVE_CALIB_STAGE_EARLY = 1,
    PROGRESSIVE_CALIB_STAGE_MID = 2,
    PROGRESSIVE_CALIB_STAGE_FULL_BLOCK1 = 3,
    PROGRESSIVE_CALIB_STAGE_FULL_BLOCK2 = 4,
    PROGRESSIVE_CALIB_STAGE_FULL_BLOCK3 = 5
} progressive_calib_stage_t;

typedef struct {
    float kl_weight;
    float kl_temperature;
} progressive_distillation_schedule_t;

typedef enum {
    STRUCTURAL_TENSOR_RULE_DIRECT = 0,
    STRUCTURAL_TENSOR_RULE_ALIAS = 1
} structural_tensor_rule_type_t;

typedef enum {
    STRUCTURAL_TENSOR_QUANT_MODE_UNSPECIFIED = 0,
    STRUCTURAL_TENSOR_QUANT_MODE_TERNARY = 1,
    STRUCTURAL_TENSOR_QUANT_MODE_PASS = 2,
    STRUCTURAL_TENSOR_QUANT_MODE_MOLD = 3
} structural_tensor_quant_mode_t;

typedef struct {
    char tensor_name[256];
    char mapped_tensor_name[256];
    structural_tensor_rule_type_t rule_type;
    structural_tensor_quant_mode_t quant_mode;
} structural_tensor_rule_t;

typedef struct {
    structural_tensor_rule_t *rules;
    size_t rule_count;
} structural_map_t;

typedef struct {
    calibration_corpus_t corpus_storage;
    calibration_corpus_t validation_corpus_storage;
    ternary_calibration_corpus_t calibration_corpus;
    ternary_student_update_checkpoint_t checkpoint_state;
    inference_context_t *activation_ctx;
    activation_tape_t *activation_tape;
    char *alignment_manifest_path;
    char *alignment_tape_path;
    sapphire_tokenizer_t *tokenizer;
    model_spec_t *model_spec;
    sapphire_tokenizer_t *previous_tokenizer_handle;
    ternary_validation_state_t validation_state;
    char *checkpoint_path;
    char *checkpoint_tmp_path;
    uint32_t activation_tape_hash;
    uint32_t hessian_sidecar_crc32;
    uint32_t structural_map_crc32;
    uint32_t resume_step_index;
    ternary_bf16_io_cache_t bf16_io_cache;
    ternary_hessian_proxy_cache_t hessian_proxy_cache;
    ternary_hessian_sidecar_t *hessian_sidecar;
    structural_map_t structural_map;
    char telemetry_path[TERNARY_TELEMETRY_PATH_MAX];
    char spatial_telemetry_path[TERNARY_TELEMETRY_PATH_MAX];
} conversion_runtime_t;

static transformer_ste_config_t default_runtime_ste_config(const ternary_conversion_config_t *config);

static int build_runtime_telemetry_path(const char *base_path,
                                        char *out_path,
                                        size_t out_path_size)
{
    const char *file_name = NULL;
    const char *extension = NULL;
    size_t dir_len = 0u;
    size_t stem_len = 0u;
    time_t now = 0;
    struct tm local_tm;
    char timestamp[32];
    int written = 0;

    if (!base_path || !out_path || out_path_size == 0u) {
        return -1;
    }

    file_name = strrchr(base_path, '/');
    file_name = file_name ? file_name + 1 : base_path;
    dir_len = (size_t)(file_name - base_path);
    extension = strrchr(file_name, '.');
    if (!extension) {
        extension = "";
        stem_len = strlen(file_name);
    } else {
        stem_len = (size_t)(extension - file_name);
    }

    now = time(NULL);
    if (now == (time_t)-1 || localtime_r(&now, &local_tm) == NULL) {
        return -1;
    }
    if (strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &local_tm) == 0u) {
        return -1;
    }

    written = snprintf(out_path,
                       out_path_size,
                       "%.*s%.*s_%s%s",
                       (int)dir_len,
                       base_path,
                       (int)stem_len,
                       file_name,
                       timestamp,
                       extension);
    if (written < 0 || (size_t)written >= out_path_size) {
        return -1;
    }

    return 0;
}

static const char *conversion_runtime_telemetry_path(const ternary_conversion_config_t *config,
                                                     const conversion_runtime_t *runtime)
{
    if (runtime && runtime->telemetry_path[0] != '\0') {
        return runtime->telemetry_path;
    }

    return default_runtime_ste_config(config).telemetry_path;
}

static int build_runtime_spatial_telemetry_path(const char *summary_path,
                                                char *out_path,
                                                size_t out_path_size)
{
    const char *file_name = NULL;
    const char *extension = NULL;
    size_t dir_len = 0u;
    size_t stem_len = 0u;
    int written = 0;

    if (!summary_path || !out_path || out_path_size == 0u) {
        return -1;
    }

    file_name = strrchr(summary_path, '/');
    file_name = file_name ? file_name + 1 : summary_path;
    dir_len = (size_t)(file_name - summary_path);
    extension = strrchr(file_name, '.');
    if (!extension) {
        extension = "";
        stem_len = strlen(file_name);
    } else {
        stem_len = (size_t)(extension - file_name);
    }

    written = snprintf(out_path,
                       out_path_size,
                       "%.*s%.*s_spatial%s",
                       (int)dir_len,
                       summary_path,
                       (int)stem_len,
                       file_name,
                       extension);
    if (written < 0 || (size_t)written >= out_path_size) {
        return -1;
    }

    return 0;
}

static const char *conversion_runtime_spatial_telemetry_path(const ternary_conversion_config_t *config,
                                                             const conversion_runtime_t *runtime)
{
    if (!config || !config->emit_spatial_telemetry) {
        return NULL;
    }
    if (runtime && runtime->spatial_telemetry_path[0] != '\0') {
        return runtime->spatial_telemetry_path;
    }

    return NULL;
}

static uint32_t config_hash_update_string(uint32_t crc32, const char *value)
{
    return io_crc32_update(crc32, value, value ? strlen(value) + 1u : 0u);
}

static int full_model_anchor_mode_enabled(const ternary_conversion_config_t *config)
{
    return config && config->use_anchor_mode &&
        (!config->layer_name || config->layer_name[0] == '\0');
}

static const char *full_model_anchor_policy_name(const ternary_conversion_config_t *config)
{
    return full_model_anchor_mode_enabled(config) ? FULL_MODEL_ANCHOR_POLICY_NAME : "";
}

static const char *full_model_anchor_policy_pattern(const ternary_conversion_config_t *config)
{
    (void)config;
    return FULL_MODEL_ANCHOR_POLICY_PATTERN;
}

static const char *full_model_anchor_policy_filter(const ternary_conversion_config_t *config)
{
    if (!config || !config->anchor_tensor_pattern || config->anchor_tensor_pattern[0] == '\0') {
        return "";
    }

    return config->anchor_tensor_pattern;
}

static uint32_t config_resume_hash_update_path_inputs(uint32_t crc32,
                                                      const ternary_conversion_config_t *config)
{
    crc32 = config_hash_update_string(crc32, config->model_name);
    crc32 = config_hash_update_string(crc32, config->output_path);
    crc32 = config_hash_update_string(crc32, config->layer_name);
    crc32 = config_hash_update_string(crc32, config->activation_tape_path);
    crc32 = config_hash_update_string(crc32, config->teacher_model_name);
    crc32 = config_hash_update_string(crc32, config->structural_map_path);
    crc32 = config_hash_update_string(crc32, config->calibration_corpus_path);
    crc32 = config_hash_update_string(crc32, config->calibration_corpus_manifest_path);
    crc32 = config_hash_update_string(crc32, config->validation_corpus_path);
    crc32 = config_hash_update_string(crc32, config->validation_corpus_manifest_path);
    crc32 = config_hash_update_string(crc32, config->hessian_sidecar_path);
    return crc32;
}

static uint32_t config_resume_hash_update_anchor_inputs(uint32_t crc32,
                                                        const ternary_conversion_config_t *config)
{
    uint32_t anchor_artifact_version = TERNARY_ANCHOR_VERSION;

    crc32 = io_crc32_update(crc32, &config->use_anchor_mode, sizeof(config->use_anchor_mode));
    crc32 = io_crc32_update(crc32, &config->anchor_budget_ppm, sizeof(config->anchor_budget_ppm));
    crc32 = io_crc32_update(crc32, &config->anchor_saliency_mode, sizeof(config->anchor_saliency_mode));
    if (config->use_anchor_mode) {
        crc32 = io_crc32_update(crc32, &anchor_artifact_version, sizeof(anchor_artifact_version));
    }
    if (full_model_anchor_mode_enabled(config)) {
        const uint32_t anchor_policy_version = FULL_MODEL_ANCHOR_POLICY_VERSION;

        crc32 = io_crc32_update(crc32, &anchor_policy_version, sizeof(anchor_policy_version));
        crc32 = config_hash_update_string(crc32, full_model_anchor_policy_name(config));
        crc32 = config_hash_update_string(crc32, full_model_anchor_policy_pattern(config));
        crc32 = config_hash_update_string(crc32, full_model_anchor_policy_filter(config));
    }

    return crc32;
}

static uint32_t config_resume_hash(const ternary_conversion_config_t *config,
                                   uint32_t hessian_sidecar_crc32,
                                   uint32_t structural_map_crc32)
{
    uint32_t crc32 = 0u;
    float early_stop_divergence_ratio = 1.25f;
    float early_stop_min_delta = 1e-4f;
    int early_stop_patience = 6;
    int kl_sample_count = 4;
    int kl_update_interval = 4;
    int ste_steps = 3;
    const uint32_t schedule_version = PROGRESSIVE_CALIB_SCHEDULE_VERSION;

    if (!config) {
        return 0u;
    }

    if (config->ste_steps > 0) {
        ste_steps = config->ste_steps;
    }
    if (config->kl_update_interval > 0) {
        kl_update_interval = config->kl_update_interval;
    }
    if (config->kl_sample_count > 0) {
        kl_sample_count = config->kl_sample_count;
    }
    if (config->early_stop_patience >= 0) {
        early_stop_patience = config->early_stop_patience;
    }
    if (config->early_stop_min_delta >= 0.0f) {
        early_stop_min_delta = config->early_stop_min_delta;
    }
    if (config->early_stop_divergence_ratio >= 0.0f) {
        early_stop_divergence_ratio = config->early_stop_divergence_ratio;
    }

    crc32 = config_resume_hash_update_path_inputs(crc32, config);
    crc32 = io_crc32_update(crc32, &config->context_len, sizeof(config->context_len));
    crc32 = io_crc32_update(crc32, &config->calibration_sample_limit, sizeof(config->calibration_sample_limit));
    crc32 = io_crc32_update(crc32, &config->validation_sample_limit, sizeof(config->validation_sample_limit));
    crc32 = io_crc32_update(crc32, &config->checkpoint_every_n_layers, sizeof(config->checkpoint_every_n_layers));
    crc32 = io_crc32_update(crc32, &config->validate_every_n, sizeof(config->validate_every_n));
    crc32 = io_crc32_update(crc32, &ste_steps, sizeof(ste_steps));
    crc32 = io_crc32_update(crc32, &config->progressive_calib, sizeof(config->progressive_calib));
    crc32 = io_crc32_update(crc32, &schedule_version, sizeof(schedule_version));
    crc32 = io_crc32_update(crc32, &config->kl_weight, sizeof(config->kl_weight));
    crc32 = io_crc32_update(crc32, &kl_update_interval, sizeof(kl_update_interval));
    crc32 = io_crc32_update(crc32, &kl_sample_count, sizeof(kl_sample_count));
    crc32 = io_crc32_update(crc32, &early_stop_patience, sizeof(early_stop_patience));
    crc32 = io_crc32_update(crc32, &early_stop_min_delta, sizeof(early_stop_min_delta));
    crc32 = io_crc32_update(crc32, &early_stop_divergence_ratio, sizeof(early_stop_divergence_ratio));
    crc32 = io_crc32_update(crc32, &config->disable_hessian_proxy, sizeof(config->disable_hessian_proxy));
    crc32 = io_crc32_update(crc32, &config->hessian_proxy_strength, sizeof(config->hessian_proxy_strength));
    crc32 = io_crc32_update(crc32, &config->hessian_proxy_floor, sizeof(config->hessian_proxy_floor));
    crc32 = io_crc32_update(crc32, &config->max_grad_norm, sizeof(config->max_grad_norm));
    crc32 = io_crc32_update(crc32,
                            &config->emit_spatial_telemetry,
                            sizeof(config->emit_spatial_telemetry));
    crc32 = io_crc32_update(crc32,
                            &config->spatial_telemetry_row_bucket_size,
                            sizeof(config->spatial_telemetry_row_bucket_size));
    crc32 = io_crc32_update(crc32,
                            &config->student_down_proj_input_rmsnorm,
                            sizeof(config->student_down_proj_input_rmsnorm));
    crc32 = config_resume_hash_update_anchor_inputs(crc32, config);
    crc32 = io_crc32_update(crc32, &hessian_sidecar_crc32, sizeof(hessian_sidecar_crc32));
    crc32 = io_crc32_update(crc32, &structural_map_crc32, sizeof(structural_map_crc32));
    return crc32;
}

static int write_conversion_config_overrides(const ternary_conversion_config_t *config)
{
    char *overrides_path = NULL;
    char *tmp_path = NULL;
    FILE *file = NULL;
    int rc = -1;
    int need_rmsnorm = 0;
    int need_anchor_flags = 0;
    int wrote_any = 0;

    if (!config ||
        !config->output_path || config->output_path[0] == '\0' ||
        (config->layer_name && config->layer_name[0] != '\0')) {
        return 0;
    }

    need_rmsnorm = config->student_down_proj_input_rmsnorm ? 1 : 0;
    need_anchor_flags = config->use_anchor_mode ? 1 : 0;
    if (!need_rmsnorm && !need_anchor_flags) {
        return 0;
    }

    overrides_path = construct_safe_path(config->output_path,
                                         CONVERSION_CONFIG_OVERRIDE_FILENAME,
                                         NULL);
    tmp_path = construct_safe_path(config->output_path,
                                   CONVERSION_CONFIG_OVERRIDE_TMP_FILENAME,
                                   NULL);
    if (!overrides_path || !tmp_path) {
        LOG_ERROR("ternary conversion: failed to build config override path under %s", config->output_path);
        goto cleanup;
    }

    file = fopen(tmp_path, "w");
    if (!file) {
        LOG_ERROR("ternary conversion: failed to open %s: %s", tmp_path, strerror(errno));
        goto cleanup;
    }
    if (fprintf(file, "{\n") < 0) {
        LOG_ERROR("ternary conversion: failed to write %s", tmp_path);
        goto cleanup;
    }
    if (need_rmsnorm) {
        if (fprintf(file, "  \"sapphire_ffn_down_proj_input_rmsnorm\": true") < 0) {
            LOG_ERROR("ternary conversion: failed to write %s", tmp_path);
            goto cleanup;
        }
        wrote_any = 1;
    }
    if (need_anchor_flags) {
        if (fprintf(file,
                    "%s\n  \"sapphire_mixed_precision_anchors\": true,\n  \"sapphire_anchor_budget_ppm\": %u",
                    wrote_any ? "," : "",
                    config->anchor_budget_ppm > 0u ? config->anchor_budget_ppm : 1000u) < 0) {
            LOG_ERROR("ternary conversion: failed to write %s", tmp_path);
            goto cleanup;
        }
        wrote_any = 1;
    }
    if (fprintf(file, "\n}\n") < 0) {
        LOG_ERROR("ternary conversion: failed to write %s", tmp_path);
        goto cleanup;
    }
    if (fclose(file) != 0) {
        file = NULL;
        LOG_ERROR("ternary conversion: failed to close %s: %s", tmp_path, strerror(errno));
        goto cleanup;
    }
    file = NULL;
    if (rename(tmp_path, overrides_path) != 0) {
        LOG_ERROR("ternary conversion: failed to publish %s: %s", overrides_path, strerror(errno));
        goto cleanup;
    }

    rc = 0;

cleanup:
    if (file) {
        fclose(file);
    }
    if (rc != 0 && tmp_path) {
        remove(tmp_path);
    }
    free(tmp_path);
    free(overrides_path);
    return rc;
}

static uint32_t conversion_stage_pass_count(const ternary_conversion_config_t *config)
{
    return (config && config->progressive_calib) ? PROGRESSIVE_CALIB_STAGE_COUNT : 1u;
}

static uint32_t conversion_total_progress_steps(const ternary_conversion_config_t *config,
                                                const model_spec_t *spec)
{
    if (!spec || spec->tensor_map_size <= 0) {
        return 0u;
    }

    return (uint32_t)spec->tensor_map_size * conversion_stage_pass_count(config);
}

static int conversion_try_parse_layer_index(const char *tensor_name,
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

static progressive_calib_stage_t progressive_calib_stage_from_progress_index(const ternary_conversion_config_t *config,
                                                                             const model_spec_t *spec,
                                                                             int progress_index)
{
    int stage_idx = 0;

    if (!config || !config->progressive_calib || !spec || spec->tensor_map_size <= 0 || progress_index < 0) {
        return PROGRESSIVE_CALIB_STAGE_DISABLED;
    }

    stage_idx = progress_index / spec->tensor_map_size;
    if (stage_idx <= 0) {
        return PROGRESSIVE_CALIB_STAGE_EARLY;
    }
    if (stage_idx == 1) {
        return PROGRESSIVE_CALIB_STAGE_MID;
    }
    if (stage_idx == 2) {
        return PROGRESSIVE_CALIB_STAGE_FULL_BLOCK1;
    }
    if (stage_idx == 3) {
        return PROGRESSIVE_CALIB_STAGE_FULL_BLOCK2;
    }

    return PROGRESSIVE_CALIB_STAGE_FULL_BLOCK3;
}

static const char *progressive_calib_stage_name(progressive_calib_stage_t stage)
{
    switch (stage) {
    case PROGRESSIVE_CALIB_STAGE_EARLY:
        return "stage-1 (layers 0-5)";
    case PROGRESSIVE_CALIB_STAGE_MID:
        return "stage-2 (layers 6-15)";
    case PROGRESSIVE_CALIB_STAGE_FULL_BLOCK1:
        return "stage-3a (layers 0-9)";
    case PROGRESSIVE_CALIB_STAGE_FULL_BLOCK2:
        return "stage-3b (layers 10-19)";
    case PROGRESSIVE_CALIB_STAGE_FULL_BLOCK3:
        return "stage-3c (layers 20-27 + shared tensors)";
    default:
        return "disabled";
    }
}

static int progressive_calib_stage_is_stage3(progressive_calib_stage_t stage)
{
    return stage == PROGRESSIVE_CALIB_STAGE_FULL_BLOCK1 ||
           stage == PROGRESSIVE_CALIB_STAGE_FULL_BLOCK2 ||
           stage == PROGRESSIVE_CALIB_STAGE_FULL_BLOCK3;
}

static int progressive_calib_stage_uses_activation_a8(progressive_calib_stage_t stage)
{
    return stage == PROGRESSIVE_CALIB_STAGE_MID ||
           progressive_calib_stage_is_stage3(stage);
}

static int conversion_tracy_end_status(sapphire_tracy_zone_t *zone, int status)
{
    sapphire_tracy_zone_end(zone);
    return status;
}

static void conversion_tracy_zone_text_if_present(const sapphire_tracy_zone_t *zone,
                                                  const char *text)
{
    if (!zone || !text || text[0] == '\0') {
        return;
    }

    sapphire_tracy_zone_text(zone, text, strlen(text));
}

static void conversion_tracy_plot_layer_index(int has_layer_index, uint32_t layer_index)
{
    if (!has_layer_index) {
        return;
    }

    sapphire_tracy_plot_i64("conversion.layer_index", (int64_t)layer_index);
}

static void conversion_tracy_emit_stage_marker(progressive_calib_stage_t stage)
{
    const char *stage_name = progressive_calib_stage_name(stage);

    sapphire_tracy_plot_i64("conversion.progressive_stage", (int64_t)stage);
    if (stage_name && stage_name[0] != '\0') {
        sapphire_tracy_message(stage_name, strlen(stage_name));
    }
}

static void conversion_tracy_plot_checkpoint(const ternary_student_update_checkpoint_t *checkpoint)
{
    if (!checkpoint) {
        return;
    }

    sapphire_tracy_plot_i64("conversion.converted_tensors",
                            (int64_t)checkpoint->converted_tensor_count);
    sapphire_tracy_plot_i64("conversion.checkpoint_next_progress",
                            (int64_t)checkpoint->next_layer_index);
}

static int progressive_calib_stage_uses_residual_recenter(progressive_calib_stage_t stage)
{
    return stage == PROGRESSIVE_CALIB_STAGE_MID;
}

static progressive_distillation_schedule_t progressive_calib_distillation_schedule(
    const ternary_conversion_config_t *config,
    progressive_calib_stage_t stage,
    int has_layer_index,
    uint32_t layer_index)
{
    progressive_distillation_schedule_t schedule;
    float base_kl_weight = 0.05f;

    if (config && config->kl_weight >= 0.0f) {
        base_kl_weight = config->kl_weight;
    }

    schedule.kl_weight = base_kl_weight;
    schedule.kl_temperature = 1.0f;

    if (!config || !config->progressive_calib || !has_layer_index) {
        return schedule;
    }

    if (stage == PROGRESSIVE_CALIB_STAGE_MID) {
        float target_kl_weight = base_kl_weight;

        if (target_kl_weight < PROGRESSIVE_CALIB_STAGE2_TARGET_KL_WEIGHT) {
            target_kl_weight = PROGRESSIVE_CALIB_STAGE2_TARGET_KL_WEIGHT;
        }
        schedule.kl_temperature = PROGRESSIVE_CALIB_STAGE2_KL_TEMPERATURE;
        if (layer_index >= PROGRESSIVE_CALIB_STAGE2_RAMP_FIRST_LAYER) {
            float progress = (float)(layer_index - PROGRESSIVE_CALIB_STAGE2_RAMP_FIRST_LAYER)
                / (float)(PROGRESSIVE_CALIB_STAGE2_LAST_LAYER - PROGRESSIVE_CALIB_STAGE2_RAMP_FIRST_LAYER);
            if (progress < 0.0f) {
                progress = 0.0f;
            }
            if (progress > 1.0f) {
                progress = 1.0f;
            }
            schedule.kl_weight = base_kl_weight +
                (target_kl_weight - base_kl_weight) * progress;
        }
        return schedule;
    }

    if (progressive_calib_stage_is_stage3(stage)) {
        schedule.kl_weight = base_kl_weight;
        if (schedule.kl_weight < PROGRESSIVE_CALIB_STAGE3_TARGET_KL_WEIGHT) {
            schedule.kl_weight = PROGRESSIVE_CALIB_STAGE3_TARGET_KL_WEIGHT;
        }
    }

    return schedule;
}

static int progressive_calib_stage_allows_tensor(progressive_calib_stage_t stage,
                                                 const char *tensor_name)
{
    uint32_t layer_index = 0u;

    if (stage == PROGRESSIVE_CALIB_STAGE_DISABLED) {
        return 1;
    }
    if (!conversion_try_parse_layer_index(tensor_name, &layer_index)) {
        return stage == PROGRESSIVE_CALIB_STAGE_FULL_BLOCK3;
    }
    if (stage == PROGRESSIVE_CALIB_STAGE_EARLY) {
        return layer_index <= PROGRESSIVE_CALIB_STAGE1_LAST_LAYER;
    }
    if (stage == PROGRESSIVE_CALIB_STAGE_MID) {
        return layer_index > PROGRESSIVE_CALIB_STAGE1_LAST_LAYER &&
               layer_index <= PROGRESSIVE_CALIB_STAGE2_LAST_LAYER;
    }
    if (stage == PROGRESSIVE_CALIB_STAGE_FULL_BLOCK1) {
        return layer_index <= PROGRESSIVE_CALIB_STAGE3_BLOCK1_LAST_LAYER;
    }
    if (stage == PROGRESSIVE_CALIB_STAGE_FULL_BLOCK2) {
        return layer_index > PROGRESSIVE_CALIB_STAGE3_BLOCK1_LAST_LAYER &&
               layer_index <= PROGRESSIVE_CALIB_STAGE3_BLOCK2_LAST_LAYER;
    }
    if (stage == PROGRESSIVE_CALIB_STAGE_FULL_BLOCK3) {
        return layer_index > PROGRESSIVE_CALIB_STAGE3_BLOCK2_LAST_LAYER &&
               layer_index <= PROGRESSIVE_CALIB_STAGE3_BLOCK3_LAST_LAYER;
    }

    return 0;
}

static int conversion_tensor_index_from_progress(const ternary_conversion_config_t *config,
                                                 const model_spec_t *spec,
                                                 int progress_index)
{
    if (!spec || spec->tensor_map_size <= 0 || progress_index < 0) {
        return -1;
    }

    if (config && config->progressive_calib) {
        return progress_index % spec->tensor_map_size;
    }

    return progress_index;
}

static float telemetry_elapsed_ms(const struct timespec *start,
                                  const struct timespec *end)
{
    time_t sec = 0;
    long nsec = 0;

    if (!start || !end) {
        return 0.0f;
    }

    sec = end->tv_sec - start->tv_sec;
    nsec = end->tv_nsec - start->tv_nsec;
    return (float)sec * 1000.0f + (float)nsec / 1000000.0f;
}

static uint32_t telemetry_checkpoint_hash_or_zero(const char *checkpoint_path)
{
    struct stat st;
    uint32_t checkpoint_hash = 0u;

    if (!checkpoint_path || checkpoint_path[0] == '\0') {
        return 0u;
    }

    if (stat(checkpoint_path, &st) != 0 || !S_ISREG(st.st_mode)) {
        return 0u;
    }

    if (ternary_student_checkpoint_compute_file_crc32(checkpoint_path, &checkpoint_hash) != 0) {
        return 0u;
    }

    return checkpoint_hash;
}

static void telemetry_prepare_layer_context(ternary_telemetry_t *telemetry,
                                            const conversion_runtime_t *runtime,
                                            int layer_index)
{
    if (!telemetry) {
        return;
    }

    memset(telemetry, 0, sizeof(*telemetry));
    telemetry->layer_idx = (uint32_t)layer_index;
    telemetry->resume_step_idx = runtime ? runtime->resume_step_index : 0u;
    telemetry->tape_hash = runtime ? runtime->activation_tape_hash : 0u;
    telemetry->student_checkpoint_hash = runtime ? telemetry_checkpoint_hash_or_zero(runtime->checkpoint_path) : 0u;
}

static int checkpoint_path_for_output(const char *output_dir, char **out_path)
{
    if (!output_dir || !out_path) {
        return -1;
    }

    *out_path = construct_safe_path(output_dir, "student_update_checkpoint.txt", NULL);
    return *out_path ? 0 : -1;
}

static int checkpoint_tmp_path_for_output(const char *output_dir, char **out_path)
{
    if (!output_dir || !out_path) {
        return -1;
    }

    *out_path = construct_safe_path(output_dir, "student_update_checkpoint.txt.tmp", NULL);
    return *out_path ? 0 : -1;
}

typedef enum {
    FULL_MODEL_TENSOR_REPRESENTATION_TERNARY = 0,
    FULL_MODEL_TENSOR_REPRESENTATION_MOLD = 1,
    FULL_MODEL_TENSOR_REPRESENTATION_ANCHOR = 2
} full_model_tensor_representation_t;

typedef struct {
    char tensor_name[256];
    char file_name[256];
    char anchor_file_name[256];
    uint32_t rows;
    uint32_t cols;
    uint32_t anchor_count;
    size_t packed_weight_bytes;
    uint32_t crc32;
    full_model_tensor_representation_t representation;
} resume_manifest_entry_t;

static int copy_text_field_local(char *dst, size_t dst_size, const char *src)
{
    size_t len = 0u;

    if (!dst || dst_size == 0u) {
        return -1;
    }
    dst[0] = '\0';
    if (!src) {
        return 0;
    }

    len = strlen(src);
    if (len >= dst_size) {
        return -1;
    }
    memcpy(dst, src, len + 1u);
    return 0;
}

static char *duplicate_text_local(const char *src)
{
    size_t len = 0u;
    char *copy = NULL;

    if (!src) {
        return NULL;
    }

    len = strlen(src);
    copy = (char *)malloc(len + 1u);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, src, len + 1u);
    return copy;
}

static int parse_u32_field(const char *text, uint32_t *out_value)
{
    char *end = NULL;
    unsigned long parsed = 0ul;

    if (!text || !out_value) {
        return -1;
    }

    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > 0xFFFFFFFFul) {
        return -1;
    }

    *out_value = (uint32_t)parsed;
    return 0;
}

static int parse_size_field(const char *text, size_t *out_value)
{
    char *end = NULL;
    unsigned long long parsed = 0ull;

    if (!text || !out_value) {
        return -1;
    }

    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed > (unsigned long long)SIZE_MAX) {
        return -1;
    }

    *out_value = (size_t)parsed;
    return 0;
}

static int parse_hex_u32_field(const char *text, uint32_t *out_value)
{
    char *end = NULL;
    unsigned long parsed = 0ul;

    if (!text || !out_value) {
        return -1;
    }

    errno = 0;
    parsed = strtoul(text, &end, 16);
    if (errno != 0 || end == text || *end != '\0' || parsed > 0xFFFFFFFFul) {
        return -1;
    }

    *out_value = (uint32_t)parsed;
    return 0;
}

static int parse_resume_manifest_line(char *line, resume_manifest_entry_t *out_entry)
{
    char *saveptr = NULL;
    const char *field = NULL;

    if (!line || !out_entry) {
        return -1;
    }

    memset(out_entry, 0, sizeof(*out_entry));
    out_entry->representation = FULL_MODEL_TENSOR_REPRESENTATION_TERNARY;
    field = strtok_r(line, "\t", &saveptr);
    if (!field || copy_text_field_local(out_entry->tensor_name, sizeof(out_entry->tensor_name), field) != 0) {
        return -1;
    }

    field = strtok_r(NULL, "\t", &saveptr);
    if (!field || copy_text_field_local(out_entry->file_name, sizeof(out_entry->file_name), field) != 0) {
        return -1;
    }

    field = strtok_r(NULL, "\t", &saveptr);
    if (!field || parse_u32_field(field, &out_entry->rows) != 0) {
        return -1;
    }

    field = strtok_r(NULL, "\t", &saveptr);
    if (!field || parse_u32_field(field, &out_entry->cols) != 0) {
        return -1;
    }

    field = strtok_r(NULL, "\t", &saveptr);
    if (!field || parse_size_field(field, &out_entry->packed_weight_bytes) != 0) {
        return -1;
    }

    field = strtok_r(NULL, "\t\r\n", &saveptr);
    if (!field || parse_hex_u32_field(field, &out_entry->crc32) != 0) {
        return -1;
    }

    field = strtok_r(NULL, "\t\r\n", &saveptr);
    if (!field) {
        return 0;
    }
    if (strcmp(field, "mold") == 0) {
        out_entry->representation = FULL_MODEL_TENSOR_REPRESENTATION_MOLD;
        return 0;
    }

    out_entry->representation = FULL_MODEL_TENSOR_REPRESENTATION_ANCHOR;
    if (strcmp(field, "anchor") == 0) {
        field = strtok_r(NULL, "\t\r\n", &saveptr);
        if (!field) {
            return -1;
        }
    }
    if (copy_text_field_local(out_entry->anchor_file_name, sizeof(out_entry->anchor_file_name), field) != 0) {
        return -1;
    }

    field = strtok_r(NULL, "\t\r\n", &saveptr);
    if (!field || parse_u32_field(field, &out_entry->anchor_count) != 0) {
        return -1;
    }

    return 0;
}

static int read_text_file(const char *path, char **out_text, size_t *out_size)
{
    char *buffer = NULL;
    size_t buffer_size = 0u;
    char *text = NULL;

    if (!path || !out_text || !out_size) {
        return -1;
    }

    if (file_read_to_buffer(path, &buffer, &buffer_size) != 0) {
        return -1;
    }

    text = (char *)malloc(buffer_size + 1u);
    if (!text) {
        free(buffer);
        return -1;
    }
    memcpy(text, buffer, buffer_size);
    text[buffer_size] = '\0';
    free(buffer);

    *out_text = text;
    *out_size = buffer_size;
    return 0;
}

static void structural_map_release(structural_map_t *map)
{
    if (!map) {
        return;
    }

    free(map->rules);
    memset(map, 0, sizeof(*map));
}

static int text_equals_ignore_case_local(const char *lhs, const char *rhs)
{
    size_t idx = 0u;

    if (!lhs || !rhs) {
        return 0;
    }

    while (lhs[idx] != '\0' && rhs[idx] != '\0') {
        char lhs_ch = lhs[idx];
        char rhs_ch = rhs[idx];

        if (lhs_ch >= 'A' && lhs_ch <= 'Z') {
            lhs_ch = (char)(lhs_ch - 'A' + 'a');
        }
        if (rhs_ch >= 'A' && rhs_ch <= 'Z') {
            rhs_ch = (char)(rhs_ch - 'A' + 'a');
        }
        if (lhs_ch != rhs_ch) {
            return 0;
        }
        idx++;
    }

    return lhs[idx] == '\0' && rhs[idx] == '\0';
}

static int text_is_decimal_local(const char *text)
{
    size_t idx = 0u;

    if (!text || text[0] == '\0') {
        return 0;
    }

    for (idx = 0u; text[idx] != '\0'; ++idx) {
        if (text[idx] < '0' || text[idx] > '9') {
            return 0;
        }
    }

    return 1;
}

static void trim_line_in_place(char *text)
{
    size_t len = 0u;

    if (!text) {
        return;
    }

    len = strlen(text);
    while (len > 0u) {
        char ch = text[len - 1u];
        if (ch != ' ' && ch != '\t' && ch != '\r') {
            break;
        }
        text[len - 1u] = '\0';
        len--;
    }
}

static int structural_map_rule_type_from_text(const char *text,
                                              structural_tensor_rule_type_t *out_type)
{
    if (!text || !out_type) {
        return -1;
    }

    if (text_equals_ignore_case_local(text, "direct")) {
        *out_type = STRUCTURAL_TENSOR_RULE_DIRECT;
        return 0;
    }
    if (text_equals_ignore_case_local(text, "alias")) {
        *out_type = STRUCTURAL_TENSOR_RULE_ALIAS;
        return 0;
    }

    return -1;
}

static int structural_map_quant_mode_from_text(const char *text,
                                               structural_tensor_quant_mode_t *out_mode)
{
    if (!text || !out_mode) {
        return -1;
    }

    if (text_equals_ignore_case_local(text, "ternary")) {
        *out_mode = STRUCTURAL_TENSOR_QUANT_MODE_TERNARY;
        return 0;
    }
    if (text_equals_ignore_case_local(text, "pass") ||
        text_equals_ignore_case_local(text, "bf16")) {
        *out_mode = STRUCTURAL_TENSOR_QUANT_MODE_PASS;
        return 0;
    }
    if (text_equals_ignore_case_local(text, "mold")) {
        *out_mode = STRUCTURAL_TENSOR_QUANT_MODE_MOLD;
        return 0;
    }

    return -1;
}

static int structural_map_parse_rule_line(const char *line,
                                          structural_tensor_rule_t *out_rule)
{
    char line_copy[1024];
    char *fields[4];
    size_t field_count = 0u;
    char *saveptr = NULL;
    char *field = NULL;
    const char *tensor_name = NULL;
    const char *mapped_tensor_name = NULL;
    structural_tensor_rule_type_t rule_type;
    structural_tensor_quant_mode_t quant_mode = STRUCTURAL_TENSOR_QUANT_MODE_UNSPECIFIED;

    if (!line || !out_rule) {
        return -1;
    }

    memset(out_rule, 0, sizeof(*out_rule));
    if (copy_text_field_local(line_copy, sizeof(line_copy), line) != 0) {
        return -1;
    }

    field = strtok_r(line_copy, " \t", &saveptr);
    while (field && field_count < (sizeof(fields) / sizeof(fields[0]))) {
        fields[field_count++] = field;
        field = strtok_r(NULL, " \t", &saveptr);
    }

    if (field_count < 3u) {
        return 0;
    }

    if (text_is_decimal_local(fields[0]) && text_is_decimal_local(fields[1])) {
        return 0;
    }

    if (field_count >= 4u &&
        structural_map_rule_type_from_text(fields[2], &rule_type) == 0 &&
        structural_map_quant_mode_from_text(fields[3], &quant_mode) == 0) {
        tensor_name = fields[0];
        mapped_tensor_name = fields[1];
    } else if (field_count == 3u &&
               structural_map_rule_type_from_text(fields[2], &rule_type) == 0) {
        tensor_name = fields[0];
        mapped_tensor_name = fields[1];
    } else if (field_count == 3u &&
               structural_map_rule_type_from_text(fields[1], &rule_type) == 0 &&
               structural_map_quant_mode_from_text(fields[2], &quant_mode) == 0) {
        if (rule_type == STRUCTURAL_TENSOR_RULE_ALIAS) {
            return -1;
        }
        tensor_name = fields[0];
        mapped_tensor_name = fields[0];
    } else {
        return 0;
    }

    if (rule_type == STRUCTURAL_TENSOR_RULE_ALIAS &&
        quant_mode == STRUCTURAL_TENSOR_QUANT_MODE_TERNARY) {
        return -1;
    }

    out_rule->rule_type = rule_type;
    out_rule->quant_mode = quant_mode;
    if (copy_text_field_local(out_rule->tensor_name, sizeof(out_rule->tensor_name), tensor_name) != 0 ||
        copy_text_field_local(out_rule->mapped_tensor_name, sizeof(out_rule->mapped_tensor_name), mapped_tensor_name) != 0) {
        return -1;
    }

    return 1;
}

static int structural_pattern_matches(const char *pattern,
                                      const char *text)
{
    const char *pattern_cursor = NULL;
    const char *text_cursor = NULL;
    const char *star = NULL;
    const char *retry_text = NULL;

    if (!pattern || !text) {
        return 0;
    }

    pattern_cursor = pattern;
    text_cursor = text;
    while (*text_cursor != '\0') {
        if (*pattern_cursor == '*') {
            star = pattern_cursor++;
            retry_text = text_cursor;
            continue;
        }
        if (*pattern_cursor == *text_cursor) {
            pattern_cursor++;
            text_cursor++;
            continue;
        }
        if (star) {
            pattern_cursor = star + 1;
            text_cursor = ++retry_text;
            continue;
        }
        return 0;
    }

    while (*pattern_cursor == '*') {
        pattern_cursor++;
    }

    return *pattern_cursor == '\0';
}

static int structural_map_append_rule(structural_map_t *map,
                                      const structural_tensor_rule_t *rule)
{
    structural_tensor_rule_t *resized_rules = NULL;

    if (!map || !rule) {
        return -1;
    }

    resized_rules = (structural_tensor_rule_t *)realloc(
        map->rules,
        (map->rule_count + 1u) * sizeof(*map->rules)
    );
    if (!resized_rules) {
        return -1;
    }

    map->rules = resized_rules;
    map->rules[map->rule_count] = *rule;
    map->rule_count++;
    return 0;
}

static int structural_map_load(const char *path, structural_map_t *out_map)
{
    char *text = NULL;
    size_t text_size = 0u;
    char *cursor = NULL;

    if (!path || !out_map) {
        return -1;
    }

    memset(out_map, 0, sizeof(*out_map));
    if (read_text_file(path, &text, &text_size) != 0) {
        LOG_ERROR("ternary structural map: failed to read %s", path);
        return -1;
    }

    cursor = text;
    while (cursor && *cursor != '\0') {
        char *line_end = strchr(cursor, '\n');
        structural_tensor_rule_t rule;

        if (line_end) {
            *line_end = '\0';
        }

        while (*cursor == ' ' || *cursor == '\t') {
            cursor++;
        }
        trim_line_in_place(cursor);

        if (cursor[0] != '\0' && cursor[0] != '#') {
            int parse_status = structural_map_parse_rule_line(cursor, &rule);
            if (parse_status < 0) {
                LOG_ERROR("ternary structural map: malformed rule in %s: %s", path, cursor);
                free(text);
                structural_map_release(out_map);
                return -1;
            }
            if (parse_status > 0 && structural_map_append_rule(out_map, &rule) != 0) {
                LOG_ERROR("ternary structural map: out of memory while parsing %s", path);
                free(text);
                structural_map_release(out_map);
                return -1;
            }
        }

        if (!line_end) {
            break;
        }
        cursor = line_end + 1;
    }

    free(text);
    LOG_INFO("ternary structural map: loaded %zu tensor rule(s) from %s", out_map->rule_count, path);
    (void)text_size;
    return 0;
}

static int structural_map_name_matches(const char *query_name, const char *rule_name)
{
    size_t query_len = 0u;
    size_t rule_len = 0u;

    if (!query_name || !rule_name) {
        return 0;
    }
    if (strchr(rule_name, '*') != NULL) {
        return structural_pattern_matches(rule_name, query_name);
    }
    if (strcmp(query_name, rule_name) == 0) {
        return 1;
    }

    query_len = strlen(query_name);
    rule_len = strlen(rule_name);
    if (query_len >= rule_len && strcmp(query_name + (query_len - rule_len), rule_name) == 0) {
        return 1;
    }
    if (rule_len >= query_len && strcmp(rule_name + (rule_len - query_len), query_name) == 0) {
        return 1;
    }

    return 0;
}

static const structural_tensor_rule_t *structural_map_find_rule(const structural_map_t *map,
                                                                const char *tensor_name)
{
    size_t rule_idx = 0u;

    if (!map || !tensor_name) {
        return NULL;
    }

    for (rule_idx = 0u; rule_idx < map->rule_count; ++rule_idx) {
        if (structural_map_name_matches(tensor_name, map->rules[rule_idx].tensor_name)) {
            return &map->rules[rule_idx];
        }
    }

    return NULL;
}

static int manifest_path_exists(const char *output_dir)
{
    char *manifest_path = NULL;
    struct stat st;
    int rc = -1;

    manifest_path = construct_safe_path(output_dir, "manifest.tsv", NULL);
    if (!manifest_path) {
        return -1;
    }

    if (stat(manifest_path, &st) == 0) {
        rc = 1;
    } else if (errno == ENOENT) {
        rc = 0;
    }

    free(manifest_path);
    return rc;
}

static int resume_validation_apply_ternary_entry(const ternary_conversion_config_t *config,
                                                 conversion_runtime_t *runtime,
                                                 const resume_manifest_entry_t *entry)
{
    char *layer_path = NULL;
    ternary_layer_payload_t payload;
    uint32_t actual_crc32 = 0u;
    int rc = -1;

    if (!config || !runtime || !entry) {
        return -1;
    }

    memset(&payload, 0, sizeof(payload));
    layer_path = construct_safe_path(config->output_path, entry->file_name, NULL);
    if (!layer_path) {
        return -1;
    }
    if (io_load_layer_ternary_payload(layer_path,
                                      entry->tensor_name,
                                      entry->rows,
                                      entry->cols,
                                      &payload) != 0) {
        goto cleanup;
    }

    actual_crc32 = io_crc32_update(0u, payload.packed_weights, payload.packed_weight_bytes);
    actual_crc32 = io_crc32_update(actual_crc32, payload.scales, payload.scale_bytes);
    if (actual_crc32 != entry->crc32) {
        LOG_ERROR("student update: manifest CRC mismatch for %s (expected=%08x actual=%08x)",
                  entry->tensor_name,
                  entry->crc32,
                  actual_crc32);
        goto cleanup;
    }

    rc = ternary_validation_apply_proxy_from_payload(&runtime->validation_state,
                                                     entry->tensor_name,
                                                     &payload,
                                                     entry->crc32,
                                                     0);

cleanup:
    io_free_layer_ternary_payload(&payload);
    free(layer_path);
    return rc;
}

static int resume_validation_apply_anchor_entry(const ternary_conversion_config_t *config,
                                                conversion_runtime_t *runtime,
                                                const resume_manifest_entry_t *entry)
{
    char *anchor_path = NULL;
    char *bulk_path = NULL;
    ternary_anchor_view_t anchor_view;
    ternary_layer_payload_t payload;
    uint32_t actual_crc32 = 0u;
    int rc = -1;

    if (!config || !runtime || !entry || entry->anchor_file_name[0] == '\0') {
        return -1;
    }

    memset(&anchor_view, 0, sizeof(anchor_view));
    memset(&payload, 0, sizeof(payload));
    anchor_path = construct_safe_path(config->output_path, entry->anchor_file_name, NULL);
    if (!anchor_path) {
        return -1;
    }
    if (ternary_anchor_load_for_tensor(anchor_path, entry->tensor_name, &anchor_view) != 0) {
        goto cleanup;
    }
    if (anchor_view.rows != entry->rows || anchor_view.cols != entry->cols) {
        LOG_ERROR("student update: anchor manifest shape mismatch for %s", entry->tensor_name);
        goto cleanup;
    }
    if (entry->anchor_count > 0u && anchor_view.anchor_count != entry->anchor_count) {
        LOG_ERROR("student update: anchor manifest count mismatch for %s (expected=%u actual=%u)",
                  entry->tensor_name,
                  entry->anchor_count,
                  anchor_view.anchor_count);
        goto cleanup;
    }

    bulk_path = construct_safe_path(config->output_path, entry->file_name, NULL);
    if (!bulk_path) {
        goto cleanup;
    }
    if (io_load_layer_ternary_payload(bulk_path,
                                      entry->tensor_name,
                                      entry->rows,
                                      entry->cols,
                                      &payload) != 0) {
        goto cleanup;
    }

    actual_crc32 = io_crc32_update(0u, payload.packed_weights, payload.packed_weight_bytes);
    actual_crc32 = io_crc32_update(actual_crc32, payload.scales, payload.scale_bytes);
    if (actual_crc32 != entry->crc32) {
        LOG_ERROR("student update: manifest CRC mismatch for %s (expected=%08x actual=%08x)",
                  entry->tensor_name,
                  entry->crc32,
                  actual_crc32);
        goto cleanup;
    }

    rc = ternary_validation_apply_proxy_from_hybrid_payload(&runtime->validation_state,
                                                            entry->tensor_name,
                                                            &payload,
                                                            &anchor_view,
                                                            entry->crc32,
                                                            0);

cleanup:
    io_free_layer_ternary_payload(&payload);
    ternary_anchor_view_release(&anchor_view);
    free(bulk_path);
    free(anchor_path);
    return rc;
}

static int resume_validation_apply_mold_entry(const ternary_conversion_config_t *config,
                                              conversion_runtime_t *runtime,
                                              const resume_manifest_entry_t *entry)
{
    char *layer_path = NULL;
    ternary_bf16_layer_map_t map;
    size_t bf16_byte_count = 0u;
    uint32_t actual_crc32 = 0u;
    int rc = -1;

    if (!config || !runtime || !entry) {
        return -1;
    }

    memset(&map, 0, sizeof(map));
    layer_path = construct_safe_path(config->output_path, entry->file_name, NULL);
    if (!layer_path) {
        return -1;
    }
    if (io_mmap_layer_bf16(layer_path, entry->tensor_name, &map) != 0) {
        goto cleanup;
    }
    if (map.rows != entry->rows || map.cols != entry->cols) {
        LOG_ERROR("student update: mold manifest shape mismatch for %s", entry->tensor_name);
        goto cleanup;
    }

    bf16_byte_count = (size_t)map.rows * map.cols * sizeof(uint16_t);
    actual_crc32 = io_crc32_update(0u, map.bf16_weights, bf16_byte_count);
    if (actual_crc32 != entry->crc32) {
        LOG_ERROR("student update: mold manifest CRC mismatch for %s (expected=%08x actual=%08x)",
                  entry->tensor_name,
                  entry->crc32,
                  actual_crc32);
        goto cleanup;
    }

    rc = ternary_validation_apply_proxy_from_bf16(&runtime->validation_state,
                                                  entry->tensor_name,
                                                  &(ternary_validation_bf16_view_t){
                                                      .weights = map.bf16_weights,
                                                      .rows = map.rows,
                                                      .cols = map.cols,
                                                  },
                                                  entry->crc32,
                                                  0);

cleanup:
    io_unmap_layer_bf16(&map);
    free(layer_path);
    return rc;
}

static int resume_validation_apply_manifest_entry(const ternary_conversion_config_t *config,
                                                  conversion_runtime_t *runtime,
                                                  const resume_manifest_entry_t *entry)
{
    if (!entry) {
        return -1;
    }

    switch (entry->representation) {
        case FULL_MODEL_TENSOR_REPRESENTATION_MOLD:
            return resume_validation_apply_mold_entry(config, runtime, entry);
        case FULL_MODEL_TENSOR_REPRESENTATION_ANCHOR:
            return resume_validation_apply_anchor_entry(config, runtime, entry);
        case FULL_MODEL_TENSOR_REPRESENTATION_TERNARY:
        default:
            return resume_validation_apply_ternary_entry(config, runtime, entry);
    }
}

static int resume_validation_from_manifest(const ternary_conversion_config_t *config,
                                           conversion_runtime_t *runtime)
{
    char *manifest_path = NULL;
    char *manifest_text = NULL;
    size_t manifest_size = 0u;
    char *cursor = NULL;
    int replayed = 0;

    if (!config || !runtime || !runtime->model_spec) {
        return -1;
    }
    if (runtime->checkpoint_state.converted_tensor_count == 0u) {
        return 0;
    }
    if (runtime->validation_state.config.sample_count <= 0) {
        return 0;
    }

    manifest_path = construct_safe_path(config->output_path, "manifest.tsv", NULL);
    if (!manifest_path) {
        return -1;
    }

    if (read_text_file(manifest_path, &manifest_text, &manifest_size) != 0) {
        free(manifest_path);
        return -1;
    }
    free(manifest_path);

    if (manifest_size == 0u) {
        if (runtime->checkpoint_state.converted_tensor_count > 0u) {
            LOG_ERROR("student update: manifest is empty but checkpoint expects %u converted tensors",
                      runtime->checkpoint_state.converted_tensor_count);
            free(manifest_text);
            return -1;
        }
        free(manifest_text);
        return 0;
    }

    cursor = manifest_text;
    while (cursor && *cursor != '\0' && replayed < (int)runtime->checkpoint_state.converted_tensor_count) {
        char *line_end = strchr(cursor, '\n');
        resume_manifest_entry_t entry;

        if (line_end) {
            *line_end = '\0';
        }
        if (cursor[0] != '\0') {
            if (parse_resume_manifest_line(cursor, &entry) != 0) {
                free(manifest_text);
                return -1;
            }
            if (resume_validation_apply_manifest_entry(config, runtime, &entry) != 0) {
                free(manifest_text);
                return -1;
            }
            replayed++;
        }

        if (!line_end) {
            break;
        }
        cursor = line_end + 1;
    }

    if (replayed != (int)runtime->checkpoint_state.converted_tensor_count) {
        if (config->progressive_calib) {
            LOG_INFO("student update: progressive calib replayed %d manifest tensors while checkpoint tracks %u staged conversions",
                     replayed,
                     runtime->checkpoint_state.converted_tensor_count);
            runtime->validation_state.last_reported_count = replayed;
            free(manifest_text);
            return 0;
        }
        LOG_ERROR("student update: manifest replay count mismatch (checkpoint=%u replayed=%d)",
                  runtime->checkpoint_state.converted_tensor_count,
                  replayed);
        free(manifest_text);
        return -1;
    }

    runtime->validation_state.last_reported_count = replayed;
    free(manifest_text);
    return 0;
}

static int write_student_checkpoint(const ternary_conversion_config_t *config,
                                    conversion_runtime_t *runtime,
                                    const char *alignment_manifest_path,
                                    const char *alignment_tape_path);

static int student_checkpoint_progress(const ternary_conversion_config_t *config,
                                       conversion_runtime_t *runtime,
                                       uint32_t next_layer_index,
                                       uint32_t converted_count,
                                       int force_write)
{
    uint32_t every_n_layers = 0u;
    uint32_t total_layers = 0u;

    if (!runtime) {
        return 0;
    }

    runtime->checkpoint_state.next_layer_index = next_layer_index;
    runtime->checkpoint_state.last_completed_layer = (next_layer_index > 0u) ? (next_layer_index - 1u) : 0u;
    runtime->checkpoint_state.converted_tensor_count = converted_count;

    every_n_layers = runtime->checkpoint_state.checkpoint_every_n_layers;
    total_layers = runtime->checkpoint_state.total_layer_count;
    if (!force_write && every_n_layers > 0u) {
        if ((next_layer_index % every_n_layers) != 0u && next_layer_index < total_layers) {
            return 0;
        }
    }

    return write_student_checkpoint(config,
                                    runtime,
                                    runtime->alignment_manifest_path,
                                    runtime->alignment_tape_path);
}

static int init_checkpoint_paths(const ternary_conversion_config_t *config,
                                 conversion_runtime_t *runtime)
{
    char *checkpoint_path = NULL;
    char *tmp_path = NULL;

    if (!config || !runtime) {
        return -1;
    }

    if (checkpoint_path_for_output(config->output_path, &checkpoint_path) != 0 ||
        checkpoint_tmp_path_for_output(config->output_path, &tmp_path) != 0) {
        free(checkpoint_path);
        return -1;
    }

    runtime->checkpoint_path = checkpoint_path;
    runtime->checkpoint_tmp_path = tmp_path;
    return 0;
}

static void seed_checkpoint_state_defaults(const ternary_conversion_config_t *config,
                                           conversion_runtime_t *runtime,
                                           uint32_t total_layer_count,
                                           uint32_t config_hash)
{
    runtime->checkpoint_state.total_layer_count = total_layer_count;
    runtime->checkpoint_state.schema_version = TERNARY_STUDENT_CHECKPOINT_VERSION;
    runtime->checkpoint_state.config_hash = config_hash;
    runtime->checkpoint_state.checkpoint_every_n_layers = (config->checkpoint_every_n_layers > 0)
        ? (uint32_t)config->checkpoint_every_n_layers
        : 1u;
    runtime->checkpoint_state.validate_every_n = (config->validate_every_n > 0) ? (uint32_t)config->validate_every_n : 0u;
    runtime->checkpoint_state.use_anchor_mode = config->use_anchor_mode ? 1u : 0u;
    runtime->checkpoint_state.anchor_budget_ppm = config->anchor_budget_ppm;
    runtime->checkpoint_state.anchor_saliency_mode = (uint32_t)config->anchor_saliency_mode;
    runtime->checkpoint_state.next_layer_index = 0u;
    runtime->checkpoint_state.last_completed_layer = 0u;
    runtime->checkpoint_state.converted_tensor_count = 0u;
    copy_text_field_local(runtime->checkpoint_state.model_name, sizeof(runtime->checkpoint_state.model_name), config->model_name);
    copy_text_field_local(runtime->checkpoint_state.teacher_model_name, sizeof(runtime->checkpoint_state.teacher_model_name), config->teacher_model_name);
    copy_text_field_local(runtime->checkpoint_state.output_dir, sizeof(runtime->checkpoint_state.output_dir), config->output_path);
    copy_text_field_local(runtime->checkpoint_state.activation_tape_path, sizeof(runtime->checkpoint_state.activation_tape_path), config->activation_tape_path);
    copy_text_field_local(runtime->checkpoint_state.hessian_sidecar_path, sizeof(runtime->checkpoint_state.hessian_sidecar_path), config->hessian_sidecar_path);
    copy_text_field_local(runtime->checkpoint_state.calibration_corpus_path, sizeof(runtime->checkpoint_state.calibration_corpus_path), config->calibration_corpus_path);
    copy_text_field_local(runtime->checkpoint_state.calibration_corpus_manifest_path, sizeof(runtime->checkpoint_state.calibration_corpus_manifest_path), config->calibration_corpus_manifest_path);
    copy_text_field_local(runtime->checkpoint_state.validation_corpus_path, sizeof(runtime->checkpoint_state.validation_corpus_path), config->validation_corpus_path);
    copy_text_field_local(runtime->checkpoint_state.validation_corpus_manifest_path, sizeof(runtime->checkpoint_state.validation_corpus_manifest_path), config->validation_corpus_manifest_path);
}

static int validate_loaded_checkpoint_state(const ternary_conversion_config_t *config,
                                            const conversion_runtime_t *runtime,
                                            uint32_t total_layer_count,
                                            uint32_t config_hash)
{
    uint32_t expected_use_anchor_mode = 0u;

    if (runtime->checkpoint_state.total_layer_count != total_layer_count) {
        LOG_ERROR("student update: checkpoint total layer count mismatch (checkpoint=%u current=%u)",
                  runtime->checkpoint_state.total_layer_count,
                  total_layer_count);
        return -1;
    }
    if (runtime->checkpoint_state.schema_version != TERNARY_STUDENT_CHECKPOINT_VERSION) {
        LOG_ERROR("student update: checkpoint schema mismatch");
        return -1;
    }
    if (runtime->checkpoint_state.config_hash != config_hash) {
        LOG_ERROR("student update: checkpoint config hash mismatch");
        return -1;
    }
    expected_use_anchor_mode = config->use_anchor_mode ? 1u : 0u;
    if (runtime->checkpoint_state.use_anchor_mode != expected_use_anchor_mode) {
        LOG_ERROR("student update: checkpoint anchor-mode mismatch (checkpoint=%u current=%u)",
                  runtime->checkpoint_state.use_anchor_mode,
                  expected_use_anchor_mode);
        return -1;
    }
    if (expected_use_anchor_mode > 0u) {
        if (runtime->checkpoint_state.anchor_budget_ppm != config->anchor_budget_ppm) {
            LOG_ERROR("student update: checkpoint anchor budget mismatch (checkpoint=%u current=%u)",
                      runtime->checkpoint_state.anchor_budget_ppm,
                      config->anchor_budget_ppm);
            return -1;
        }
        if (runtime->checkpoint_state.anchor_saliency_mode != (uint32_t)config->anchor_saliency_mode) {
            LOG_ERROR("student update: checkpoint anchor saliency mismatch (checkpoint=%u current=%u)",
                      runtime->checkpoint_state.anchor_saliency_mode,
                      (uint32_t)config->anchor_saliency_mode);
            return -1;
        }
    }
    if (config->hessian_sidecar_path && config->hessian_sidecar_path[0] != '\0') {
        if (runtime->checkpoint_state.hessian_sidecar_path[0] == '\0' ||
            strcmp(runtime->checkpoint_state.hessian_sidecar_path, config->hessian_sidecar_path) != 0) {
            LOG_ERROR("student update: checkpoint Hessian sidecar path mismatch");
            return -1;
        }
        if (runtime->checkpoint_state.hessian_sidecar_crc32 != runtime->hessian_sidecar_crc32) {
            LOG_ERROR("student update: checkpoint Hessian sidecar checksum mismatch");
            return -1;
        }
    }

    return 0;
}

static int restore_checkpoint_alignment_paths(conversion_runtime_t *runtime)
{
    char *alignment_manifest_path = NULL;
    char *alignment_tape_path = NULL;

    if (!runtime) {
        return -1;
    }

    if (runtime->checkpoint_state.alignment_manifest_path[0] != '\0') {
        alignment_manifest_path = duplicate_text_local(runtime->checkpoint_state.alignment_manifest_path);
        if (!alignment_manifest_path) {
            goto fail;
        }
        free(runtime->alignment_manifest_path);
        runtime->alignment_manifest_path = alignment_manifest_path;
        alignment_manifest_path = NULL;
    }
    if (runtime->checkpoint_state.alignment_tape_path[0] != '\0') {
        alignment_tape_path = duplicate_text_local(runtime->checkpoint_state.alignment_tape_path);
        if (!alignment_tape_path) {
            goto fail;
        }
        free(runtime->alignment_tape_path);
        runtime->alignment_tape_path = alignment_tape_path;
        alignment_tape_path = NULL;
    }

    return 0;

fail:
    free(alignment_manifest_path);
    free(alignment_tape_path);
    return -1;
}

static int init_checkpoint_state(const ternary_conversion_config_t *config,
                                 conversion_runtime_t *runtime)
{
    int load_rc = 0;
    uint32_t config_hash = 0u;
    uint32_t total_layer_count = 0u;

    if (!config || !runtime) {
        return -1;
    }
    if (!runtime->model_spec) {
        return -1;
    }

    if (init_checkpoint_paths(config, runtime) != 0) {
        return -1;
    }

    total_layer_count = conversion_total_progress_steps(config, runtime->model_spec);
    config_hash = config_resume_hash(config,
                                     runtime->hessian_sidecar ? runtime->hessian_sidecar_crc32 : 0u,
                                     runtime->structural_map_crc32);
    seed_checkpoint_state_defaults(config, runtime, total_layer_count, config_hash);

    load_rc = ternary_student_checkpoint_load(runtime->checkpoint_path, &runtime->checkpoint_state);
    if (load_rc == 1) {
        LOG_INFO("student update: no checkpoint found at %s; starting fresh", runtime->checkpoint_path);
        if (manifest_path_exists(config->output_path) > 0) {
            LOG_ERROR("student update: output directory %s already contains a manifest but no checkpoint", config->output_path);
            return -1;
        }
        return 0;
    }
    if (load_rc != 0) {
        return -1;
    }
    if (validate_loaded_checkpoint_state(config, runtime, total_layer_count, config_hash) != 0) {
        return -1;
    }
    if (restore_checkpoint_alignment_paths(runtime) != 0) {
        return -1;
    }

    if (ternary_student_checkpoint_validate_alignment(&runtime->checkpoint_state, runtime->activation_tape) != 0) {
        return -1;
    }

    return 0;
}

static int write_student_checkpoint(const ternary_conversion_config_t *config,
                                    conversion_runtime_t *runtime,
                                    const char *alignment_manifest_path,
                                    const char *alignment_tape_path)
{
    ternary_student_update_checkpoint_t checkpoint;
    const char *effective_alignment_manifest_path = NULL;
    const char *effective_alignment_tape_path = NULL;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "write_student_checkpoint");

    if (!config || !runtime || !runtime->checkpoint_path) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, runtime->checkpoint_path);

    effective_alignment_manifest_path = runtime->alignment_manifest_path
        ? runtime->alignment_manifest_path
        : alignment_manifest_path;
    effective_alignment_tape_path = runtime->alignment_tape_path
        ? runtime->alignment_tape_path
        : alignment_tape_path;

    checkpoint = runtime->checkpoint_state;
    copy_text_field_local(checkpoint.alignment_manifest_path,
                          sizeof(checkpoint.alignment_manifest_path),
                          effective_alignment_manifest_path);
    copy_text_field_local(checkpoint.alignment_tape_path,
                          sizeof(checkpoint.alignment_tape_path),
                          effective_alignment_tape_path);
    copy_text_field_local(checkpoint.hessian_sidecar_path,
                          sizeof(checkpoint.hessian_sidecar_path),
                          config->hessian_sidecar_path);
    checkpoint.alignment_manifest_crc32 = 0u;
    checkpoint.alignment_tape_provenance_hash = 0u;
    checkpoint.hessian_sidecar_crc32 = runtime->hessian_sidecar_crc32;
    checkpoint.use_anchor_mode = config->use_anchor_mode ? 1u : 0u;
    checkpoint.anchor_budget_ppm = config->anchor_budget_ppm;
    checkpoint.anchor_saliency_mode = (uint32_t)config->anchor_saliency_mode;

    if (effective_alignment_manifest_path && effective_alignment_manifest_path[0] != '\0') {
        if (ternary_student_checkpoint_compute_manifest_crc32(effective_alignment_manifest_path,
                                                              &checkpoint.alignment_manifest_crc32) != 0) {
            return conversion_tracy_end_status(&tracy_zone, -1);
        }
    }
    if (effective_alignment_tape_path && effective_alignment_tape_path[0] != '\0' && runtime->activation_tape) {
        if (ternary_student_checkpoint_compute_tape_provenance_hash(&checkpoint,
                                                                    runtime->activation_tape,
                                                                    &checkpoint.alignment_tape_provenance_hash) != 0) {
            return conversion_tracy_end_status(&tracy_zone, -1);
        }
    }

    int rc = ternary_student_checkpoint_write_atomic(runtime->checkpoint_path, &checkpoint);
    if (rc == 0) {
        conversion_tracy_plot_checkpoint(&checkpoint);
    }

    return conversion_tracy_end_status(&tracy_zone, rc);
}

static int spec_copy_layer_prefix_local(const model_spec_t *spec,
                                        char *out_prefix,
                                        size_t out_prefix_size)
{
    const char *layer_marker = ".layers.";
    size_t layer_marker_len = strlen(layer_marker);

    if (!spec || !spec->tensor_map || !out_prefix || out_prefix_size == 0u) {
        return -1;
    }

    out_prefix[0] = '\0';
    for (int idx = 0; spec->tensor_map[idx].hf_name; ++idx) {
        const char *name = spec->tensor_map[idx].hf_name;
        const char *layer_pos = name ? strstr(name, layer_marker) : NULL;
        size_t prefix_len = 0u;

        if (!layer_pos) {
            continue;
        }

        prefix_len = (size_t)((layer_pos - name) + layer_marker_len);
        if (prefix_len == 0u || prefix_len >= out_prefix_size) {
            return -1;
        }

        memcpy(out_prefix, name, prefix_len);
        out_prefix[prefix_len] = '\0';
        return 0;
    }

    return -1;
}

static int activation_tape_matches_student_layout(const activation_tape_t *tape,
                                                  const model_spec_t *spec)
{
    const gemma3_270m_config_t *cfg = NULL;
    const tape_file_header_t *header = NULL;
    const tape_manifest_entry_t *entry = NULL;
    char expected_prefix[128];

    if (!tape || !spec || !spec->variant_config) {
        return 0;
    }

    cfg = (const gemma3_270m_config_t *)spec->variant_config;
    if (cfg->num_hidden_layers <= 0 || cfg->hidden_size <= 0) {
        return 0;
    }

    header = activation_tape_header(tape);
    if (!header) {
        return 0;
    }
    if (activation_tape_entry_count(tape) !=
        (uint32_t)cfg->num_hidden_layers * TAPE_TENSORS_PER_LAYER) {
        return 0;
    }
    if (header->hidden_size != (uint32_t)cfg->hidden_size) {
        return 0;
    }

    if (spec_copy_layer_prefix_local(spec, expected_prefix, sizeof(expected_prefix)) != 0) {
        return 0;
    }

    entry = activation_tape_entry(tape, 0u);
    if (!entry) {
        return 0;
    }

    return strncmp(entry->tensor_name,
                   expected_prefix,
                   strlen(expected_prefix)) == 0;
}

static char *derive_alignment_manifest_path_from_tape(const char *alignment_tape_path)
{
    struct stat st;
    char *manifest_path = NULL;
    size_t path_len = 0u;

    if (!alignment_tape_path) {
        return NULL;
    }

    path_len = strlen(alignment_tape_path);
    if (path_len <= 5u || strcmp(alignment_tape_path + (path_len - 5u), ".tape") != 0) {
        return NULL;
    }

    manifest_path = duplicate_text_local(alignment_tape_path);
    if (!manifest_path) {
        return NULL;
    }

    strcpy(manifest_path + (path_len - 5u), ".tsv");
    if (stat(manifest_path, &st) != 0 || !S_ISREG(st.st_mode)) {
        free(manifest_path);
        return NULL;
    }

    return manifest_path;
}

static int reuse_supplied_aligned_tape(const ternary_conversion_config_t *config,
                                       conversion_runtime_t *runtime)
{
    char *alignment_tape_path = NULL;
    char *alignment_manifest_path = NULL;

    if (!config || !runtime || !runtime->activation_tape || !runtime->model_spec ||
        !config->activation_tape_path || config->activation_tape_path[0] == '\0') {
        return 0;
    }

    if (!activation_tape_matches_student_layout(runtime->activation_tape, runtime->model_spec)) {
        return 0;
    }

    alignment_tape_path = duplicate_text_local(config->activation_tape_path);
    alignment_manifest_path = derive_alignment_manifest_path_from_tape(config->activation_tape_path);
    if (!alignment_tape_path || !alignment_manifest_path) {
        LOG_ERROR("ternary alignment: supplied activation tape already matches the student layout, but the companion alignment manifest is missing");
        free(alignment_tape_path);
        free(alignment_manifest_path);
        return -1;
    }

    free(runtime->alignment_manifest_path);
    runtime->alignment_manifest_path = alignment_manifest_path;
    alignment_manifest_path = NULL;
    free(runtime->alignment_tape_path);
    runtime->alignment_tape_path = alignment_tape_path;
    alignment_tape_path = NULL;

    LOG_INFO("ternary alignment: reusing supplied student-aligned tape %s",
             runtime->alignment_tape_path);
    LOG_INFO("ternary alignment: manifest=%s", runtime->alignment_manifest_path);
    return 1;
}

static int prepare_teacher_student_alignment(const ternary_conversion_config_t *config,
                                             conversion_runtime_t *runtime)
{
    activation_alignment_request_t request;
    char *aligned_tape_path = NULL;
    char *aligned_manifest_path = NULL;
    activation_tape_t *aligned_tape = NULL;
    const model_spec_t *teacher_spec = NULL;
    const activation_tape_t *teacher_tape = NULL;
    int reuse_status = 0;
    int rc = -1;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "prepare_teacher_student_alignment");

    if (!config || !runtime) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, config->teacher_model_name);

    if (!config->teacher_model_name || config->teacher_model_name[0] == '\0') {
        return conversion_tracy_end_status(&tracy_zone, 0);
    }

    if (config->layer_name && config->layer_name[0] != '\0') {
        LOG_ERROR("ternary alignment: --teacher-model requires full-model conversion");
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    if (!runtime->activation_tape) {
        LOG_ERROR("ternary alignment: --teacher-model requires --activation-tape");
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    reuse_status = reuse_supplied_aligned_tape(config, runtime);
    if (reuse_status < 0) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    if (reuse_status > 0) {
        return conversion_tracy_end_status(&tracy_zone, 0);
    }

    teacher_tape = runtime->activation_tape;
    teacher_spec = get_model_spec(config->teacher_model_name);
    if (!teacher_spec) {
        LOG_ERROR("ternary alignment: failed to resolve teacher spec for %s", config->teacher_model_name);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    memset(&request, 0, sizeof(request));
    request.teacher_spec = teacher_spec;
    request.student_spec = runtime->model_spec;
    request.teacher_tape = teacher_tape;
    request.structural_map_path = config->structural_map_path;
    request.depth_strategy = ACTIVATION_ALIGNMENT_DEPTH_BUCKET;
    request.width_strategy = ACTIVATION_ALIGNMENT_WIDTH_AUTO;

    if (activation_alignment_prepare_artifacts(&request,
                                              config->output_path,
                                              &aligned_tape_path,
                                              &aligned_manifest_path) != 0) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    aligned_tape = activation_tape_open(aligned_tape_path);
    if (!aligned_tape) {
        LOG_ERROR("ternary alignment: failed to open aligned tape %s", aligned_tape_path);
        goto alignment_cleanup;
    }

    LOG_INFO("ternary alignment: manifest=%s", aligned_manifest_path);
    LOG_INFO("ternary alignment: tape=%s", aligned_tape_path);

    free(runtime->alignment_manifest_path);
    runtime->alignment_manifest_path = duplicate_text_local(aligned_manifest_path);
    if (!runtime->alignment_manifest_path) {
        goto alignment_cleanup;
    }
    free(runtime->alignment_tape_path);
    runtime->alignment_tape_path = duplicate_text_local(aligned_tape_path);
    if (!runtime->alignment_tape_path) {
        goto alignment_cleanup;
    }

    activation_tape_close(runtime->activation_tape);
    runtime->activation_tape = aligned_tape;
    aligned_tape = NULL;
    rc = 0;

alignment_cleanup:
    free(aligned_tape_path);
    free(aligned_manifest_path);
    if (aligned_tape) {
        activation_tape_close(aligned_tape);
    }
    return conversion_tracy_end_status(&tracy_zone, rc);
}

static int load_runtime_corpus(const char *manifest_path,
                               const char *source_path,
                               int sample_limit,
                               calibration_corpus_t *storage) {
    if (!storage) {
        return -1;
    }
    if (manifest_path && manifest_path[0] != '\0') {
        return calibration_corpus_load_manifest(manifest_path, sample_limit, storage);
    }
    if (source_path && source_path[0] != '\0') {
        return calibration_corpus_load(source_path, sample_limit, storage);
    }
    return 0;
}

static int positive_int_or_default(int value, int default_value)
{
    return value > 0 ? value : default_value;
}

static int nonnegative_int_or_default(int value, int default_value)
{
    return value >= 0 ? value : default_value;
}

static float nonnegative_float_or_default(float value, float default_value)
{
    return value >= 0.0f ? value : default_value;
}

static int normalize_flag(int value)
{
    return value ? 1 : 0;
}

static transformer_ste_config_t default_runtime_ste_config(const ternary_conversion_config_t *config) {
    transformer_ste_config_t ste_config;
    static const ternary_conversion_config_t zero_config = {0};
    const ternary_conversion_config_t *cfg = config ? config : &zero_config;

    ste_config.ste_steps = positive_int_or_default(cfg->ste_steps, 3);
    ste_config.learning_rate = cfg->ste_learning_rate > 0.0f ? cfg->ste_learning_rate : 0.03f;
    ste_config.zero_threshold = 0.05f;
    ste_config.momentum = 0.85f;
    ste_config.regularization_strength = 0.01f;
    ste_config.non_collapse_weight = 0.02f;
    ste_config.zero_occupancy_floor = 0.75f;
    ste_config.clip_value = 1.0f;
    ste_config.calibration_samples = positive_int_or_default(cfg->calibration_sample_limit, 4);
    ste_config.kl_weight = nonnegative_float_or_default(cfg->kl_weight, 0.05f);
    ste_config.kl_temperature = 1.0f;
    ste_config.kl_update_interval = positive_int_or_default(cfg->kl_update_interval, 4);
    ste_config.kl_sample_count = positive_int_or_default(cfg->kl_sample_count, 4);
    ste_config.early_stop_patience = nonnegative_int_or_default(cfg->early_stop_patience, 6);
    ste_config.early_stop_min_delta = nonnegative_float_or_default(cfg->early_stop_min_delta, 1e-4f);
    ste_config.early_stop_divergence_ratio =
        nonnegative_float_or_default(cfg->early_stop_divergence_ratio, 1.25f);
    ste_config.simulate_activation_a8 = 0;
    ste_config.use_hessian_proxy = cfg->disable_hessian_proxy ? 0 : 1;
    ste_config.hessian_proxy_strength = nonnegative_float_or_default(cfg->hessian_proxy_strength, 1.0f);
    ste_config.hessian_proxy_floor = nonnegative_float_or_default(cfg->hessian_proxy_floor, 0.05f);
    ste_config.max_grad_norm = cfg->max_grad_norm > 0.0f ? cfg->max_grad_norm : 1.0f;
    ste_config.adam_beta2 = 0.95f;
    ste_config.adam_epsilon = 1e-8f;
    ste_config.telemetry_interval = 10;
    ste_config.telemetry_path = "./out/ternary_telemetry.jsonl";
    ste_config.spatial_telemetry_path = NULL;
    ste_config.spatial_telemetry_row_bucket_size = positive_int_or_default(cfg->spatial_telemetry_row_bucket_size, 64);
    ste_config.hessian_sidecar_path = cfg->hessian_sidecar_path;
    ste_config.hessian_sidecar_crc32 = 0u;
    ste_config.student_down_proj_input_rmsnorm = normalize_flag(cfg->student_down_proj_input_rmsnorm);
    ste_config.telemetry = NULL;
    ste_config.use_anchor_mode = normalize_flag(cfg->use_anchor_mode);
    ste_config.anchor_budget_ppm = cfg->anchor_budget_ppm;
    ste_config.anchor_saliency_mode = cfg->anchor_saliency_mode;
    ste_config.anchor_learning_rate_mult = 1.0f;
    ste_config.protected_anchor_entries = NULL;
    ste_config.protected_anchor_row_offsets = NULL;
    ste_config.protected_anchor_count = 0u;
    ste_config.telemetry_reference_weights = NULL;
    return ste_config;
}

static int tensor_name_is_layer_tensor(const char *tensor_name)
{
    return tensor_name && strstr(tensor_name, ".layers.") != NULL;
}

static int tensor_name_has_suffix(const char *tensor_name, const char *suffix)
{
    size_t tensor_len = 0u;
    size_t suffix_len = 0u;

    if (!tensor_name || !suffix) {
        return 0;
    }

    tensor_len = strlen(tensor_name);
    suffix_len = strlen(suffix);
    if (tensor_len < suffix_len) {
        return 0;
    }

    return strcmp(tensor_name + (tensor_len - suffix_len), suffix) == 0;
}

static int tensor_name_is_tied_embedding_tensor(const char *tensor_name)
{
    return tensor_name_has_suffix(tensor_name, "embed_tokens.weight");
}

static int tensor_name_is_tied_output_tensor(const char *tensor_name)
{
    return tensor_name_has_suffix(tensor_name, "lm_head.weight");
}

static const structural_tensor_rule_t *runtime_structural_rule_for_tensor(const conversion_runtime_t *runtime,
                                                                          const char *tensor_name)
{
    if (!runtime || !tensor_name) {
        return NULL;
    }

    return structural_map_find_rule(&runtime->structural_map, tensor_name);
}

static int structural_rule_is_pass_through(const structural_tensor_rule_t *rule)
{
    return rule && rule->quant_mode == STRUCTURAL_TENSOR_QUANT_MODE_PASS;
}

static int structural_rule_is_mold(const structural_tensor_rule_t *rule)
{
    return rule && rule->quant_mode == STRUCTURAL_TENSOR_QUANT_MODE_MOLD;
}

/**
 * @brief Extract num_hidden_layers from the model spec's variant config.
 *
 * @return The number of hidden layers, or -1 if not available.
 */
static int spec_get_num_hidden_layers(const model_spec_t *spec)
{
    const gemma3_270m_config_t *cfg = NULL;

    if (!spec || !spec->variant_config) {
        return -1;
    }

    cfg = (const gemma3_270m_config_t *)spec->variant_config;
    if (cfg->num_hidden_layers <= 0) {
        return -1;
    }

    return cfg->num_hidden_layers;
}

/**
 * @brief Validate that the activation tape's layer count matches the model's expected layers.
 *
 * This prevents buffer overflows when processing tapes with more layers than the
 * student model config expects.
 *
 * @param tape The activation tape to validate.
 * @param spec The model spec with the expected layer count.
 * @return 0 if valid, -1 if there's a mismatch.
 */
static int validate_tape_layer_count(const activation_tape_t *tape,
                                     const model_spec_t *spec)
{
    int expected_layers = 0;
    uint32_t tape_entry_count = 0u;
    int tape_layer_count = 0;

    if (!tape || !spec) {
        /* No tape to validate, or no spec to compare against */
        return 0;
    }

    expected_layers = spec_get_num_hidden_layers(spec);
    if (expected_layers <= 0) {
        LOG_WARN("ternary conversion: could not determine expected layer count from model spec");
        return 0;
    }

    tape_entry_count = activation_tape_entry_count(tape);
    if (tape_entry_count == 0u) {
        LOG_ERROR("ternary conversion: activation tape has no entries");
        return -1;
    }

    /* Each layer has TAPE_TENSORS_PER_LAYER entries */
    if ((tape_entry_count % TAPE_TENSORS_PER_LAYER) != 0u) {
        LOG_ERROR("ternary conversion: tape entry count (%u) is not a multiple of %d tensors per layer",
                  tape_entry_count, TAPE_TENSORS_PER_LAYER);
        return -1;
    }

    tape_layer_count = (int)(tape_entry_count / TAPE_TENSORS_PER_LAYER);

    if (tape_layer_count != expected_layers) {
        LOG_ERROR("ternary conversion: ARCHITECTURE MISMATCH - tape contains %d layers but model expects %d layers",
                  tape_layer_count, expected_layers);
        LOG_ERROR("ternary conversion: ensure the activation tape was expanded with the correct structural map");
        LOG_ERROR("ternary conversion: for 1B teacher -> 7B student, use: --structural-map configs/manifests/gemma3_7b_structural.tsv");
        return -1;
    }

    LOG_INFO("ternary conversion: tape layer count (%d) matches model config (%d layers)",
             tape_layer_count, expected_layers);
    return 0;
}

static void prefetch_activation_tape_lookahead(const conversion_runtime_t *runtime,
                                               const model_spec_t *spec,
                                               int start_idx,
                                               int lookahead_count,
                                               int *last_prefetched_entry_idx,
                                               float *out_io_ms)
{
    int prefetched = 0;

    if (!runtime || !runtime->activation_tape || !spec || lookahead_count <= 0) {
        return;
    }

    for (int i = start_idx; i < spec->tensor_map_size && prefetched < lookahead_count; ++i) {
        const char *tensor_name = spec->tensor_map[i].hf_name;
        int entry_idx = -1;

        if (!tensor_name || !tensor_name_is_layer_tensor(tensor_name)) {
            continue;
        }

        entry_idx = activation_tape_entry_index(runtime->activation_tape, tensor_name);
        if (entry_idx < 0) {
            continue;
        }
        if (last_prefetched_entry_idx && *last_prefetched_entry_idx == entry_idx) {
            continue;
        }

        if (out_io_ms) {
            struct timespec prefetch_start;
            struct timespec prefetch_end;

            if (clock_gettime(CLOCK_MONOTONIC, &prefetch_start) == 0) {
                activation_tape_prefetch_entry(runtime->activation_tape, (uint32_t)entry_idx);
                if (clock_gettime(CLOCK_MONOTONIC, &prefetch_end) == 0) {
                    *out_io_ms += telemetry_elapsed_ms(&prefetch_start, &prefetch_end);
                }
            } else {
                activation_tape_prefetch_entry(runtime->activation_tape, (uint32_t)entry_idx);
            }
        } else {
            activation_tape_prefetch_entry(runtime->activation_tape, (uint32_t)entry_idx);
        }
        if (last_prefetched_entry_idx) {
            *last_prefetched_entry_idx = entry_idx;
        }
        prefetched++;
    }
}

static void destroy_conversion_runtime(conversion_runtime_t *runtime) {
    if (!runtime) {
        return;
    }

    if (runtime->hessian_sidecar) {
        ternary_hessian_sidecar_close(runtime->hessian_sidecar);
    }
    if (runtime->activation_tape) {
        activation_tape_close(runtime->activation_tape);
    }
    io_release_bf16_io_cache(&runtime->bf16_io_cache);
    ternary_hessian_proxy_cache_release(&runtime->hessian_proxy_cache);
    structural_map_release(&runtime->structural_map);
    free(runtime->checkpoint_path);
    free(runtime->checkpoint_tmp_path);
    runtime->checkpoint_path = NULL;
    runtime->checkpoint_tmp_path = NULL;
    free(runtime->alignment_manifest_path);
    free(runtime->alignment_tape_path);
    if (runtime->activation_ctx) {
        ternary_validation_destroy(&runtime->validation_state);
        destroy_inference_context(runtime->activation_ctx);
    } else {
        tokenizer_free(runtime->tokenizer);
        if (runtime->model_spec) {
            runtime->model_spec->tokenizer_handle = runtime->previous_tokenizer_handle;
        }
    }
    calibration_corpus_free(&runtime->corpus_storage);
    calibration_corpus_free(&runtime->validation_corpus_storage);
    memset(runtime, 0, sizeof(*runtime));
}

static void init_conversion_validation(const ternary_conversion_config_t *config,
                                       conversion_runtime_t *runtime) {
    ternary_validation_config_t validation_config;
    const char *validation_manifest = NULL;
    const char *validation_source = NULL;
    int validation_samples = 0;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "init_conversion_validation");

    if (!config || !runtime || !runtime->activation_ctx) {
        sapphire_tracy_zone_end(&tracy_zone);
        return;
    }
    if (config->validate_every_n <= 0) {
        sapphire_tracy_zone_end(&tracy_zone);
        return;
    }

    validation_source = config->validation_corpus_path;
    if (config->validation_corpus_manifest_path && config->validation_corpus_manifest_path[0] != '\0') {
        validation_manifest = config->validation_corpus_manifest_path;
    }
    if ((!validation_manifest || validation_manifest[0] == '\0') &&
        (!validation_source || validation_source[0] == '\0') &&
        config->calibration_corpus_manifest_path && config->calibration_corpus_manifest_path[0] != '\0') {
        validation_manifest = config->calibration_corpus_manifest_path;
    }
    if ((!validation_manifest || validation_manifest[0] == '\0') &&
        (!validation_source || validation_source[0] == '\0')) {
        validation_source = config->calibration_corpus_path;
    }
    validation_samples = (config->validation_sample_limit > 0)
        ? config->validation_sample_limit
        : ((config->calibration_sample_limit > 0) ? config->calibration_sample_limit : 4);
    sapphire_tracy_plot_i64("validation.sample_count", validation_samples);

    if ((!validation_manifest || validation_manifest[0] == '\0') &&
        (!validation_source || validation_source[0] == '\0')) {
        LOG_WARN("ternary validation: disabled because no validation corpus was provided");
        sapphire_tracy_zone_end(&tracy_zone);
        return;
    }
    if (load_runtime_corpus(validation_manifest,
                            validation_source,
                            validation_samples,
                            &runtime->validation_corpus_storage) != 0) {
        LOG_WARN("ternary validation: failed to load held-out corpus; disabling checkpoints");
        memset(&runtime->validation_corpus_storage, 0, sizeof(runtime->validation_corpus_storage));
        sapphire_tracy_zone_end(&tracy_zone);
        return;
    }

    memset(&validation_config, 0, sizeof(validation_config));
    validation_config.validate_every_n = config->validate_every_n;
    validation_config.student_down_proj_input_rmsnorm = config->student_down_proj_input_rmsnorm ? 1 : 0;
    validation_config.output_dir = config->output_path;
    validation_config.telemetry_path = conversion_runtime_telemetry_path(config, runtime);
    validation_config.sample_texts = (const char *const *)runtime->validation_corpus_storage.samples;
    validation_config.sample_count = runtime->validation_corpus_storage.sample_count;

    if (ternary_validation_init(&runtime->validation_state,
                                &validation_config,
                                runtime->activation_ctx) != 0) {
        LOG_WARN("ternary validation: failed to initialize checkpoint state; disabling checkpoints");
        ternary_validation_destroy(&runtime->validation_state);
    }

    sapphire_tracy_zone_end(&tracy_zone);
}

static int open_runtime_activation_tape(const ternary_conversion_config_t *config,
                                        conversion_runtime_t *runtime)
{
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "open_activation_tape");
    if (!config || !runtime) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, config->activation_tape_path);

    if (!config->activation_tape_path || config->activation_tape_path[0] == '\0') {
        return conversion_tracy_end_status(&tracy_zone, 0);
    }

    runtime->activation_tape = activation_tape_open(config->activation_tape_path);
    if (!runtime->activation_tape) {
        LOG_ERROR("ternary conversion: failed to open activation tape %s", config->activation_tape_path);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    runtime->activation_tape_hash = activation_tape_crc32(runtime->activation_tape);
    return conversion_tracy_end_status(&tracy_zone, 0);
}

static int open_runtime_hessian_sidecar(const ternary_conversion_config_t *config,
                                        conversion_runtime_t *runtime)
{
    const char *sidecar_teacher_name = NULL;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "open_hessian_sidecar");

    if (!config || !runtime) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, config->hessian_sidecar_path);

    if (!config->hessian_sidecar_path || config->hessian_sidecar_path[0] == '\0') {
        return conversion_tracy_end_status(&tracy_zone, 0);
    }
    if (!runtime->activation_tape) {
        LOG_ERROR("ternary conversion: --hessian-sidecar requires --activation-tape");
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    runtime->hessian_sidecar = ternary_hessian_sidecar_open(config->hessian_sidecar_path);
    if (!runtime->hessian_sidecar) {
        LOG_ERROR("ternary conversion: failed to open Hessian sidecar %s", config->hessian_sidecar_path);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    runtime->hessian_sidecar_crc32 = ternary_hessian_sidecar_crc32(runtime->hessian_sidecar);
    if (activation_tape_crc32(runtime->activation_tape) != ternary_hessian_sidecar_tape_crc32(runtime->hessian_sidecar)) {
        LOG_ERROR("ternary conversion: Hessian sidecar tape CRC mismatch for %s", config->hessian_sidecar_path);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    sidecar_teacher_name = ternary_hessian_sidecar_teacher_model_name(runtime->hessian_sidecar);
    if (config->teacher_model_name && config->teacher_model_name[0] != '\0' &&
        sidecar_teacher_name && strcmp(config->teacher_model_name, sidecar_teacher_name) != 0) {
        LOG_ERROR("ternary conversion: Hessian sidecar teacher model mismatch (config=%s sidecar=%s)",
                  config->teacher_model_name,
                  sidecar_teacher_name);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    return conversion_tracy_end_status(&tracy_zone, 0);
}

static void attach_runtime_activation_context(conversion_runtime_t *runtime)
{
    runtime->model_spec = runtime->activation_ctx->spec;
    runtime->tokenizer = runtime->activation_ctx->tokenizer;
    runtime->calibration_corpus.tokenizer = runtime->activation_ctx->tokenizer;
    runtime->calibration_corpus.model_spec = runtime->activation_ctx->spec;
    runtime->calibration_corpus.session = runtime->activation_ctx->session;
}

static int attach_runtime_tokenizer_fallback(const char *model_dir,
                                             model_spec_t *spec,
                                             conversion_runtime_t *runtime)
{
    runtime->tokenizer = tokenizer_load(model_dir);
    if (!runtime->tokenizer) {
        LOG_ERROR("ternary conversion: failed to load tokenizer from %s", model_dir);
        return -1;
    }

    runtime->model_spec = spec;
    runtime->previous_tokenizer_handle = spec->tokenizer_handle;
    spec->tokenizer_handle = runtime->tokenizer;
    runtime->calibration_corpus.tokenizer = runtime->tokenizer;
    runtime->calibration_corpus.model_spec = spec;
    runtime->calibration_corpus.session = NULL;
    return 0;
}

static int load_runtime_corpus_if_needed(const ternary_conversion_config_t *config,
                                         conversion_runtime_t *runtime)
{
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "load_calibration_corpus");
    if (!config || !runtime) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    if ((!config->calibration_corpus_manifest_path || config->calibration_corpus_manifest_path[0] == '\0') &&
        (!config->calibration_corpus_path || config->calibration_corpus_path[0] == '\0')) {
        return conversion_tracy_end_status(&tracy_zone, 0);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone,
                                          config->calibration_corpus_manifest_path ? config->calibration_corpus_manifest_path : config->calibration_corpus_path);

    if (load_runtime_corpus(config->calibration_corpus_manifest_path,
                            config->calibration_corpus_path,
                            config->calibration_sample_limit,
                            &runtime->corpus_storage) != 0) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    runtime->calibration_corpus.sample_texts = (const char *const *)runtime->corpus_storage.samples;
    runtime->calibration_corpus.sample_count = runtime->corpus_storage.sample_count;
    return conversion_tracy_end_status(&tracy_zone, 0);
}

static int prepare_conversion_output_path(const ternary_conversion_config_t *config)
{
    const char *slash = NULL;
    char *parent_dir = NULL;
    size_t parent_len = 0u;
    int rc = 0;

    if (!config || !config->output_path || config->output_path[0] == '\0') {
        return -1;
    }
    if (config->use_anchor_mode && config->layer_name && config->layer_name[0] != '\0') {
        return io_prepare_ternary_output_dir(config->output_path);
    }
    if (!config->layer_name || config->layer_name[0] == '\0') {
        return io_prepare_ternary_output_dir(config->output_path);
    }

    slash = strrchr(config->output_path, '/');
    if (!slash) {
        return 0;
    }

    parent_len = (size_t)(slash - config->output_path);
    if (parent_len == 0u) {
        return 0;
    }

    parent_dir = (char *)malloc(parent_len + 1u);
    if (!parent_dir) {
        LOG_ERROR("ternary conversion: failed to allocate output parent directory path");
        return -1;
    }

    memcpy(parent_dir, config->output_path, parent_len);
    parent_dir[parent_len] = '\0';
    rc = io_prepare_ternary_output_dir(parent_dir);
    free(parent_dir);
    return rc;
}

static int finalize_conversion_runtime_state(const ternary_conversion_config_t *config,
                                             conversion_runtime_t *runtime)
{
    if (config && config->layer_name && config->layer_name[0] != '\0') {
        runtime->resume_step_index = 0u;
        return 0;
    }

    if (prepare_teacher_student_alignment(config, runtime) != 0) {
        return -1;
    }
    if (init_checkpoint_state(config, runtime) != 0) {
        return -1;
    }

    runtime->resume_step_index = runtime->checkpoint_state.next_layer_index;
    init_conversion_validation(config, runtime);
    if (resume_validation_from_manifest(config, runtime) != 0) {
        return -1;
    }

    return 0;
}

static int init_conversion_runtime(const ternary_conversion_config_t *config,
                                   const char *model_dir,
                                   conversion_runtime_t *out_runtime) {
    model_spec_t *spec = NULL;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "init_conversion_runtime");

    if (!config || !model_dir || !out_runtime) {
        LOG_ERROR("init_conversion_runtime: invalid arguments");
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, config->model_name);

    memset(out_runtime, 0, sizeof(*out_runtime));
    if (config->structural_map_path && config->structural_map_path[0] != '\0') {
        if (ternary_student_checkpoint_compute_file_crc32(config->structural_map_path,
                                                          &out_runtime->structural_map_crc32) != 0 ||
            structural_map_load(config->structural_map_path, &out_runtime->structural_map) != 0) {
            destroy_conversion_runtime(out_runtime);
            return conversion_tracy_end_status(&tracy_zone, -1);
        }
    }

    spec = get_model_spec(config->model_name);
    if (!spec) {
        LOG_ERROR("ternary conversion: failed to resolve model spec for %s", config->model_name);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    if (config->output_path && config->output_path[0] != '\0' &&
        prepare_conversion_output_path(config) != 0) {
        LOG_ERROR("ternary conversion: failed to prepare output path %s", config->output_path);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    if (write_conversion_config_overrides(config) != 0) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    if (build_runtime_telemetry_path(default_runtime_ste_config(config).telemetry_path,
                                     out_runtime->telemetry_path,
                                     sizeof(out_runtime->telemetry_path)) != 0) {
        LOG_WARN("ternary conversion: failed to timestamp telemetry path; using default %s",
                 default_runtime_ste_config(config).telemetry_path);
        snprintf(out_runtime->telemetry_path,
                 sizeof(out_runtime->telemetry_path),
                 "%s",
                 default_runtime_ste_config(config).telemetry_path);
    }
    if (config->emit_spatial_telemetry &&
        build_runtime_spatial_telemetry_path(out_runtime->telemetry_path,
                                             out_runtime->spatial_telemetry_path,
                                             sizeof(out_runtime->spatial_telemetry_path)) != 0) {
        LOG_WARN("ternary conversion: failed to derive spatial telemetry path from %s",
                 out_runtime->telemetry_path);
        out_runtime->spatial_telemetry_path[0] = '\0';
    }

    if (open_runtime_activation_tape(config, out_runtime) != 0 ||
        open_runtime_hessian_sidecar(config, out_runtime) != 0) {
        destroy_conversion_runtime(out_runtime);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    out_runtime->activation_ctx = create_inference_context(0.0f,
                                                           0,
                                                           config->context_len > 0 ? config->context_len : 2048,
                                                           config->model_name);
    if (out_runtime->activation_ctx) {
        if (config->student_down_proj_input_rmsnorm && out_runtime->activation_ctx->session) {
            inference_session_set_ffn_down_proj_input_rmsnorm_default(out_runtime->activation_ctx->session, 1);
        }
        attach_runtime_activation_context(out_runtime);
    } else {
        LOG_WARN("ternary conversion: failed to initialize activation replay context; using tokenized fallback vectors");
        if (attach_runtime_tokenizer_fallback(model_dir, spec, out_runtime) != 0) {
            destroy_conversion_runtime(out_runtime);
            return conversion_tracy_end_status(&tracy_zone, -1);
        }
        if ((!config->calibration_corpus_manifest_path || config->calibration_corpus_manifest_path[0] == '\0') &&
            (!config->calibration_corpus_path || config->calibration_corpus_path[0] == '\0')) {
        LOG_WARN("ternary conversion: no calibration corpus provided; using built-in fallback prompts");
        }
    }

    if (load_runtime_corpus_if_needed(config, out_runtime) != 0 ||
        finalize_conversion_runtime_state(config, out_runtime) != 0) {
        destroy_conversion_runtime(out_runtime);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    return conversion_tracy_end_status(&tracy_zone, 0);
}

static int mmap_tensor_for_conversion(const char *model_dir,
                                      const char *model_path,
                                      ternary_bf16_io_cache_t *bf16_io_cache,
                                      const char *tensor_name,
                                      ternary_bf16_layer_map_t *out_map);

static int validate_single_layer_target(const ternary_conversion_config_t *config,
                                        const conversion_runtime_t *runtime)
{
    const structural_tensor_rule_t *structural_rule = NULL;

    if (!config) {
        return -1;
    }

    structural_rule = runtime_structural_rule_for_tensor(runtime, config->layer_name);
    if (structural_rule_is_pass_through(structural_rule)) {
        if (structural_rule->rule_type == STRUCTURAL_TENSOR_RULE_ALIAS) {
            LOG_ERROR("Prompt 2 keeps PASS alias tensors in BF16; requested layer %s maps to %s",
                      config->layer_name,
                      structural_rule->mapped_tensor_name);
        } else {
            LOG_ERROR("Prompt 2 keeps PASS tensors in BF16; requested layer is excluded: %s",
                      config->layer_name);
        }
        return -1;
    }

    if (structural_rule && structural_rule->rule_type == STRUCTURAL_TENSOR_RULE_ALIAS) {
        LOG_ERROR("Prompt 2 skips structural alias tensors; requested layer %s maps to %s",
                  config->layer_name,
                  structural_rule->mapped_tensor_name);
        return -1;
    }

    if (tensor_name_is_tied_embedding_tensor(config->layer_name) ||
        tensor_name_is_tied_output_tensor(config->layer_name)) {
        LOG_ERROR("Prompt 2 keeps embeddings / tied output weights in BF16; requested layer is excluded: %s",
                  config->layer_name);
        return -1;
    }

    return 0;
}

static const ternary_calibration_source_t *init_single_layer_calibration_source(
    const ternary_conversion_config_t *config,
    const conversion_runtime_t *runtime,
    ternary_calibration_corpus_t *calibration_corpus,
    ternary_activation_tape_context_t *tape_context,
    ternary_calibration_source_t *calibration_source)
{
    if (!config || !runtime || !calibration_corpus || !tape_context || !calibration_source) {
        return NULL;
    }

    calibration_corpus->sample_texts = runtime->calibration_corpus.sample_texts;
    calibration_corpus->sample_count = runtime->calibration_corpus.sample_count;
    calibration_corpus->tokenizer = runtime->calibration_corpus.tokenizer;
    calibration_corpus->model_spec = runtime->calibration_corpus.model_spec;
    calibration_corpus->session = runtime->calibration_corpus.session;
    calibration_corpus->tensor_name = config->layer_name;

    calibration_source->corpus = calibration_corpus;
    calibration_source->tape_context = NULL;
    if (runtime->activation_tape && tensor_name_is_layer_tensor(config->layer_name)) {
        tape_context->tape = runtime->activation_tape;
        tape_context->tensor_name = config->layer_name;
        tape_context->proxy_cache = NULL;
        calibration_source->tape_context = tape_context;
    }
    calibration_source->sidecar = runtime->hessian_sidecar;
    return calibration_source;
}

static int calibrate_single_layer_tensor(const ternary_conversion_config_t *config,
                                         const ternary_bf16_layer_map_t *map,
                                         const conversion_runtime_t *runtime,
                                         transformer_ste_config_t *out_ste_config,
                                         ternary_calibration_result_t *out_result)
{
    transformer_ste_config_t ste_config;
    ternary_calibration_source_t calibration_source;
    ternary_calibration_corpus_t calibration_corpus;
    ternary_activation_tape_context_t tape_context;
    const ternary_calibration_source_t *calibration_source_ptr = NULL;

    if (!config || !map || !out_ste_config || !out_result) {
        return -1;
    }

    memset(&calibration_source, 0, sizeof(calibration_source));
    memset(&calibration_corpus, 0, sizeof(calibration_corpus));
    memset(&tape_context, 0, sizeof(tape_context));

    ste_config = default_runtime_ste_config(config);
    ste_config.telemetry_path = conversion_runtime_telemetry_path(config, runtime);
    ste_config.spatial_telemetry_path = conversion_runtime_spatial_telemetry_path(config, runtime);
    ste_config.hessian_sidecar_path = config->hessian_sidecar_path;
    ste_config.hessian_sidecar_crc32 = runtime ? runtime->hessian_sidecar_crc32 : 0u;

    calibration_source_ptr = init_single_layer_calibration_source(config,
                                                                  runtime,
                                                                  &calibration_corpus,
                                                                  &tape_context,
                                                                  &calibration_source);
    if (ste_config.use_anchor_mode) {
        if (transformer_calibrate_layer_ste_hybrid(map->bf16_weights,
                                                   map->rows,
                                                   map->cols,
                                                   &ste_config,
                                                   calibration_source_ptr,
                                                   out_result) != 0) {
            return -1;
        }
    } else if (transformer_calibrate_layer_ste_with_tape(map->bf16_weights,
                                                         map->rows,
                                                         map->cols,
                                                         &ste_config,
                                                         calibration_source_ptr,
                                                         out_result) != 0) {
        return -1;
    }

    *out_ste_config = ste_config;
    return 0;
}

static uint32_t single_layer_max_row_nnz(const ternary_calibration_result_t *result)
{
    uint32_t max_row_nnz = 0u;

    if (!result || !result->anchor_row_offsets) {
        return 0u;
    }

    for (uint32_t row = 0; row < result->rows; ++row) {
        uint32_t row_nnz = result->anchor_row_offsets[row + 1] - result->anchor_row_offsets[row];
        if (row_nnz > max_row_nnz) {
            max_row_nnz = row_nnz;
        }
    }

    return max_row_nnz;
}

static void populate_ternary_layer_from_result(const ternary_calibration_result_t *result,
                                               ternary_layer_t *out_layer)
{
    if (!result || !out_layer) {
        return;
    }

    memset(out_layer, 0, sizeof(*out_layer));
    out_layer->packed_weights = result->packed_weights;
    out_layer->packed_weight_bytes = result->packed_weight_bytes;
    out_layer->scales = result->scales;
    out_layer->scale_count = result->scale_count;
    out_layer->scale_bytes = result->scale_count * sizeof(float);
    out_layer->scale_dtype = SAFETENSORS_F32;
    out_layer->rows = result->rows;
    out_layer->cols = result->cols;
    out_layer->scale_group_size = result->scale_group_size;
    out_layer->groups_per_row = result->rows > 0u ? (uint32_t)(result->scale_count / result->rows) : 0u;
    out_layer->integrity = TERNARY_IO_INTEGRITY_CRC32;
}

static void populate_hybrid_layer_from_result(const transformer_ste_config_t *ste_config,
                                              const ternary_calibration_result_t *result,
                                              ternary_hybrid_layer_t *out_layer)
{
    if (!ste_config || !result || !out_layer) {
        return;
    }

    memset(out_layer, 0, sizeof(*out_layer));
    out_layer->packed_weights = result->packed_weights;
    out_layer->packed_weight_bytes = result->packed_weight_bytes;
    out_layer->scales = result->scales;
    out_layer->scale_count = result->scale_count;
    out_layer->scale_bytes = result->scale_count * sizeof(float);
    out_layer->scale_dtype = SAFETENSORS_F32;
    out_layer->rows = result->rows;
    out_layer->cols = result->cols;
    out_layer->scale_group_size = result->scale_group_size;
    out_layer->groups_per_row = result->rows > 0u ? (uint32_t)(result->scale_count / result->rows) : 0u;
    out_layer->integrity = TERNARY_IO_INTEGRITY_CRC32;
    out_layer->anchor_entries = result->anchor_entries;
    out_layer->anchor_count = result->anchor_count;
    out_layer->anchor_metadata.magic = TERNARY_ANCHOR_MAGIC;
    out_layer->anchor_metadata.version = TERNARY_ANCHOR_VERSION;
    out_layer->anchor_metadata.rows = result->rows;
    out_layer->anchor_metadata.cols = result->cols;
    out_layer->anchor_metadata.anchor_count = result->anchor_count;
    out_layer->anchor_metadata.scale_group_size = result->scale_group_size;
    out_layer->anchor_metadata.groups_per_row = out_layer->groups_per_row;
    out_layer->anchor_metadata.saliency_mode = (uint32_t)ste_config->anchor_saliency_mode;
    out_layer->anchor_metadata.budget_ppm = ste_config->anchor_budget_ppm;
    out_layer->anchor_metadata.saliency_cutoff = result->anchor_saliency_cutoff;
    out_layer->anchor_metadata.anchor_value_rms = result->anchor_value_rms;
    out_layer->anchor_metadata.bulk_gamma_mean = result->bulk_gamma_mean;
    out_layer->anchor_metadata.max_row_nnz = single_layer_max_row_nnz(result);
}

static int write_hybrid_output_to_dir(const char *output_dir,
                                      const char *tensor_name,
                                      const transformer_ste_config_t *ste_config,
                                      const ternary_calibration_result_t *result,
                                      uint32_t *out_crc32)
{
    ternary_hybrid_layer_t hybrid_layer;
    uint32_t crc32 = 0u;
    int rc = -1;

    if (!output_dir || !tensor_name || !ste_config || !result) {
        return -1;
    }

    populate_hybrid_layer_from_result(ste_config, result, &hybrid_layer);
    rc = ternary_anchor_write_layer(output_dir, tensor_name, &hybrid_layer, &crc32);
    if (rc == 0) {
        LOG_INFO("Converted tensor %s as anchor-bearing hybrid (crc32=%08x anchors=%u)",
                 tensor_name,
                 crc32,
                 result->anchor_count);
    }
    if (out_crc32) {
        *out_crc32 = crc32;
    }
    return rc;
}

static int write_ternary_output_to_dir(const char *output_dir,
                                       const char *tensor_name,
                                       const ternary_calibration_result_t *result,
                                       uint32_t *out_crc32)
{
    ternary_layer_t layer;
    uint32_t crc32 = 0u;
    int rc = -1;

    if (!output_dir || !tensor_name || !result) {
        return -1;
    }

    populate_ternary_layer_from_result(result, &layer);
    rc = io_write_layer_ternary_into_dir(output_dir, tensor_name, &layer, &crc32);
    if (rc == 0) {
        LOG_INFO("Converted tensor %s as plain ternary (crc32=%08x)", tensor_name, crc32);
    }
    if (out_crc32) {
        *out_crc32 = crc32;
    }
    return rc;
}

static int write_single_layer_hybrid_output(const ternary_conversion_config_t *config,
                                            const transformer_ste_config_t *ste_config,
                                            const ternary_calibration_result_t *result,
                                            uint32_t *out_crc32)
{
    uint32_t crc32 = 0u;
    int rc = -1;

    if (!config || !ste_config || !result) {
        return -1;
    }

    rc = write_hybrid_output_to_dir(config->output_path,
                                    config->layer_name,
                                    ste_config,
                                    result,
                                    &crc32);
    if (rc == 0) {
        LOG_INFO("Hybrid ternary conversion complete for %s (crc32=%08x anchors=%u output_dir=%s)",
                 config->layer_name,
                 crc32,
                 result->anchor_count,
                 config->output_path);
        sapphire_tracy_plot_i64("conversion.converted_tensors", 1);
    }
    if (out_crc32) {
        *out_crc32 = crc32;
    }
    return rc;
}

static int write_single_layer_ternary_output(const ternary_conversion_config_t *config,
                                             const ternary_calibration_result_t *result,
                                             uint32_t *out_crc32)
{
    ternary_layer_t layer;
    uint32_t crc32 = 0u;
    int rc = -1;

    if (!config || !result) {
        return -1;
    }

    populate_ternary_layer_from_result(result, &layer);
    rc = io_write_layer_ternary(config->output_path, config->layer_name, &layer, &crc32);
    if (rc == 0) {
        LOG_INFO("Ternary conversion complete for %s (crc32=%08x)", config->layer_name, crc32);
        sapphire_tracy_plot_i64("conversion.converted_tensors", 1);
    }
    if (out_crc32) {
        *out_crc32 = crc32;
    }
    return rc;
}

static int run_single_layer_conversion(const ternary_conversion_config_t *config,
                                       const char *model_dir,
                                       const char *model_path,
                                       const conversion_runtime_t *runtime) {
    ternary_bf16_layer_map_t map;
    ternary_calibration_result_t result;
    transformer_ste_config_t ste_config;
    uint32_t crc32 = 0;
    int rc = -1;
    uint32_t layer_index = 0u;
    int has_layer_index = 0;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "run_single_layer_conversion");

    if (!config) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, config->layer_name);
    has_layer_index = conversion_try_parse_layer_index(config->layer_name, &layer_index);
    conversion_tracy_plot_layer_index(has_layer_index, layer_index);

    memset(&map, 0, sizeof(map));
    memset(&result, 0, sizeof(result));
    memset(&ste_config, 0, sizeof(ste_config));

    if (validate_single_layer_target(config, runtime) != 0) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    if (mmap_tensor_for_conversion(model_dir, model_path, NULL, config->layer_name, &map) != 0) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    if (calibrate_single_layer_tensor(config, &map, runtime, &ste_config, &result) != 0) {
        goto cleanup;
    }

    if (ste_config.use_anchor_mode) {
        rc = write_single_layer_hybrid_output(config, &ste_config, &result, &crc32);
    } else {
        rc = write_single_layer_ternary_output(config, &result, &crc32);
    }

cleanup:
    transformer_free_ternary_calibration_result(&result);
    io_unmap_layer_bf16(&map);
    return conversion_tracy_end_status(&tracy_zone, rc);
}

typedef struct {
    const char *model_path;
    const char *model_dir;
    const char *output_dir;
    const char *tensor_name;
    const activation_tape_t *activation_tape;
    const ternary_hessian_sidecar_t *hessian_sidecar;
    ternary_bf16_io_cache_t *bf16_io_cache;
    ternary_hessian_proxy_cache_t *hessian_proxy_cache;
    const transformer_ste_config_t *ste_config;
    const ternary_calibration_corpus_t *calibration_corpus;
    ternary_calibration_result_t *out_result;
    uint32_t *out_crc32;
    int *out_skipped_vector;
    float *out_io_ms;
    full_model_tensor_representation_t representation;
    progressive_calib_stage_t progressive_stage;
} convert_tensor_job_t;

typedef enum {
    FULL_MODEL_TENSOR_STATUS_CONVERTED = 0,
    FULL_MODEL_TENSOR_STATUS_SKIPPED_VECTOR = 1,
    FULL_MODEL_TENSOR_STATUS_SKIPPED_OTHER = 2
} full_model_tensor_status_t;

typedef struct {
    const ternary_conversion_config_t *config;
    conversion_runtime_t *runtime;
    const model_spec_t *spec;
    const transformer_ste_config_t *ste_config;
    const char *model_dir;
    const char *model_path;
    int layer_index;
    uint32_t progress_index;
    progressive_calib_stage_t progressive_stage;
    const char *tensor_name;
    int *last_prefetched_entry_idx;
    full_model_tensor_representation_t representation;
} full_model_tensor_task_t;

static int mmap_tensor_for_conversion(const char *model_dir,
                                      const char *model_path,
                                      ternary_bf16_io_cache_t *bf16_io_cache,
                                      const char *tensor_name,
                                      ternary_bf16_layer_map_t *out_map);

static int mmap_tensor_for_conversion(const char *model_dir,
                                      const char *model_path,
                                      ternary_bf16_io_cache_t *bf16_io_cache,
                                      const char *tensor_name,
                                      ternary_bf16_layer_map_t *out_map)
{
    struct stat st;

    if (!tensor_name || !out_map) {
        return -1;
    }

    if (model_path && stat(model_path, &st) == 0 && S_ISREG(st.st_mode)) {
        return io_mmap_layer_bf16(model_path, tensor_name, out_map);
    }

    if (model_dir && model_dir[0] != '\0') {
        if (bf16_io_cache) {
            return io_mmap_layer_bf16_sharded_cached(model_dir, tensor_name, bf16_io_cache, out_map);
        }
        return io_mmap_layer_bf16_sharded(model_dir, tensor_name, out_map);
    }

    if (model_path) {
        return io_mmap_layer_bf16(model_path, tensor_name, out_map);
    }

    return -1;
}

static void store_convert_result(const convert_tensor_job_t *job,
                                 ternary_calibration_result_t *result,
                                 uint32_t crc32)
{
    if (job->out_result) {
        *job->out_result = *result;
        memset(result, 0, sizeof(*result));
    }
    if (job->out_crc32) {
        *job->out_crc32 = crc32;
    }
}

static int replace_tensor_suffix(const char *tensor_name,
                                 const char *suffix,
                                 const char *replacement,
                                 char *out_name,
                                 size_t out_name_size)
{
    size_t tensor_name_len = 0u;
    size_t suffix_len = 0u;
    size_t replacement_len = 0u;

    if (!tensor_name || !suffix || !replacement || !out_name || out_name_size == 0u) {
        return 0;
    }

    tensor_name_len = strlen(tensor_name);
    suffix_len = strlen(suffix);
    replacement_len = strlen(replacement);
    if (tensor_name_len < suffix_len ||
        strcmp(tensor_name + (tensor_name_len - suffix_len), suffix) != 0 ||
        tensor_name_len - suffix_len + replacement_len + 1u > out_name_size) {
        return 0;
    }

    memcpy(out_name, tensor_name, tensor_name_len - suffix_len);
    memcpy(out_name + (tensor_name_len - suffix_len), replacement, replacement_len + 1u);
    return 1;
}

static int norm_residual_probe_tensor_name(const char *norm_tensor_name,
                                           char *out_name,
                                           size_t out_name_size)
{
    if (replace_tensor_suffix(norm_tensor_name,
                              "input_layernorm.weight",
                              "self_attn.q_proj.weight",
                              out_name,
                              out_name_size)) {
        return 1;
    }
    if (replace_tensor_suffix(norm_tensor_name,
                              "post_attention_layernorm.weight",
                              "mlp.gate_proj.weight",
                              out_name,
                              out_name_size)) {
        return 1;
    }
    if (replace_tensor_suffix(norm_tensor_name,
                              "pre_feedforward_layernorm.weight",
                              "mlp.gate_proj.weight",
                              out_name,
                              out_name_size)) {
        return 1;
    }

    return 0;
}

static float activation_vector_variance(const float *vector, uint32_t vector_dim)
{
    double mean = 0.0;
    double mean_sq = 0.0;

    if (!vector || vector_dim == 0u) {
        return 0.0f;
    }

    for (uint32_t idx = 0; idx < vector_dim; ++idx) {
        double value = vector[idx];
        mean += value;
        mean_sq += value * value;
    }

    mean /= (double)vector_dim;
    mean_sq /= (double)vector_dim;
    if (mean_sq <= mean * mean) {
        return 0.0f;
    }
    return (float)(mean_sq - mean * mean);
}

static int mean_tape_activation_variance(const activation_tape_t *tape,
                                         const char *tensor_name,
                                         int sample_count,
                                         float *scratch_vector,
                                         float *out_variance)
{
    float variance_sum = 0.0f;

    if (!tape || !tensor_name || sample_count <= 0 || !scratch_vector || !out_variance) {
        return -1;
    }

    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        if (activation_tape_get_vector(tape, tensor_name, sample_idx, scratch_vector) != 0) {
            return -1;
        }
        variance_sum += activation_vector_variance(scratch_vector,
                                                   activation_tape_vector_dim(tape, tensor_name));
    }

    *out_variance = variance_sum / (float)sample_count;
    return 0;
}

static int mean_student_activation_variance(const ternary_calibration_corpus_t *corpus,
                                            const char *tensor_name,
                                            int sample_count,
                                            uint32_t vector_dim,
                                            float *scratch_vector,
                                            float *out_variance)
{
    float variance_sum = 0.0f;

    if (!corpus || !corpus->session || !corpus->tokenizer || !corpus->model_spec ||
        !corpus->sample_texts || corpus->sample_count <= 0 || !tensor_name ||
        sample_count <= 0 || vector_dim == 0u || !scratch_vector || !out_variance) {
        return -1;
    }

    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        transformer_activation_capture_request_t activation_request;

        memset(&activation_request, 0, sizeof(activation_request));
        activation_request.spec = corpus->model_spec;
        activation_request.text = corpus->sample_texts[sample_idx % corpus->sample_count];
        activation_request.tensor_name = tensor_name;
        activation_request.out_vector = scratch_vector;
        activation_request.out_dim = vector_dim;
        if (sapphire_collect_tensor_activation(corpus->session,
                                              corpus->tokenizer,
                                              &activation_request) != 0) {
            return -1;
        }
        variance_sum += activation_vector_variance(scratch_vector, vector_dim);
    }

    *out_variance = variance_sum / (float)sample_count;
    return 0;
}

static float compute_norm_residual_correction(const convert_tensor_job_t *job)
{
    char probe_tensor_name[256];
    uint32_t vector_dim = 0u;
    float teacher_variance = 0.0f;
    float student_variance = 0.0f;
    float *teacher_vector = NULL;
    float *student_vector = NULL;
    float correction = 1.0f;
    int tape_samples = 0;
    int sample_count = 0;

    if (!job || !progressive_calib_stage_uses_residual_recenter(job->progressive_stage) ||
        !job->tensor_name || !job->activation_tape || !job->calibration_corpus) {
        return 1.0f;
    }
    if (!norm_residual_probe_tensor_name(job->tensor_name,
                                         probe_tensor_name,
                                         sizeof(probe_tensor_name))) {
        return 1.0f;
    }

    vector_dim = activation_tape_vector_dim(job->activation_tape, probe_tensor_name);
    tape_samples = activation_tape_sample_count(job->activation_tape);
    if (vector_dim == 0u || tape_samples <= 0 || !job->calibration_corpus->sample_texts ||
        job->calibration_corpus->sample_count <= 0) {
        return 1.0f;
    }

    sample_count = tape_samples;
    if (sample_count > job->calibration_corpus->sample_count) {
        sample_count = job->calibration_corpus->sample_count;
    }
    if (sample_count > PROGRESSIVE_RESIDUAL_CORRECTION_MAX_SAMPLES) {
        sample_count = PROGRESSIVE_RESIDUAL_CORRECTION_MAX_SAMPLES;
    }
    if (sample_count <= 0) {
        return 1.0f;
    }

    teacher_vector = (float *)malloc((size_t)vector_dim * sizeof(float));
    student_vector = (float *)malloc((size_t)vector_dim * sizeof(float));
    if (!teacher_vector || !student_vector) {
        free(teacher_vector);
        free(student_vector);
        return 1.0f;
    }

    if (mean_tape_activation_variance(job->activation_tape,
                                      probe_tensor_name,
                                      sample_count,
                                      teacher_vector,
                                      &teacher_variance) != 0 ||
        mean_student_activation_variance(job->calibration_corpus,
                                         probe_tensor_name,
                                         sample_count,
                                         vector_dim,
                                         student_vector,
                                         &student_variance) != 0) {
        free(teacher_vector);
        free(student_vector);
        return 1.0f;
    }

    free(teacher_vector);
    free(student_vector);
    if (teacher_variance <= 1e-12f || student_variance <= 1e-12f ||
        (student_variance / teacher_variance) >= PROGRESSIVE_RESIDUAL_CORRECTION_MIN_VARIANCE_RATIO) {
        return 1.0f;
    }

    correction = sqrtf(teacher_variance / student_variance);
    if (correction < 1.0f) {
        correction = 1.0f;
    }
    if (correction > PROGRESSIVE_RESIDUAL_CORRECTION_MAX_SCALE) {
        correction = PROGRESSIVE_RESIDUAL_CORRECTION_MAX_SCALE;
    }
    LOG_INFO("Residual re-center: tensor=%s probe=%s teacher_var=%.6f student_var=%.6f correction=%.4f samples=%d",
             job->tensor_name,
             probe_tensor_name,
             (double)teacher_variance,
             (double)student_variance,
             (double)correction,
             sample_count);
    return correction;
}

static int convert_vector_mold(const convert_tensor_job_t *job,
                               const ternary_bf16_layer_map_t *map)
{
    float *f32_tmp = (float *)malloc((size_t)map->rows * sizeof(float));
    float residual_correction = 1.0f;
    int rc = 0;

    if (!f32_tmp) {
        LOG_ERROR("convert_vector_mold: alloc failed for %s", job->tensor_name);
        return -1;
    }
    bf16_to_f32_vec(f32_tmp, map->bf16_weights, (int)map->rows);
    residual_correction = compute_norm_residual_correction(job);
    if (residual_correction != 1.0f) {
        for (uint32_t row = 0; row < map->rows; ++row) {
            f32_tmp[row] *= residual_correction;
        }
    }
    rc = io_write_layer_molded_bf16_into_dir(job->output_dir, job->tensor_name,
                                             f32_tmp, map->rows, 1u,
                                             job->out_crc32);
    free(f32_tmp);
    if (rc == 0) {
        LOG_INFO("Wrote MOLD BF16 vector %s (source, residual_correction=%.4f)",
                 job->tensor_name,
                 (double)residual_correction);
    }
    return rc;
}

static float f32_roundtrip_bf16_local(float value)
{
    union {
        uint32_t u32;
        float f32;
    } bits;

    bits.f32 = value;
    bits.u32 &= 0xFFFF0000u;
    return bits.f32;
}

static void round_dense_to_bf16_inplace(float *weights, size_t weight_count)
{
    if (!weights) {
        return;
    }

    for (size_t idx = 0u; idx < weight_count; ++idx) {
        weights[idx] = f32_roundtrip_bf16_local(weights[idx]);
    }
}

static int convert_tensor_to_dir(const convert_tensor_job_t *job) {
    ternary_bf16_layer_map_t map;
    ternary_calibration_result_t result;
    struct timespec load_start;
    struct timespec load_end;
    int measure_io = 0;
    uint32_t crc32 = 0;
    int rc = -1;

    if (!job || !job->model_path || !job->output_dir || !job->tensor_name || !job->ste_config) {
        LOG_ERROR("convert_tensor_to_dir: invalid arguments");
        return -1;
    }

    if (job->out_skipped_vector) {
        *job->out_skipped_vector = 0;
    }
    if (job->out_result) {
        memset(job->out_result, 0, sizeof(*job->out_result));
    }
    if (job->out_crc32) {
        *job->out_crc32 = 0u;
    }

    memset(&map, 0, sizeof(map));
    memset(&result, 0, sizeof(result));

    measure_io = job->out_io_ms ? 1 : 0;
    if (measure_io && clock_gettime(CLOCK_MONOTONIC, &load_start) != 0) {
        measure_io = 0;
    }
    if (mmap_tensor_for_conversion(job->model_dir,
                                   job->model_path,
                                   job->bf16_io_cache,
                                   job->tensor_name,
                                   &map) != 0) {
        return -1;
    }
    if (measure_io && clock_gettime(CLOCK_MONOTONIC, &load_end) == 0) {
        *job->out_io_ms += telemetry_elapsed_ms(&load_start, &load_end);
    }

    if (map.cols == 1u) {
        if (job->representation == FULL_MODEL_TENSOR_REPRESENTATION_MOLD) {
            int vrc = convert_vector_mold(job, &map);
            io_unmap_layer_bf16(&map);
            return vrc;
        }
        LOG_INFO("Skipping vector tensor in full conversion: %s", job->tensor_name);
        if (job->out_skipped_vector) {
            *job->out_skipped_vector = 1;
        }
        io_unmap_layer_bf16(&map);
        return 0;
    }

    if (tensor_name_is_layer_tensor(job->tensor_name) && job->activation_tape) {
        int tape_entry_idx = activation_tape_entry_index(job->activation_tape, job->tensor_name);
        ternary_calibration_source_t calibration_source;

        if (tape_entry_idx < 0) {
            LOG_ERROR("Activation tape does not contain tensor %s", job->tensor_name);
            io_unmap_layer_bf16(&map);
            return -1;
        }

        memset(&calibration_source, 0, sizeof(calibration_source));
        calibration_source.corpus = job->calibration_corpus;
        calibration_source.tape_context = &(ternary_activation_tape_context_t){
            .tape = job->activation_tape,
            .tensor_name = job->tensor_name,
            .proxy_cache = job->hessian_proxy_cache
        };
        calibration_source.sidecar = job->hessian_sidecar;

        if (transformer_calibrate_layer_ste_with_tape(map.bf16_weights,
                                                      map.rows,
                                                      map.cols,
                                                      job->ste_config,
                                                      &calibration_source,
                                                      &result) != 0) {
            io_unmap_layer_bf16(&map);
            return -1;
        }
    } else if (transformer_calibrate_layer_ste(map.bf16_weights,
                                               map.rows,
                                               map.cols,
                                               job->ste_config,
                                               job->calibration_corpus,
                                               &result) != 0) {
        io_unmap_layer_bf16(&map);
        return -1;
    }

    if (job->representation == FULL_MODEL_TENSOR_REPRESENTATION_MOLD) {
        rc = io_write_layer_molded_bf16_into_dir(job->output_dir, job->tensor_name,
                                                 result.latent_weights,
                                                 result.rows, result.cols,
                                                 &crc32);
        if (rc == 0) {
            round_dense_to_bf16_inplace(result.latent_weights, result.weight_count);
            LOG_INFO("Converted tensor %s as mold BF16 (crc32=%08x)", job->tensor_name, crc32);
            store_convert_result(job, &result, crc32);
        }
        transformer_free_ternary_calibration_result(&result);
        io_unmap_layer_bf16(&map);
        return rc;
    }

    if (job->representation == FULL_MODEL_TENSOR_REPRESENTATION_ANCHOR) {
        rc = write_hybrid_output_to_dir(job->output_dir,
                                        job->tensor_name,
                                        job->ste_config,
                                        &result,
                                        &crc32);
    } else {
        rc = write_ternary_output_to_dir(job->output_dir,
                                         job->tensor_name,
                                         &result,
                                         &crc32);
    }
    if (rc == 0) {
        store_convert_result(job, &result, crc32);
    }

    transformer_free_ternary_calibration_result(&result);
    io_unmap_layer_bf16(&map);
    return rc;
}

static int process_full_model_tensor_skip_other(const full_model_tensor_task_t *task,
                                                uint32_t converted_count,
                                                const char *skip_message)
{
    if (skip_message) {
        LOG_INFO(skip_message, task->tensor_name);
    }

    if (student_checkpoint_progress(task->config,
                                    task->runtime,
                                    task->progress_index + 1u,
                                    converted_count,
                                    0) != 0) {
        LOG_WARN("student update: checkpoint write failed after layer %d", task->layer_index);
    }

    return FULL_MODEL_TENSOR_STATUS_SKIPPED_OTHER;
}

static int process_full_model_tensor_skipped_vector(const full_model_tensor_task_t *task,
                                                    ternary_calibration_result_t *result,
                                                    uint32_t converted_count)
{
    transformer_free_ternary_calibration_result(result);
    if (student_checkpoint_progress(task->config,
                                    task->runtime,
                                    task->progress_index + 1u,
                                    converted_count,
                                    0) != 0) {
        LOG_WARN("student update: checkpoint write failed after skipped vector %s", task->tensor_name);
    }

    return FULL_MODEL_TENSOR_STATUS_SKIPPED_VECTOR;
}

static int process_full_model_tensor_failed(const full_model_tensor_task_t *task,
                                            ternary_calibration_result_t *result,
                                            uint32_t converted_count)
{
    LOG_WARN("Failed to convert tensor: %s", task->tensor_name);
    transformer_free_ternary_calibration_result(result);
    if (student_checkpoint_progress(task->config,
                                    task->runtime,
                                    task->progress_index,
                                    converted_count,
                                    1) != 0) {
        LOG_WARN("student update: checkpoint write failed while handling tensor failure at %s", task->tensor_name);
    }

    return -1;
}

static int process_full_model_tensor_complete(const full_model_tensor_task_t *task,
                                              ternary_calibration_result_t *result,
                                              uint32_t converted_count,
                                              uint32_t crc32)
{
    if (task->runtime && task->runtime->validation_state.config.sample_count > 0) {
        int validation_rc = 0;

        if (task->representation == FULL_MODEL_TENSOR_REPRESENTATION_MOLD) {
            validation_rc = ternary_validation_apply_proxy_from_dense(&task->runtime->validation_state,
                                                                      task->tensor_name,
                                                                      &(ternary_validation_dense_view_t){
                                                                          .weights = result->latent_weights,
                                                                          .rows = result->rows,
                                                                          .cols = result->cols,
                                                                      },
                                                                      crc32,
                                                                      (int)(converted_count + 1u));
        } else {
            validation_rc = ternary_validation_apply_proxy(&task->runtime->validation_state,
                                                           task->tensor_name,
                                                           result,
                                                           crc32,
                                                           (int)(converted_count + 1u));
        }
        if (validation_rc != 0) {
            LOG_WARN("Validation checkpoint failed after tensor: %s", task->tensor_name);
        }
    }

    transformer_free_ternary_calibration_result(result);
    if (student_checkpoint_progress(task->config,
                                    task->runtime,
                                    task->progress_index + 1u,
                                    converted_count + 1u,
                                    0) != 0) {
        LOG_WARN("student update: checkpoint write failed after tensor %s", task->tensor_name);
    }

    return FULL_MODEL_TENSOR_STATUS_CONVERTED;
}

static int prepare_full_model_layer_ste_config(const full_model_tensor_task_t *task,
                                               uint32_t layer_index,
                                               int has_layer_index,
                                               transformer_ste_config_t *out_ste_config,
                                               ternary_telemetry_t *out_telemetry)
{
    progressive_distillation_schedule_t distillation_schedule;

    if (!task || !task->runtime || !task->runtime->activation_tape ||
        !tensor_name_is_layer_tensor(task->tensor_name) || !out_ste_config || !out_telemetry) {
        return 0;
    }

    *out_ste_config = task->ste_config ? *task->ste_config : default_runtime_ste_config(task->config);
    out_ste_config->telemetry_path = conversion_runtime_telemetry_path(task->config, task->runtime);
    out_ste_config->spatial_telemetry_path = conversion_runtime_spatial_telemetry_path(task->config, task->runtime);
    distillation_schedule = progressive_calib_distillation_schedule(task->config,
                                                                   task->progressive_stage,
                                                                   has_layer_index,
                                                                   layer_index);
    out_ste_config->kl_weight = distillation_schedule.kl_weight;
    out_ste_config->kl_temperature = distillation_schedule.kl_temperature;
    out_ste_config->simulate_activation_a8 = progressive_calib_stage_uses_activation_a8(task->progressive_stage);
    if (progressive_calib_stage_is_stage3(task->progressive_stage)) {
        out_ste_config->learning_rate *= PROGRESSIVE_CALIB_STAGE3_LR_SCALE;
    }
    out_ste_config->hessian_sidecar_path = task->config->hessian_sidecar_path;
    out_ste_config->hessian_sidecar_crc32 = task->runtime->hessian_sidecar_crc32;
    if (task->config->progressive_calib && has_layer_index) {
        LOG_INFO("Progressive distillation: tensor=%s stage=%s layer=%u kl_weight=%.4f kl_temp=%.2f lr=%.4f a8=%u",
                 task->tensor_name,
                 progressive_calib_stage_name(task->progressive_stage),
                 layer_index,
                 (double)out_ste_config->kl_weight,
                 (double)out_ste_config->kl_temperature,
             (double)out_ste_config->learning_rate,
             (unsigned)out_ste_config->simulate_activation_a8);
    }
    telemetry_prepare_layer_context(out_telemetry, task->runtime, task->layer_index);
    out_ste_config->telemetry = out_telemetry;
    prefetch_activation_tape_lookahead(task->runtime,
                                       task->spec,
                                       task->layer_index,
                                       2,
                                       task->last_prefetched_entry_idx,
                                       &out_telemetry->io_ms);
    return 1;
}

static int full_model_anchor_policy_allows_tensor(const full_model_tensor_task_t *task,
                                                  const structural_tensor_rule_t *structural_rule)
{
    if (!task || !task->config || !task->tensor_name || !full_model_anchor_mode_enabled(task->config)) {
        return 0;
    }
    if (structural_rule_is_mold(structural_rule) || structural_rule_is_pass_through(structural_rule)) {
        return 0;
    }
    if (!tensor_name_is_layer_tensor(task->tensor_name) ||
        !structural_map_name_matches(task->tensor_name, full_model_anchor_policy_pattern(task->config))) {
        return 0;
    }
    if (full_model_anchor_policy_filter(task->config)[0] != '\0' &&
        !structural_map_name_matches(task->tensor_name, full_model_anchor_policy_filter(task->config))) {
        return 0;
    }

    return 1;
}

static int classify_full_model_tensor(full_model_tensor_task_t *task,
                                      uint32_t converted_count)
{
    const structural_tensor_rule_t *structural_rule = NULL;

    if (!task) {
        return -1;
    }

    task->representation = FULL_MODEL_TENSOR_REPRESENTATION_TERNARY;
    if (!task->tensor_name || task->tensor_name[0] == '\0') {
        return process_full_model_tensor_skip_other(task, converted_count, NULL);
    }
    if (task->progressive_stage != PROGRESSIVE_CALIB_STAGE_DISABLED &&
        !progressive_calib_stage_allows_tensor(task->progressive_stage, task->tensor_name)) {
        LOG_INFO("Progressive calib %s: freezing tensor outside active window: %s",
                 progressive_calib_stage_name(task->progressive_stage),
                 task->tensor_name);
        return process_full_model_tensor_skip_other(task, converted_count, NULL);
    }

    structural_rule = runtime_structural_rule_for_tensor(task->runtime, task->tensor_name);
    if (structural_rule && structural_rule->rule_type == STRUCTURAL_TENSOR_RULE_ALIAS) {
        if (structural_rule_is_pass_through(structural_rule)) {
            LOG_INFO("Skipping structural PASS alias tensor in full conversion: %s -> %s",
                     task->tensor_name,
                     structural_rule->mapped_tensor_name);
            return process_full_model_tensor_skip_other(task, converted_count, NULL);
        }
        LOG_INFO("Skipping structural alias tensor in full conversion: %s -> %s",
                 task->tensor_name,
                 structural_rule->mapped_tensor_name);
        return process_full_model_tensor_skip_other(task,
                                                    converted_count,
                                                    "Skipping structural alias tensor in full conversion: %s");
    }
    if (structural_rule_is_pass_through(structural_rule)) {
        LOG_INFO("Skipping structural PASS tensor in full conversion: %s", task->tensor_name);
        return process_full_model_tensor_skip_other(task, converted_count, NULL);
    }

    if (structural_rule_is_mold(structural_rule)) {
        task->representation = FULL_MODEL_TENSOR_REPRESENTATION_MOLD;
    } else if (full_model_anchor_policy_allows_tensor(task, structural_rule)) {
        task->representation = FULL_MODEL_TENSOR_REPRESENTATION_ANCHOR;
    }

    if (task->representation != FULL_MODEL_TENSOR_REPRESENTATION_MOLD &&
        tensor_name_is_tied_embedding_tensor(task->tensor_name)) {
        LOG_INFO("Skipping embedding tensor in full conversion: %s", task->tensor_name);
        return process_full_model_tensor_skip_other(task,
                                                    converted_count,
                                                    "Skipping embedding tensor in full conversion: %s");
    }
    if (task->representation != FULL_MODEL_TENSOR_REPRESENTATION_MOLD &&
        tensor_name_is_tied_output_tensor(task->tensor_name)) {
        LOG_INFO("Skipping tied output tensor in full conversion: %s", task->tensor_name);
        return process_full_model_tensor_skip_other(task,
                                                    converted_count,
                                                    "Skipping tied output tensor in full conversion: %s");
    }

    return FULL_MODEL_TENSOR_STATUS_CONVERTED;
}

static int process_full_model_tensor(full_model_tensor_task_t *task)
{
    convert_tensor_job_t job;
    ternary_calibration_corpus_t active_corpus;
    ternary_calibration_result_t result;
    ternary_telemetry_t layer_telemetry;
    transformer_ste_config_t effective_ste_config;
    uint32_t crc32 = 0u;
    int skipped_vector = 0;
    int rc = 0;
    uint32_t converted_count = 0u;
    int use_layer_telemetry = 0;
    uint32_t layer_index = 0u;
    int has_layer_index = 0;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "process_full_model_tensor");

    if (!task || !task->config) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, task->tensor_name);
    sapphire_tracy_plot_i64("conversion.progress_index", (int64_t)task->progress_index);

    if (task->runtime) {
        converted_count = task->runtime->checkpoint_state.converted_tensor_count;
    }
    sapphire_tracy_plot_i64("conversion.converted_tensors", (int64_t)converted_count);

    memset(&result, 0, sizeof(result));
    memset(&layer_telemetry, 0, sizeof(layer_telemetry));
    memset(&effective_ste_config, 0, sizeof(effective_ste_config));

    rc = classify_full_model_tensor(task, converted_count);
    if (rc != FULL_MODEL_TENSOR_STATUS_CONVERTED) {
        return conversion_tracy_end_status(&tracy_zone, rc);
    }

    has_layer_index = conversion_try_parse_layer_index(task->tensor_name, &layer_index);
    conversion_tracy_plot_layer_index(has_layer_index, layer_index);

    use_layer_telemetry = prepare_full_model_layer_ste_config(task,
                                                              layer_index,
                                                              has_layer_index,
                                                              &effective_ste_config,
                                                              &layer_telemetry);
    if (!use_layer_telemetry) {
        effective_ste_config = task->ste_config ? *task->ste_config : default_runtime_ste_config(task->config);
    }
    effective_ste_config.use_anchor_mode =
        (task->representation == FULL_MODEL_TENSOR_REPRESENTATION_ANCHOR) ? 1 : 0;

    memset(&active_corpus, 0, sizeof(active_corpus));
    if (task->runtime) {
        active_corpus = task->runtime->calibration_corpus;
        active_corpus.tensor_name = task->tensor_name;
    }

    memset(&job, 0, sizeof(job));
    job.model_path = task->model_path;
    job.model_dir = task->model_dir;
    job.output_dir = task->config->output_path;
    job.tensor_name = task->tensor_name;
    job.activation_tape = task->runtime ? task->runtime->activation_tape : NULL;
    job.hessian_sidecar = task->runtime ? task->runtime->hessian_sidecar : NULL;
    job.bf16_io_cache = task->runtime ? &task->runtime->bf16_io_cache : NULL;
    job.hessian_proxy_cache = task->runtime ? &task->runtime->hessian_proxy_cache : NULL;
    job.ste_config = &effective_ste_config;
    job.calibration_corpus = task->runtime ? &active_corpus : NULL;
    job.out_result = &result;
    job.out_crc32 = &crc32;
    job.out_skipped_vector = &skipped_vector;
    job.out_io_ms = use_layer_telemetry ? &layer_telemetry.io_ms : NULL;
    job.representation = task->representation;
    job.progressive_stage = task->progressive_stage;

    rc = convert_tensor_to_dir(&job);
    if (rc != 0) {
        return conversion_tracy_end_status(&tracy_zone,
                                           process_full_model_tensor_failed(task, &result, converted_count));
    }

    if (skipped_vector) {
        return conversion_tracy_end_status(&tracy_zone,
                                           process_full_model_tensor_skipped_vector(task, &result, converted_count));
    }

    return conversion_tracy_end_status(&tracy_zone,
                                       process_full_model_tensor_complete(task, &result, converted_count, crc32));
}

static int run_full_model_conversion(const ternary_conversion_config_t *config,
                                     const char *model_dir,
                                     const char *model_path,
                                     conversion_runtime_t *runtime) {
    model_spec_t *spec = NULL;
    transformer_ste_config_t ste_config;
    int converted = 0;
    int skipped_vectors = 0;
    int start_progress = 0;
    int total_progress_steps = 0;
    progressive_calib_stage_t logged_stage = PROGRESSIVE_CALIB_STAGE_DISABLED;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "run_full_model_conversion");

    if (!config) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, config->model_name);

    spec = runtime ? runtime->model_spec : get_model_spec(config->model_name);
    if (!spec || !spec->tensor_map || spec->tensor_map_size <= 0) {
        LOG_ERROR("Full ternary conversion: failed to resolve model spec for %s", config->model_name);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    /* Validate tape layer count matches model config to prevent buffer overflow */
    if (runtime && runtime->activation_tape) {
        if (validate_tape_layer_count(runtime->activation_tape, spec) != 0) {
            LOG_ERROR("Full ternary conversion: tape/model layer mismatch prevents safe conversion");
            return conversion_tracy_end_status(&tracy_zone, -1);
        }
    }

    total_progress_steps = (int)conversion_total_progress_steps(config, spec);
    if (total_progress_steps <= 0) {
        LOG_ERROR("Full ternary conversion: no tensor progress steps available for %s", config->model_name);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    sapphire_tracy_plot_i64("conversion.total_progress_steps", (int64_t)total_progress_steps);

    if (runtime) {
        start_progress = (runtime->checkpoint_state.next_layer_index > 0)
            ? (int)runtime->checkpoint_state.next_layer_index
            : 0;
        converted = (int)runtime->checkpoint_state.converted_tensor_count;
        if (converted > 0) {
            LOG_INFO("student update: resuming from progress step %d with %d converted tensors",
                     start_progress,
                     converted);
        }
    }
    sapphire_tracy_plot_i64("conversion.progress_index", (int64_t)start_progress);
    sapphire_tracy_plot_i64("conversion.converted_tensors", (int64_t)converted);
    ste_config = default_runtime_ste_config(config);
    ste_config.telemetry_path = conversion_runtime_telemetry_path(config, runtime);
    ste_config.spatial_telemetry_path = conversion_runtime_spatial_telemetry_path(config, runtime);
    ste_config.hessian_sidecar_path = config->hessian_sidecar_path;
    ste_config.hessian_sidecar_crc32 = runtime ? runtime->hessian_sidecar_crc32 : 0u;

    int last_prefetched_entry_idx = -1;

    for (int progress_idx = start_progress; progress_idx < total_progress_steps; ++progress_idx) {
        int tensor_index = conversion_tensor_index_from_progress(config, spec, progress_idx);
        progressive_calib_stage_t stage = progressive_calib_stage_from_progress_index(config, spec, progress_idx);
        const char *tensor_name = NULL;
        full_model_tensor_task_t task;
        int status = 0;

        if (tensor_index < 0 || tensor_index >= spec->tensor_map_size) {
            LOG_ERROR("Full ternary conversion: invalid tensor index %d at progress step %d",
                      tensor_index,
                      progress_idx);
            return conversion_tracy_end_status(&tracy_zone, -1);
        }
        tensor_name = spec->tensor_map[tensor_index].hf_name;
        if (config->progressive_calib && stage != logged_stage) {
            LOG_INFO("Progressive calib: starting %s", progressive_calib_stage_name(stage));
            conversion_tracy_emit_stage_marker(stage);
            logged_stage = stage;
        }

        sapphire_tracy_plot_i64("conversion.progress_index", (int64_t)progress_idx);

        memset(&task, 0, sizeof(task));
        task.config = config;
        task.runtime = runtime;
        task.spec = spec;
        task.ste_config = &ste_config;
        task.model_dir = model_dir;
        task.model_path = model_path;
        task.layer_index = tensor_index;
        task.progress_index = (uint32_t)progress_idx;
        task.progressive_stage = stage;
        task.tensor_name = tensor_name;
        task.last_prefetched_entry_idx = &last_prefetched_entry_idx;
        task.representation = FULL_MODEL_TENSOR_REPRESENTATION_TERNARY;
        status = process_full_model_tensor(&task);

        if (status < 0) {
            return conversion_tracy_end_status(&tracy_zone, -1);
        }
        if (status == FULL_MODEL_TENSOR_STATUS_SKIPPED_VECTOR) {
            skipped_vectors++;
        } else if (status == FULL_MODEL_TENSOR_STATUS_CONVERTED) {
            converted++;
            sapphire_tracy_plot_i64("conversion.converted_tensors", (int64_t)converted);
        }
    }

    if (runtime && runtime->validation_state.config.sample_count > 0) {
        if (ternary_validation_finish(&runtime->validation_state, converted) != 0) {
            LOG_WARN("Validation final checkpoint failed");
        }
    }

    if (student_checkpoint_progress(config,
                                    runtime,
                                    runtime ? runtime->checkpoint_state.next_layer_index : 0u,
                                    runtime ? runtime->checkpoint_state.converted_tensor_count : 0u,
                                    1) != 0) {
        if (runtime) {
            LOG_WARN("student update: final checkpoint write failed");
        }
    }

    LOG_INFO("Full ternary conversion summary: converted=%d skipped_vectors=%d",
             converted, skipped_vectors);
    return conversion_tracy_end_status(&tracy_zone, (converted > 0) ? 0 : -1);
}

static int config_has_text(const char *value)
{
    return value && value[0] != '\0';
}

static void log_conversion_corpus_inputs(const ternary_conversion_config_t *config)
{
    if (config_has_text(config->calibration_corpus_manifest_path)) {
        if (config_has_text(config->calibration_corpus_path)) {
            LOG_INFO("  calibration_corpus: %s", config->calibration_corpus_path);
        } else {
            LOG_INFO("  calibration_corpus: <loaded from manifest>");
        }
        LOG_INFO("  calibration_manifest: %s", config->calibration_corpus_manifest_path);
        return;
    }

    if (config_has_text(config->calibration_corpus_path)) {
        LOG_INFO("  calibration_corpus: %s", config->calibration_corpus_path);
    } else {
        LOG_INFO("  calibration_corpus: <built-in fallback>");
    }
}

static void log_conversion_validation_inputs(const ternary_conversion_config_t *config)
{
    if (config_has_text(config->validation_corpus_path)) {
        LOG_INFO("  validation_corpus: %s", config->validation_corpus_path);
    }
    if (config_has_text(config->validation_corpus_manifest_path)) {
        LOG_INFO("  validation_manifest: %s", config->validation_corpus_manifest_path);
    }
}

static void log_conversion_config(const ternary_conversion_config_t *config)
{
    LOG_INFO("Ternary conversion mode selected");
    LOG_INFO("  model: %s", config->model_name);
    LOG_INFO("  output: %s", config->output_path);
    LOG_INFO("  context_len: %d", config->context_len);
    LOG_INFO("  calibration_samples: %d",
             (config->calibration_sample_limit > 0) ? config->calibration_sample_limit : 4);
    LOG_INFO("  ste_steps: %d", (config->ste_steps > 0) ? config->ste_steps : 3);
    LOG_INFO("  progressive_calib: %s", config->progressive_calib ? "enabled" : "disabled");
    LOG_INFO("  student_down_proj_rmsnorm: %s",
             config->student_down_proj_input_rmsnorm ? "enabled" : "disabled");
    LOG_INFO("  max_grad_norm: %.4f", (double)((config->max_grad_norm > 0.0f) ? config->max_grad_norm : 1.0f));
    LOG_INFO("  validate_every: %d", config->validate_every_n);
    LOG_INFO("  kl_weight: %.4f", (double)((config->kl_weight >= 0.0f) ? config->kl_weight : 0.05f));
    LOG_INFO("  kl_update_freq: %d", (config->kl_update_interval > 0) ? config->kl_update_interval : 4);
    LOG_INFO("  kl_samples: %d", (config->kl_sample_count > 0) ? config->kl_sample_count : 4);
    LOG_INFO("  ste_early_stop: patience=%d delta=%.4f divergence=%.2f",
             (config->early_stop_patience >= 0) ? config->early_stop_patience : 6,
             (double)((config->early_stop_min_delta >= 0.0f) ? config->early_stop_min_delta : 1e-4f),
             (double)((config->early_stop_divergence_ratio >= 0.0f) ? config->early_stop_divergence_ratio : 1.25f));
    if (config->progressive_calib) {
        LOG_INFO("  progressive_kl: stage2 layers 6-10 hold, 11-15 ramp to %.4f, stage3 %.4f, stage2 temp=%.2f",
                 (double)PROGRESSIVE_CALIB_STAGE2_TARGET_KL_WEIGHT,
                 (double)PROGRESSIVE_CALIB_STAGE3_TARGET_KL_WEIGHT,
                 (double)PROGRESSIVE_CALIB_STAGE2_KL_TEMPERATURE);
        LOG_INFO("  progressive_stage3: blocks 0-9, 10-19, 20-27+shared with lr_scale=%.2f",
                 (double)PROGRESSIVE_CALIB_STAGE3_LR_SCALE);
    }
    LOG_INFO("  hessian_proxy: %s", config->disable_hessian_proxy ? "disabled" : "activation-diagonal");
    LOG_INFO("  hessian_proxy_strength: %.4f",
             (double)((config->hessian_proxy_strength >= 0.0f) ? config->hessian_proxy_strength : 1.0f));
    LOG_INFO("  hessian_proxy_floor: %.4f",
             (double)((config->hessian_proxy_floor >= 0.0f) ? config->hessian_proxy_floor : 0.05f));
    LOG_INFO("  layer filter: %s",
             config_has_text(config->layer_name) ? config->layer_name : "<all layers>");
    LOG_INFO("  anchor_mode: %s", config->use_anchor_mode ? "enabled" : "disabled");
    if (config->use_anchor_mode) {
        LOG_INFO("  anchor_budget_ppm: %u", config->anchor_budget_ppm);
        LOG_INFO("  anchor_saliency_mode: %d", config->anchor_saliency_mode);
        if (full_model_anchor_mode_enabled(config)) {
            LOG_INFO("  full_model_anchor_policy: %s pattern=%s artifact_version=%u",
                     full_model_anchor_policy_name(config),
                     full_model_anchor_policy_pattern(config),
                     TERNARY_ANCHOR_VERSION);
            if (full_model_anchor_policy_filter(config)[0] != '\0') {
                LOG_INFO("  full_model_anchor_filter: %s", full_model_anchor_policy_filter(config));
            }
        }
    }

    if (config_has_text(config->activation_tape_path)) {
        LOG_INFO("  activation_tape: %s", config->activation_tape_path);
    }
    if (config_has_text(config->teacher_model_name)) {
        LOG_INFO("  teacher_model: %s", config->teacher_model_name);
    }
    if (config_has_text(config->structural_map_path)) {
        LOG_INFO("  structural_map: %s", config->structural_map_path);
    }
    if (config_has_text(config->hessian_sidecar_path)) {
        LOG_INFO("  hessian_sidecar: %s", config->hessian_sidecar_path);
    }

    log_conversion_corpus_inputs(config);
    log_conversion_validation_inputs(config);
}

static int run_requested_conversion(const ternary_conversion_config_t *config,
                                    const char *model_dir,
                                    const char *model_path,
                                    const conversion_runtime_t *runtime)
{
    if (config_has_text(config->layer_name)) {
        return run_single_layer_conversion(config, model_dir, model_path, runtime);
    }

    return run_full_model_conversion(config, model_dir, model_path, (conversion_runtime_t *)runtime);
}

int transformer_run_ternary_conversion(const ternary_conversion_config_t *config) {
    conversion_runtime_t runtime;
    char *model_dir = NULL;
    char *model_path = NULL;
    int rc = -1;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "transformer_run_ternary_conversion");

    if (!config || !config->model_name || !config->output_path) {
        LOG_ERROR("ternary conversion: invalid configuration");
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    conversion_tracy_zone_text_if_present(&tracy_zone, config->model_name);

    log_conversion_config(config);

    model_dir = construct_safe_path("./models", config->model_name, NULL);
    if (!model_dir) {
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    if (init_conversion_runtime(config, model_dir, &runtime) != 0) {
        free(model_dir);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }
    model_path = construct_safe_path(model_dir, "model.safetensors", NULL);
    if (!model_path) {
        destroy_conversion_runtime(&runtime);
        free(model_dir);
        return conversion_tracy_end_status(&tracy_zone, -1);
    }

    rc = run_requested_conversion(config, model_dir, model_path, &runtime);
    free(model_path);
    free(model_dir);
    destroy_conversion_runtime(&runtime);
    return conversion_tracy_end_status(&tracy_zone, rc);
}
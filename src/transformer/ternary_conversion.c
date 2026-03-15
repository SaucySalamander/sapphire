/**
 * @file ternary_conversion.c
 * @brief Prompt 2 conversion entry-point shell.
 */

#include "ternary_conversion.h"

#include "calibration_corpus.h"
#include "file_reader.h"
#include "inference.h"
#include "log.h"
#include "model_reader.h"
#include "model_spec.h"
#include "ternary_calibration.h"
#include "ternary_io.h"
#include "ternary_validation.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    calibration_corpus_t corpus_storage;
    calibration_corpus_t validation_corpus_storage;
    ternary_calibration_corpus_t calibration_corpus;
    inference_context_t *activation_ctx;
    sapphire_tokenizer_t *tokenizer;
    model_spec_t *model_spec;
    sapphire_tokenizer_t *previous_tokenizer_handle;
    ternary_validation_state_t validation_state;
} conversion_runtime_t;

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

static transformer_ste_config_t default_runtime_ste_config(const ternary_conversion_config_t *config) {
    transformer_ste_config_t ste_config;

    ste_config.ste_steps = 3;
    ste_config.learning_rate = 0.03f;
    ste_config.zero_threshold = 0.05f;
    ste_config.momentum = 0.85f;
    ste_config.regularization_strength = 0.01f;
    ste_config.clip_value = 1.0f;
    ste_config.calibration_samples = (config && config->calibration_sample_limit > 0)
        ? config->calibration_sample_limit
        : 4;
    ste_config.kl_weight = (config && config->kl_weight >= 0.0f) ? config->kl_weight : 0.05f;
    return ste_config;
}

static void destroy_conversion_runtime(conversion_runtime_t *runtime) {
    if (!runtime) {
        return;
    }

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

    if (!config || !runtime || !runtime->activation_ctx) {
        return;
    }
    if (config->validate_every_n <= 0) {
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

    if ((!validation_manifest || validation_manifest[0] == '\0') &&
        (!validation_source || validation_source[0] == '\0')) {
        LOG_WARN("ternary validation: disabled because no validation corpus was provided");
        return;
    }
    if (load_runtime_corpus(validation_manifest,
                            validation_source,
                            validation_samples,
                            &runtime->validation_corpus_storage) != 0) {
        LOG_WARN("ternary validation: failed to load held-out corpus; disabling checkpoints");
        memset(&runtime->validation_corpus_storage, 0, sizeof(runtime->validation_corpus_storage));
        return;
    }

    memset(&validation_config, 0, sizeof(validation_config));
    validation_config.validate_every_n = config->validate_every_n;
    validation_config.output_dir = config->output_path;
    validation_config.sample_texts = (const char *const *)runtime->validation_corpus_storage.samples;
    validation_config.sample_count = runtime->validation_corpus_storage.sample_count;

    if (ternary_validation_init(&runtime->validation_state,
                                &validation_config,
                                runtime->activation_ctx) != 0) {
        LOG_WARN("ternary validation: failed to initialize checkpoint state; disabling checkpoints");
        ternary_validation_destroy(&runtime->validation_state);
    }
}

static int init_conversion_runtime(const ternary_conversion_config_t *config,
                                   const char *model_dir,
                                   conversion_runtime_t *out_runtime) {
    model_spec_t *spec = NULL;

    if (!config || !model_dir || !out_runtime) {
        LOG_ERROR("init_conversion_runtime: invalid arguments");
        return -1;
    }

    memset(out_runtime, 0, sizeof(*out_runtime));
    spec = get_model_spec(config->model_name);
    if (!spec) {
        LOG_ERROR("ternary conversion: failed to resolve model spec for %s", config->model_name);
        return -1;
    }

    out_runtime->activation_ctx = create_inference_context(0.0f,
                                                           0,
                                                           config->context_len > 0 ? config->context_len : 2048,
                                                           config->model_name);
    if (out_runtime->activation_ctx) {
        out_runtime->model_spec = out_runtime->activation_ctx->spec;
        out_runtime->tokenizer = out_runtime->activation_ctx->tokenizer;
        out_runtime->calibration_corpus.tokenizer = out_runtime->activation_ctx->tokenizer;
        out_runtime->calibration_corpus.model_spec = out_runtime->activation_ctx->spec;
        out_runtime->calibration_corpus.session = out_runtime->activation_ctx->session;
    } else {
        LOG_WARN("ternary conversion: failed to initialize activation replay context; using tokenized fallback vectors");
    }

    if (out_runtime->activation_ctx) {
        if ((config->calibration_corpus_manifest_path && config->calibration_corpus_manifest_path[0] != '\0') ||
            (config->calibration_corpus_path && config->calibration_corpus_path[0] != '\0')) {
            if (load_runtime_corpus(config->calibration_corpus_manifest_path,
                                    config->calibration_corpus_path,
                                    config->calibration_sample_limit,
                                    &out_runtime->corpus_storage) != 0) {
                destroy_conversion_runtime(out_runtime);
                return -1;
            }
            out_runtime->calibration_corpus.sample_texts =
                (const char *const *)out_runtime->corpus_storage.samples;
            out_runtime->calibration_corpus.sample_count = out_runtime->corpus_storage.sample_count;
        }
        init_conversion_validation(config, out_runtime);
        return 0;
    }

    out_runtime->tokenizer = tokenizer_load(model_dir);
    if (!out_runtime->tokenizer) {
        LOG_ERROR("ternary conversion: failed to load tokenizer from %s", model_dir);
        destroy_conversion_runtime(out_runtime);
        return -1;
    }

    out_runtime->model_spec = spec;
    out_runtime->previous_tokenizer_handle = spec->tokenizer_handle;
    spec->tokenizer_handle = out_runtime->tokenizer;
    out_runtime->calibration_corpus.tokenizer = out_runtime->tokenizer;
    out_runtime->calibration_corpus.model_spec = spec;
    out_runtime->calibration_corpus.session = NULL;

    if ((config->calibration_corpus_manifest_path && config->calibration_corpus_manifest_path[0] != '\0') ||
        (config->calibration_corpus_path && config->calibration_corpus_path[0] != '\0')) {
        if (load_runtime_corpus(config->calibration_corpus_manifest_path,
                                config->calibration_corpus_path,
                                config->calibration_sample_limit,
                                &out_runtime->corpus_storage) != 0) {
            destroy_conversion_runtime(out_runtime);
            return -1;
        }
        out_runtime->calibration_corpus.sample_texts =
            (const char *const *)out_runtime->corpus_storage.samples;
        out_runtime->calibration_corpus.sample_count = out_runtime->corpus_storage.sample_count;
    } else {
        LOG_WARN("ternary conversion: no calibration corpus provided; using built-in fallback prompts");
    }

    return 0;
}

static int run_single_layer_conversion(const ternary_conversion_config_t *config,
                                       const char *model_path,
                                       const conversion_runtime_t *runtime) {
    ternary_bf16_layer_map_t map;
    ternary_calibration_result_t result;
    ternary_layer_t layer;
    transformer_ste_config_t ste_config;
    uint32_t crc32 = 0;
    int rc = -1;

    memset(&map, 0, sizeof(map));
    memset(&result, 0, sizeof(result));
    memset(&layer, 0, sizeof(layer));

    if (strcmp(config->layer_name, "model.embed_tokens.weight") == 0 ||
        strcmp(config->layer_name, "lm_head.weight") == 0) {
        LOG_ERROR("Prompt 2 keeps embeddings / tied output weights in BF16; requested layer is excluded: %s",
                  config->layer_name);
        return -1;
    }

    if (io_mmap_layer_bf16(model_path, config->layer_name, &map) != 0) {
        return -1;
    }

    ste_config = default_runtime_ste_config(config);

    if (transformer_calibrate_layer_ste(map.bf16_weights,
                                        map.rows,
                                        map.cols,
                                        &ste_config,
                                        runtime ? &(ternary_calibration_corpus_t){
                                            .sample_texts = runtime->calibration_corpus.sample_texts,
                                            .sample_count = runtime->calibration_corpus.sample_count,
                                            .tokenizer = runtime->calibration_corpus.tokenizer,
                                            .model_spec = runtime->calibration_corpus.model_spec,
                                            .session = runtime->calibration_corpus.session,
                                            .tensor_name = config->layer_name
                                        } : NULL,
                                        &result) != 0) {
        io_unmap_layer_bf16(&map);
        return -1;
    }

    layer.packed_weights = result.packed_weights;
    layer.packed_weight_bytes = result.packed_weight_bytes;
    layer.scales = result.scales;
    layer.scale_count = result.rows;
    layer.scale_bytes = (size_t)result.rows * sizeof(float);
    layer.scale_dtype = SAFETENSORS_F32;
    layer.rows = result.rows;
    layer.cols = result.cols;
    layer.integrity = TERNARY_IO_INTEGRITY_CRC32;

    rc = io_write_layer_ternary(config->output_path, config->layer_name, &layer, &crc32);
    if (rc == 0) {
        LOG_INFO("Ternary conversion complete for %s (crc32=%08x)", config->layer_name, crc32);
    }

    transformer_free_ternary_calibration_result(&result);
    io_unmap_layer_bf16(&map);
    return rc;
}

typedef struct {
    const char *model_path;
    const char *output_dir;
    const char *tensor_name;
    const transformer_ste_config_t *ste_config;
    const ternary_calibration_corpus_t *calibration_corpus;
    ternary_calibration_result_t *out_result;
    uint32_t *out_crc32;
    int *out_skipped_vector;
} convert_tensor_job_t;

static int convert_tensor_to_dir(const convert_tensor_job_t *job) {
    ternary_bf16_layer_map_t map;
    ternary_calibration_result_t result;
    ternary_layer_t layer;
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
    memset(&layer, 0, sizeof(layer));

    if (io_mmap_layer_bf16(job->model_path, job->tensor_name, &map) != 0) {
        return -1;
    }

    if (map.cols == 1u) {
        LOG_INFO("Skipping vector tensor in full conversion: %s", job->tensor_name);
        if (job->out_skipped_vector) {
            *job->out_skipped_vector = 1;
        }
        io_unmap_layer_bf16(&map);
        return 0;
    }

    if (transformer_calibrate_layer_ste(map.bf16_weights,
                                        map.rows,
                                        map.cols,
                                        job->ste_config,
                                        job->calibration_corpus,
                                        &result) != 0) {
        io_unmap_layer_bf16(&map);
        return -1;
    }

    layer.packed_weights = result.packed_weights;
    layer.packed_weight_bytes = result.packed_weight_bytes;
    layer.scales = result.scales;
    layer.scale_count = result.rows;
    layer.scale_bytes = (size_t)result.rows * sizeof(float);
    layer.scale_dtype = SAFETENSORS_F32;
    layer.rows = result.rows;
    layer.cols = result.cols;
    layer.integrity = TERNARY_IO_INTEGRITY_CRC32;

    rc = io_write_layer_ternary_into_dir(job->output_dir, job->tensor_name, &layer, &crc32);
    if (rc == 0) {
        LOG_INFO("Converted tensor %s (crc32=%08x)", job->tensor_name, crc32);
        if (job->out_result) {
            *job->out_result = result;
            memset(&result, 0, sizeof(result));
        }
        if (job->out_crc32) {
            *job->out_crc32 = crc32;
        }
    }

    transformer_free_ternary_calibration_result(&result);
    io_unmap_layer_bf16(&map);
    return rc;
}

static int run_full_model_conversion(const ternary_conversion_config_t *config,
                                     const char *model_path,
                                     conversion_runtime_t *runtime) {
    model_spec_t *spec = NULL;
    transformer_ste_config_t ste_config;
    int converted = 0;
    int skipped_vectors = 0;
    int failed = 0;

    spec = runtime ? runtime->model_spec : get_model_spec(config->model_name);
    if (!spec || !spec->tensor_map || spec->tensor_map_size <= 0) {
        LOG_ERROR("Full ternary conversion: failed to resolve model spec for %s", config->model_name);
        return -1;
    }
    ste_config = default_runtime_ste_config(config);
    if (io_prepare_ternary_output_dir(config->output_path) != 0) {
        return -1;
    }

    for (int i = 0; i < spec->tensor_map_size; ++i) {
        const char *tensor_name = spec->tensor_map[i].hf_name;
        convert_tensor_job_t job;
        ternary_calibration_corpus_t active_corpus;
        ternary_calibration_result_t result;
        uint32_t crc32 = 0;
        int skipped_vector = 0;
        int rc = 0;

        memset(&result, 0, sizeof(result));

        if (!tensor_name || tensor_name[0] == '\0') {
            continue;
        }
        if (strcmp(tensor_name, "model.embed_tokens.weight") == 0) {
            LOG_INFO("Skipping embedding tensor in full conversion: %s", tensor_name);
            continue;
        }
        if (strcmp(tensor_name, "lm_head.weight") == 0) {
            LOG_INFO("Skipping tied output tensor in full conversion: %s", tensor_name);
            continue;
        }

        memset(&active_corpus, 0, sizeof(active_corpus));
        if (runtime) {
            active_corpus = runtime->calibration_corpus;
            active_corpus.tensor_name = tensor_name;
        }

        memset(&job, 0, sizeof(job));
        job.model_path = model_path;
        job.output_dir = config->output_path;
        job.tensor_name = tensor_name;
        job.ste_config = &ste_config;
        job.calibration_corpus = runtime ? &active_corpus : NULL;
        job.out_result = &result;
        job.out_crc32 = &crc32;
        job.out_skipped_vector = &skipped_vector;
        rc = convert_tensor_to_dir(&job);
        if (rc != 0) {
            LOG_WARN("Failed to convert tensor: %s", tensor_name);
            transformer_free_ternary_calibration_result(&result);
            failed++;
            continue;
        }
        if (skipped_vector) {
            transformer_free_ternary_calibration_result(&result);
            skipped_vectors++;
            continue;
        }
        converted++;

        if (runtime && runtime->validation_state.config.sample_count > 0) {
            if (ternary_validation_apply_proxy(&runtime->validation_state,
                                              tensor_name,
                                              &result,
                                              crc32,
                                              converted) != 0) {
                LOG_WARN("Validation checkpoint failed after tensor: %s", tensor_name);
            }
        }
        transformer_free_ternary_calibration_result(&result);
    }

    if (runtime && runtime->validation_state.config.sample_count > 0) {
        if (ternary_validation_finish(&runtime->validation_state, converted) != 0) {
            LOG_WARN("Validation final checkpoint failed");
        }
    }

    LOG_INFO("Full ternary conversion summary: converted=%d skipped_vectors=%d failed=%d",
             converted, skipped_vectors, failed);
    return (converted > 0) ? 0 : -1;
}

int transformer_run_ternary_conversion(const ternary_conversion_config_t *config) {
    conversion_runtime_t runtime;
    char *model_dir = NULL;
    char *model_path = NULL;
    int rc = -1;

    if (!config || !config->model_name || !config->output_path) {
        LOG_ERROR("ternary conversion: invalid configuration");
        return -1;
    }

    LOG_INFO("Ternary conversion mode selected");
    LOG_INFO("  model: %s", config->model_name);
    LOG_INFO("  output: %s", config->output_path);
    LOG_INFO("  context_len: %d", config->context_len);
    LOG_INFO("  calibration_samples: %d",
             (config->calibration_sample_limit > 0) ? config->calibration_sample_limit : 4);
    LOG_INFO("  validate_every: %d", config->validate_every_n);
    LOG_INFO("  kl_weight: %.4f", (double)((config->kl_weight >= 0.0f) ? config->kl_weight : 0.05f));
    if (config->layer_name && config->layer_name[0] != '\0') {
        LOG_INFO("  layer filter: %s", config->layer_name);
    } else {
        LOG_INFO("  layer filter: <all layers>");
    }
    if (config->calibration_corpus_path && config->calibration_corpus_path[0] != '\0') {
        LOG_INFO("  calibration_corpus: %s", config->calibration_corpus_path);
    } else {
        LOG_INFO("  calibration_corpus: <built-in fallback>");
    }
    if (config->calibration_corpus_manifest_path && config->calibration_corpus_manifest_path[0] != '\0') {
        LOG_INFO("  calibration_manifest: %s", config->calibration_corpus_manifest_path);
    }
    if (config->validation_corpus_path && config->validation_corpus_path[0] != '\0') {
        LOG_INFO("  validation_corpus: %s", config->validation_corpus_path);
    }
    if (config->validation_corpus_manifest_path && config->validation_corpus_manifest_path[0] != '\0') {
        LOG_INFO("  validation_manifest: %s", config->validation_corpus_manifest_path);
    }

    model_dir = construct_safe_path("./models", config->model_name, NULL);
    if (!model_dir) {
        return -1;
    }
    if (init_conversion_runtime(config, model_dir, &runtime) != 0) {
        free(model_dir);
        return -1;
    }
    model_path = construct_safe_path(model_dir, "model.safetensors", NULL);
    if (!model_path) {
        destroy_conversion_runtime(&runtime);
        free(model_dir);
        return -1;
    }

    if (config->layer_name && config->layer_name[0] != '\0') {
        rc = run_single_layer_conversion(config, model_path, &runtime);
    } else {
        rc = run_full_model_conversion(config, model_path, &runtime);
    }
    free(model_path);
    free(model_dir);
    destroy_conversion_runtime(&runtime);
    return rc;
}
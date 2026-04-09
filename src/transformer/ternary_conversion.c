/**
 * @file ternary_conversion.c
 * @brief Prompt 2 conversion entry-point shell.
 */

#include "ternary_conversion.h"

#include "calibration_corpus.h"
#include "activation_alignment.h"
#include "activation_tape.h"
#include "file_reader.h"
#include "inference.h"
#include "log.h"
#include "model_reader.h"
#include "model_spec.h"
#include "ternary_checkpoint.h"
#include "ternary_calibration.h"
#include "ternary_io.h"
#include "ternary_validation.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

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
    uint32_t resume_step_index;
} conversion_runtime_t;

static uint32_t config_resume_hash(const ternary_conversion_config_t *config)
{
    uint32_t crc32 = 0u;
    int ste_steps = 3;

    if (!config) {
        return 0u;
    }

    if (config->ste_steps > 0) {
        ste_steps = config->ste_steps;
    }

    crc32 = io_crc32_update(crc32, config->model_name, strlen(config->model_name) + 1u);
    crc32 = io_crc32_update(crc32, config->output_path, strlen(config->output_path) + 1u);
    crc32 = io_crc32_update(crc32, config->layer_name, config->layer_name ? strlen(config->layer_name) + 1u : 0u);
    crc32 = io_crc32_update(crc32, config->activation_tape_path, config->activation_tape_path ? strlen(config->activation_tape_path) + 1u : 0u);
    crc32 = io_crc32_update(crc32, config->teacher_model_name, config->teacher_model_name ? strlen(config->teacher_model_name) + 1u : 0u);
    crc32 = io_crc32_update(crc32, config->calibration_corpus_path, config->calibration_corpus_path ? strlen(config->calibration_corpus_path) + 1u : 0u);
    crc32 = io_crc32_update(crc32, config->calibration_corpus_manifest_path, config->calibration_corpus_manifest_path ? strlen(config->calibration_corpus_manifest_path) + 1u : 0u);
    crc32 = io_crc32_update(crc32, config->validation_corpus_path, config->validation_corpus_path ? strlen(config->validation_corpus_path) + 1u : 0u);
    crc32 = io_crc32_update(crc32, config->validation_corpus_manifest_path, config->validation_corpus_manifest_path ? strlen(config->validation_corpus_manifest_path) + 1u : 0u);
    crc32 = io_crc32_update(crc32, &config->context_len, sizeof(config->context_len));
    crc32 = io_crc32_update(crc32, &config->calibration_sample_limit, sizeof(config->calibration_sample_limit));
    crc32 = io_crc32_update(crc32, &config->validation_sample_limit, sizeof(config->validation_sample_limit));
    crc32 = io_crc32_update(crc32, &config->checkpoint_every_n_layers, sizeof(config->checkpoint_every_n_layers));
    crc32 = io_crc32_update(crc32, &config->validate_every_n, sizeof(config->validate_every_n));
    crc32 = io_crc32_update(crc32, &ste_steps, sizeof(ste_steps));
    crc32 = io_crc32_update(crc32, &config->kl_weight, sizeof(config->kl_weight));
    return crc32;
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

typedef struct {
    char tensor_name[256];
    char file_name[256];
    uint32_t rows;
    uint32_t cols;
    size_t packed_weight_bytes;
    uint32_t crc32;
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
            char *layer_path = NULL;
            ternary_layer_payload_t payload;
            uint32_t actual_crc32 = 0u;

            if (parse_resume_manifest_line(cursor, &entry) != 0) {
                free(manifest_text);
                return -1;
            }

            layer_path = construct_safe_path(config->output_path, entry.file_name, NULL);
            if (!layer_path) {
                free(manifest_text);
                return -1;
            }

            if (io_load_layer_ternary_payload(layer_path,
                                              entry.tensor_name,
                                              entry.rows,
                                              entry.cols,
                                              &payload) != 0) {
                free(layer_path);
                free(manifest_text);
                return -1;
            }

            actual_crc32 = io_crc32_update(0u, payload.packed_weights, payload.packed_weight_bytes);
            actual_crc32 = io_crc32_update(actual_crc32, payload.scales, payload.scale_bytes);
            if (actual_crc32 != entry.crc32) {
                LOG_ERROR("student update: manifest CRC mismatch for %s (expected=%08x actual=%08x)",
                          entry.tensor_name,
                          entry.crc32,
                          actual_crc32);
                io_free_layer_ternary_payload(&payload);
                free(layer_path);
                free(manifest_text);
                return -1;
            }

            if (ternary_validation_apply_proxy_from_payload(&runtime->validation_state,
                                                            entry.tensor_name,
                                                            &payload,
                                                            entry.crc32,
                                                            0) != 0) {
                io_free_layer_ternary_payload(&payload);
                free(layer_path);
                free(manifest_text);
                return -1;
            }

            io_free_layer_ternary_payload(&payload);
            free(layer_path);
            replayed++;
        }

        if (!line_end) {
            break;
        }
        cursor = line_end + 1;
    }

    if (replayed != (int)runtime->checkpoint_state.converted_tensor_count) {
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

static int init_checkpoint_state(const ternary_conversion_config_t *config,
                                 conversion_runtime_t *runtime)
{
    char *checkpoint_path = NULL;
    char *tmp_path = NULL;
    char *alignment_manifest_path = NULL;
    char *alignment_tape_path = NULL;
    int load_rc = 0;
    uint32_t total_layer_count = 0u;

    if (!config || !runtime) {
        return -1;
    }
    if (!runtime->model_spec) {
        return -1;
    }

    if (checkpoint_path_for_output(config->output_path, &checkpoint_path) != 0 ||
        checkpoint_tmp_path_for_output(config->output_path, &tmp_path) != 0) {
        free(checkpoint_path);
        return -1;
    }

    runtime->checkpoint_path = checkpoint_path;
    runtime->checkpoint_tmp_path = tmp_path;
    total_layer_count = (runtime->model_spec->tensor_map_size > 0)
        ? (uint32_t)runtime->model_spec->tensor_map_size
        : 0u;
    runtime->checkpoint_state.total_layer_count = total_layer_count;
    runtime->checkpoint_state.schema_version = TERNARY_STUDENT_CHECKPOINT_VERSION;
    runtime->checkpoint_state.config_hash = config_resume_hash(config);
    runtime->checkpoint_state.checkpoint_every_n_layers = (config->checkpoint_every_n_layers > 0)
        ? (uint32_t)config->checkpoint_every_n_layers
        : 1u;
    runtime->checkpoint_state.validate_every_n = (config->validate_every_n > 0) ? (uint32_t)config->validate_every_n : 0u;
    runtime->checkpoint_state.next_layer_index = 0u;
    runtime->checkpoint_state.last_completed_layer = 0u;
    runtime->checkpoint_state.converted_tensor_count = 0u;
    copy_text_field_local(runtime->checkpoint_state.model_name, sizeof(runtime->checkpoint_state.model_name), config->model_name);
    copy_text_field_local(runtime->checkpoint_state.teacher_model_name, sizeof(runtime->checkpoint_state.teacher_model_name), config->teacher_model_name);
    copy_text_field_local(runtime->checkpoint_state.output_dir, sizeof(runtime->checkpoint_state.output_dir), config->output_path);
    copy_text_field_local(runtime->checkpoint_state.activation_tape_path, sizeof(runtime->checkpoint_state.activation_tape_path), config->activation_tape_path);
    copy_text_field_local(runtime->checkpoint_state.calibration_corpus_path, sizeof(runtime->checkpoint_state.calibration_corpus_path), config->calibration_corpus_path);
    copy_text_field_local(runtime->checkpoint_state.calibration_corpus_manifest_path, sizeof(runtime->checkpoint_state.calibration_corpus_manifest_path), config->calibration_corpus_manifest_path);
    copy_text_field_local(runtime->checkpoint_state.validation_corpus_path, sizeof(runtime->checkpoint_state.validation_corpus_path), config->validation_corpus_path);
    copy_text_field_local(runtime->checkpoint_state.validation_corpus_manifest_path, sizeof(runtime->checkpoint_state.validation_corpus_manifest_path), config->validation_corpus_manifest_path);

    load_rc = ternary_student_checkpoint_load(checkpoint_path, &runtime->checkpoint_state);
    if (load_rc == 1) {
        LOG_INFO("student update: no checkpoint found at %s; starting fresh", checkpoint_path);
        if (manifest_path_exists(config->output_path) > 0) {
            LOG_ERROR("student update: output directory %s already contains a manifest but no checkpoint", config->output_path);
            return -1;
        }
        return 0;
    }
    if (load_rc != 0) {
        return -1;
    }
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
    if (runtime->checkpoint_state.config_hash != config_resume_hash(config)) {
        LOG_ERROR("student update: checkpoint config hash mismatch");
        return -1;
    }

    if (runtime->checkpoint_state.alignment_manifest_path[0] != '\0') {
        free(runtime->alignment_manifest_path);
        runtime->alignment_manifest_path = NULL;
        alignment_manifest_path = duplicate_text_local(runtime->checkpoint_state.alignment_manifest_path);
        if (!alignment_manifest_path) {
            return -1;
        }
        runtime->alignment_manifest_path = alignment_manifest_path;
        alignment_manifest_path = NULL;
    }
    if (runtime->checkpoint_state.alignment_tape_path[0] != '\0') {
        free(runtime->alignment_tape_path);
        runtime->alignment_tape_path = NULL;
        alignment_tape_path = duplicate_text_local(runtime->checkpoint_state.alignment_tape_path);
        if (!alignment_tape_path) {
            return -1;
        }
        runtime->alignment_tape_path = alignment_tape_path;
        alignment_tape_path = NULL;
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

    if (!config || !runtime || !runtime->checkpoint_path) {
        return -1;
    }

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
    checkpoint.alignment_manifest_crc32 = 0u;
    checkpoint.alignment_tape_provenance_hash = 0u;

    if (effective_alignment_manifest_path && effective_alignment_manifest_path[0] != '\0') {
        if (ternary_student_checkpoint_compute_manifest_crc32(effective_alignment_manifest_path,
                                                              &checkpoint.alignment_manifest_crc32) != 0) {
            return -1;
        }
    }
    if (effective_alignment_tape_path && effective_alignment_tape_path[0] != '\0' && runtime->activation_tape) {
        if (ternary_student_checkpoint_compute_tape_provenance_hash(&checkpoint,
                                                                    runtime->activation_tape,
                                                                    &checkpoint.alignment_tape_provenance_hash) != 0) {
            return -1;
        }
    }

    return ternary_student_checkpoint_write_atomic(runtime->checkpoint_path, &checkpoint);
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
    int rc = -1;

    if (!config || !runtime) {
        return -1;
    }

    if (!config->teacher_model_name || config->teacher_model_name[0] == '\0') {
        return 0;
    }

    if (config->layer_name && config->layer_name[0] != '\0') {
        LOG_ERROR("ternary alignment: --teacher-model requires full-model conversion");
        return -1;
    }
    if (!runtime->activation_tape) {
        LOG_ERROR("ternary alignment: --teacher-model requires --activation-tape");
        return -1;
    }

    teacher_tape = runtime->activation_tape;
    teacher_spec = get_model_spec(config->teacher_model_name);
    if (!teacher_spec) {
        LOG_ERROR("ternary alignment: failed to resolve teacher spec for %s", config->teacher_model_name);
        return -1;
    }

    memset(&request, 0, sizeof(request));
    request.teacher_spec = teacher_spec;
    request.student_spec = runtime->model_spec;
    request.teacher_tape = teacher_tape;
    request.depth_strategy = ACTIVATION_ALIGNMENT_DEPTH_BUCKET;
    request.width_strategy = ACTIVATION_ALIGNMENT_WIDTH_AUTO;

    if (activation_alignment_prepare_artifacts(&request,
                                              config->output_path,
                                              &aligned_tape_path,
                                              &aligned_manifest_path) != 0) {
        return -1;
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
    return rc;
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

static transformer_ste_config_t default_runtime_ste_config(const ternary_conversion_config_t *config) {
    transformer_ste_config_t ste_config;
    int ste_steps = 3;

    if (config && config->ste_steps > 0) {
        ste_steps = config->ste_steps;
    }

    ste_config.ste_steps = ste_steps;
    ste_config.learning_rate = 0.03f;
    ste_config.zero_threshold = 0.05f;
    ste_config.momentum = 0.85f;
    ste_config.regularization_strength = 0.01f;
    ste_config.non_collapse_weight = 0.02f;
    ste_config.zero_occupancy_floor = 0.75f;
    ste_config.clip_value = 1.0f;
    ste_config.calibration_samples = (config && config->calibration_sample_limit > 0)
        ? config->calibration_sample_limit
        : 4;
    ste_config.kl_weight = (config && config->kl_weight >= 0.0f) ? config->kl_weight : 0.05f;
    ste_config.telemetry_interval = 10;
    ste_config.telemetry_path = "./out/ternary_telemetry.jsonl";
    ste_config.telemetry = NULL;
    return ste_config;
}

static int tensor_name_is_layer_tensor(const char *tensor_name)
{
    return tensor_name && strstr(tensor_name, ".layers.") != NULL;
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

    if (runtime->activation_tape) {
        activation_tape_close(runtime->activation_tape);
    }
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

    if (config->activation_tape_path && config->activation_tape_path[0] != '\0') {
        out_runtime->activation_tape = activation_tape_open(config->activation_tape_path);
        if (!out_runtime->activation_tape) {
            LOG_ERROR("ternary conversion: failed to open activation tape %s", config->activation_tape_path);
            destroy_conversion_runtime(out_runtime);
            return -1;
        }
        out_runtime->activation_tape_hash = activation_tape_crc32(out_runtime->activation_tape);
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
        if (prepare_teacher_student_alignment(config, out_runtime) != 0) {
            destroy_conversion_runtime(out_runtime);
            return -1;
        }
        if (init_checkpoint_state(config, out_runtime) != 0) {
            destroy_conversion_runtime(out_runtime);
            return -1;
        }
        out_runtime->resume_step_index = out_runtime->checkpoint_state.next_layer_index;
        init_conversion_validation(config, out_runtime);
        if (resume_validation_from_manifest(config, out_runtime) != 0) {
            destroy_conversion_runtime(out_runtime);
            return -1;
        }
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

    if (prepare_teacher_student_alignment(config, out_runtime) != 0) {
        destroy_conversion_runtime(out_runtime);
        return -1;
    }

    if (init_checkpoint_state(config, out_runtime) != 0) {
        destroy_conversion_runtime(out_runtime);
        return -1;
    }

    out_runtime->resume_step_index = out_runtime->checkpoint_state.next_layer_index;

    init_conversion_validation(config, out_runtime);
    if (resume_validation_from_manifest(config, out_runtime) != 0) {
        destroy_conversion_runtime(out_runtime);
        return -1;
    }

    return 0;
}

static int mmap_tensor_for_conversion(const char *model_dir,
                                      const char *model_path,
                                      const char *tensor_name,
                                      ternary_bf16_layer_map_t *out_map);

static int run_single_layer_conversion(const ternary_conversion_config_t *config,
                                       const char *model_dir,
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

    if (mmap_tensor_for_conversion(model_dir, model_path, config->layer_name, &map) != 0) {
        return -1;
    }

    ste_config = default_runtime_ste_config(config);

    if (transformer_calibrate_layer_ste_with_tape(map.bf16_weights,
                                                  map.rows,
                                                  map.cols,
                                                  &ste_config,
                                                  runtime ? &(ternary_calibration_source_t){
                                                      .corpus = &(ternary_calibration_corpus_t){
                                                          .sample_texts = runtime->calibration_corpus.sample_texts,
                                                          .sample_count = runtime->calibration_corpus.sample_count,
                                                          .tokenizer = runtime->calibration_corpus.tokenizer,
                                                          .model_spec = runtime->calibration_corpus.model_spec,
                                                          .session = runtime->calibration_corpus.session,
                                                          .tensor_name = config->layer_name
                                                      },
                                                      .tape_context = (runtime->activation_tape && tensor_name_is_layer_tensor(config->layer_name))
                                                          ? &(ternary_activation_tape_context_t){
                                                                .tape = runtime->activation_tape,
                                                                .tensor_name = config->layer_name
                                                            }
                                                          : NULL
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
    const char *model_dir;
    const char *output_dir;
    const char *tensor_name;
    const activation_tape_t *activation_tape;
    const transformer_ste_config_t *ste_config;
    const ternary_calibration_corpus_t *calibration_corpus;
    ternary_calibration_result_t *out_result;
    uint32_t *out_crc32;
    int *out_skipped_vector;
    float *out_io_ms;
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
    const char *tensor_name;
    int *last_prefetched_entry_idx;
} full_model_tensor_task_t;

static int mmap_tensor_for_conversion(const char *model_dir,
                                      const char *model_path,
                                      const char *tensor_name,
                                      ternary_bf16_layer_map_t *out_map);

static int mmap_tensor_for_conversion(const char *model_dir,
                                      const char *model_path,
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
        return io_mmap_layer_bf16_sharded(model_dir, tensor_name, out_map);
    }

    if (model_path) {
        return io_mmap_layer_bf16(model_path, tensor_name, out_map);
    }

    return -1;
}

static int convert_tensor_to_dir(const convert_tensor_job_t *job) {
    ternary_bf16_layer_map_t map;
    ternary_calibration_result_t result;
    ternary_layer_t layer;
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
    memset(&layer, 0, sizeof(layer));

    measure_io = job->out_io_ms ? 1 : 0;
    if (measure_io && clock_gettime(CLOCK_MONOTONIC, &load_start) != 0) {
        measure_io = 0;
    }
    if (mmap_tensor_for_conversion(job->model_dir, job->model_path, job->tensor_name, &map) != 0) {
        return -1;
    }
    if (measure_io && clock_gettime(CLOCK_MONOTONIC, &load_end) == 0) {
        *job->out_io_ms += telemetry_elapsed_ms(&load_start, &load_end);
    }

    if (map.cols == 1u) {
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
            .tensor_name = job->tensor_name
        };

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

static int process_full_model_tensor_skip_other(const full_model_tensor_task_t *task,
                                                uint32_t converted_count,
                                                const char *skip_message)
{
    if (skip_message) {
        LOG_INFO(skip_message, task->tensor_name);
    }

    if (student_checkpoint_progress(task->config,
                                    task->runtime,
                                    (uint32_t)(task->layer_index + 1),
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
                                    (uint32_t)(task->layer_index + 1),
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
                                    (uint32_t)task->layer_index,
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
        if (ternary_validation_apply_proxy(&task->runtime->validation_state,
                                          task->tensor_name,
                                          result,
                                          crc32,
                                          (int)(converted_count + 1u)) != 0) {
            LOG_WARN("Validation checkpoint failed after tensor: %s", task->tensor_name);
        }
    }

    transformer_free_ternary_calibration_result(result);
    if (student_checkpoint_progress(task->config,
                                    task->runtime,
                                    (uint32_t)(task->layer_index + 1),
                                    converted_count + 1u,
                                    0) != 0) {
        LOG_WARN("student update: checkpoint write failed after tensor %s", task->tensor_name);
    }

    return FULL_MODEL_TENSOR_STATUS_CONVERTED;
}

static int process_full_model_tensor(const full_model_tensor_task_t *task)
{
    convert_tensor_job_t job;
    ternary_calibration_corpus_t active_corpus;
    ternary_calibration_result_t result;
    ternary_telemetry_t layer_telemetry;
    transformer_ste_config_t layer_ste_config;
    uint32_t crc32 = 0u;
    int skipped_vector = 0;
    int rc = 0;
    uint32_t converted_count = 0u;
    int use_layer_telemetry = 0;

    if (!task) {
        return -1;
    }

    if (task->runtime) {
        converted_count = task->runtime->checkpoint_state.converted_tensor_count;
    }

    memset(&result, 0, sizeof(result));
    memset(&layer_telemetry, 0, sizeof(layer_telemetry));
    memset(&layer_ste_config, 0, sizeof(layer_ste_config));

    if (!task->tensor_name || task->tensor_name[0] == '\0') {
        return process_full_model_tensor_skip_other(task, converted_count, NULL);
    }
    if (strcmp(task->tensor_name, "model.embed_tokens.weight") == 0) {
        LOG_INFO("Skipping embedding tensor in full conversion: %s", task->tensor_name);
        return process_full_model_tensor_skip_other(task,
                                                    converted_count,
                                                    "Skipping embedding tensor in full conversion: %s");
    }
    if (strcmp(task->tensor_name, "lm_head.weight") == 0) {
        LOG_INFO("Skipping tied output tensor in full conversion: %s", task->tensor_name);
        return process_full_model_tensor_skip_other(task,
                                                    converted_count,
                                                    "Skipping tied output tensor in full conversion: %s");
    }

    if (task->runtime && task->runtime->activation_tape && tensor_name_is_layer_tensor(task->tensor_name)) {
        use_layer_telemetry = 1;
        layer_ste_config = task->ste_config ? *task->ste_config : default_runtime_ste_config(task->config);
        telemetry_prepare_layer_context(&layer_telemetry, task->runtime, task->layer_index);
        layer_ste_config.telemetry = &layer_telemetry;
        prefetch_activation_tape_lookahead(task->runtime,
                                           task->spec,
                                           task->layer_index,
                                           2,
                                           task->last_prefetched_entry_idx,
                                           &layer_telemetry.io_ms);
    }

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
    job.ste_config = use_layer_telemetry ? &layer_ste_config : task->ste_config;
    job.calibration_corpus = task->runtime ? &active_corpus : NULL;
    job.out_result = &result;
    job.out_crc32 = &crc32;
    job.out_skipped_vector = &skipped_vector;
    job.out_io_ms = use_layer_telemetry ? &layer_telemetry.io_ms : NULL;

    rc = convert_tensor_to_dir(&job);
    if (rc != 0) {
        return process_full_model_tensor_failed(task, &result, converted_count);
    }

    if (skipped_vector) {
        return process_full_model_tensor_skipped_vector(task, &result, converted_count);
    }

    return process_full_model_tensor_complete(task, &result, converted_count, crc32);
}

static int run_full_model_conversion(const ternary_conversion_config_t *config,
                                     const char *model_dir,
                                     const char *model_path,
                                     conversion_runtime_t *runtime) {
    model_spec_t *spec = NULL;
    transformer_ste_config_t ste_config;
    int converted = 0;
    int skipped_vectors = 0;
    int start_layer = 0;

    spec = runtime ? runtime->model_spec : get_model_spec(config->model_name);
    if (!spec || !spec->tensor_map || spec->tensor_map_size <= 0) {
        LOG_ERROR("Full ternary conversion: failed to resolve model spec for %s", config->model_name);
        return -1;
    }
    if (runtime) {
        start_layer = (runtime->checkpoint_state.next_layer_index > 0)
            ? (int)runtime->checkpoint_state.next_layer_index
            : 0;
        converted = (int)runtime->checkpoint_state.converted_tensor_count;
        if (converted > 0) {
            LOG_INFO("student update: resuming from layer %d with %d converted tensors", start_layer, converted);
        }
    }
    ste_config = default_runtime_ste_config(config);
    if (io_prepare_ternary_output_dir(config->output_path) != 0) {
        return -1;
    }

    int last_prefetched_entry_idx = -1;

    for (int i = start_layer; i < spec->tensor_map_size; ++i) {
        const char *tensor_name = spec->tensor_map[i].hf_name;
        full_model_tensor_task_t task;
        int status = 0;

        memset(&task, 0, sizeof(task));
        task.config = config;
        task.runtime = runtime;
        task.spec = spec;
        task.ste_config = &ste_config;
        task.model_dir = model_dir;
        task.model_path = model_path;
        task.layer_index = i;
        task.tensor_name = tensor_name;
        task.last_prefetched_entry_idx = &last_prefetched_entry_idx;
        status = process_full_model_tensor(&task);

        if (status < 0) {
            return -1;
        }
        if (status == FULL_MODEL_TENSOR_STATUS_SKIPPED_VECTOR) {
            skipped_vectors++;
        } else if (status == FULL_MODEL_TENSOR_STATUS_CONVERTED) {
            converted++;
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
    LOG_INFO("  ste_steps: %d", (config->ste_steps > 0) ? config->ste_steps : 3);
    LOG_INFO("  validate_every: %d", config->validate_every_n);
    LOG_INFO("  kl_weight: %.4f", (double)((config->kl_weight >= 0.0f) ? config->kl_weight : 0.05f));
    if (config->layer_name && config->layer_name[0] != '\0') {
        LOG_INFO("  layer filter: %s", config->layer_name);
    } else {
        LOG_INFO("  layer filter: <all layers>");
    }
    if (config->activation_tape_path && config->activation_tape_path[0] != '\0') {
        LOG_INFO("  activation_tape: %s", config->activation_tape_path);
    }
    if (config->teacher_model_name && config->teacher_model_name[0] != '\0') {
        LOG_INFO("  teacher_model: %s", config->teacher_model_name);
    }
    if (config->calibration_corpus_manifest_path && config->calibration_corpus_manifest_path[0] != '\0') {
        if (config->calibration_corpus_path && config->calibration_corpus_path[0] != '\0') {
            LOG_INFO("  calibration_corpus: %s", config->calibration_corpus_path);
        } else {
            LOG_INFO("  calibration_corpus: <loaded from manifest>");
        }
        LOG_INFO("  calibration_manifest: %s", config->calibration_corpus_manifest_path);
    } else if (config->calibration_corpus_path && config->calibration_corpus_path[0] != '\0') {
        LOG_INFO("  calibration_corpus: %s", config->calibration_corpus_path);
    } else {
        LOG_INFO("  calibration_corpus: <built-in fallback>");
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
        rc = run_single_layer_conversion(config, model_dir, model_path, &runtime);
    } else {
        rc = run_full_model_conversion(config, model_dir, model_path, &runtime);
    }
    free(model_path);
    free(model_dir);
    destroy_conversion_runtime(&runtime);
    return rc;
}
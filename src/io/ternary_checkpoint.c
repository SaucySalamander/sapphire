/*
 * @file ternary_checkpoint.c
 * @brief Atomic student-update checkpoint I/O and provenance validation.
 */

#include "ternary_checkpoint.h"

#include "file_reader.h"
#include "log.h"
#include "ternary_io.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static size_t text_len(const char *text)
{
    return text ? strlen(text) : 0u;
}

static int append_line(char **cursor, size_t *remaining, const char *key, const char *value)
{
    int written = 0;

    if (!cursor || !*cursor || !remaining || !key) {
        return -1;
    }

    written = snprintf(*cursor, *remaining, "%s\t%s\n", key, value ? value : "");
    if (written < 0 || (size_t)written >= *remaining) {
        return -1;
    }

    *cursor += (size_t)written;
    *remaining -= (size_t)written;
    return 0;
}

static int append_u32_line(char **cursor, size_t *remaining, const char *key, uint32_t value)
{
    char value_buf[32];

    if (snprintf(value_buf, sizeof(value_buf), "%u", value) < 0) {
        return -1;
    }
    return append_line(cursor, remaining, key, value_buf);
}

static int append_hex32_line(char **cursor, size_t *remaining, const char *key, uint32_t value)
{
    char value_buf[32];

    if (snprintf(value_buf, sizeof(value_buf), "%08x", value) < 0) {
        return -1;
    }
    return append_line(cursor, remaining, key, value_buf);
}

static size_t checkpoint_payload_size(const ternary_student_update_checkpoint_t *checkpoint)
{
    size_t size = 0u;

    if (!checkpoint) {
        return 0u;
    }

    size += sizeof("checkpoint_version\t1\n") - 1u;
    size += sizeof("config_hash\t00000000\n") - 1u;
    size += sizeof("model_name\t\n") - 1u + text_len(checkpoint->model_name);
    size += sizeof("teacher_model_name\t\n") - 1u + text_len(checkpoint->teacher_model_name);
    size += sizeof("output_dir\t\n") - 1u + text_len(checkpoint->output_dir);
    size += sizeof("activation_tape_path\t\n") - 1u + text_len(checkpoint->activation_tape_path);
    size += sizeof("alignment_manifest_path\t\n") - 1u + text_len(checkpoint->alignment_manifest_path);
    size += sizeof("alignment_manifest_crc32\t00000000\n") - 1u;
    size += sizeof("alignment_tape_path\t\n") - 1u + text_len(checkpoint->alignment_tape_path);
    size += sizeof("alignment_tape_provenance_hash\t00000000\n") - 1u;
    size += sizeof("calibration_corpus_path\t\n") - 1u + text_len(checkpoint->calibration_corpus_path);
    size += sizeof("calibration_corpus_manifest_path\t\n") - 1u + text_len(checkpoint->calibration_corpus_manifest_path);
    size += sizeof("validation_corpus_path\t\n") - 1u + text_len(checkpoint->validation_corpus_path);
    size += sizeof("validation_corpus_manifest_path\t\n") - 1u + text_len(checkpoint->validation_corpus_manifest_path);
    size += sizeof("total_layer_count\t4294967295\n") - 1u;
    size += sizeof("next_layer_index\t4294967295\n") - 1u;
    size += sizeof("last_completed_layer\t4294967295\n") - 1u;
    size += sizeof("converted_tensor_count\t4294967295\n") - 1u;
    size += sizeof("checkpoint_every_n_layers\t4294967295\n") - 1u;
    size += sizeof("validate_every_n\t4294967295\n") - 1u;
    size += sizeof("checkpoint_crc32\t00000000\n") - 1u;
    return size;
}

static int build_checkpoint_payload(const ternary_student_update_checkpoint_t *checkpoint,
                                    char **out_payload,
                                    size_t *out_payload_size)
{
    char *payload = NULL;
    char *cursor = NULL;
    size_t remaining = 0u;
    size_t payload_size = 0u;

    if (!checkpoint || !out_payload || !out_payload_size) {
        return -1;
    }

    payload_size = checkpoint_payload_size(checkpoint);
    payload = (char *)malloc(payload_size + 1u);
    if (!payload) {
        LOG_ERROR("checkpoint: payload allocation failed");
        return -1;
    }

    cursor = payload;
    remaining = payload_size + 1u;
    if (append_u32_line(&cursor, &remaining, "checkpoint_version", TERNARY_STUDENT_CHECKPOINT_VERSION) != 0 ||
        append_hex32_line(&cursor, &remaining, "config_hash", checkpoint->config_hash) != 0 ||
        append_line(&cursor, &remaining, "model_name", checkpoint->model_name) != 0 ||
        append_line(&cursor, &remaining, "teacher_model_name", checkpoint->teacher_model_name) != 0 ||
        append_line(&cursor, &remaining, "output_dir", checkpoint->output_dir) != 0 ||
        append_line(&cursor, &remaining, "activation_tape_path", checkpoint->activation_tape_path) != 0 ||
        append_line(&cursor, &remaining, "alignment_manifest_path", checkpoint->alignment_manifest_path) != 0 ||
        append_hex32_line(&cursor, &remaining, "alignment_manifest_crc32", checkpoint->alignment_manifest_crc32) != 0 ||
        append_line(&cursor, &remaining, "alignment_tape_path", checkpoint->alignment_tape_path) != 0 ||
        append_hex32_line(&cursor, &remaining, "alignment_tape_provenance_hash", checkpoint->alignment_tape_provenance_hash) != 0 ||
        append_line(&cursor, &remaining, "calibration_corpus_path", checkpoint->calibration_corpus_path) != 0 ||
        append_line(&cursor, &remaining, "calibration_corpus_manifest_path", checkpoint->calibration_corpus_manifest_path) != 0 ||
        append_line(&cursor, &remaining, "validation_corpus_path", checkpoint->validation_corpus_path) != 0 ||
        append_line(&cursor, &remaining, "validation_corpus_manifest_path", checkpoint->validation_corpus_manifest_path) != 0 ||
        append_u32_line(&cursor, &remaining, "total_layer_count", checkpoint->total_layer_count) != 0 ||
        append_u32_line(&cursor, &remaining, "next_layer_index", checkpoint->next_layer_index) != 0 ||
        append_u32_line(&cursor, &remaining, "last_completed_layer", checkpoint->last_completed_layer) != 0 ||
        append_u32_line(&cursor, &remaining, "converted_tensor_count", checkpoint->converted_tensor_count) != 0 ||
        append_u32_line(&cursor, &remaining, "checkpoint_every_n_layers", checkpoint->checkpoint_every_n_layers) != 0 ||
        append_u32_line(&cursor, &remaining, "validate_every_n", checkpoint->validate_every_n) != 0) {
        free(payload);
        return -1;
    }

    *cursor = '\0';
    *out_payload = payload;
    *out_payload_size = payload_size;
    return 0;
}

static int build_temp_path(const char *path, char **out_tmp_path)
{
    size_t path_len = 0u;
    char *tmp_path = NULL;

    if (!path || !out_tmp_path) {
        return -1;
    }

    path_len = strlen(path);
    tmp_path = (char *)malloc(path_len + 5u);
    if (!tmp_path) {
        return -1;
    }

    if (snprintf(tmp_path, path_len + 5u, "%s.tmp", path) < 0) {
        free(tmp_path);
        return -1;
    }

    *out_tmp_path = tmp_path;
    return 0;
}

static int parse_u32_value(const char *value, uint32_t *out_value)
{
    char *end = NULL;
    unsigned long parsed = 0;

    if (!value || !out_value) {
        return -1;
    }

    errno = 0;
    parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > 0xFFFFFFFFul) {
        return -1;
    }

    *out_value = (uint32_t)parsed;
    return 0;
}

static int parse_hex_u32_value(const char *value, uint32_t *out_value)
{
    char *end = NULL;
    unsigned long parsed = 0;

    if (!value || !out_value) {
        return -1;
    }

    errno = 0;
    parsed = strtoul(value, &end, 16);
    if (errno != 0 || end == value || *end != '\0' || parsed > 0xFFFFFFFFul) {
        return -1;
    }

    *out_value = (uint32_t)parsed;
    return 0;
}

static int copy_line_field(char *dst, size_t dst_size, const char *value)
{
    if (!dst || dst_size == 0u || !value) {
        return -1;
    }
    if (strlen(value) >= dst_size) {
        return -1;
    }
    memcpy(dst, value, strlen(value) + 1u);
    return 0;
}

static int parse_checkpoint_line(ternary_student_update_checkpoint_t *checkpoint,
                                 const char *key,
                                 const char *value,
                                 uint32_t *stored_crc32,
                                 int *seen_crc32)
{
    if (!checkpoint || !key || !value || !stored_crc32 || !seen_crc32) {
        return -1;
    }

    if (strcmp(key, "checkpoint_version") == 0) {
        return parse_u32_value(value, &checkpoint->schema_version);
    }
    if (strcmp(key, "config_hash") == 0) {
        return parse_hex_u32_value(value, &checkpoint->config_hash);
    }
    if (strcmp(key, "model_name") == 0) {
        return copy_line_field(checkpoint->model_name, sizeof(checkpoint->model_name), value);
    }
    if (strcmp(key, "teacher_model_name") == 0) {
        return copy_line_field(checkpoint->teacher_model_name, sizeof(checkpoint->teacher_model_name), value);
    }
    if (strcmp(key, "output_dir") == 0) {
        return copy_line_field(checkpoint->output_dir, sizeof(checkpoint->output_dir), value);
    }
    if (strcmp(key, "activation_tape_path") == 0) {
        return copy_line_field(checkpoint->activation_tape_path, sizeof(checkpoint->activation_tape_path), value);
    }
    if (strcmp(key, "alignment_manifest_path") == 0) {
        return copy_line_field(checkpoint->alignment_manifest_path, sizeof(checkpoint->alignment_manifest_path), value);
    }
    if (strcmp(key, "alignment_manifest_crc32") == 0) {
        return parse_hex_u32_value(value, &checkpoint->alignment_manifest_crc32);
    }
    if (strcmp(key, "alignment_tape_path") == 0) {
        return copy_line_field(checkpoint->alignment_tape_path, sizeof(checkpoint->alignment_tape_path), value);
    }
    if (strcmp(key, "alignment_tape_provenance_hash") == 0) {
        return parse_hex_u32_value(value, &checkpoint->alignment_tape_provenance_hash);
    }
    if (strcmp(key, "calibration_corpus_path") == 0) {
        return copy_line_field(checkpoint->calibration_corpus_path, sizeof(checkpoint->calibration_corpus_path), value);
    }
    if (strcmp(key, "calibration_corpus_manifest_path") == 0) {
        return copy_line_field(checkpoint->calibration_corpus_manifest_path, sizeof(checkpoint->calibration_corpus_manifest_path), value);
    }
    if (strcmp(key, "validation_corpus_path") == 0) {
        return copy_line_field(checkpoint->validation_corpus_path, sizeof(checkpoint->validation_corpus_path), value);
    }
    if (strcmp(key, "validation_corpus_manifest_path") == 0) {
        return copy_line_field(checkpoint->validation_corpus_manifest_path, sizeof(checkpoint->validation_corpus_manifest_path), value);
    }
    if (strcmp(key, "total_layer_count") == 0) {
        return parse_u32_value(value, &checkpoint->total_layer_count);
    }
    if (strcmp(key, "next_layer_index") == 0) {
        return parse_u32_value(value, &checkpoint->next_layer_index);
    }
    if (strcmp(key, "last_completed_layer") == 0) {
        return parse_u32_value(value, &checkpoint->last_completed_layer);
    }
    if (strcmp(key, "converted_tensor_count") == 0) {
        return parse_u32_value(value, &checkpoint->converted_tensor_count);
    }
    if (strcmp(key, "checkpoint_every_n_layers") == 0) {
        return parse_u32_value(value, &checkpoint->checkpoint_every_n_layers);
    }
    if (strcmp(key, "validate_every_n") == 0) {
        return parse_u32_value(value, &checkpoint->validate_every_n);
    }
    if (strcmp(key, "checkpoint_crc32") == 0) {
        if (parse_hex_u32_value(value, stored_crc32) != 0) {
            return -1;
        }
        *seen_crc32 = 1;
        return 0;
    }

    return -1;
}

static int checkpoint_has_required_fields(const ternary_student_update_checkpoint_t *checkpoint)
{
    if (!checkpoint) {
        return 0;
    }
    if (checkpoint->schema_version != TERNARY_STUDENT_CHECKPOINT_VERSION) {
        return 0;
    }
    if (checkpoint->model_name[0] == '\0' || checkpoint->output_dir[0] == '\0') {
        return 0;
    }
    if (checkpoint->total_layer_count == 0u) {
        return 0;
    }
    if (checkpoint->next_layer_index > checkpoint->total_layer_count) {
        return 0;
    }
    return 1;
}

int ternary_student_checkpoint_write_atomic(const char *checkpoint_path,
                                            const ternary_student_update_checkpoint_t *checkpoint)
{
    char *payload = NULL;
    char *tmp_path = NULL;
    uint32_t payload_crc32 = 0u;
    char checksum_line[64];
    size_t payload_len = 0u;
    size_t checksum_len = 0u;
    int fd = -1;
    int rc = -1;

    if (!checkpoint_path || !checkpoint) {
        LOG_ERROR("checkpoint: invalid write arguments");
        return -1;
    }

    if (build_checkpoint_payload(checkpoint, &payload, &payload_len) != 0) {
        return -1;
    }
    payload_crc32 = io_crc32_update(0u, payload, payload_len);
    checksum_len = (size_t)snprintf(checksum_line,
                                    sizeof(checksum_line),
                                    "checkpoint_crc32\t%08x\n",
                                    payload_crc32);
    if (checksum_len == 0u || checksum_len >= sizeof(checksum_line)) {
        free(payload);
        return -1;
    }

    if (build_temp_path(checkpoint_path, &tmp_path) != 0) {
        free(payload);
        return -1;
    }

    fd = open(tmp_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("checkpoint: cannot open %s: %s", tmp_path, strerror(errno));
        goto write_cleanup;
    }

    if (write(fd, payload, payload_len) != (ssize_t)payload_len ||
        write(fd, checksum_line, checksum_len) != (ssize_t)checksum_len) {
        LOG_ERROR("checkpoint: write failed for %s: %s", tmp_path, strerror(errno));
        goto write_cleanup;
    }
    if (fsync(fd) != 0) {
        LOG_ERROR("checkpoint: fsync failed for %s: %s", tmp_path, strerror(errno));
        goto write_cleanup;
    }
    if (close(fd) != 0) {
        fd = -1;
        LOG_ERROR("checkpoint: close failed for %s: %s", tmp_path, strerror(errno));
        goto write_cleanup;
    }
    fd = -1;

    if (rename(tmp_path, checkpoint_path) != 0) {
        LOG_ERROR("checkpoint: rename %s -> %s failed: %s", tmp_path, checkpoint_path, strerror(errno));
        goto write_cleanup;
    }

    rc = 0;

write_cleanup:
    if (fd >= 0) {
        close(fd);
    }
    if (rc != 0 && tmp_path) {
        unlink(tmp_path);
    }
    free(tmp_path);
    free(payload);
    return rc;
}

int ternary_student_checkpoint_load(const char *checkpoint_path,
                                    ternary_student_update_checkpoint_t *out_checkpoint)
{
    char *buffer = NULL;
    size_t buffer_size = 0u;
    char *tmp_path = NULL;
    size_t pos = 0u;
    uint32_t stored_crc32 = 0u;
    int seen_crc32 = 0;
    uint32_t computed_crc32 = 0u;
    const char *checksum_key = "checkpoint_crc32";
    const size_t checksum_key_len = strlen(checksum_key);

    if (!checkpoint_path || !out_checkpoint) {
        return -1;
    }

    memset(out_checkpoint, 0, sizeof(*out_checkpoint));

    if (build_temp_path(checkpoint_path, &tmp_path) != 0) {
        return -1;
    }
    if (access(tmp_path, F_OK) == 0) {
        LOG_ERROR("checkpoint: temporary file present, refusing to resume: %s", tmp_path);
        free(tmp_path);
        return -1;
    }
    free(tmp_path);

    if (file_read_to_buffer(checkpoint_path, &buffer, &buffer_size) != 0) {
        return 1;
    }

    while (pos < buffer_size) {
        char *line = buffer + pos;
        char *newline = memchr(line, '\n', buffer_size - pos);
        char *tab = NULL;
        const char *value = NULL;
        size_t line_len = 0u;
        int is_checksum_line = 0;

        if (newline) {
            line_len = (size_t)(newline - line);
        } else {
            line_len = buffer_size - pos;
        }

        if (line_len >= checksum_key_len &&
            strncmp(line, checksum_key, checksum_key_len) == 0 &&
            line[checksum_key_len] == '\t') {
            is_checksum_line = 1;
        }

        if (!is_checksum_line && line_len > 0u) {
            computed_crc32 = io_crc32_update(computed_crc32,
                                             line,
                                             line_len + (newline ? 1u : 0u));
        }

        if (newline) {
            *newline = '\0';
            pos = (size_t)(newline - buffer) + 1u;
        } else {
            pos = buffer_size;
        }

        if (line[0] == '\0') {
            continue;
        }

        tab = strchr(line, '\t');
        if (!tab || tab == line || tab[1] == '\0') {
            LOG_ERROR("checkpoint: malformed line in %s", checkpoint_path);
            free(buffer);
            return -1;
        }
        *tab = '\0';
        value = tab + 1;

        if (is_checksum_line) {
            if (parse_hex_u32_value(value, &stored_crc32) != 0) {
                LOG_ERROR("checkpoint: invalid checksum line in %s", checkpoint_path);
                free(buffer);
                return -1;
            }
            seen_crc32 = 1;
            continue;
        }

        if (parse_checkpoint_line(out_checkpoint, line, value, &stored_crc32, &seen_crc32) != 0) {
            LOG_ERROR("checkpoint: failed to parse field %s", line);
            free(buffer);
            return -1;
        }
    }

    if (!seen_crc32) {
        LOG_ERROR("checkpoint: checksum missing in %s", checkpoint_path);
        free(buffer);
        return -1;
    }

    if (!checkpoint_has_required_fields(out_checkpoint)) {
        LOG_ERROR("checkpoint: required fields missing or invalid in %s", checkpoint_path);
        free(buffer);
        return -1;
    }

    if (computed_crc32 != stored_crc32) {
        LOG_ERROR("checkpoint: checksum mismatch in %s", checkpoint_path);
        free(buffer);
        return -1;
    }

    out_checkpoint->schema_version = TERNARY_STUDENT_CHECKPOINT_VERSION;
    free(buffer);
    return 0;
}

int ternary_student_checkpoint_compute_manifest_crc32(const char *manifest_path,
                                                      uint32_t *out_crc32)
{
    char *buffer = NULL;
    size_t buffer_size = 0u;
    uint32_t crc32 = 0u;

    if (!manifest_path || !out_crc32) {
        return -1;
    }

    if (file_read_to_buffer(manifest_path, &buffer, &buffer_size) != 0) {
        return -1;
    }

    crc32 = io_crc32_update(0u, buffer, buffer_size);
    free(buffer);
    *out_crc32 = crc32;
    return 0;
}

int ternary_student_checkpoint_compute_tape_provenance_hash(const ternary_student_update_checkpoint_t *checkpoint,
                                                            const activation_tape_t *alignment_tape,
                                                            uint32_t *out_hash)
{
    const tape_file_header_t *header = NULL;
    uint32_t crc32 = 0u;

    if (!checkpoint || !out_hash || !alignment_tape) {
        return -1;
    }

    header = activation_tape_header(alignment_tape);
    if (!header) {
        return -1;
    }

    crc32 = io_crc32_update(crc32, checkpoint->model_name, text_len(checkpoint->model_name) + 1u);
    crc32 = io_crc32_update(crc32, checkpoint->teacher_model_name, text_len(checkpoint->teacher_model_name) + 1u);
    crc32 = io_crc32_update(crc32, checkpoint->output_dir, text_len(checkpoint->output_dir) + 1u);
    crc32 = io_crc32_update(crc32, checkpoint->activation_tape_path, text_len(checkpoint->activation_tape_path) + 1u);
    crc32 = io_crc32_update(crc32, checkpoint->alignment_manifest_path, text_len(checkpoint->alignment_manifest_path) + 1u);
    crc32 = io_crc32_update(crc32, &checkpoint->alignment_manifest_crc32, sizeof(checkpoint->alignment_manifest_crc32));
    crc32 = io_crc32_update(crc32, checkpoint->alignment_tape_path, text_len(checkpoint->alignment_tape_path) + 1u);
    crc32 = io_crc32_update(crc32, &header->entry_count, sizeof(header->entry_count));
    crc32 = io_crc32_update(crc32, &header->sample_count, sizeof(header->sample_count));
    crc32 = io_crc32_update(crc32, &header->hidden_size, sizeof(header->hidden_size));
    crc32 = io_crc32_update(crc32, &header->data_section_offset, sizeof(header->data_section_offset));
    crc32 = io_crc32_update(crc32, &header->data_section_size, sizeof(header->data_section_size));
    crc32 = io_crc32_update(crc32, &header->crc32, sizeof(header->crc32));
    *out_hash = crc32;
    return 0;
}

int ternary_student_checkpoint_validate_alignment(const ternary_student_update_checkpoint_t *checkpoint,
                                                  const activation_tape_t *alignment_tape)
{
    uint32_t actual_manifest_crc32 = 0u;
    uint32_t actual_provenance_hash = 0u;

    if (!checkpoint) {
        return -1;
    }
    if (checkpoint->teacher_model_name[0] == '\0') {
        return 0;
    }
    if (checkpoint->alignment_manifest_path[0] == '\0' ||
        checkpoint->alignment_tape_path[0] == '\0' ||
        !alignment_tape) {
        LOG_ERROR("checkpoint: alignment artifacts are missing from checkpoint");
        return -1;
    }

    if (ternary_student_checkpoint_compute_manifest_crc32(checkpoint->alignment_manifest_path,
                                                          &actual_manifest_crc32) != 0) {
        LOG_ERROR("checkpoint: failed to verify alignment manifest %s",
                  checkpoint->alignment_manifest_path);
        return -1;
    }
    if (actual_manifest_crc32 != checkpoint->alignment_manifest_crc32) {
        LOG_ERROR("checkpoint: alignment manifest CRC mismatch (expected=%08x actual=%08x)",
                  checkpoint->alignment_manifest_crc32,
                  actual_manifest_crc32);
        return -1;
    }
    if (ternary_student_checkpoint_compute_tape_provenance_hash(checkpoint,
                                                                 alignment_tape,
                                                                 &actual_provenance_hash) != 0) {
        LOG_ERROR("checkpoint: failed to compute alignment tape provenance hash");
        return -1;
    }
    if (actual_provenance_hash != checkpoint->alignment_tape_provenance_hash) {
        LOG_ERROR("checkpoint: alignment tape provenance mismatch (expected=%08x actual=%08x)",
                  checkpoint->alignment_tape_provenance_hash,
                  actual_provenance_hash);
        return -1;
    }

    return 0;
}
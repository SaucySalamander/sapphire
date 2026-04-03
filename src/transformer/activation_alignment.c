/*
 * @file activation_alignment.c
 * @brief Deterministic teacher-to-student activation alignment writer.
 */

#include "activation_alignment.h"

#include "file_reader.h"
#include "gemma3_270m_config.h"
#include "log.h"
#include "ternary_io.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    const char *suffix;
    capture_target_t target;
    int alias_idx;
} alignment_wt_t;

static const alignment_wt_t k_alignment_wt[7] = {
    {"self_attn.q_proj.weight",  CAPTURE_TARGET_QKV_INPUT,  -1},
    {"self_attn.k_proj.weight",  CAPTURE_TARGET_QKV_INPUT,   0},
    {"self_attn.v_proj.weight",  CAPTURE_TARGET_QKV_INPUT,   0},
    {"self_attn.o_proj.weight",  CAPTURE_TARGET_OUT_INPUT,  -1},
    {"mlp.gate_proj.weight",     CAPTURE_TARGET_FFN_INPUT,  -1},
    {"mlp.up_proj.weight",       CAPTURE_TARGET_FFN_INPUT,   4},
    {"mlp.down_proj.weight",     CAPTURE_TARGET_DOWN_INPUT, -1},
};

static const char *capture_target_name(capture_target_t target)
{
    switch (target) {
    case CAPTURE_TARGET_QKV_INPUT:
        return "qkv_input";
    case CAPTURE_TARGET_OUT_INPUT:
        return "out_input";
    case CAPTURE_TARGET_FFN_INPUT:
        return "ffn_input";
    case CAPTURE_TARGET_DOWN_INPUT:
        return "down_input";
    default:
        return "unknown";
    }
}

static const char *depth_strategy_name(activation_alignment_depth_strategy_t strategy)
{
    switch (strategy) {
    case ACTIVATION_ALIGNMENT_DEPTH_BUCKET:
        return "bucket";
    case ACTIVATION_ALIGNMENT_DEPTH_REPEAT:
        return "repeat";
    default:
        return "unknown";
    }
}

static const char *width_strategy_name(activation_alignment_width_strategy_t strategy)
{
    switch (strategy) {
    case ACTIVATION_ALIGNMENT_WIDTH_BLOCK_REPLICATION:
        return "block_replication";
    case ACTIVATION_ALIGNMENT_WIDTH_INTERPOLATION:
        return "interpolation";
    case ACTIVATION_ALIGNMENT_WIDTH_AUTO:
        return "auto";
    default:
        return "unknown";
    }
}

static int write_all(int fd, const void *buffer, size_t size)
{
    const uint8_t *cursor = (const uint8_t *)buffer;

    while (size > 0) {
        ssize_t written = write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (written == 0) {
            return -1;
        }
        cursor += (size_t)written;
        size -= (size_t)written;
    }

    return 0;
}

static void sanitize_component(const char *input, char *output, size_t output_size)
{
    size_t i = 0;

    if (!output || output_size == 0) {
        return;
    }

    output[0] = '\0';
    if (!input) {
        return;
    }

    for (; input[i] != '\0' && i + 1 < output_size; ++i) {
        char ch = input[i];

        if ((ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '-' || ch == '_') {
            output[i] = ch;
        } else {
            output[i] = '_';
        }
    }
    output[i] = '\0';
}

static uint32_t layer_type_for_idx(uint32_t layer_idx)
{
    return ((layer_idx % 6u) == 5u) ? 1u : 0u;
}

static uint32_t target_dim_for_config(capture_target_t target, const gemma3_270m_config_t *cfg)
{
    if (!cfg) {
        return 0u;
    }

    switch (target) {
    case CAPTURE_TARGET_QKV_INPUT:
        return (uint32_t)cfg->hidden_size;
    case CAPTURE_TARGET_OUT_INPUT:
        return (uint32_t)(cfg->num_attention_heads * cfg->head_dim);
    case CAPTURE_TARGET_FFN_INPUT:
        return (uint32_t)cfg->hidden_size;
    case CAPTURE_TARGET_DOWN_INPUT:
        return (uint32_t)cfg->intermediate_size;
    default:
        return 0u;
    }
}

static int copy_tensor_prefix(const model_spec_t *spec, char *out_prefix, size_t out_prefix_size)
{
    const char *layer_marker = ".layers.";
    size_t layer_marker_len = strlen(layer_marker);

    if (!spec || !spec->tensor_map || !out_prefix || out_prefix_size == 0) {
        return -1;
    }

    out_prefix[0] = '\0';
    for (size_t i = 0; spec->tensor_map[i].hf_name; ++i) {
        const char *name = spec->tensor_map[i].hf_name;
        const char *layer_pos = name ? strstr(name, layer_marker) : NULL;
        size_t prefix_len = 0;

        if (!layer_pos) {
            continue;
        }

        prefix_len = (size_t)((layer_pos - name) + layer_marker_len);
        if (prefix_len == 0 || prefix_len >= out_prefix_size) {
            return -1;
        }
        memcpy(out_prefix, name, prefix_len);
        out_prefix[prefix_len] = '\0';
        return 0;
    }

    return -1;
}

static uint32_t infer_layer_count(const model_spec_t *spec)
{
    const gemma3_270m_config_t *cfg = NULL;
    const char *layer_marker = ".layers.";

    if (!spec) {
        return 0u;
    }

    cfg = (const gemma3_270m_config_t *)spec->variant_config;
    if (cfg && cfg->num_hidden_layers > 0) {
        return (uint32_t)cfg->num_hidden_layers;
    }

    if (spec->tensor_map_size <= 0 || !spec->tensor_map) {
        return 0u;
    }

    {
        uint32_t max_layer = 0u;
        int seen = 0;
        const size_t layer_marker_len = strlen(layer_marker);

        for (int i = 0; i < spec->tensor_map_size; ++i) {
            const char *name = spec->tensor_map[i].hf_name;
            const char *layer_pos = name ? strstr(name, layer_marker) : NULL;
            char *end = NULL;
            long layer_idx = 0;

            if (!layer_pos) {
                continue;
            }

            layer_idx = strtol(layer_pos + layer_marker_len, &end, 10);
            if (end == layer_pos + layer_marker_len || layer_idx < 0) {
                continue;
            }

            if (!seen || (uint32_t)layer_idx > max_layer) {
                max_layer = (uint32_t)layer_idx;
                seen = 1;
            }
        }

        if (seen) {
            return max_layer + 1u;
        }
    }

    return 0u;
}

static uint32_t map_teacher_layer(uint32_t student_layer,
                                  uint32_t teacher_layer_count,
                                  uint32_t student_layer_count,
                                  activation_alignment_depth_strategy_t depth_strategy)
{
    if (teacher_layer_count == 0u || student_layer_count == 0u) {
        return 0u;
    }

    if (depth_strategy == ACTIVATION_ALIGNMENT_DEPTH_REPEAT) {
        return student_layer % teacher_layer_count;
    }

    return (student_layer * teacher_layer_count) / student_layer_count;
}

static int format_tensor_name(char *out,
                              size_t out_size,
                              const char *prefix,
                              uint32_t layer_idx,
                              const char *suffix)
{
    int written = 0;

    if (!out || !prefix || !suffix || out_size == 0) {
        return -1;
    }

    written = snprintf(out, out_size, "%s%u.%s", prefix, layer_idx, suffix);
    if (written < 0 || (size_t)written >= out_size) {
        return -1;
    }
    return 0;
}

static uint32_t primary_entry_index(uint32_t layer_idx, int wt_idx)
{
    return (uint32_t)(layer_idx * 7u + (uint32_t)wt_idx);
}

static activation_alignment_width_strategy_t resolve_width_strategy(activation_alignment_width_strategy_t requested,
                                                                     uint32_t source_dim,
                                                                     uint32_t target_dim)
{
    if (requested == ACTIVATION_ALIGNMENT_WIDTH_INTERPOLATION) {
        return ACTIVATION_ALIGNMENT_WIDTH_INTERPOLATION;
    }

    if (source_dim > 0u && target_dim >= source_dim && (target_dim % source_dim) == 0u) {
        return ACTIVATION_ALIGNMENT_WIDTH_BLOCK_REPLICATION;
    }

    return ACTIVATION_ALIGNMENT_WIDTH_INTERPOLATION;
}

static void resample_vector(const float *source,
                            uint32_t source_dim,
                            float *target,
                            uint32_t target_dim,
                            activation_alignment_width_strategy_t strategy)
{
    uint32_t i = 0u;

    if (!source || !target || source_dim == 0u || target_dim == 0u) {
        return;
    }

    if (source_dim == target_dim) {
        memcpy(target, source, (size_t)source_dim * sizeof(float));
        return;
    }

    if (strategy == ACTIVATION_ALIGNMENT_WIDTH_BLOCK_REPLICATION &&
        target_dim >= source_dim && (target_dim % source_dim) == 0u) {
        uint32_t repeat = target_dim / source_dim;

        for (i = 0u; i < source_dim; ++i) {
            uint32_t r = 0u;
            for (r = 0u; r < repeat; ++r) {
                target[i * repeat + r] = source[i];
            }
        }
        return;
    }

    if (source_dim == 1u || target_dim == 1u) {
        for (i = 0u; i < target_dim; ++i) {
            target[i] = source[0];
        }
        return;
    }

    for (i = 0u; i < target_dim; ++i) {
        double position = ((double)i * (double)(source_dim - 1u)) / (double)(target_dim - 1u);
        uint32_t left = (uint32_t)position;
        uint32_t right = (left + 1u < source_dim) ? (left + 1u) : left;
        double mix = position - (double)left;

        target[i] = (float)(((1.0 - mix) * (double)source[left]) + (mix * (double)source[right]));
    }
}

static int validate_request(const activation_alignment_request_t *request)
{
    const gemma3_270m_config_t *student_cfg = NULL;

    if (!request || !request->teacher_spec || !request->student_spec || !request->teacher_tape) {
        LOG_ERROR("activation alignment: invalid request");
        return -1;
    }
    if (!request->teacher_spec->model_id || !request->student_spec->model_id) {
        LOG_ERROR("activation alignment: model identifiers are missing");
        return -1;
    }
    if (!request->teacher_spec->tensor_map || !request->student_spec->tensor_map) {
        LOG_ERROR("activation alignment: tensor maps are missing");
        return -1;
    }
    student_cfg = (const gemma3_270m_config_t *)request->student_spec->variant_config;
    if (!student_cfg || student_cfg->num_hidden_layers <= 0) {
        LOG_ERROR("activation alignment: student variant_config is missing");
        return -1;
    }
    if (activation_tape_sample_count(request->teacher_tape) <= 0) {
        LOG_ERROR("activation alignment: teacher tape has no samples");
        return -1;
    }
    if (infer_layer_count(request->teacher_spec) == 0u || infer_layer_count(request->student_spec) == 0u) {
        LOG_ERROR("activation alignment: could not infer teacher/student layer counts");
        return -1;
    }
    return 0;
}

int activation_alignment_build_manifest(activation_alignment_manifest_t *manifest,
                                        const activation_alignment_request_t *request)
{
    const gemma3_270m_config_t *student_cfg = NULL;
    char teacher_prefix[128];
    char student_prefix[128];
    uint32_t teacher_layer_count = 0u;
    uint32_t student_layer_count = 0u;
    uint32_t sample_count = 0u;
    size_t entry_count = 0u;
    uint64_t current_offset = 0u;

    if (!manifest) {
        LOG_ERROR("activation alignment: manifest is NULL");
        return -1;
    }

    memset(manifest, 0, sizeof(*manifest));
    if (validate_request(request) != 0) {
        return -1;
    }

    student_cfg = (const gemma3_270m_config_t *)request->student_spec->variant_config;
    teacher_layer_count = infer_layer_count(request->teacher_spec);
    student_layer_count = infer_layer_count(request->student_spec);
    sample_count = (uint32_t)activation_tape_sample_count(request->teacher_tape);

    if (copy_tensor_prefix(request->teacher_spec, teacher_prefix, sizeof(teacher_prefix)) != 0 ||
        copy_tensor_prefix(request->student_spec, student_prefix, sizeof(student_prefix)) != 0) {
        LOG_ERROR("activation alignment: failed to resolve tensor prefixes");
        return -1;
    }

    entry_count = (size_t)student_layer_count * 7u;
    manifest->entries = (activation_alignment_entry_t *)calloc(entry_count, sizeof(*manifest->entries));
    if (!manifest->entries) {
        LOG_ERROR("activation alignment: manifest allocation failed");
        return -1;
    }

    if (snprintf(manifest->teacher_model_id,
                 sizeof(manifest->teacher_model_id),
                 "%s",
                 request->teacher_spec->model_id) < 0 ||
        snprintf(manifest->student_model_id,
                 sizeof(manifest->student_model_id),
                 "%s",
                 request->student_spec->model_id) < 0 ||
        snprintf(manifest->teacher_prefix,
                 sizeof(manifest->teacher_prefix),
                 "%s",
                 teacher_prefix) < 0 ||
        snprintf(manifest->student_prefix,
                 sizeof(manifest->student_prefix),
                 "%s",
                 student_prefix) < 0) {
        LOG_ERROR("activation alignment: failed to copy metadata");
        activation_alignment_manifest_free(manifest);
        return -1;
    }

    manifest->teacher_layer_count = teacher_layer_count;
    manifest->student_layer_count = student_layer_count;
    manifest->sample_count = sample_count;
    manifest->depth_strategy = request->depth_strategy;
    manifest->width_strategy = request->width_strategy;
    manifest->entry_count = entry_count;

    for (uint32_t student_layer_idx = 0u; student_layer_idx < student_layer_count; ++student_layer_idx) {
        uint32_t teacher_layer_idx = map_teacher_layer(student_layer_idx,
                                                       teacher_layer_count,
                                                       student_layer_count,
                                                       request->depth_strategy);

        if (teacher_layer_idx >= teacher_layer_count) {
            LOG_ERROR("activation alignment: invalid depth mapping for student layer %u", student_layer_idx);
            activation_alignment_manifest_free(manifest);
            return -1;
        }

        for (int wt_idx = 0; wt_idx < 7; ++wt_idx) {
            activation_alignment_entry_t *entry = &manifest->entries[primary_entry_index(student_layer_idx, wt_idx)];
            uint32_t source_dim = 0u;
            uint32_t target_dim = 0u;

            if (format_tensor_name(entry->teacher_tensor_name,
                                   sizeof(entry->teacher_tensor_name),
                                   manifest->teacher_prefix,
                                   teacher_layer_idx,
                                   k_alignment_wt[wt_idx].suffix) != 0 ||
                format_tensor_name(entry->student_tensor_name,
                                   sizeof(entry->student_tensor_name),
                                   manifest->student_prefix,
                                   student_layer_idx,
                                   k_alignment_wt[wt_idx].suffix) != 0) {
                LOG_ERROR("activation alignment: failed to build tensor names");
                activation_alignment_manifest_free(manifest);
                return -1;
            }

            source_dim = activation_tape_vector_dim(request->teacher_tape, entry->teacher_tensor_name);
            target_dim = target_dim_for_config(k_alignment_wt[wt_idx].target, student_cfg);
            if (source_dim == 0u || target_dim == 0u) {
                LOG_ERROR("activation alignment: shape mismatch for %s -> %s",
                          entry->teacher_tensor_name,
                          entry->student_tensor_name);
                activation_alignment_manifest_free(manifest);
                return -1;
            }

            entry->teacher_layer_idx = teacher_layer_idx;
            entry->student_layer_idx = student_layer_idx;
            entry->target = k_alignment_wt[wt_idx].target;
            entry->teacher_entry_idx = primary_entry_index(teacher_layer_idx, wt_idx);
            entry->student_entry_idx = primary_entry_index(student_layer_idx, wt_idx);
            entry->source_dim = source_dim;
            entry->target_dim = target_dim;
            entry->sample_count = sample_count;
            entry->alias_of_entry = (k_alignment_wt[wt_idx].alias_idx < 0)
                ? TAPE_NO_ALIAS
                : primary_entry_index(student_layer_idx, k_alignment_wt[wt_idx].alias_idx);
            entry->teacher_layer_type = layer_type_for_idx(teacher_layer_idx);
            entry->student_layer_type = layer_type_for_idx(student_layer_idx);

            if (entry->alias_of_entry == TAPE_NO_ALIAS) {
                uint64_t bytes = (uint64_t)entry->target_dim * (uint64_t)entry->sample_count * (uint64_t)sizeof(float);

                if (bytes == 0u || current_offset > UINT64_MAX - bytes) {
                    LOG_ERROR("activation alignment: tape data section overflow");
                    activation_alignment_manifest_free(manifest);
                    return -1;
                }
                current_offset += bytes;
            }
        }
    }

    return 0;
}

void activation_alignment_manifest_free(activation_alignment_manifest_t *manifest)
{
    if (!manifest) {
        return;
    }

    free(manifest->entries);
    memset(manifest, 0, sizeof(*manifest));
}

static int write_manifest_line(int fd, const char *line)
{
    size_t line_len = line ? strlen(line) : 0u;

    if (!line || line_len == 0u) {
        return 0;
    }
    return write_all(fd, line, line_len);
}

typedef struct {
    const char *key;
    const char *value;
} manifest_text_field_t;

typedef struct {
    const char *key;
    uint32_t value;
} manifest_u32_field_t;

static int write_manifest_text_fields(int fd,
                                      const manifest_text_field_t *fields,
                                      size_t field_count)
{
    char line[1024];

    for (size_t i = 0; i < field_count; ++i) {
        int written = snprintf(line,
                               sizeof(line),
                               "# %s\t%s\n",
                               fields[i].key,
                               fields[i].value ? fields[i].value : "");
        if (written < 0 || (size_t)written >= sizeof(line) || write_manifest_line(fd, line) != 0) {
            return -1;
        }
    }

    return 0;
}

static int write_manifest_u32_fields(int fd,
                                     const manifest_u32_field_t *fields,
                                     size_t field_count)
{
    char line[128];

    for (size_t i = 0; i < field_count; ++i) {
        int written = snprintf(line,
                               sizeof(line),
                               "# %s\t%u\n",
                               fields[i].key,
                               fields[i].value);
        if (written < 0 || (size_t)written >= sizeof(line) || write_manifest_line(fd, line) != 0) {
            return -1;
        }
    }

    return 0;
}

static int write_manifest_entry_header(int fd)
{
    return write_manifest_line(fd,
                               "student_entry_idx\tstudent_layer_idx\tstudent_target\tteacher_entry_idx\tteacher_layer_idx\tteacher_target\tteacher_tensor_name\tstudent_tensor_name\tsource_dim\ttarget_dim\tteacher_layer_type\tstudent_layer_type\talias_of_entry\n");
}

static int write_manifest_entry_line(int fd, const activation_alignment_entry_t *entry)
{
    char line[1024];
    char alias_buf[32];
    const char *alias_text = alias_buf;
    int written = 0;

    if (!entry) {
        return -1;
    }

    if (entry->alias_of_entry == TAPE_NO_ALIAS) {
        alias_text = "-";
    } else {
        written = snprintf(alias_buf, sizeof(alias_buf), "%u", entry->alias_of_entry);
        if (written < 0 || (size_t)written >= sizeof(alias_buf)) {
            return -1;
        }
    }

    written = snprintf(line,
                       sizeof(line),
                       "%u\t%u\t%s\t%u\t%u\t%s\t%s\t%s\t%u\t%u\t%u\t%u\t%s\n",
                       entry->student_entry_idx,
                       entry->student_layer_idx,
                       capture_target_name(entry->target),
                       entry->teacher_entry_idx,
                       entry->teacher_layer_idx,
                       capture_target_name(entry->target),
                       entry->teacher_tensor_name,
                       entry->student_tensor_name,
                       entry->source_dim,
                       entry->target_dim,
                       entry->teacher_layer_type,
                       entry->student_layer_type,
                       alias_text);
    if (written < 0 || (size_t)written >= sizeof(line)) {
        return -1;
    }

    return write_manifest_line(fd, line);
}

int activation_alignment_write_manifest(const activation_alignment_manifest_t *manifest,
                                        const char *path)
{
    int fd = -1;
    int rc = -1;

    if (!manifest || !manifest->entries || !path) {
        LOG_ERROR("activation alignment: invalid manifest write request");
        return -1;
    }

    const manifest_text_field_t text_fields[] = {
        {"teacher_model_id", manifest->teacher_model_id},
        {"student_model_id", manifest->student_model_id},
        {"teacher_prefix", manifest->teacher_prefix},
        {"student_prefix", manifest->student_prefix},
        {"depth_strategy", depth_strategy_name(manifest->depth_strategy)},
        {"width_strategy", width_strategy_name(manifest->width_strategy)},
    };
    const manifest_u32_field_t numeric_fields[] = {
        {"teacher_layer_count", manifest->teacher_layer_count},
        {"student_layer_count", manifest->student_layer_count},
        {"sample_count", manifest->sample_count},
    };

    fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("activation alignment: cannot open %s: %s", path, strerror(errno));
        return -1;
    }

    if (write_manifest_line(fd, "# Sapphire activation alignment manifest v1\n") != 0) {
        goto manifest_cleanup;
    }
    if (write_manifest_text_fields(fd, text_fields, sizeof(text_fields) / sizeof(text_fields[0])) != 0 ||
        write_manifest_u32_fields(fd, numeric_fields, sizeof(numeric_fields) / sizeof(numeric_fields[0])) != 0) {
        goto manifest_cleanup;
    }
    if (write_manifest_entry_header(fd) != 0) {
        goto manifest_cleanup;
    }

    for (size_t i = 0; i < manifest->entry_count; ++i) {
        if (write_manifest_entry_line(fd, &manifest->entries[i]) != 0) {
            goto manifest_cleanup;
        }
    }

    rc = 0;

manifest_cleanup:
    if (fd >= 0 && close(fd) != 0) {
        LOG_ERROR("activation alignment: close failed for %s: %s", path, strerror(errno));
        rc = -1;
    }
    return rc;
}

static int build_tape_entries(const activation_alignment_manifest_t *manifest,
                              tape_manifest_entry_t **out_entries,
                              uint64_t *out_data_size)
{
    tape_manifest_entry_t *entries = NULL;
    uint64_t current_offset = 0u;

    if (!manifest || !out_entries || !out_data_size) {
        return -1;
    }

    entries = (tape_manifest_entry_t *)calloc(manifest->entry_count, sizeof(*entries));
    if (!entries) {
        LOG_ERROR("activation alignment: tape manifest allocation failed");
        return -1;
    }

    for (size_t i = 0; i < manifest->entry_count; ++i) {
        const activation_alignment_entry_t *entry = &manifest->entries[i];
        tape_manifest_entry_t *tape_entry = &entries[i];

        if (snprintf(tape_entry->tensor_name,
                     sizeof(tape_entry->tensor_name),
                     "%s",
                     entry->student_tensor_name) < 0) {
            free(entries);
            return -1;
        }
        tape_entry->vector_dim = entry->target_dim;
        tape_entry->sample_count = manifest->sample_count;
        tape_entry->layer_type = entry->student_layer_type;
        tape_entry->alias_of_entry = entry->alias_of_entry;

        if (entry->alias_of_entry == TAPE_NO_ALIAS) {
            uint64_t bytes = (uint64_t)entry->target_dim * (uint64_t)manifest->sample_count * (uint64_t)sizeof(float);

            if (bytes == 0u || current_offset > UINT64_MAX - bytes) {
                free(entries);
                LOG_ERROR("activation alignment: tape size overflow");
                return -1;
            }
            tape_entry->data_offset = current_offset;
            tape_entry->data_bytes = bytes;
            current_offset += bytes;
        } else {
            if (entry->alias_of_entry >= i) {
                free(entries);
                LOG_ERROR("activation alignment: alias index out of range");
                return -1;
            }
            tape_entry->data_offset = entries[entry->alias_of_entry].data_offset;
            tape_entry->data_bytes = entries[entry->alias_of_entry].data_bytes;
        }
    }

    *out_entries = entries;
    *out_data_size = current_offset;
    return 0;
}

static int write_primary_entry_data(int fd,
                                    const activation_alignment_manifest_t *manifest,
                                    const activation_alignment_request_t *request,
                                    const activation_alignment_entry_t *entry)
{
    float *source = NULL;
    float *aligned = NULL;
    activation_alignment_width_strategy_t strategy = ACTIVATION_ALIGNMENT_WIDTH_INTERPOLATION;
    size_t aligned_bytes = 0u;
    uint32_t sample_idx = 0u;

    if (!manifest || !request || !entry || !request->teacher_tape) {
        return -1;
    }

    strategy = resolve_width_strategy(request->width_strategy, entry->source_dim, entry->target_dim);
    aligned_bytes = (size_t)entry->target_dim * (size_t)manifest->sample_count * sizeof(float);
    source = (float *)malloc((size_t)entry->source_dim * sizeof(float));
    aligned = (float *)malloc(aligned_bytes);
    if (!source || !aligned) {
        LOG_ERROR("activation alignment: data buffer allocation failed");
        free(source);
        free(aligned);
        return -1;
    }

    for (sample_idx = 0u; sample_idx < manifest->sample_count; ++sample_idx) {
        float *target_row = aligned + ((size_t)sample_idx * entry->target_dim);

        if (activation_tape_get_vector(request->teacher_tape,
                                       entry->teacher_tensor_name,
                                       (int)sample_idx,
                                       source) != 0) {
            LOG_ERROR("activation alignment: failed to read %s sample %u",
                      entry->teacher_tensor_name,
                      sample_idx);
            free(source);
            free(aligned);
            return -1;
        }

        resample_vector(source,
                        entry->source_dim,
                        target_row,
                        entry->target_dim,
                        strategy);
    }

    if (write_all(fd, aligned, aligned_bytes) != 0) {
        LOG_ERROR("activation alignment: data write failed for %s", entry->student_tensor_name);
        free(source);
        free(aligned);
        return -1;
    }

    free(source);
    free(aligned);
    return 0;
}

int activation_alignment_write_aligned_tape(const activation_alignment_manifest_t *manifest,
                                            const activation_alignment_request_t *request,
                                            const char *path)
{
    const gemma3_270m_config_t *student_cfg = NULL;
    tape_file_header_t hdr;
    tape_manifest_entry_t *entries = NULL;
    uint64_t data_size = 0u;
    int fd = -1;
    int rc = -1;

    if (!manifest || !request || !path || !manifest->entries) {
        LOG_ERROR("activation alignment: invalid aligned tape request");
        return -1;
    }
    student_cfg = (const gemma3_270m_config_t *)request->student_spec->variant_config;
    if (!student_cfg || student_cfg->hidden_size <= 0) {
        LOG_ERROR("activation alignment: student config is missing");
        return -1;
    }

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = TAPE_MAGIC;
    hdr.version = TAPE_VERSION;
    hdr.hidden_size = (uint32_t)student_cfg->hidden_size;
    hdr.sample_count = manifest->sample_count;

    if (build_tape_entries(manifest, &entries, &data_size) != 0) {
        return -1;
    }

    hdr.entry_count = (uint32_t)manifest->entry_count;
    hdr.data_section_offset = (uint64_t)sizeof(tape_file_header_t) + (uint64_t)manifest->entry_count * sizeof(tape_manifest_entry_t);
    hdr.data_section_size = data_size;
    hdr.crc32 = 0u;

    fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("activation alignment: cannot open %s: %s", path, strerror(errno));
        free(entries);
        return -1;
    }

    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    if (write_all(fd, &hdr, sizeof(hdr)) != 0 ||
        write_all(fd, entries, (size_t)manifest->entry_count * sizeof(*entries)) != 0) {
        LOG_ERROR("activation alignment: header or manifest write failed");
        goto tape_cleanup;
    }

    for (size_t i = 0; i < manifest->entry_count; ++i) {
        const activation_alignment_entry_t *entry = &manifest->entries[i];

        if (entry->alias_of_entry != TAPE_NO_ALIAS) {
            continue;
        }
        if (write_primary_entry_data(fd, manifest, request, entry) != 0) {
            goto tape_cleanup;
        }
    }

    hdr.crc32 = io_crc32_update(0u, &hdr, offsetof(tape_file_header_t, crc32));
    if (lseek(fd, (off_t)offsetof(tape_file_header_t, crc32), SEEK_SET) < 0 ||
        write_all(fd, &hdr.crc32, sizeof(hdr.crc32)) != 0) {
        LOG_ERROR("activation alignment: CRC patch failed for %s", path);
        goto tape_cleanup;
    }

    if (fsync(fd) != 0) {
        LOG_ERROR("activation alignment: fsync failed for %s: %s", path, strerror(errno));
        goto tape_cleanup;
    }

    rc = 0;

tape_cleanup:
    if (fd >= 0 && close(fd) != 0) {
        LOG_ERROR("activation alignment: close failed for %s: %s", path, strerror(errno));
        rc = -1;
    }
    free(entries);
    return rc;
}

int activation_alignment_prepare_artifacts(const activation_alignment_request_t *request,
                                           const char *output_dir,
                                           char **out_tape_path,
                                           char **out_manifest_path)
{
    activation_alignment_manifest_t manifest;
    const char alignment_dir_component[] = "alignment";
    char teacher_component[128];
    char student_component[128];
    char *alignment_dir = NULL;
    char *tape_path = NULL;
    char *manifest_path = NULL;
    char *tape_name = NULL;
    char *manifest_name = NULL;
    int rc = -1;

    if (!out_tape_path || !out_manifest_path) {
        LOG_ERROR("activation alignment: output pointers are NULL");
        return -1;
    }
    *out_tape_path = NULL;
    *out_manifest_path = NULL;

    if (activation_alignment_build_manifest(&manifest, request) != 0) {
        return -1;
    }

    if (!output_dir || output_dir[0] == '\0') {
        LOG_ERROR("activation alignment: output directory is empty");
        activation_alignment_manifest_free(&manifest);
        return -1;
    }

    if (io_prepare_ternary_output_dir(output_dir) != 0) {
        activation_alignment_manifest_free(&manifest);
        return -1;
    }

    alignment_dir = construct_safe_path(output_dir, alignment_dir_component, NULL);
    if (!alignment_dir) {
        activation_alignment_manifest_free(&manifest);
        return -1;
    }
    if (io_prepare_ternary_output_dir(alignment_dir) != 0) {
        free(alignment_dir);
        activation_alignment_manifest_free(&manifest);
        return -1;
    }

    sanitize_component(manifest.teacher_model_id, teacher_component, sizeof(teacher_component));
    sanitize_component(manifest.student_model_id, student_component, sizeof(student_component));
    if (teacher_component[0] == '\0' || student_component[0] == '\0') {
        LOG_ERROR("activation alignment: failed to sanitize model identifiers");
        free(alignment_dir);
        activation_alignment_manifest_free(&manifest);
        return -1;
    }

    {
        char file_name[512];
        int written = snprintf(file_name,
                               sizeof(file_name),
                               "%s__to__%s__alignment_v1.tape",
                               teacher_component,
                               student_component);
        if (written < 0 || (size_t)written >= sizeof(file_name)) {
            LOG_ERROR("activation alignment: failed to build tape filename");
            free(alignment_dir);
            activation_alignment_manifest_free(&manifest);
            return -1;
        }
        tape_name = construct_safe_path(alignment_dir, file_name, NULL);
        if (!tape_name) {
            free(alignment_dir);
            activation_alignment_manifest_free(&manifest);
            return -1;
        }
    }

    {
        char file_name[512];
        int written = snprintf(file_name,
                               sizeof(file_name),
                               "%s__to__%s__alignment_v1.tsv",
                               teacher_component,
                               student_component);
        if (written < 0 || (size_t)written >= sizeof(file_name)) {
            LOG_ERROR("activation alignment: failed to build manifest filename");
            free(tape_name);
            free(alignment_dir);
            activation_alignment_manifest_free(&manifest);
            return -1;
        }
        manifest_name = construct_safe_path(alignment_dir, file_name, NULL);
        if (!manifest_name) {
            free(tape_name);
            free(alignment_dir);
            activation_alignment_manifest_free(&manifest);
            return -1;
        }
    }

    if (activation_alignment_write_manifest(&manifest, manifest_name) != 0) {
        goto prepare_cleanup;
    }
    if (activation_alignment_write_aligned_tape(&manifest, request, tape_name) != 0) {
        goto prepare_cleanup;
    }

    tape_path = tape_name;
    manifest_path = manifest_name;
    tape_name = NULL;
    manifest_name = NULL;
    rc = 0;

prepare_cleanup:
    free(tape_name);
    free(manifest_name);
    free(alignment_dir);
    activation_alignment_manifest_free(&manifest);
    if (rc == 0) {
        *out_tape_path = tape_path;
        *out_manifest_path = manifest_path;
    } else {
        free(tape_path);
        free(manifest_path);
    }
    return rc;
}

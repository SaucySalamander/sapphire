#include "llm_model.h"
#include "model_spec.h"
#include "safetensors_reader.h"
#include "ternary_anchor.h"
#include "gemma3_config.h"
#include "tensor_mapper.h"
#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <dirent.h>
#include <sys/mman.h>
#include <log.h>
#include <file_reader.h>

typedef enum {
    MODEL_FORMAT_UNKNOWN = 0,
    MODEL_FORMAT_GGML = 1,
    MODEL_FORMAT_SAFETENSORS = 2
} model_format_t;

typedef enum {
    TERNARY_MANIFEST_KIND_TERNARY = 0,
    TERNARY_MANIFEST_KIND_MOLD = 1,
    TERNARY_MANIFEST_KIND_ANCHOR = 2
} ternary_manifest_loader_kind_t;

typedef enum {
    TERNARY_MANIFEST_FORM_NONE = 0,
    TERNARY_MANIFEST_FORM_BULK = 1 << 0,
    TERNARY_MANIFEST_FORM_MOLD = 1 << 1,
    TERNARY_MANIFEST_FORM_LEGACY_ANCHOR = 1 << 2,
    TERNARY_MANIFEST_FORM_UNIFIED_ANCHOR = 1 << 3
} ternary_manifest_loader_form_t;

typedef struct {
    char tensor_name[256];
    char bulk_file_name[256];
    char anchor_file_name[256];
    uint32_t rows;
    uint32_t cols;
    size_t packed_weight_bytes;
    uint32_t crc32;
    uint32_t anchor_count;
    ternary_manifest_loader_kind_t kind;
    uint32_t source_forms;
} ternary_manifest_loader_entry_t;

static tensor_t **resolve_tensor_slot_from_entry(llm_model_t *model,
                                                 const tensor_map_entry_t *entry) {
    int layer_idx = -1;

    if (!model || !entry || !entry->internal_key || !entry->field_name) {
        return NULL;
    }

    if (strcmp(entry->internal_key, "embedding") == 0) {
        return &model->embedding_weight;
    }
    if (strcmp(entry->internal_key, "final") == 0) {
        if (strcmp(entry->field_name, "norm_final_weight") == 0) return &model->norm_final_weight;
        if (strcmp(entry->field_name, "lm_head_weight") == 0) return &model->lm_head_weight;
        return NULL;
    }
    if (sscanf(entry->internal_key, "blk.%d", &layer_idx) != 1 ||
        layer_idx < 0 || layer_idx >= model->num_layers) {
        return NULL;
    }

    if (strcmp(entry->field_name, "norm_attn_weight") == 0) return &model->layers[layer_idx].norm_attn_weight;
    if (strcmp(entry->field_name, "norm_attn_post_weight") == 0) return &model->layers[layer_idx].norm_attn_post_weight;
    if (strcmp(entry->field_name, "q_proj_weight") == 0) return &model->layers[layer_idx].q_proj_weight;
    if (strcmp(entry->field_name, "k_proj_weight") == 0) return &model->layers[layer_idx].k_proj_weight;
    if (strcmp(entry->field_name, "v_proj_weight") == 0) return &model->layers[layer_idx].v_proj_weight;
    if (strcmp(entry->field_name, "q_norm_weight") == 0) return &model->layers[layer_idx].q_norm_weight;
    if (strcmp(entry->field_name, "k_norm_weight") == 0) return &model->layers[layer_idx].k_norm_weight;
    if (strcmp(entry->field_name, "out_proj_weight") == 0) return &model->layers[layer_idx].out_proj_weight;
    if (strcmp(entry->field_name, "norm_ffn_weight") == 0) return &model->layers[layer_idx].norm_ffn_weight;
    if (strcmp(entry->field_name, "norm_ffn_post_weight") == 0) return &model->layers[layer_idx].norm_ffn_post_weight;
    if (strcmp(entry->field_name, "up_proj_weight") == 0) return &model->layers[layer_idx].up_proj_weight;
    if (strcmp(entry->field_name, "gate_proj_weight") == 0) return &model->layers[layer_idx].gate_proj_weight;
    if (strcmp(entry->field_name, "down_proj_weight") == 0) return &model->layers[layer_idx].down_proj_weight;
    return NULL;
}

static tensor_t **resolve_tensor_slot_by_name(llm_model_t *model,
                                              const model_spec_t *model_spec,
                                              const char *tensor_name) {
    int i = 0;

    if (!model || !model_spec || !model_spec->tensor_map || !tensor_name) {
        return NULL;
    }

    for (i = 0; i < model_spec->tensor_map_size; ++i) {
        const tensor_map_entry_t *entry = &model_spec->tensor_map[i];
        if (!entry->hf_name) {
            continue;
        }
        if (strcmp(entry->hf_name, tensor_name) == 0) {
            return resolve_tensor_slot_from_entry(model, entry);
        }
    }

    return NULL;
}

static const char *path_basename_local(const char *path) {
    const char *last_slash = NULL;

    if (!path) {
        return NULL;
    }
    last_slash = strrchr(path, '/');
    return last_slash ? (last_slash + 1) : path;
}

static int parse_manifest_u32_local(const char *text,
                                    int base,
                                    uint32_t *out_value) {
    char *end = NULL;
    unsigned long value = 0ul;

    if (!text || !*text || !out_value) {
        return -1;
    }

    value = strtoul(text, &end, base);
    if (end == text || *end != '\0' || value > UINT_MAX) {
        return -1;
    }
    *out_value = (uint32_t)value;
    return 0;
}

static int parse_manifest_size_local(const char *text,
                                     size_t *out_value) {
    char *end = NULL;
    unsigned long long value = 0ull;

    if (!text || !*text || !out_value) {
        return -1;
    }

    value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value > (unsigned long long)((size_t)-1)) {
        return -1;
    }
    *out_value = (size_t)value;
    return 0;
}

static int split_manifest_fields_local(char *line,
                                       char **fields,
                                       int max_fields) {
    char *cursor = line;
    char *next = NULL;
    int field_idx = 0;

    if (!line || !fields || max_fields <= 0) {
        return -1;
    }

    while (field_idx < max_fields && cursor) {
        next = strchr(cursor, '\t');
        if (next) {
            *next = '\0';
            fields[field_idx++] = cursor;
            cursor = next + 1;
        } else {
            fields[field_idx++] = cursor;
            cursor = NULL;
        }
    }
    return field_idx;
}

static int parse_manifest_common_entry_local(char **fields,
                                             const char *manifest_path,
                                             unsigned int line_number,
                                             ternary_manifest_loader_entry_t *out_entry) {
    if (!fields || !manifest_path || !out_entry) {
        return -1;
    }

    if (!fields[0] || fields[0][0] == '\0') {
        LOG_ERROR("Invalid manifest row at %s:%u: missing tensor name", manifest_path, line_number);
        return -1;
    }
    snprintf(out_entry->tensor_name, sizeof(out_entry->tensor_name), "%s", fields[0]);
    if (parse_manifest_u32_local(fields[2], 10, &out_entry->rows) != 0 ||
        parse_manifest_u32_local(fields[3], 10, &out_entry->cols) != 0 ||
        parse_manifest_size_local(fields[4], &out_entry->packed_weight_bytes) != 0 ||
        parse_manifest_u32_local(fields[5], 16, &out_entry->crc32) != 0) {
        LOG_ERROR("Invalid numeric field in manifest row at %s:%u for tensor %s",
                  manifest_path,
                  line_number,
                  out_entry->tensor_name);
        return -1;
    }

    return 0;
}

static int parse_manifest_payload_row_local(const char *payload_path,
                                            ternary_manifest_loader_kind_t kind,
                                            uint32_t source_form,
                                            const char *manifest_path,
                                            unsigned int line_number,
                                            ternary_manifest_loader_entry_t *out_entry) {
    const char *row_kind = (kind == TERNARY_MANIFEST_KIND_MOLD) ? "mold" : "ternary";

    if (!payload_path || payload_path[0] == '\0') {
        LOG_ERROR("Invalid %s manifest row at %s:%u: missing payload for %s",
                  row_kind,
                  manifest_path,
                  line_number,
                  out_entry->tensor_name);
        return -1;
    }

    snprintf(out_entry->bulk_file_name, sizeof(out_entry->bulk_file_name), "%s", payload_path);
    out_entry->kind = kind;
    out_entry->source_forms = source_form;
    return 0;
}

static int parse_manifest_anchor_row_local(char **fields,
                                           int field_idx,
                                           const char *manifest_path,
                                           unsigned int line_number,
                                           ternary_manifest_loader_entry_t *out_entry) {
    if (!fields || !manifest_path || !out_entry) {
        return -1;
    }

    out_entry->kind = TERNARY_MANIFEST_KIND_ANCHOR;
    if (field_idx == 8) {
        if (!fields[1] || fields[1][0] == '\0') {
            LOG_ERROR("Invalid legacy anchor manifest row at %s:%u: missing anchor sidecar for %s",
                      manifest_path,
                      line_number,
                      out_entry->tensor_name);
            return -1;
        }
        snprintf(out_entry->anchor_file_name, sizeof(out_entry->anchor_file_name), "%s", fields[1]);
        if (parse_manifest_u32_local(fields[7], 10, &out_entry->anchor_count) != 0) {
            LOG_ERROR("Invalid anchor_count in manifest row at %s:%u for %s",
                      manifest_path,
                      line_number,
                      out_entry->tensor_name);
            return -1;
        }
        out_entry->source_forms = TERNARY_MANIFEST_FORM_LEGACY_ANCHOR;
        return 0;
    }
    if (field_idx == 9) {
        if (!fields[1] || fields[1][0] == '\0' || !fields[7] || fields[7][0] == '\0') {
            LOG_ERROR("Invalid unified anchor manifest row at %s:%u: missing payload path for %s",
                      manifest_path,
                      line_number,
                      out_entry->tensor_name);
            return -1;
        }
        snprintf(out_entry->bulk_file_name, sizeof(out_entry->bulk_file_name), "%s", fields[1]);
        snprintf(out_entry->anchor_file_name, sizeof(out_entry->anchor_file_name), "%s", fields[7]);
        if (parse_manifest_u32_local(fields[8], 10, &out_entry->anchor_count) != 0) {
            LOG_ERROR("Invalid anchor_count in manifest row at %s:%u for %s",
                      manifest_path,
                      line_number,
                      out_entry->tensor_name);
            return -1;
        }
        out_entry->source_forms = TERNARY_MANIFEST_FORM_UNIFIED_ANCHOR;
        return 0;
    }

    return -1;
}

static int parse_manifest_line_local(char *line,
                                     const char *manifest_path,
                                     unsigned int line_number,
                                     ternary_manifest_loader_entry_t *out_entry) {
    char *fields[9] = {0};
    int field_idx = 0;

    if (!line || !manifest_path || !out_entry) {
        return -1;
    }

    field_idx = split_manifest_fields_local(line, fields, 9);
    if (field_idx != 6 && field_idx != 7 && field_idx != 8 && field_idx != 9) {
        LOG_ERROR("Invalid manifest row at %s:%u: expected 6, 7, 8, or 9 tab-separated fields",
                  manifest_path,
                  line_number);
        return -1;
    }

    memset(out_entry, 0, sizeof(*out_entry));
    if (parse_manifest_common_entry_local(fields, manifest_path, line_number, out_entry) != 0) {
        return -1;
    }

    if (field_idx == 6) {
        return parse_manifest_payload_row_local(fields[1],
                                                TERNARY_MANIFEST_KIND_TERNARY,
                                                TERNARY_MANIFEST_FORM_BULK,
                                                manifest_path,
                                                line_number,
                                                out_entry);
    }

    if (field_idx == 7 && strcmp(fields[6], "mold") == 0) {
        return parse_manifest_payload_row_local(fields[1],
                                                TERNARY_MANIFEST_KIND_MOLD,
                                                TERNARY_MANIFEST_FORM_MOLD,
                                                manifest_path,
                                                line_number,
                                                out_entry);
    }

    if (strcmp(fields[6], "anchor") != 0) {
        LOG_ERROR("Invalid manifest row at %s:%u: unsupported kind '%s' for %s",
                  manifest_path,
                  line_number,
                  fields[6],
                  out_entry->tensor_name);
        return -1;
    }

    return parse_manifest_anchor_row_local(fields,
                                           field_idx,
                                           manifest_path,
                                           line_number,
                                           out_entry);
}

static ternary_manifest_loader_entry_t *find_manifest_entry_local(ternary_manifest_loader_entry_t *entries,
                                                                  int count,
                                                                  const char *tensor_name) {
    int i = 0;

    if (!entries || count <= 0 || !tensor_name) {
        return NULL;
    }

    for (i = 0; i < count; ++i) {
        if (strcmp(entries[i].tensor_name, tensor_name) == 0) {
            return &entries[i];
        }
    }
    return NULL;
}

static int merge_manifest_shape_local(ternary_manifest_loader_entry_t *dst,
                                      const ternary_manifest_loader_entry_t *src) {
    if (!dst || !src) {
        return -1;
    }
    if ((dst->rows != 0u && src->rows != 0u && dst->rows != src->rows) ||
        (dst->cols != 0u && src->cols != 0u && dst->cols != src->cols)) {
        return -1;
    }
    if (dst->rows == 0u) dst->rows = src->rows;
    if (dst->cols == 0u) dst->cols = src->cols;
    return 0;
}

static int merge_manifest_bulk_local(ternary_manifest_loader_entry_t *dst,
                                     const ternary_manifest_loader_entry_t *src) {
    if (!dst || !src) {
        return -1;
    }
    if (src->packed_weight_bytes != 0u) {
        if (dst->packed_weight_bytes != 0u && dst->packed_weight_bytes != src->packed_weight_bytes) {
            return -1;
        }
        dst->packed_weight_bytes = src->packed_weight_bytes;
    }
    if (src->kind != TERNARY_MANIFEST_KIND_ANCHOR || src->bulk_file_name[0] != '\0') {
        if (src->crc32 != 0u) {
            if (dst->crc32 != 0u && dst->crc32 != src->crc32) {
                return -1;
            }
            dst->crc32 = src->crc32;
        }
    }
    if (src->bulk_file_name[0] != '\0') {
        if (dst->bulk_file_name[0] != '\0' && strcmp(dst->bulk_file_name, src->bulk_file_name) != 0) {
            return -1;
        }
        snprintf(dst->bulk_file_name, sizeof(dst->bulk_file_name), "%s", src->bulk_file_name);
    }
    return 0;
}

static int merge_manifest_anchor_local(ternary_manifest_loader_entry_t *dst,
                                       const ternary_manifest_loader_entry_t *src) {
    if (!dst || !src) {
        return -1;
    }
    if (src->anchor_file_name[0] != '\0') {
        if (dst->anchor_file_name[0] != '\0' && strcmp(dst->anchor_file_name, src->anchor_file_name) != 0) {
            return -1;
        }
        snprintf(dst->anchor_file_name, sizeof(dst->anchor_file_name), "%s", src->anchor_file_name);
    }
    if (src->anchor_count != 0u) {
        if (dst->anchor_count != 0u && dst->anchor_count != src->anchor_count) {
            return -1;
        }
        dst->anchor_count = src->anchor_count;
    }
    return 0;
}

static int merge_manifest_kind_local(ternary_manifest_loader_entry_t *dst,
                                     const ternary_manifest_loader_entry_t *src) {
    if (!dst || !src) {
        return -1;
    }

    if ((dst->source_forms & TERNARY_MANIFEST_FORM_MOLD) ||
        (src->source_forms & TERNARY_MANIFEST_FORM_MOLD)) {
        LOG_ERROR("Invalid manifest contract for %s: mold rows cannot be duplicated or mixed",
                  dst->tensor_name);
        return -1;
    }
    if ((dst->source_forms & TERNARY_MANIFEST_FORM_UNIFIED_ANCHOR) ||
        (src->source_forms & TERNARY_MANIFEST_FORM_UNIFIED_ANCHOR)) {
        LOG_ERROR("Invalid manifest contract for %s: unified anchor rows cannot be mixed with duplicate rows",
                  dst->tensor_name);
        return -1;
    }
    if ((dst->source_forms & TERNARY_MANIFEST_FORM_BULK) &&
        (src->source_forms & TERNARY_MANIFEST_FORM_BULK)) {
        LOG_ERROR("Invalid manifest contract for %s: duplicate ternary bulk rows are not allowed",
                  dst->tensor_name);
        return -1;
    }
    if ((dst->source_forms & TERNARY_MANIFEST_FORM_LEGACY_ANCHOR) &&
        (src->source_forms & TERNARY_MANIFEST_FORM_LEGACY_ANCHOR)) {
        LOG_ERROR("Invalid manifest contract for %s: duplicate legacy anchor rows are not allowed",
                  dst->tensor_name);
        return -1;
    }
    if (((dst->source_forms & TERNARY_MANIFEST_FORM_BULK) &&
         (src->source_forms & TERNARY_MANIFEST_FORM_LEGACY_ANCHOR)) ||
        ((dst->source_forms & TERNARY_MANIFEST_FORM_LEGACY_ANCHOR) &&
         (src->source_forms & TERNARY_MANIFEST_FORM_BULK))) {
        dst->kind = TERNARY_MANIFEST_KIND_ANCHOR;
        dst->source_forms |= src->source_forms;
        return 0;
    }

    LOG_ERROR("Invalid manifest contract for %s: unsupported duplicate row combination",
              dst->tensor_name);
    return -1;
}

static int merge_manifest_entry_local(ternary_manifest_loader_entry_t *dst,
                                      const ternary_manifest_loader_entry_t *src) {
    if (!dst || !src || strcmp(dst->tensor_name, src->tensor_name) != 0) {
        return -1;
    }
    if (merge_manifest_kind_local(dst, src) != 0 ||
        merge_manifest_shape_local(dst, src) != 0 ||
        merge_manifest_bulk_local(dst, src) != 0 ||
        merge_manifest_anchor_local(dst, src) != 0) {
        return -1;
    }
    if (dst->source_forms == TERNARY_MANIFEST_FORM_BULK) {
        dst->kind = TERNARY_MANIFEST_KIND_TERNARY;
    } else if (dst->source_forms == (TERNARY_MANIFEST_FORM_BULK | TERNARY_MANIFEST_FORM_LEGACY_ANCHOR)) {
        dst->kind = TERNARY_MANIFEST_KIND_ANCHOR;
    }
    return 0;
}

static int validate_manifest_entry_local(const ternary_manifest_loader_entry_t *entry,
                                         const char *manifest_path) {
    if (!entry || !manifest_path) {
        return -1;
    }
    if (entry->tensor_name[0] == '\0' || entry->rows == 0u || entry->cols == 0u) {
        LOG_ERROR("Invalid manifest entry in %s: tensor name/shape missing", manifest_path);
        return -1;
    }
    if (entry->source_forms == TERNARY_MANIFEST_FORM_MOLD) {
        if (entry->bulk_file_name[0] == '\0') {
            LOG_ERROR("Invalid mold manifest entry for %s in %s: missing payload",
                      entry->tensor_name,
                      manifest_path);
            return -1;
        }
        return 0;
    }
    if (entry->source_forms != TERNARY_MANIFEST_FORM_BULK &&
        entry->source_forms != (TERNARY_MANIFEST_FORM_BULK | TERNARY_MANIFEST_FORM_LEGACY_ANCHOR) &&
        entry->source_forms != TERNARY_MANIFEST_FORM_UNIFIED_ANCHOR) {
        LOG_ERROR("Invalid manifest contract for %s in %s: incomplete or ambiguous migration row set",
                  entry->tensor_name,
                  manifest_path);
        return -1;
    }
    if (entry->bulk_file_name[0] == '\0' || entry->packed_weight_bytes == 0u) {
        LOG_ERROR("Invalid manifest entry for %s in %s: missing ternary bulk payload",
                  entry->tensor_name,
                  manifest_path);
        return -1;
    }
    if (entry->source_forms == (TERNARY_MANIFEST_FORM_BULK | TERNARY_MANIFEST_FORM_LEGACY_ANCHOR) ||
        entry->source_forms == TERNARY_MANIFEST_FORM_UNIFIED_ANCHOR) {
        if (entry->anchor_file_name[0] == '\0') {
            LOG_ERROR("Invalid anchor manifest entry for %s in %s: missing anchor sidecar metadata",
                      entry->tensor_name,
                      manifest_path);
            return -1;
        }
    }
    return 0;
}

static int load_ternary_manifest_local(const char *model_dir,
                                       ternary_manifest_loader_entry_t **out_entries,
                                       int *out_count) {
    char *manifest_path = NULL;
    FILE *manifest_file = NULL;
    ternary_manifest_loader_entry_t *entries = NULL;
    char line[1024];
    unsigned int line_number = 0u;
    int count = 0;
    int capacity = 0;
    int rc = -1;

    if (!model_dir || !out_entries || !out_count) {
        return -1;
    }

    *out_entries = NULL;
    *out_count = 0;
    manifest_path = construct_safe_path(model_dir, "manifest.tsv", NULL);
    if (!manifest_path) {
        return -1;
    }
    if (access(manifest_path, F_OK) != 0) {
        free(manifest_path);
        return 0;
    }

    manifest_file = fopen(manifest_path, "r");
    if (!manifest_file) {
        free(manifest_path);
        return -1;
    }

    while (fgets(line, sizeof(line), manifest_file) != NULL) {
        ternary_manifest_loader_entry_t entry;
        ternary_manifest_loader_entry_t *existing = NULL;
        size_t len = strlen(line);
        ++line_number;

        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) {
            continue;
        }
        if (parse_manifest_line_local(line, manifest_path, line_number, &entry) != 0) {
            goto cleanup;
        }
        existing = find_manifest_entry_local(entries, count, entry.tensor_name);
        if (existing) {
            if (merge_manifest_entry_local(existing, &entry) != 0) {
                LOG_ERROR("Invalid duplicate manifest row for %s at %s:%u",
                          entry.tensor_name,
                          manifest_path,
                          line_number);
                goto cleanup;
            }
            continue;
        }
        if (count >= capacity) {
            int new_capacity = capacity ? (capacity * 2) : 64;
            ternary_manifest_loader_entry_t *new_entries = (ternary_manifest_loader_entry_t *)realloc(entries,
                                                                                                       (size_t)new_capacity * sizeof(*entries));
            if (!new_entries) {
                LOG_ERROR("Failed to grow manifest normalization buffer for %s", manifest_path);
                goto cleanup;
            }
            entries = new_entries;
            capacity = new_capacity;
        }
        entries[count++] = entry;
    }

    if (ferror(manifest_file)) {
        LOG_ERROR("Failed to read manifest %s", manifest_path);
        goto cleanup;
    }
    if (count == 0) {
        LOG_ERROR("Ternary manifest is empty: %s", manifest_path);
        goto cleanup;
    }
    for (int i = 0; i < count; ++i) {
        if (validate_manifest_entry_local(&entries[i], manifest_path) != 0) {
            goto cleanup;
        }
    }
    *out_entries = entries;
    *out_count = count;
    entries = NULL;
    rc = 1;

cleanup:
    if (manifest_file) {
        fclose(manifest_file);
    }
    free(manifest_path);
    free(entries);
    return rc;
}

static int find_shard_index_by_name(char **shard_paths,
                                    int shard_count,
                                    const char *file_name) {
    int i = 0;

    if (!shard_paths || shard_count <= 0 || !file_name) {
        return -1;
    }

    for (i = 0; i < shard_count; ++i) {
        const char *base_name = path_basename_local(shard_paths[i]);
        if (base_name && strcmp(base_name, file_name) == 0) {
            return i;
        }
    }
    return -1;
}

static tensor_t *create_hybrid_tensor_from_manifest_entry(const ternary_manifest_loader_entry_t *entry,
                                                          const char *model_dir,
                                                          const safetensors_file_t *bulk_handle) {
    tensor_t *bulk_tensor = NULL;
    tensor_t *hybrid_tensor = NULL;
    const tensor_ternary_view_t *bulk_view = NULL;
    tensor_hybrid_payload_t payload;
    ternary_anchor_view_t anchor_view;
    char *anchor_path = NULL;

    if (!entry || !model_dir || !bulk_handle || entry->kind != TERNARY_MANIFEST_KIND_ANCHOR) {
        return NULL;
    }

    memset(&payload, 0, sizeof(payload));
    memset(&anchor_view, 0, sizeof(anchor_view));

    bulk_tensor = safetensors_create_ternary_tensor_ref(bulk_handle,
                                                        entry->tensor_name,
                                                        entry->rows,
                                                        entry->cols,
                                                        entry->packed_weight_bytes,
                                                        entry->crc32);
    if (!bulk_tensor) {
        return NULL;
    }
    bulk_view = tensor_data_ternary(bulk_tensor);
    if (!bulk_view) {
        tensor_release(bulk_tensor);
        return NULL;
    }

    anchor_path = construct_safe_path(model_dir, entry->anchor_file_name, NULL);
    if (!anchor_path) {
        tensor_release(bulk_tensor);
        return NULL;
    }
    if (ternary_anchor_load_for_tensor(anchor_path, entry->tensor_name, &anchor_view) != 0) {
        free(anchor_path);
        tensor_release(bulk_tensor);
        return NULL;
    }
    free(anchor_path);
    anchor_path = NULL;

    if (anchor_view.rows != entry->rows || anchor_view.cols != entry->cols ||
        anchor_view.anchor_count != entry->anchor_count) {
        LOG_ERROR("Hybrid manifest payload mismatch for %s: manifest rows=%u cols=%u anchors=%u, sidecar rows=%u cols=%u anchors=%u",
                  entry->tensor_name,
                  entry->rows,
                  entry->cols,
                  entry->anchor_count,
                  anchor_view.rows,
                  anchor_view.cols,
                  anchor_view.anchor_count);
        ternary_anchor_view_release(&anchor_view);
        tensor_release(bulk_tensor);
        return NULL;
    }
    if ((anchor_view.metadata.scale_group_size != 0u &&
         anchor_view.metadata.scale_group_size != bulk_view->scale_group_size) ||
        (anchor_view.metadata.groups_per_row != 0u &&
         anchor_view.metadata.groups_per_row != bulk_view->groups_per_row)) {
        LOG_ERROR("Hybrid scale layout mismatch for %s: bulk group_size=%u groups=%u, sidecar group_size=%u groups=%u",
                  entry->tensor_name,
                  bulk_view->scale_group_size,
                  bulk_view->groups_per_row,
                  anchor_view.metadata.scale_group_size,
                  anchor_view.metadata.groups_per_row);
        ternary_anchor_view_release(&anchor_view);
        tensor_release(bulk_tensor);
        return NULL;
    }

    payload.bulk.packed_weights = bulk_view->packed_weights;
    payload.bulk.packed_weight_bytes = bulk_view->packed_weight_bytes;
    payload.bulk.scales = bulk_view->scales;
    payload.bulk.scale_count = bulk_view->scale_count;
    payload.bulk.scale_group_size = bulk_view->scale_group_size;
    payload.anchor_entries = anchor_view.entries;
    payload.anchor_row_offsets = anchor_view.row_offsets;
    payload.anchor_count = anchor_view.anchor_count;
    payload.owns_anchor_memory = 1;

    hybrid_tensor = tensor_create_hybrid_view(entry->rows,
                                              entry->cols,
                                              &payload,
                                              1);
    if (!hybrid_tensor) {
        ternary_anchor_view_release(&anchor_view);
        tensor_release(bulk_tensor);
        return NULL;
    }

    anchor_view.owns_memory = 0;
    tensor_release(bulk_tensor);
    return hybrid_tensor;
}

static int apply_ternary_manifest_overrides(llm_model_t *model,
                                            const model_spec_t *model_spec,
                                            const char *model_dir,
                                            safetensors_file_t **handles,
                                            char **shard_paths,
                                            int shard_count) {
    ternary_manifest_loader_entry_t *entries = NULL;
    int entry_count = 0;
    int manifest_rc = 0;
    int i = 0;

    if (!model || !model_spec || !model_dir || !handles || !shard_paths || shard_count <= 0) {
        return -1;
    }

    manifest_rc = load_ternary_manifest_local(model_dir, &entries, &entry_count);
    if (manifest_rc <= 0) {
        return manifest_rc;
    }

    for (i = 0; i < entry_count; ++i) {
        tensor_t **slot = resolve_tensor_slot_by_name(model, model_spec, entries[i].tensor_name);
        int shard_index = -1;
        tensor_t *tensor = NULL;

        if (!slot || *slot) {
            continue;
        }

        shard_index = find_shard_index_by_name(shard_paths, shard_count, entries[i].bulk_file_name);
        if (shard_index < 0 || !handles[shard_index]) {
            LOG_ERROR("Ternary manifest shard missing for %s: %s",
                      entries[i].tensor_name,
                      entries[i].bulk_file_name);
            free(entries);
            return -1;
        }

        if (entries[i].kind == TERNARY_MANIFEST_KIND_MOLD) {
            const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_name(handles[shard_index],
                                                                                    entries[i].tensor_name);
            if (!meta) {
                LOG_ERROR("MOLD manifest shard missing tensor %s in %s",
                          entries[i].tensor_name,
                          entries[i].bulk_file_name);
                free(entries);
                return -1;
            }
            tensor = safetensors_create_tensor_ref(handles[shard_index], meta);
            if (!tensor) {
                LOG_ERROR("Failed to create MOLD BF16 tensor for %s from %s",
                          entries[i].tensor_name,
                          entries[i].bulk_file_name);
                free(entries);
                return -1;
            }
        } else if (entries[i].kind == TERNARY_MANIFEST_KIND_ANCHOR) {
            tensor = create_hybrid_tensor_from_manifest_entry(&entries[i],
                                                             model_dir,
                                                             handles[shard_index]);
            if (!tensor) {
                LOG_ERROR("Failed to create hybrid tensor for %s from %s + %s",
                          entries[i].tensor_name,
                          entries[i].bulk_file_name,
                          entries[i].anchor_file_name);
                free(entries);
                return -1;
            }
        } else {
            tensor = safetensors_create_ternary_tensor_ref(handles[shard_index],
                                                           entries[i].tensor_name,
                                                           entries[i].rows,
                                                           entries[i].cols,
                                                           entries[i].packed_weight_bytes,
                                                           entries[i].crc32);
            if (!tensor) {
                LOG_ERROR("Failed to create ternary tensor for %s from %s",
                          entries[i].tensor_name,
                          entries[i].bulk_file_name);
                free(entries);
                return -1;
            }
        }

        *slot = tensor;
    }

    free(entries);
    return 1;
}

static int validate_loaded_model(const llm_model_t *model,
                                 const model_spec_t *model_spec) {
    int i = 0;

    if (!model || !model_spec || !model_spec->tensor_map) {
        return -1;
    }

    for (i = 0; i < model_spec->tensor_map_size; ++i) {
        const tensor_map_entry_t *entry = &model_spec->tensor_map[i];
        tensor_t **slot = NULL;

        if (!entry->hf_name) {
            continue;
        }
        slot = resolve_tensor_slot_from_entry((llm_model_t *)model, entry);
        if (!slot) {
            continue;
        }
        if (!*slot) {
            LOG_ERROR("Missing tensor after load: %s", entry->hf_name);
            return -1;
        }
    }

    return 0;
}

static int finalize_loaded_model(llm_model_t *model,
                                 const model_spec_t *model_spec,
                                 const char *model_dir,
                                 safetensors_file_t **handles,
                                 char **shard_paths,
                                 int shard_count) {
    if (!model || !model_spec) {
        return -1;
    }

    if (!model->lm_head_weight && model->embedding_weight) {
        model->lm_head_weight = model->embedding_weight;
        tensor_ref_inc(model->lm_head_weight);
        LOG_INFO("lm_head weight tied to embedding");
    }

    if (model_dir && handles && shard_paths && shard_count > 0) {
        if (apply_ternary_manifest_overrides(model, model_spec, model_dir, handles, shard_paths, shard_count) < 0) {
            return -1;
        }
    }

    if (!model->lm_head_weight && model->embedding_weight) {
        model->lm_head_weight = model->embedding_weight;
        tensor_ref_inc(model->lm_head_weight);
        LOG_INFO("lm_head weight tied to embedding");
    }

    return validate_loaded_model(model, model_spec);
}

static int skip_missing_tensor(const safetensors_file_t *st,
                               const safetensors_tensor_meta_t *meta,
                               llm_model_t *model) {
    (void)st;
    (void)meta;
    (void)model;
    return 0;
}

static char *duplicate_parent_dir(const char *path) {
    const char *last_slash = NULL;
    size_t dir_len = 0u;
    char *dir = NULL;

    if (!path) {
        return NULL;
    }

    last_slash = strrchr(path, '/');
    if (!last_slash) {
        dir = (char *)malloc(2u);
        if (!dir) {
            return NULL;
        }
        strcpy(dir, ".");
        return dir;
    }

    dir_len = (size_t)(last_slash - path);
    dir = (char *)malloc(dir_len + 1u);
    if (!dir) {
        return NULL;
    }
    memcpy(dir, path, dir_len);
    dir[dir_len] = '\0';
    return dir;
}


/**
 * @brief Load a model from Safetensors file
 * 
 * Uses safetensors_reader to:
 * 1. Open the safetensors file (mmapped zero-copy)
 * 2. Parse the JSON header and extract tensor metadata
 * 3. Create an llm_model_t structure
 * 4. Map all tensors using the tensor_map from model_spec
 * 5. Store the model pointer in the model_spec_t
 *
 * @param model_spec The model specification containing the tensor mapping table
 * @param safetensors_path Path to the safetensors file
 *
 * @return Pointer to populated llm_model_t, or NULL on error
 */
static llm_model_t* load_model_safetensors(const model_spec_t *model_spec, 
                                           const char *safetensors_path) {
    if (!model_spec || !safetensors_path) {
        LOG_ERROR("model_spec or safetensors_path is NULL");
        return NULL;
    }
    
    // Open the safetensors file
    safetensors_file_t *st = safetensors_open(safetensors_path);
    if (!st) {
        LOG_ERROR("Failed to open safetensors file: %s", safetensors_path);
        return NULL;
    }
    
    // Allocate and initialize the model structure
    llm_model_t *model = (llm_model_t *)malloc(sizeof(llm_model_t));
    if (!model) {
        LOG_ERROR("Failed to allocate model structure");
        safetensors_close(st);
        return NULL;
    }
    memset(model, 0, sizeof(llm_model_t));
    
    // Derive layer count from spec's variant_config when available
    int num_layers = 18;  /* safe default for Gemma3 270M */
    if (model_spec->variant_config) {
        const gemma3_270m_config_t *cfg =
            (const gemma3_270m_config_t *)model_spec->variant_config;
        if (cfg->num_hidden_layers > 0 &&
            cfg->num_hidden_layers <= SAPPHIRE_MAX_LAYERS) {
            num_layers = cfg->num_hidden_layers;
        }
    }
    model->layers = (model_layer_weights_t *)malloc(num_layers * sizeof(model_layer_weights_t));
    if (!model->layers) {
        LOG_ERROR("Failed to allocate layer array");
        free(model);
        safetensors_close(st);
        return NULL;
    }
    memset(model->layers, 0, num_layers * sizeof(model_layer_weights_t));
    model->num_layers = num_layers;
    int rc = safetensors_map_all_tensors_with_table(st,
                                                    model_spec->tensor_map,
                                                    model_spec->tensor_map_size,
                                                    skip_missing_tensor,
                                                    model);
    
    if (rc != 0) {
        LOG_ERROR("Failed to map tensors");
        free(model->layers);
        free(model);
        safetensors_close(st);
        return NULL;
    }
    
    // Store the safetensors file handle for cleanup in llm_model_destroy()
    // The file must remain open since tensors are zero-copy references into mmapped memory
    model->safetensors_handle = st;

    {
        char *model_dir = duplicate_parent_dir(safetensors_path);
        safetensors_file_t *handles[1] = { st };
        char *paths[1] = { (char *)safetensors_path };

        if (!model_dir) {
            LOG_ERROR("Failed to derive model directory for %s", safetensors_path);
            free(model->layers);
            free(model);
            safetensors_close(st);
            return NULL;
        }
        if (finalize_loaded_model(model, model_spec, model_dir, handles, paths, 1) != 0) {
            free(model_dir);
            free(model->layers);
            free(model);
            safetensors_close(st);
            return NULL;
        }
        free(model_dir);
    }

    LOG_INFO("Successfully loaded all tensors from %s", safetensors_path);
    
    return model;
}

/* -------------------------------------------------------------------------
 * Sharded safetensors loader
 * -------------------------------------------------------------------------*/

/** qsort comparator for shard filename strings */
static int shard_name_cmp(const void *a, const void *b) {
    return strcmp(*(const char * const *)a, *(const char * const *)b);
}

/**
 * @brief Collect sorted shard filenames from model_dir.
 *
 * Scans for files matching "model-NNNNN-of-NNNNN.safetensors" and returns
 * a heap-allocated, sorted array of heap-allocated path strings.
 * Caller must free each element and the array itself.
 *
 * @param model_dir Directory to scan.
 * @param out_count Number of shards found.
 * @return Sorted array of paths, or NULL on error / none found.
 */
static char **collect_shard_paths(const char *model_dir, int *out_count) {
    *out_count = 0;
    DIR *d = opendir(model_dir);
    if (!d) return NULL;

    char **names = NULL;
    int   cap    = 0;
    int   cnt    = 0;

    const struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        const char *n = ent->d_name;
        /* Match "model-NNNNN-of-NNNNN.safetensors" */
        if (strncmp(n, "model-", 6) != 0) continue;
        const char *ext = strstr(n, ".safetensors");
        if (!ext || ext[sizeof(".safetensors") - 1] != '\0') continue;

        if (cnt >= cap) {
            int new_cap = cap ? cap * 2 : 8;
            char **tmp = (char **)realloc(names, new_cap * sizeof(char *));
            if (!tmp) { closedir(d); goto oom; }
            names = tmp;
            cap   = new_cap;
        }
        names[cnt] = construct_safe_path(model_dir, n, NULL);
        if (!names[cnt]) { closedir(d); goto oom; }
        cnt++;
    }
    closedir(d);

    if (cnt == 0) { free(names); return NULL; }

    qsort(names, cnt, sizeof(char *), shard_name_cmp);
    *out_count = cnt;
    return names;

oom:
    for (int i = 0; i < cnt; i++) free(names[i]);
    free(names);
    return NULL;
}

/**
 * @brief Load a sharded safetensors model by mapping all shards in order.
 *
 * Opens each shard file (sorted lexicographically), mmaps it, and maps
 * its tensors into the model.  All shard handles are kept open and stored
 * in model->safetensors_shard_handles so that the zero-copy tensor views
 * remain valid for the model's lifetime.
 */
static llm_model_t *load_model_sharded(const model_spec_t *model_spec,
                                        const char *model_dir) {
    int    shard_count = 0;
    char **shard_paths = collect_shard_paths(model_dir, &shard_count);
    if (!shard_paths || shard_count == 0) {
        LOG_ERROR("load_model_sharded: no shards found in %s", model_dir);
        return NULL;
    }
    LOG_INFO("Sharded safetensors: found %d shard(s) in %s", shard_count, model_dir);

    llm_model_t *model = (llm_model_t *)malloc(sizeof(llm_model_t));
    if (!model) { goto fail_paths; }
    memset(model, 0, sizeof(llm_model_t));

    int num_layers = 18;
    if (model_spec->variant_config) {
        const gemma3_270m_config_t *cfg =
            (const gemma3_270m_config_t *)model_spec->variant_config;
        if (cfg->num_hidden_layers > 0 &&
            cfg->num_hidden_layers <= SAPPHIRE_MAX_LAYERS) {
            num_layers = cfg->num_hidden_layers;
        }
    }

    model->layers = (model_layer_weights_t *)malloc(
        num_layers * sizeof(model_layer_weights_t));
    if (!model->layers) { goto fail_model; }
    memset(model->layers, 0, num_layers * sizeof(model_layer_weights_t));
    model->num_layers = num_layers;

    model->safetensors_shard_handles =
        (void **)malloc(shard_count * sizeof(void *));
    if (!model->safetensors_shard_handles) { goto fail_layers; }
    memset(model->safetensors_shard_handles, 0, shard_count * sizeof(void *));

    for (int s = 0; s < shard_count; s++) {
        LOG_INFO("  Loading shard %d/%d: %s", s + 1, shard_count, shard_paths[s]);
        safetensors_file_t *st = safetensors_open(shard_paths[s]);
        if (!st) {
            LOG_ERROR("Failed to open shard: %s", shard_paths[s]);
            goto fail_shards;
        }
        model->safetensors_shard_handles[s] = st;
        model->safetensors_shard_count      = s + 1;

        int rc = safetensors_map_all_tensors_with_table(
            st,
            model_spec->tensor_map,
            model_spec->tensor_map_size,
            skip_missing_tensor,
            model);
        if (rc != 0) {
            LOG_ERROR("Failed to map tensors from shard: %s", shard_paths[s]);
            goto fail_shards;
        }
    }

    int loaded_shard_count = shard_count;

    if (finalize_loaded_model(model,
                              model_spec,
                              model_dir,
                              (safetensors_file_t **)model->safetensors_shard_handles,
                              shard_paths,
                              loaded_shard_count) != 0) {
        goto fail_shards;
    }

    for (int i = 0; i < shard_count; i++) free(shard_paths[i]);
    free(shard_paths);
    shard_paths = NULL;
    shard_count = 0;

    /* Check for low-memory mode (CLI flag exports SAPPHIRE_LOW_MEMORY=1) */
    const char *low_mem_env = getenv("SAPPHIRE_LOW_MEMORY");
    if (low_mem_env && (strcmp(low_mem_env, "1") == 0 ||
                        strcasecmp(low_mem_env, "true") == 0)) {
        model->low_memory_mode = 1;
        LOG_INFO("Low-memory mode enabled: inference will use sliding-window layer eviction");
    }

    LOG_INFO("All %d shards loaded successfully", loaded_shard_count);
    return model;

fail_shards:
    for (int i = 0; i < model->safetensors_shard_count; i++) {
        if (model->safetensors_shard_handles[i])
            safetensors_close((safetensors_file_t *)model->safetensors_shard_handles[i]);
    }
    free(model->safetensors_shard_handles);
fail_layers:
    free(model->layers);
fail_model:
    free(model);
fail_paths:
    for (int i = 0; i < shard_count; i++) free(shard_paths[i]);
    free(shard_paths);
    return NULL;
}

/**
 * @brief Load a model from directory (looks for model.safetensors or model.gguf)
 */
llm_model_t* load_model(const char *model_dir, const model_spec_t *model_spec) {
    if (!model_dir) {
        LOG_ERROR("Model directory path is NULL");
        return NULL;
    }
    
    if (!model_spec) {
        LOG_ERROR("Model specification is NULL");
        return NULL;
    }
    
    LOG_INFO("Loading model from directory: %s", model_dir);
    
    // Check if directory exists
    if (access(model_dir, F_OK) == -1) {
        LOG_ERROR("Model directory not found: %s", model_dir);
        return NULL;
    }
    
    // Try to find model.safetensors first
    char *model_path = construct_safe_path(model_dir, "model.safetensors", NULL);
    if (!model_path) {
        return NULL;
    }
    
    // Check if model.safetensors exists
    if (access(model_path, F_OK) != -1) {
        LOG_INFO("✓ Found model.safetensors, loading...");
        
        LOG_INFO("Loading Safetensors format model...");
        llm_model_t *model = load_model_safetensors(model_spec, model_path);
        free(model_path);
        
        if (model) {
            LOG_INFO("Model loaded successfully from Safetensors");
            return model;
        } else {
            LOG_ERROR("Failed to load model from Safetensors");
            return NULL;
        }
    }
    
    free(model_path);

    /* Try sharded safetensors (model-00001-of-NNNNN.safetensors, ...) */
    {
        llm_model_t *model = load_model_sharded(model_spec, model_dir);
        if (model) {
            LOG_INFO("Model loaded successfully from sharded Safetensors");
            return model;
        }
    }

    LOG_ERROR("No supported model file found in directory: %s", model_dir);
    return NULL;
}

/**
 * @brief Free all model memory and close associated files.
 *
 * Releases all tensors and closes the safetensors file handle if present.
 * After calling this, the model pointer becomes invalid.
 *
 * @param model Model to destroy (may be NULL; safe noop)
 */
/* Legacy destroy that accepts a direct model pointer. This preserves
 * compatibility with unit tests and older call-sites. When possible prefer
 * `llm_model_destroy_ex(spec)` which can consult the owning `spec` for
 * accurate layer counts. */
void llm_model_destroy(llm_model_t *model) {
    if (!model) return;

    if (model->embedding_weight) tensor_release(model->embedding_weight);
    if (model->norm_final_weight) tensor_release(model->norm_final_weight);
    if (model->lm_head_weight) tensor_release(model->lm_head_weight);

    if (model->layers) {
        int n = (model->num_layers > 0) ? model->num_layers : SAPPHIRE_MAX_LAYERS;
        for (int i = 0; i < n; i++) {
            model_layer_weights_t *layer = &model->layers[i];
            if (layer->norm_attn_weight) tensor_release(layer->norm_attn_weight);
            if (layer->norm_attn_post_weight) tensor_release(layer->norm_attn_post_weight);
            if (layer->q_proj_weight) tensor_release(layer->q_proj_weight);
            if (layer->k_proj_weight) tensor_release(layer->k_proj_weight);
            if (layer->v_proj_weight) tensor_release(layer->v_proj_weight);
            if (layer->q_norm_weight) tensor_release(layer->q_norm_weight);
            if (layer->k_norm_weight) tensor_release(layer->k_norm_weight);
            if (layer->out_proj_weight) tensor_release(layer->out_proj_weight);
            if (layer->norm_ffn_weight) tensor_release(layer->norm_ffn_weight);
            if (layer->norm_ffn_post_weight) tensor_release(layer->norm_ffn_post_weight);
            if (layer->up_proj_weight) tensor_release(layer->up_proj_weight);
            if (layer->gate_proj_weight) tensor_release(layer->gate_proj_weight);
            if (layer->down_proj_weight) tensor_release(layer->down_proj_weight);
        }
        free(model->layers);
    }

    if (model->safetensors_handle)
        safetensors_close((safetensors_file_t *)model->safetensors_handle);
    if (model->safetensors_shard_handles) {
        for (int i = 0; i < model->safetensors_shard_count; i++) {
            if (model->safetensors_shard_handles[i])
                safetensors_close((safetensors_file_t *)model->safetensors_shard_handles[i]);
        }
        free(model->safetensors_shard_handles);
    }
    free(model->layer_page_info);
    free(model);
}

/* Spec-aware destroy: free the model referenced by `spec->llm_model`. This
 * allows using `spec->variant_config` to determine the exact number of
 * layers (e.g., `num_hidden_layers`) to free. */
void llm_model_destroy_ex(const struct model_spec *spec) {
    if (!spec) return;
    llm_model_t *model = (llm_model_t *)spec->llm_model;
    if (!model) return;

    if (model->embedding_weight) tensor_release(model->embedding_weight);
    if (model->norm_final_weight) tensor_release(model->norm_final_weight);
    if (model->lm_head_weight) tensor_release(model->lm_head_weight);

    int n_layers = SAPPHIRE_MAX_LAYERS;
    if (spec->variant_config) {
        /* Best-effort: many variant configs expose `num_hidden_layers` as an int. */
        /* We attempt to read that field from the common Gemma3 config if available. */
        const gemma3_270m_config_t *cfg = (const gemma3_270m_config_t *)spec->variant_config;
        if (cfg->num_hidden_layers > 0 && cfg->num_hidden_layers <= SAPPHIRE_MAX_LAYERS) {
            n_layers = cfg->num_hidden_layers;
        }
    }

    if (model->layers) {
        for (int i = 0; i < n_layers; i++) {
            model_layer_weights_t *layer = &model->layers[i];
            if (layer->norm_attn_weight) tensor_release(layer->norm_attn_weight);
            if (layer->norm_attn_post_weight) tensor_release(layer->norm_attn_post_weight);
            if (layer->q_proj_weight) tensor_release(layer->q_proj_weight);
            if (layer->k_proj_weight) tensor_release(layer->k_proj_weight);
            if (layer->v_proj_weight) tensor_release(layer->v_proj_weight);
            if (layer->q_norm_weight) tensor_release(layer->q_norm_weight);
            if (layer->k_norm_weight) tensor_release(layer->k_norm_weight);
            if (layer->out_proj_weight) tensor_release(layer->out_proj_weight);
            if (layer->norm_ffn_weight) tensor_release(layer->norm_ffn_weight);
            if (layer->norm_ffn_post_weight) tensor_release(layer->norm_ffn_post_weight);
            if (layer->up_proj_weight) tensor_release(layer->up_proj_weight);
            if (layer->gate_proj_weight) tensor_release(layer->gate_proj_weight);
            if (layer->down_proj_weight) tensor_release(layer->down_proj_weight);
        }
        free(model->layers);
    }

    if (model->safetensors_handle)
        safetensors_close((safetensors_file_t *)model->safetensors_handle);
    if (model->safetensors_shard_handles) {
        for (int i = 0; i < model->safetensors_shard_count; i++) {
            if (model->safetensors_shard_handles[i])
                safetensors_close((safetensors_file_t *)model->safetensors_shard_handles[i]);
        }
        free(model->safetensors_shard_handles);
    }
    free(model->layer_page_info);

    /* Null out spec->llm_model to avoid dangling pointer in the spec */
    ((struct model_spec *)spec)->llm_model = NULL;

    free(model);
}

/* Helper: call madvise on a tensor's data if it's mmap-backed.
 * Only external (file-mmap) tensors are safe: MADV_DONTNEED on anonymous
 * malloc pages returns zero on next read, permanently corrupting weights. */
static void madvise_tensor(const tensor_t *t, int advice) {
    if (!t) return;
    if (!tensor_is_external(t)) return;   /* malloc-backed: skip */

    const void *data = tensor_data(t);
    size_t nbytes = tensor_nbytes(t);
    if (!data || nbytes == 0) return;
    
    /* Round to page boundaries */
    size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
    uintptr_t addr = (uintptr_t)data;
    uintptr_t aligned_addr = (addr / page_size) * page_size;
    size_t offset_delta = addr - aligned_addr;
    size_t aligned_size = ((nbytes + offset_delta + page_size - 1) / page_size) * page_size;
    
    madvise((void *)aligned_addr, aligned_size, advice);
}

/**
 * Evict layer pages from resident memory (MADV_DONTNEED).
 *
 * Use this in a sliding-window inference loop to bound RSS. The mmap
 * stays valid; pages fault back in on next access.
 */
void llm_model_evict_layer(llm_model_t *model, int layer_idx) {
    if (!model || !model->low_memory_mode) {
        return;
    }
    if (layer_idx < 0 || layer_idx >= model->num_layers) {
        return;
    }
    if (!model->layers) {
        return;
    }

    const model_layer_weights_t *layer = &model->layers[layer_idx];
    
    madvise_tensor(layer->norm_attn_weight, MADV_DONTNEED);
    madvise_tensor(layer->norm_attn_post_weight, MADV_DONTNEED);
    madvise_tensor(layer->q_proj_weight, MADV_DONTNEED);
    madvise_tensor(layer->k_proj_weight, MADV_DONTNEED);
    madvise_tensor(layer->v_proj_weight, MADV_DONTNEED);
    madvise_tensor(layer->q_norm_weight, MADV_DONTNEED);
    madvise_tensor(layer->k_norm_weight, MADV_DONTNEED);
    madvise_tensor(layer->out_proj_weight, MADV_DONTNEED);
    madvise_tensor(layer->norm_ffn_weight, MADV_DONTNEED);
    madvise_tensor(layer->norm_ffn_post_weight, MADV_DONTNEED);
    madvise_tensor(layer->up_proj_weight, MADV_DONTNEED);
    madvise_tensor(layer->gate_proj_weight, MADV_DONTNEED);
    madvise_tensor(layer->down_proj_weight, MADV_DONTNEED);
}

/**
 * Prefetch layer pages for upcoming access (MADV_WILLNEED).
 *
 * Starts async readahead for the layer's mmapped pages.
 */
void llm_model_prefetch_layer(llm_model_t *model, int layer_idx) {
    if (!model || !model->low_memory_mode) {
        return;
    }
    if (layer_idx < 0 || layer_idx >= model->num_layers) {
        return;
    }
    if (!model->layers) {
        return;
    }

    const model_layer_weights_t *layer = &model->layers[layer_idx];
    
    madvise_tensor(layer->norm_attn_weight, MADV_WILLNEED);
    madvise_tensor(layer->norm_attn_post_weight, MADV_WILLNEED);
    madvise_tensor(layer->q_proj_weight, MADV_WILLNEED);
    madvise_tensor(layer->k_proj_weight, MADV_WILLNEED);
    madvise_tensor(layer->v_proj_weight, MADV_WILLNEED);
    madvise_tensor(layer->q_norm_weight, MADV_WILLNEED);
    madvise_tensor(layer->k_norm_weight, MADV_WILLNEED);
    madvise_tensor(layer->out_proj_weight, MADV_WILLNEED);
    madvise_tensor(layer->norm_ffn_weight, MADV_WILLNEED);
    madvise_tensor(layer->norm_ffn_post_weight, MADV_WILLNEED);
    madvise_tensor(layer->up_proj_weight, MADV_WILLNEED);
    madvise_tensor(layer->gate_proj_weight, MADV_WILLNEED);
    madvise_tensor(layer->down_proj_weight, MADV_WILLNEED);
}
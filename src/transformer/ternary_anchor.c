/**
 * @file ternary_anchor.c
 * @brief Implementation of saliency-driven mixed-precision anchor selection and I/O.
 *
 * This implements:
 * - Saliency oracle using weight magnitude × Hessian diagonal
 * - Deterministic top-K selection with min-heap
 * - Sorted COO serialization with 8-byte aligned entries
 * - CSR-like row offset construction for runtime traversal
 */

#include "ternary_anchor.h"

#include "file_reader.h"
#include "log.h"
#include "safetensors_reader.h"
#include "simple_json.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* =========================================================================
 * Internal Helpers
 * ========================================================================= */

static float bf16_to_f32_scalar(uint16_t value) {
    union {
        uint32_t u32;
        float f32;
    } bits;
    bits.u32 = ((uint32_t)value) << 16;
    return bits.f32;
}

static uint16_t f32_to_bf16_scalar(float value) {
    union {
        float f32;
        uint32_t u32;
    } bits;
    bits.f32 = value;
    /* Simple truncation (no rounding). */
    return (uint16_t)(bits.u32 >> 16);
}

static void ternary_anchor_sanitize_tensor_name_local(const char *tensor_name,
                                                      char *safe_name,
                                                      size_t safe_name_size)
{
    size_t i = 0u;

    if (!safe_name || safe_name_size == 0u) {
        return;
    }

    safe_name[0] = '\0';
    if (!tensor_name) {
        return;
    }

    for (; tensor_name[i] != '\0' && i + 1u < safe_name_size; ++i) {
        char ch = tensor_name[i];
        if ((ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '-' || ch == '_') {
            safe_name[i] = ch;
        } else {
            safe_name[i] = '_';
        }
    }
    safe_name[i] = '\0';
}

static int ternary_anchor_name_has_suffix(const char *name, const char *suffix)
{
    size_t name_len = 0u;
    size_t suffix_len = 0u;

    if (!name || !suffix) {
        return 0;
    }

    name_len = strlen(name);
    suffix_len = strlen(suffix);
    if (name_len < suffix_len) {
        return 0;
    }

    return strcmp(name + name_len - suffix_len, suffix) == 0;
}

static char *ternary_anchor_duplicate_parent_dir_local(const char *path)
{
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

static int ternary_anchor_fsync_parent_dir_local(const char *path)
{
    char *dir = NULL;
    int fd = -1;
    int rc = -1;

    dir = ternary_anchor_duplicate_parent_dir_local(path);
    if (!dir) {
        return -1;
    }

    fd = open(dir, O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("anchor write: failed to open parent dir %s: %s", dir, strerror(errno));
        goto cleanup;
    }
    if (fsync(fd) != 0) {
        LOG_ERROR("anchor write: failed to fsync parent dir %s: %s", dir, strerror(errno));
        goto cleanup;
    }

    rc = 0;

cleanup:
    if (fd >= 0) {
        close(fd);
    }
    free(dir);
    return rc;
}

static int ternary_anchor_write_atomic_safetensors_local(const char *output_path,
                                                         const char *header,
                                                         uint64_t header_len,
                                                         const void *data,
                                                         size_t data_size)
{
    char *tmp_path = NULL;
    FILE *file = NULL;
    int rc = -1;
    int fd = -1;

    if (!output_path || !header) {
        return -1;
    }

    tmp_path = (char *)malloc(strlen(output_path) + 5u);
    if (!tmp_path) {
        return -1;
    }
    snprintf(tmp_path, strlen(output_path) + 5u, "%s.tmp", output_path);

    file = fopen(tmp_path, "wb");
    if (!file) {
        LOG_ERROR("anchor write: failed to open %s: %s", tmp_path, strerror(errno));
        goto cleanup;
    }
    if (fwrite(&header_len, sizeof(header_len), 1, file) != 1) {
        LOG_ERROR("anchor write: failed to write safetensors header length to %s", tmp_path);
        goto cleanup;
    }
    if (header_len > 0u && fwrite(header, 1u, (size_t)header_len, file) != (size_t)header_len) {
        LOG_ERROR("anchor write: failed to write safetensors header to %s", tmp_path);
        goto cleanup;
    }
    if (data_size > 0u && data && fwrite(data, 1u, data_size, file) != data_size) {
        LOG_ERROR("anchor write: failed to write anchor payload to %s", tmp_path);
        goto cleanup;
    }
    if (fflush(file) != 0) {
        LOG_ERROR("anchor write: fflush failed for %s: %s", tmp_path, strerror(errno));
        goto cleanup;
    }
    fd = fileno(file);
    if (fd < 0 || fsync(fd) != 0) {
        LOG_ERROR("anchor write: fsync failed for %s: %s", tmp_path, strerror(errno));
        goto cleanup;
    }
    if (fclose(file) != 0) {
        file = NULL;
        LOG_ERROR("anchor write: close failed for %s: %s", tmp_path, strerror(errno));
        goto cleanup;
    }
    file = NULL;

    if (rename(tmp_path, output_path) != 0) {
        LOG_ERROR("anchor write: rename failed %s -> %s: %s", tmp_path, output_path, strerror(errno));
        goto cleanup;
    }
    if (ternary_anchor_fsync_parent_dir_local(output_path) != 0) {
        goto cleanup;
    }

    rc = 0;

cleanup:
    if (file) {
        fclose(file);
    }
    if (rc != 0 && tmp_path) {
        unlink(tmp_path);
    }
    free(tmp_path);
    return rc;
}

static int ternary_anchor_build_safetensors_header_local(char *header,
                                                         size_t header_size,
                                                         const char *tensor_name,
                                                         const ternary_hybrid_layer_t *layer,
                                                         uint32_t entry_crc32)
{
    size_t entry_bytes = 0u;
    int rc = 0;

    if (!header || header_size == 0u || !tensor_name || !layer) {
        return -1;
    }

    entry_bytes = (size_t)layer->anchor_count * sizeof(ternary_anchor_entry_t);
    rc = snprintf(
        header,
        header_size,
        "{\"__metadata__\":{\"sapphire_quant\":\"ternary-anchor-v2\","
        "\"format_version\":\"%u\",\"tensor_name\":\"%s\",\"rows\":\"%u\",\"cols\":\"%u\","
        "\"anchor_count\":\"%u\",\"scale_group_size\":\"%u\","
        "\"groups_per_row\":\"%u\",\"value_dtype\":\"BF16\","
        "\"saliency_mode\":\"%u\",\"budget_ppm\":\"%u\","
        "\"entry_layout\":\"row,col,value_bf16,pad\","
        "\"entry_order\":\"row_col\",\"entry_crc32\":\"%08x\","
        "\"saliency_cutoff\":\"%.9g\",\"anchor_value_rms\":\"%.9g\","
        "\"bulk_gamma_mean\":\"%.9g\",\"max_row_nnz\":\"%u\"},"
        "\"%s.anchor_entries\":{\"dtype\":\"U16\",\"shape\":[%u,4],\"data_offsets\":[0,%zu]}}",
        TERNARY_ANCHOR_VERSION,
        tensor_name,
        layer->rows,
        layer->cols,
        layer->anchor_count,
        layer->scale_group_size,
        layer->groups_per_row,
        layer->anchor_metadata.saliency_mode,
        layer->anchor_metadata.budget_ppm,
        entry_crc32,
        (double)layer->anchor_metadata.saliency_cutoff,
        (double)layer->anchor_metadata.anchor_value_rms,
        (double)layer->anchor_metadata.bulk_gamma_mean,
        layer->anchor_metadata.max_row_nnz,
        tensor_name,
        layer->anchor_count,
        entry_bytes);
    if (rc < 0 || (size_t)rc >= header_size) {
        return -1;
    }
    return rc;
}

static int ternary_anchor_parse_optional_f32_metadata_local(const safetensors_file_t *st,
                                                            const char *key,
                                                            float *out_value)
{
    char value_text[64];
    char *end = NULL;
    float parsed = 0.0f;

    if (!out_value) {
        return -1;
    }

    *out_value = 0.0f;
    if (!safetensors_metadata_get_string(st, key, value_text, sizeof(value_text))) {
        return 0;
    }

    errno = 0;
    parsed = strtof(value_text, &end);
    if (errno != 0 || end == value_text || *end != '\0') {
        return -1;
    }

    *out_value = parsed;
    return 0;
}

static int ternary_anchor_expected_entries_name_local(const char *tensor_name,
                                                      char *name_buf,
                                                      size_t name_buf_size)
{
    int written = 0;

    if (!tensor_name || !name_buf || name_buf_size == 0u) {
        return -1;
    }

    written = snprintf(name_buf, name_buf_size, "%s.anchor_entries", tensor_name);
    if (written < 0 || (size_t)written >= name_buf_size) {
        return -1;
    }
    return 0;
}

static const safetensors_tensor_meta_t *ternary_anchor_find_entries_meta_local(const safetensors_file_t *st,
                                                                               const char *expected_tensor_name)
{
    const safetensors_tensor_meta_t *found = NULL;
    int tensor_count = 0;
    char expected_name[320];
    int require_exact_name = 0;

    if (!st) {
        return NULL;
    }

    memset(expected_name, 0, sizeof(expected_name));
    if (expected_tensor_name && expected_tensor_name[0] != '\0') {
        if (ternary_anchor_expected_entries_name_local(expected_tensor_name,
                                                       expected_name,
                                                       sizeof(expected_name)) != 0) {
            LOG_ERROR("anchor load: expected tensor name is too long");
            return NULL;
        }
        require_exact_name = 1;
    }

    tensor_count = safetensors_tensor_count(st);
    for (int i = 0; i < tensor_count; ++i) {
        const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_index(st, i);
        if (!meta) {
            continue;
        }
        if (!ternary_anchor_name_has_suffix(meta->name, ".anchor_entries")) {
            continue;
        }
        if (require_exact_name && strcmp(meta->name, expected_name) != 0) {
            continue;
        }
        if (found) {
            LOG_ERROR("anchor load: multiple anchor_entries tensors found in sidecar");
            return NULL;
        }
        found = meta;
    }

    if (!found) {
        if (require_exact_name) {
            LOG_ERROR("anchor load: missing %s tensor in sidecar", expected_name);
        } else {
            LOG_ERROR("anchor load: missing .anchor_entries tensor in sidecar");
        }
    }
    return found;
}

static int ternary_anchor_validate_tensor_identity_local(const safetensors_file_t *st,
                                                         const safetensors_tensor_meta_t *entries_meta,
                                                         const char *expected_tensor_name,
                                                         const char *anchor_path)
{
    char metadata_tensor_name[256];
    char expected_entries_name[320];

    if (!st || !entries_meta || !anchor_path) {
        return -1;
    }
    if (!expected_tensor_name || expected_tensor_name[0] == '\0') {
        return 0;
    }

    if (ternary_anchor_expected_entries_name_local(expected_tensor_name,
                                                   expected_entries_name,
                                                   sizeof(expected_entries_name)) != 0) {
        LOG_ERROR("anchor load: expected tensor name is too long");
        return -1;
    }
    if (strcmp(entries_meta->name, expected_entries_name) != 0) {
        LOG_ERROR("anchor load: tensor name mismatch in %s (expected %s, found %s)",
                  anchor_path,
                  expected_entries_name,
                  entries_meta->name);
        return -1;
    }
    if (safetensors_metadata_get_string(st,
                                        "tensor_name",
                                        metadata_tensor_name,
                                        sizeof(metadata_tensor_name)) &&
        strcmp(metadata_tensor_name, expected_tensor_name) != 0) {
        LOG_ERROR("anchor load: tensor_name metadata mismatch in %s (expected %s, found %s)",
                  anchor_path,
                  expected_tensor_name,
                  metadata_tensor_name);
        return -1;
    }

    return 0;
}

static int ternary_anchor_read_safetensors_metadata_local(const safetensors_file_t *st,
                                                          const char *anchor_path,
                                                          ternary_anchor_metadata_t *out_meta)
{
    char value_dtype[32];

    if (!st || !anchor_path || !out_meta) {
        return -1;
    }

    memset(out_meta, 0, sizeof(*out_meta));
    if (!safetensors_metadata_get_u32(st, "format_version", &out_meta->version) ||
        out_meta->version != TERNARY_ANCHOR_VERSION) {
        LOG_ERROR("anchor load: unsupported safetensors format_version in %s", anchor_path);
        return -1;
    }
    if (!safetensors_metadata_get_u32(st, "rows", &out_meta->rows) ||
        !safetensors_metadata_get_u32(st, "cols", &out_meta->cols) ||
        !safetensors_metadata_get_u32(st, "anchor_count", &out_meta->anchor_count) ||
        !safetensors_metadata_get_u32(st, "scale_group_size", &out_meta->scale_group_size) ||
        !safetensors_metadata_get_u32(st, "groups_per_row", &out_meta->groups_per_row) ||
        !safetensors_metadata_get_u32(st, "saliency_mode", &out_meta->saliency_mode) ||
        !safetensors_metadata_get_u32(st, "budget_ppm", &out_meta->budget_ppm) ||
        !safetensors_metadata_get_u32(st, "max_row_nnz", &out_meta->max_row_nnz) ||
        !safetensors_metadata_get_u32(st, "entry_crc32", &out_meta->crc32)) {
        LOG_ERROR("anchor load: missing required safetensors metadata in %s", anchor_path);
        return -1;
    }
    if (!safetensors_metadata_get_string(st, "value_dtype", value_dtype, sizeof(value_dtype)) ||
        strcmp(value_dtype, "BF16") != 0) {
        LOG_ERROR("anchor load: unsupported value_dtype in %s", anchor_path);
        return -1;
    }
    if (out_meta->rows == 0u || out_meta->cols == 0u) {
        LOG_ERROR("anchor load: invalid safetensors rows/cols in %s", anchor_path);
        return -1;
    }

    return 0;
}

static int ternary_anchor_validate_entries_tensor_local(const safetensors_tensor_meta_t *entries_meta,
                                                        const ternary_anchor_metadata_t *meta,
                                                        const char *anchor_path,
                                                        size_t *out_entry_bytes)
{
    size_t entry_bytes = 0u;

    if (!entries_meta || !meta || !anchor_path || !out_entry_bytes) {
        return -1;
    }

    if (entries_meta->dtype != SAFETENSORS_U16 ||
        entries_meta->ndim != 2 ||
        entries_meta->shape[0] != meta->anchor_count ||
        entries_meta->shape[1] != 4u) {
        LOG_ERROR("anchor load: invalid anchor_entries tensor layout in %s", anchor_path);
        return -1;
    }

    entry_bytes = (size_t)meta->anchor_count * sizeof(ternary_anchor_entry_t);
    if (entries_meta->size_bytes != entry_bytes) {
        LOG_ERROR("anchor load: anchor_entries byte mismatch in %s", anchor_path);
        return -1;
    }

    *out_entry_bytes = entry_bytes;
    return 0;
}

static int ternary_anchor_load_entries_from_safetensors_local(const safetensors_file_t *st,
                                                              const safetensors_tensor_meta_t *entries_meta,
                                                              const ternary_anchor_metadata_t *meta,
                                                              size_t entry_bytes,
                                                              ternary_anchor_entry_t **out_entries,
                                                              const char *anchor_path)
{
    const uint16_t *entries_words = NULL;
    ternary_anchor_entry_t *entries = NULL;

    if (!st || !entries_meta || !meta || !out_entries || !anchor_path) {
        return -1;
    }

    *out_entries = NULL;
    if (meta->anchor_count == 0u) {
        return 0;
    }

    entries_words = (const uint16_t *)safetensors_data_ptr(st, entries_meta);
    if (!entries_words) {
        return -1;
    }

    entries = (ternary_anchor_entry_t *)malloc(entry_bytes);
    if (!entries) {
        LOG_ERROR("anchor load: failed to allocate safetensors entries");
        return -1;
    }
    memcpy(entries, entries_words, entry_bytes);

    if (meta->crc32 != 0u) {
        uint32_t actual_crc32 = io_crc32_update(0u, entries, entry_bytes);
        if (actual_crc32 != meta->crc32) {
            LOG_ERROR("anchor load: safetensors CRC mismatch for %s (metadata=%08x actual=%08x)",
                      anchor_path,
                      meta->crc32,
                      actual_crc32);
            free(entries);
            return -1;
        }
    }

    *out_entries = entries;
    return 0;
}

static int ternary_anchor_init_safetensors_view_local(const safetensors_file_t *st,
                                                      const ternary_anchor_metadata_t *meta,
                                                      const ternary_anchor_entry_t *entries,
                                                      const uint32_t *row_offsets,
                                                      ternary_anchor_view_t *out_view,
                                                      const char *anchor_path)
{
    if (!st || !meta || !row_offsets || !out_view || !anchor_path) {
        return -1;
    }

    memset(out_view, 0, sizeof(*out_view));
    out_view->entries = entries;
    out_view->row_offsets = row_offsets;
    out_view->anchor_count = meta->anchor_count;
    out_view->rows = meta->rows;
    out_view->cols = meta->cols;
    out_view->metadata = *meta;
    out_view->owns_memory = 1;

    if (ternary_anchor_parse_optional_f32_metadata_local(st, "saliency_cutoff", &out_view->metadata.saliency_cutoff) != 0 ||
        ternary_anchor_parse_optional_f32_metadata_local(st, "anchor_value_rms", &out_view->metadata.anchor_value_rms) != 0 ||
        ternary_anchor_parse_optional_f32_metadata_local(st, "bulk_gamma_mean", &out_view->metadata.bulk_gamma_mean) != 0) {
        LOG_ERROR("anchor load: invalid floating-point metadata in %s", anchor_path);
        ternary_anchor_view_release(out_view);
        return -1;
    }
    if (ternary_anchor_validate(out_view) != 0) {
        ternary_anchor_view_release(out_view);
        return -1;
    }

    return 0;
}

static int ternary_anchor_load_legacy_binary_local(const char *anchor_path,
                                                   ternary_anchor_view_t *out_view)
{
    FILE *file = NULL;
    ternary_anchor_metadata_t meta;
    ternary_anchor_entry_t *entries = NULL;
    uint32_t *row_offsets = NULL;
    int rc = -1;

    file = fopen(anchor_path, "rb");
    if (!file) {
        LOG_ERROR("anchor load: failed to open %s: %s", anchor_path, strerror(errno));
        return -1;
    }

    if (fread(&meta, sizeof(meta), 1, file) != 1) {
        LOG_ERROR("anchor load: failed to read legacy header from %s", anchor_path);
        goto cleanup;
    }
    if (meta.magic != TERNARY_ANCHOR_MAGIC) {
        LOG_ERROR("anchor load: invalid legacy magic 0x%08x in %s", meta.magic, anchor_path);
        goto cleanup;
    }
    if (meta.version != TERNARY_ANCHOR_LEGACY_VERSION) {
        LOG_ERROR("anchor load: unsupported legacy version %u in %s", meta.version, anchor_path);
        goto cleanup;
    }
    if (meta.rows == 0u || meta.cols == 0u) {
        LOG_ERROR("anchor load: invalid legacy rows/cols in %s", anchor_path);
        goto cleanup;
    }

    if (meta.anchor_count > 0u) {
        entries = (ternary_anchor_entry_t *)malloc(meta.anchor_count * sizeof(*entries));
        if (!entries) {
            LOG_ERROR("anchor load: failed to allocate legacy entries");
            goto cleanup;
        }
        if (fread(entries, sizeof(*entries), meta.anchor_count, file) != meta.anchor_count) {
            LOG_ERROR("anchor load: failed to read legacy entries from %s", anchor_path);
            goto cleanup;
        }
        if (meta.crc32 != 0u) {
            uint32_t actual_crc32 = io_crc32_update(0u,
                                                    entries,
                                                    meta.anchor_count * sizeof(*entries));
            if (actual_crc32 != meta.crc32) {
                LOG_ERROR("anchor load: legacy CRC mismatch for %s (header=%08x actual=%08x)",
                          anchor_path,
                          meta.crc32,
                          actual_crc32);
                goto cleanup;
            }
        }
    }

    row_offsets = (uint32_t *)malloc((meta.rows + 1u) * sizeof(uint32_t));
    if (!row_offsets) {
        LOG_ERROR("anchor load: failed to allocate legacy row_offsets");
        goto cleanup;
    }
    if (ternary_anchor_build_row_offsets(entries, meta.anchor_count, meta.rows, row_offsets) != 0) {
        goto cleanup;
    }

    memset(out_view, 0, sizeof(*out_view));
    out_view->entries = entries;
    out_view->row_offsets = row_offsets;
    out_view->anchor_count = meta.anchor_count;
    out_view->rows = meta.rows;
    out_view->cols = meta.cols;
    out_view->metadata = meta;
    out_view->owns_memory = 1;
    entries = NULL;
    row_offsets = NULL;

    if (ternary_anchor_validate(out_view) != 0) {
        ternary_anchor_view_release(out_view);
        goto cleanup;
    }

    LOG_INFO("anchor load: loaded %u legacy anchors from %s", meta.anchor_count, anchor_path);
    rc = 0;

cleanup:
    if (file) {
        fclose(file);
    }
    free(entries);
    free(row_offsets);
    return rc;
}

static int ternary_anchor_load_safetensors_local(const char *anchor_path,
                                                 const char *expected_tensor_name,
                                                 ternary_anchor_view_t *out_view)
{
    safetensors_file_t *st = NULL;
    const safetensors_tensor_meta_t *entries_meta = NULL;
    ternary_anchor_entry_t *entries = NULL;
    uint32_t *row_offsets = NULL;
    ternary_anchor_metadata_t meta;
    size_t entry_bytes = 0u;
    int rc = -1;

    memset(&meta, 0, sizeof(meta));

    st = safetensors_open(anchor_path);
    if (!st) {
        return -1;
    }

    entries_meta = ternary_anchor_find_entries_meta_local(st, expected_tensor_name);
    if (!entries_meta) {
        goto cleanup;
    }
    if (ternary_anchor_validate_tensor_identity_local(st,
                                                     entries_meta,
                                                     expected_tensor_name,
                                                     anchor_path) != 0) {
        goto cleanup;
    }
    if (ternary_anchor_read_safetensors_metadata_local(st, anchor_path, &meta) != 0) {
        goto cleanup;
    }
    if (ternary_anchor_validate_entries_tensor_local(entries_meta, &meta, anchor_path, &entry_bytes) != 0) {
        goto cleanup;
    }
    if (ternary_anchor_load_entries_from_safetensors_local(
            st, entries_meta, &meta, entry_bytes, &entries, anchor_path) != 0) {
        goto cleanup;
    }

    row_offsets = (uint32_t *)malloc((meta.rows + 1u) * sizeof(uint32_t));
    if (!row_offsets) {
        LOG_ERROR("anchor load: failed to allocate safetensors row_offsets");
        goto cleanup;
    }
    if (ternary_anchor_build_row_offsets(entries, meta.anchor_count, meta.rows, row_offsets) != 0) {
        goto cleanup;
    }

    if (ternary_anchor_init_safetensors_view_local(st, &meta, entries, row_offsets, out_view, anchor_path) != 0) {
        entries = NULL;
        row_offsets = NULL;
        goto cleanup;
    }

    entries = NULL;
    row_offsets = NULL;
    LOG_INFO("anchor load: loaded %u safetensors anchors from %s", meta.anchor_count, anchor_path);
    rc = 0;

cleanup:
    safetensors_close(st);
    free(entries);
    free(row_offsets);
    return rc;
}

/**
 * @brief Candidate entry for selection.
 */
typedef struct {
    float score;
    uint32_t row;
    uint32_t col;
} anchor_candidate_t;

typedef struct {
    const uint16_t *bf16_weights;
    uint32_t rows;
    uint32_t cols;
    const float *hessian_diag;
    const ternary_anchor_config_t *config;
} anchor_selection_request_t;

/**
 * @brief Compare candidates for min-heap (lowest score at top).
 */
static int candidate_compare_min(const void *a, const void *b) {
    const anchor_candidate_t *ca = (const anchor_candidate_t *)a;
    const anchor_candidate_t *cb = (const anchor_candidate_t *)b;
    
    if (ca->score < cb->score) return -1;
    if (ca->score > cb->score) return 1;
    /* Tie-break by position for determinism */
    if (ca->row < cb->row) return -1;
    if (ca->row > cb->row) return 1;
    if (ca->col < cb->col) return -1;
    if (ca->col > cb->col) return 1;
    return 0;
}

/**
 * @brief Compare candidates for sorting by (row, col).
 */
static int candidate_compare_position(const void *a, const void *b) {
    const anchor_candidate_t *ca = (const anchor_candidate_t *)a;
    const anchor_candidate_t *cb = (const anchor_candidate_t *)b;
    
    if (ca->row < cb->row) return -1;
    if (ca->row > cb->row) return 1;
    if (ca->col < cb->col) return -1;
    if (ca->col > cb->col) return 1;
    return 0;
}

/**
 * @brief Compare anchor entries by (row, col).
 */
static int entry_compare_position(const void *a, const void *b) {
    const ternary_anchor_entry_t *ea = (const ternary_anchor_entry_t *)a;
    const ternary_anchor_entry_t *eb = (const ternary_anchor_entry_t *)b;
    
    if (ea->row < eb->row) return -1;
    if (ea->row > eb->row) return 1;
    if (ea->col < eb->col) return -1;
    if (ea->col > eb->col) return 1;
    return 0;
}

/**
 * @brief Min-heap operations for top-K selection.
 */
static void heap_sift_down(anchor_candidate_t *heap, size_t size, size_t i) {
    size_t smallest = i;
    size_t left = 2 * i + 1;
    size_t right = 2 * i + 2;
    
    if (left < size && candidate_compare_min(&heap[left], &heap[smallest]) < 0) {
        smallest = left;
    }
    if (right < size && candidate_compare_min(&heap[right], &heap[smallest]) < 0) {
        smallest = right;
    }
    if (smallest != i) {
        anchor_candidate_t tmp = heap[i];
        heap[i] = heap[smallest];
        heap[smallest] = tmp;
        heap_sift_down(heap, size, smallest);
    }
}

static void heap_sift_up(anchor_candidate_t *heap, size_t i) {
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (candidate_compare_min(&heap[i], &heap[parent]) < 0) {
            anchor_candidate_t tmp = heap[i];
            heap[i] = heap[parent];
            heap[parent] = tmp;
            i = parent;
        } else {
            break;
        }
    }
}

static void heap_push(anchor_candidate_t *heap, size_t *size, size_t capacity,
                      const anchor_candidate_t *candidate) {
    if (*size < capacity) {
        heap[*size] = *candidate;
        heap_sift_up(heap, *size);
        (*size)++;
    } else if (candidate_compare_min(candidate, &heap[0]) > 0) {
        /* Replace minimum with new candidate (which has higher score) */
        heap[0] = *candidate;
        heap_sift_down(heap, *size, 0);
    }
}

/* =========================================================================
 * Public API Implementation
 * ========================================================================= */

ternary_anchor_config_t ternary_anchor_default_config(void) {
    ternary_anchor_config_t config;
    config.saliency_mode = TERNARY_ANCHOR_SALIENCY_WEIGHT_TIMES_HESSIAN;
    config.budget_ppm = 1000; /* 0.1% default */
    config.min_anchors_per_tensor = 128;
    config.max_anchors_per_tensor = 0; /* No cap by default */
    config.max_anchors_per_row = 0;    /* No per-row cap by default */
    config.deterministic_tiebreak = 1;
    return config;
}

const char *ternary_anchor_saliency_mode_name(ternary_anchor_saliency_mode_t mode) {
    switch (mode) {
    case TERNARY_ANCHOR_SALIENCY_NONE:
        return "none";
    case TERNARY_ANCHOR_SALIENCY_WEIGHT_TIMES_HESSIAN:
        return "weight_times_hessian";
    case TERNARY_ANCHOR_SALIENCY_HESSIAN_ONLY:
        return "hessian_only";
    case TERNARY_ANCHOR_SALIENCY_WEIGHT_ONLY:
        return "weight_only";
    default:
        return "unknown";
    }
}

static int anchor_selection_validate_request(const anchor_selection_request_t *request,
                                             const ternary_anchor_selection_t *out_selection)
{
    if (!request || !request->bf16_weights || !request->config || !out_selection) {
        LOG_ERROR("anchor select: NULL argument");
        return -1;
    }
    if (request->rows == 0 || request->cols == 0) {
        LOG_ERROR("anchor select: zero dimensions");
        return -1;
    }
    if (request->rows > UINT16_MAX || request->cols > UINT16_MAX) {
        LOG_ERROR("anchor select: dimensions exceed 16-bit index range (rows=%u, cols=%u)",
                  request->rows,
                  request->cols);
        return -1;
    }
    if (request->config->saliency_mode == TERNARY_ANCHOR_SALIENCY_WEIGHT_TIMES_HESSIAN && !request->hessian_diag) {
        LOG_ERROR("anchor select: saliency_mode requires hessian_diag");
        return -1;
    }
    if (request->config->saliency_mode == TERNARY_ANCHOR_SALIENCY_HESSIAN_ONLY && !request->hessian_diag) {
        LOG_ERROR("anchor select: hessian_only mode requires hessian_diag");
        return -1;
    }
    return 0;
}

static size_t anchor_selection_target_count(const anchor_selection_request_t *request)
{
    size_t total_weights = (size_t)request->rows * request->cols;
    size_t target_count = (total_weights * request->config->budget_ppm + 500000u) / 1000000u;

    if (target_count < request->config->min_anchors_per_tensor && request->config->min_anchors_per_tensor > 0u) {
        target_count = request->config->min_anchors_per_tensor;
    }
    if (request->config->max_anchors_per_tensor > 0u &&
        target_count > request->config->max_anchors_per_tensor) {
        target_count = request->config->max_anchors_per_tensor;
    }
    if (target_count > total_weights) {
        target_count = total_weights;
    }
    if (target_count == 0u) {
        LOG_WARN("anchor select: computed target count is 0, using minimum of 1");
        target_count = 1u;
    }
    return target_count;
}

static float anchor_candidate_score(float weight,
                                    uint32_t col,
                                    const float *hessian_diag,
                                    ternary_anchor_saliency_mode_t saliency_mode)
{
    switch (saliency_mode) {
    case TERNARY_ANCHOR_SALIENCY_WEIGHT_TIMES_HESSIAN:
        return fabsf(weight) * hessian_diag[col];
    case TERNARY_ANCHOR_SALIENCY_HESSIAN_ONLY:
        return hessian_diag[col];
    case TERNARY_ANCHOR_SALIENCY_WEIGHT_ONLY:
        return fabsf(weight);
    default:
        return fabsf(weight);
    }
}

static void anchor_selection_fill_heap(const anchor_selection_request_t *request,
                                       anchor_candidate_t *heap,
                                       size_t *heap_size,
                                       size_t target_count)
{
    for (uint32_t row = 0; row < request->rows; ++row) {
        size_t row_base = (size_t)row * request->cols;

        for (uint32_t col = 0; col < request->cols; ++col) {
            anchor_candidate_t candidate;
            float weight = bf16_to_f32_scalar(request->bf16_weights[row_base + col]);

            candidate.score = anchor_candidate_score(weight,
                                                     col,
                                                     request->hessian_diag,
                                                     request->config->saliency_mode);
            candidate.row = row;
            candidate.col = col;
            heap_push(heap, heap_size, target_count, &candidate);
        }
    }
}

static void anchor_selection_apply_row_cap(anchor_candidate_t *heap,
                                           size_t *heap_size,
                                           const ternary_anchor_config_t *config,
                                           uint32_t rows,
                                           uint32_t *row_counts)
{
    size_t write_idx = 0u;

    if (!heap || !heap_size || !config || !row_counts || config->max_anchors_per_row == 0u) {
        return;
    }

    memset(row_counts, 0, rows * sizeof(uint32_t));
    for (size_t i = 0; i < *heap_size; ++i) {
        uint32_t row = heap[i].row;
        if (row_counts[row] < config->max_anchors_per_row) {
            heap[write_idx++] = heap[i];
            row_counts[row]++;
        }
    }

    if (write_idx < *heap_size) {
        LOG_INFO("anchor select: per-row cap reduced count from %zu to %zu",
                 *heap_size,
                 write_idx);
        *heap_size = write_idx;
    }
}

static int anchor_selection_build_output(const anchor_selection_request_t *request,
                                         anchor_candidate_t *heap,
                                         size_t heap_size,
                                         float saliency_cutoff,
                                         ternary_anchor_selection_t *out_selection)
{
    double anchor_sum_sq = 0.0;

    out_selection->entries = (ternary_anchor_entry_t *)malloc(heap_size * sizeof(ternary_anchor_entry_t));
    out_selection->row_offsets = (uint32_t *)malloc((request->rows + 1u) * sizeof(uint32_t));
    if (!out_selection->entries || !out_selection->row_offsets) {
        LOG_ERROR("anchor select: failed to allocate output");
        return -1;
    }

    for (size_t i = 0; i < heap_size; ++i) {
        uint32_t row = heap[i].row;
        uint32_t col = heap[i].col;
        float weight = bf16_to_f32_scalar(request->bf16_weights[(size_t)row * request->cols + col]);

        out_selection->entries[i].row = (uint16_t)row;
        out_selection->entries[i].col = (uint16_t)col;
        out_selection->entries[i].value_bf16 = request->bf16_weights[(size_t)row * request->cols + col];
        out_selection->entries[i]._pad = 0u;
        anchor_sum_sq += (double)weight * (double)weight;
    }

    memset(out_selection->row_offsets, 0, (request->rows + 1u) * sizeof(uint32_t));
    for (size_t i = 0; i < heap_size; ++i) {
        out_selection->row_offsets[out_selection->entries[i].row + 1u]++;
    }
    for (uint32_t row = 1; row <= request->rows; ++row) {
        out_selection->row_offsets[row] += out_selection->row_offsets[row - 1u];
    }

    out_selection->max_row_nnz = 0u;
    for (uint32_t row = 0; row < request->rows; ++row) {
        uint32_t row_nnz = out_selection->row_offsets[row + 1u] - out_selection->row_offsets[row];
        if (row_nnz > out_selection->max_row_nnz) {
            out_selection->max_row_nnz = row_nnz;
        }
    }

    out_selection->anchor_count = (uint32_t)heap_size;
    out_selection->rows = request->rows;
    out_selection->cols = request->cols;
    out_selection->saliency_cutoff = saliency_cutoff;
    out_selection->anchor_value_rms = heap_size > 0u
        ? (float)sqrt(anchor_sum_sq / (double)heap_size)
        : 0.0f;
    return 0;
}

int ternary_anchor_select(const uint16_t *bf16_weights,
                          uint32_t rows,
                          uint32_t cols,
                          const float *hessian_diag,
                          const ternary_anchor_config_t *config,
                          ternary_anchor_selection_t *out_selection)
{
    anchor_selection_request_t request;
    anchor_candidate_t *heap = NULL;
    uint32_t *row_counts = NULL;
    size_t heap_size = 0u;
    size_t target_count = 0u;
    float saliency_cutoff = 0.0f;
    int rc = -1;

    request.bf16_weights = bf16_weights;
    request.rows = rows;
    request.cols = cols;
    request.hessian_diag = hessian_diag;
    request.config = config;

    if (anchor_selection_validate_request(&request, out_selection) != 0) {
        return -1;
    }

    memset(out_selection, 0, sizeof(*out_selection));
    target_count = anchor_selection_target_count(&request);

    LOG_INFO("anchor select: rows=%u cols=%u budget_ppm=%u target_count=%zu mode=%s",
             rows,
             cols,
             config->budget_ppm,
             target_count,
             ternary_anchor_saliency_mode_name(config->saliency_mode));

    heap = (anchor_candidate_t *)malloc(target_count * sizeof(anchor_candidate_t));
    if (!heap) {
        LOG_ERROR("anchor select: failed to allocate heap");
        return -1;
    }

    if (config->max_anchors_per_row > 0u) {
        row_counts = (uint32_t *)calloc(rows, sizeof(uint32_t));
        if (!row_counts) {
            LOG_ERROR("anchor select: failed to allocate row_counts");
            goto cleanup;
        }
    }

    anchor_selection_fill_heap(&request,
                               heap,
                               &heap_size,
                               target_count);

    LOG_DEBUG("anchor select: heap contains %zu candidates", heap_size);

    if (heap_size > 0u) {
        saliency_cutoff = heap[0].score;
    }
    qsort(heap, heap_size, sizeof(anchor_candidate_t), candidate_compare_position);

    if (config->max_anchors_per_row > 0u && heap_size > 0u) {
        anchor_selection_apply_row_cap(heap, &heap_size, config, rows, row_counts);
    }

    if (anchor_selection_build_output(&request,
                                      heap,
                                      heap_size,
                                      saliency_cutoff,
                                      out_selection) != 0) {
        goto cleanup;
    }

    LOG_INFO("anchor select: selected %u anchors, cutoff=%.6e, rms=%.6e, max_row_nnz=%u",
             out_selection->anchor_count,
             out_selection->saliency_cutoff,
             out_selection->anchor_value_rms,
             out_selection->max_row_nnz);
    rc = 0;

cleanup:
    free(heap);
    free(row_counts);

    if (rc != 0) {
        ternary_anchor_selection_free(out_selection);
    }

    return rc;
}

void ternary_anchor_selection_free(ternary_anchor_selection_t *selection)
{
    if (!selection) {
        return;
    }

    free(selection->entries);
    free(selection->row_offsets);
    memset(selection, 0, sizeof(*selection));
}

int ternary_anchor_build_row_offsets(const ternary_anchor_entry_t *entries,
                                     uint32_t anchor_count,
                                     uint32_t rows,
                                     uint32_t *out_offsets) {
    if (!out_offsets) {
        return -1;
    }
    
    memset(out_offsets, 0, (rows + 1) * sizeof(uint32_t));
    
    if (!entries || anchor_count == 0) {
        return 0;
    }
    
    /* Count entries per row */
    for (uint32_t i = 0; i < anchor_count; ++i) {
        if (entries[i].row >= rows) {
            LOG_ERROR("anchor build_row_offsets: entry %u has invalid row %u >= %u",
                      i, entries[i].row, rows);
            return -1;
        }
        out_offsets[entries[i].row + 1]++;
    }
    
    /* Prefix sum */
    for (uint32_t r = 1; r <= rows; ++r) {
        out_offsets[r] += out_offsets[r - 1];
    }
    
    return 0;
}

/* =========================================================================
 * I/O Implementation
 * ========================================================================= */

int ternary_anchor_write_layer(const char *output_dir,
                               const char *tensor_name,
                               const ternary_hybrid_layer_t *layer,
                               uint32_t *out_crc32) {
    char *bulk_path = NULL;
    char *anchors_path = NULL;
    char safe_name[256];
    char bulk_file_name[512];
    char anchor_file_name[512];
    char header[4096];
    uint32_t bulk_crc32 = 0u;
    uint32_t final_crc32 = 0u;
    uint32_t anchor_crc32 = 0u;
    size_t entry_bytes = 0u;
    int rc = -1;
    
    if (!output_dir || !tensor_name || !layer || !out_crc32) {
        LOG_ERROR("anchor write: NULL argument");
        return -1;
    }
    if (layer->anchor_count > 0u && !layer->anchor_entries) {
        LOG_ERROR("anchor write: non-zero anchor_count without anchor entries");
        return -1;
    }
    
    *out_crc32 = 0;
    
    if (io_prepare_ternary_output_dir(output_dir) != 0) {
        return -1;
    }

    /* First write the ternary bulk without appending a plain manifest row. */
    ternary_layer_t bulk_layer;
    bulk_layer.packed_weights = layer->packed_weights;
    bulk_layer.packed_weight_bytes = layer->packed_weight_bytes;
    bulk_layer.scales = layer->scales;
    bulk_layer.scale_count = layer->scale_count;
    bulk_layer.scale_bytes = layer->scale_bytes;
    bulk_layer.scale_dtype = layer->scale_dtype;
    bulk_layer.rows = layer->rows;
    bulk_layer.cols = layer->cols;
    bulk_layer.scale_group_size = layer->scale_group_size;
    bulk_layer.groups_per_row = layer->groups_per_row;
    bulk_layer.integrity = layer->integrity;

    ternary_anchor_sanitize_tensor_name_local(tensor_name, safe_name, sizeof(safe_name));

    if (snprintf(bulk_file_name, sizeof(bulk_file_name), "%s.safetensors", safe_name) < 0) {
        LOG_ERROR("anchor write: failed to build bulk filename for %s", tensor_name);
        return -1;
    }
    bulk_path = construct_safe_path(output_dir, bulk_file_name, NULL);
    if (!bulk_path) {
        LOG_ERROR("anchor write: failed to construct bulk path");
        return -1;
    }
    if (io_write_layer_ternary(bulk_path, tensor_name, &bulk_layer, &bulk_crc32) != 0) {
        LOG_ERROR("anchor write: failed to write ternary bulk for %s", tensor_name);
        goto cleanup;
    }
    final_crc32 = bulk_crc32;
    
    /* Write anchor payload as an explicit safetensors sidecar. */
    snprintf(anchor_file_name, sizeof(anchor_file_name), "%s.anchors.safetensors", safe_name);
    anchors_path = construct_safe_path(output_dir, anchor_file_name, NULL);
    if (!anchors_path) {
        LOG_ERROR("anchor write: failed to construct path");
        goto cleanup;
    }
    entry_bytes = (size_t)layer->anchor_count * sizeof(ternary_anchor_entry_t);
    if (entry_bytes > 0u) {
        anchor_crc32 = io_crc32_update(0u, layer->anchor_entries, entry_bytes);
    }
    if (ternary_anchor_build_safetensors_header_local(header,
                                                      sizeof(header),
                                                      tensor_name,
                                                      layer,
                                                      anchor_crc32) < 0) {
        LOG_ERROR("anchor write: failed to build safetensors header for %s", tensor_name);
        goto cleanup;
    }
    if (ternary_anchor_write_atomic_safetensors_local(anchors_path,
                                                      header,
                                                      (uint64_t)strlen(header),
                                                      layer->anchor_entries,
                                                      entry_bytes) != 0) {
        goto cleanup;
    }
    final_crc32 = io_crc32_update(final_crc32, header, strlen(header));
    if (entry_bytes > 0u) {
        final_crc32 = io_crc32_update(final_crc32, layer->anchor_entries, entry_bytes);
    }

    if (ternary_anchor_append_manifest(output_dir,
                                       tensor_name,
                                       bulk_file_name,
                                       anchor_file_name,
                                       layer,
                                       bulk_crc32) != 0) {
        LOG_ERROR("anchor write: failed to append manifest entry for %s", tensor_name);
        goto cleanup;
    }
    
    *out_crc32 = final_crc32;
    rc = 0;
    
    LOG_INFO("anchor write: wrote %u anchors to %s (crc32=%08x)",
             layer->anchor_count, anchors_path, final_crc32);
    
cleanup:
    free(bulk_path);
    free(anchors_path);
    return rc;
}

int ternary_anchor_append_manifest(const char *output_dir,
                                   const char *tensor_name,
                                   const char *bulk_file_name,
                                   const char *anchor_file_name,
                                   const ternary_hybrid_layer_t *layer,
                                   uint32_t bulk_crc32) {
    char *manifest_path = NULL;
    char line[1024];
    int fd = -1;
    int rc = -1;
    int n = 0;
    ssize_t wrote = 0;
    
    if (!output_dir || !tensor_name || !bulk_file_name || !anchor_file_name || !layer) {
        return -1;
    }
    
    manifest_path = construct_safe_path(output_dir, "manifest.tsv", NULL);
    if (!manifest_path) {
        return -1;
    }
    
    fd = open(manifest_path, O_CREAT | O_APPEND | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("anchor manifest: cannot open %s: %s", manifest_path, strerror(errno));
        free(manifest_path);
        return -1;
    }
    
    /* Format: tensor_name, bulk_file_name, rows, cols, packed_bytes, bulk_crc32, anchor, anchor_file_name, anchor_count */
    n = snprintf(line, sizeof(line),
                 "%s\t%s\t%u\t%u\t%zu\t%08x\tanchor\t%s\t%u\n",
                 tensor_name,
                 bulk_file_name,
                 layer->rows,
                 layer->cols,
                 layer->packed_weight_bytes,
                 bulk_crc32,
                 anchor_file_name,
                 layer->anchor_count);
    
    if (n < 0 || (size_t)n >= sizeof(line)) {
        LOG_ERROR("anchor manifest: line construction failed for %s", tensor_name);
        goto cleanup;
    }
    
    wrote = write(fd, line, (size_t)n);
    if (wrote != (ssize_t)n) {
        LOG_ERROR("anchor manifest: write failed for %s: %s", manifest_path, strerror(errno));
        goto cleanup;
    }
    
    rc = 0;
    
cleanup:
    if (fd >= 0 && close(fd) != 0) {
        LOG_ERROR("anchor manifest: close failed for %s: %s", manifest_path, strerror(errno));
        rc = -1;
    }
    free(manifest_path);
    return rc;
}

int ternary_anchor_load_for_tensor(const char *anchor_path,
                                   const char *expected_tensor_name,
                                   ternary_anchor_view_t *out_view) {
    if (!anchor_path || !out_view) {
        LOG_ERROR("anchor load: NULL argument");
        return -1;
    }

    memset(out_view, 0, sizeof(*out_view));

    if (ternary_anchor_name_has_suffix(anchor_path, ".safetensors")) {
        return ternary_anchor_load_safetensors_local(anchor_path, expected_tensor_name, out_view);
    }
    return ternary_anchor_load_legacy_binary_local(anchor_path, out_view);
}

int ternary_anchor_load(const char *anchor_path,
                        ternary_anchor_view_t *out_view) {
    return ternary_anchor_load_for_tensor(anchor_path, NULL, out_view);
}

void ternary_anchor_view_release(ternary_anchor_view_t *view) {
    if (!view) {
        return;
    }
    
    if (view->owns_memory) {
        free((void *)view->entries);
        free((void *)view->row_offsets);
    }
    
    memset(view, 0, sizeof(*view));
}

/* =========================================================================
 * Validation and Debug
 * ========================================================================= */

int ternary_anchor_validate(const ternary_anchor_view_t *view) {
    if (!view) {
        LOG_ERROR("anchor validate: NULL view");
        return -1;
    }
    
    if (view->anchor_count == 0) {
        return 0; /* Empty is valid */
    }
    
    if (!view->entries) {
        LOG_ERROR("anchor validate: NULL entries with non-zero count");
        return -1;
    }
    
    /* Check bounds and sorting */
    uint16_t prev_row = 0;
    uint16_t prev_col = 0;
    
    for (uint32_t i = 0; i < view->anchor_count; ++i) {
        const ternary_anchor_entry_t *e = &view->entries[i];
        
        if (e->row >= view->rows) {
            LOG_ERROR("anchor validate: entry %u has row %u >= %u", i, e->row, view->rows);
            return -1;
        }
        if (e->col >= view->cols) {
            LOG_ERROR("anchor validate: entry %u has col %u >= %u", i, e->col, view->cols);
            return -1;
        }
        
        /* Check sorting */
        if (i > 0) {
            if (e->row < prev_row || (e->row == prev_row && e->col <= prev_col)) {
                LOG_ERROR("anchor validate: entries not sorted at index %u", i);
                return -1;
            }
        }
        
        prev_row = e->row;
        prev_col = e->col;
    }
    
    /* Validate row_offsets if present */
    if (view->row_offsets) {
        if (view->row_offsets[0] != 0) {
            LOG_ERROR("anchor validate: row_offsets[0] != 0");
            return -1;
        }
        if (view->row_offsets[view->rows] != view->anchor_count) {
            LOG_ERROR("anchor validate: row_offsets[rows] != anchor_count");
            return -1;
        }
        for (uint32_t r = 1; r <= view->rows; ++r) {
            if (view->row_offsets[r] < view->row_offsets[r - 1]) {
                LOG_ERROR("anchor validate: row_offsets not monotonic at row %u", r);
                return -1;
            }
        }
    }
    
    return 0;
}

float ternary_anchor_row_dot(const ternary_anchor_view_t *view,
                             uint32_t row,
                             const float *x) {
    float acc = 0.0f;
    uint32_t start, end;
    
    if (!view || !x || row >= view->rows || !view->row_offsets || !view->entries) {
        return 0.0f;
    }
    
    start = view->row_offsets[row];
    end = view->row_offsets[row + 1];
    
    for (uint32_t i = start; i < end; ++i) {
        const ternary_anchor_entry_t *e = &view->entries[i];
        float value = bf16_to_f32_scalar(e->value_bf16);
        acc += value * x[e->col];
    }
    
    return acc;
}

int ternary_anchor_is_protected(const ternary_anchor_view_t *view,
                                uint32_t row,
                                uint32_t col) {
    uint32_t start, end;
    
    if (!view || row >= view->rows || col >= view->cols || 
        !view->row_offsets || !view->entries) {
        return 0;
    }
    
    start = view->row_offsets[row];
    end = view->row_offsets[row + 1];
    
    /* Binary search within the row */
    while (start < end) {
        uint32_t mid = start + (end - start) / 2;
        uint16_t entry_col = view->entries[mid].col;
        
        if (entry_col == col) {
            return 1;
        } else if (entry_col < col) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }
    
    return 0;
}

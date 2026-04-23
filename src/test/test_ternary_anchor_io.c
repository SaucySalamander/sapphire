#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../include/ternary_anchor.h"

static void cleanup_anchor_dir(const char *dir_path)
{
    char path[512];

    if (!dir_path) {
        return;
    }

    snprintf(path, sizeof(path), "%s/%s", dir_path, "layer_weight.safetensors");
    unlink(path);
    snprintf(path, sizeof(path), "%s/%s", dir_path, "layer_weight.anchors.safetensors");
    unlink(path);
    snprintf(path, sizeof(path), "%s/%s", dir_path, "legacy.anchors.bin");
    unlink(path);
    snprintf(path, sizeof(path), "%s/%s", dir_path, "manifest.tsv");
    unlink(path);
    rmdir(dir_path);
}

static int read_manifest_line(const char *dir_path, char *line, size_t line_size)
{
    char manifest_path[512];
    FILE *file = NULL;

    snprintf(manifest_path, sizeof(manifest_path), "%s/%s", dir_path, "manifest.tsv");
    file = fopen(manifest_path, "r");
    if (!file) {
        fprintf(stderr, "ERROR: failed to open manifest %s: %s\n", manifest_path, strerror(errno));
        return -1;
    }
    if (!fgets(line, (int)line_size, file)) {
        fprintf(stderr, "ERROR: failed to read manifest line from %s\n", manifest_path);
        fclose(file);
        return -1;
    }
    fclose(file);
    return 0;
}

static void init_test_anchor_entries(ternary_anchor_entry_t entries[2])
{
    memset(entries, 0, 2u * sizeof(*entries));
    entries[0].row = 0u;
    entries[0].col = 1u;
    entries[0].value_bf16 = 0x3f80u;
    entries[1].row = 1u;
    entries[1].col = 3u;
    entries[1].value_bf16 = 0x4000u;
}

static void init_test_anchor_metadata(ternary_anchor_metadata_t *meta, uint32_t anchor_count)
{
    if (!meta) {
        return;
    }

    memset(meta, 0, sizeof(*meta));
    meta->magic = TERNARY_ANCHOR_MAGIC;
    meta->version = TERNARY_ANCHOR_VERSION;
    meta->rows = 2u;
    meta->cols = 4u;
    meta->anchor_count = anchor_count;
    meta->scale_group_size = 4u;
    meta->groups_per_row = 1u;
    meta->saliency_mode = TERNARY_ANCHOR_SALIENCY_HESSIAN_ONLY;
    meta->budget_ppm = 5000u;
    meta->saliency_cutoff = 0.5f;
    meta->anchor_value_rms = 1.5f;
    meta->bulk_gamma_mean = 0.25f;
    meta->max_row_nnz = 1u;
}

static int write_anchor_safetensors_file(const char *path,
                                         const char *tensor_name,
                                         const ternary_anchor_metadata_t *meta,
                                         uint32_t tensor_shape_anchor_count,
                                         size_t claimed_entry_bytes,
                                         const ternary_anchor_entry_t *entries,
                                         size_t written_entry_bytes)
{
    char header[2048];
    uint64_t header_len = 0u;
    FILE *file = NULL;
    int written = 0;

    if (!path || !tensor_name || !meta) {
        return -1;
    }

    written = snprintf(
        header,
        sizeof(header),
        "{\"__metadata__\":{\"sapphire_quant\":\"ternary-anchor-v2\"," \
        "\"format_version\":\"%u\",\"tensor_name\":\"%s\",\"rows\":\"%u\"," \
        "\"cols\":\"%u\",\"anchor_count\":\"%u\",\"scale_group_size\":\"%u\"," \
        "\"groups_per_row\":\"%u\",\"value_dtype\":\"BF16\",\"saliency_mode\":\"%u\"," \
        "\"budget_ppm\":\"%u\",\"entry_layout\":\"row,col,value_bf16,pad\"," \
        "\"entry_order\":\"row_col\",\"entry_crc32\":\"%08x\",\"saliency_cutoff\":\"%.9g\"," \
        "\"anchor_value_rms\":\"%.9g\",\"bulk_gamma_mean\":\"%.9g\",\"max_row_nnz\":\"%u\"},"
        "\"%s.anchor_entries\":{\"dtype\":\"U16\",\"shape\":[%u,4],\"data_offsets\":[0,%zu]}}",
        meta->version,
        tensor_name,
        meta->rows,
        meta->cols,
        meta->anchor_count,
        meta->scale_group_size,
        meta->groups_per_row,
        meta->saliency_mode,
        meta->budget_ppm,
        meta->crc32,
        (double)meta->saliency_cutoff,
        (double)meta->anchor_value_rms,
        (double)meta->bulk_gamma_mean,
        meta->max_row_nnz,
        tensor_name,
        tensor_shape_anchor_count,
        claimed_entry_bytes);
    if (written < 0 || (size_t)written >= sizeof(header)) {
        return -1;
    }

    header_len = (uint64_t)written;
    file = fopen(path, "wb");
    if (!file) {
        return -1;
    }
    if (fwrite(&header_len, sizeof(header_len), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (fwrite(header, 1u, (size_t)header_len, file) != (size_t)header_len) {
        fclose(file);
        return -1;
    }
    if (written_entry_bytes > 0u && entries) {
        if (fwrite(entries, 1u, written_entry_bytes, file) != written_entry_bytes) {
            fclose(file);
            return -1;
        }
    }
    if (fclose(file) != 0) {
        return -1;
    }

    return 0;
}

static int write_legacy_anchor_file_with_metadata(const char *path,
                                                  const ternary_anchor_metadata_t *meta,
                                                  const ternary_anchor_entry_t *entries,
                                                  uint32_t entry_count)
{
    FILE *file = NULL;

    if (!path || !meta) {
        return -1;
    }

    file = fopen(path, "wb");
    if (!file) {
        return -1;
    }
    if (fwrite(meta, sizeof(*meta), 1, file) != 1) {
        fclose(file);
        return -1;
    }
    if (entry_count > 0u && entries) {
        if (fwrite(entries, sizeof(*entries), entry_count, file) != entry_count) {
            fclose(file);
            return -1;
        }
    }
    if (fclose(file) != 0) {
        return -1;
    }
    return 0;
}

static int expect_anchor_load_failure(const char *anchor_path, const char *expected_tensor_name)
{
    ternary_anchor_view_t view;

    memset(&view, 0, sizeof(view));
    if (ternary_anchor_load_for_tensor(anchor_path, expected_tensor_name, &view) == 0) {
        fprintf(stderr, "ERROR: expected anchor load failure for %s\n", anchor_path);
        ternary_anchor_view_release(&view);
        return -1;
    }
    return 0;
}

static int test_anchor_write_and_load_safetensors(void)
{
    char dir_template[] = "/tmp/sapphire_anchor_ioXXXXXX";
    char *dir_path = NULL;
    char anchor_path[512];
    char manifest_line[1024];
    uint8_t packed_weights[2] = {0x01u, 0x02u};
    float scales[2] = {1.0f, 2.0f};
    ternary_anchor_entry_t entries[2];
    ternary_hybrid_layer_t layer;
    ternary_anchor_view_t view;
    uint32_t crc32 = 0u;

    dir_path = mkdtemp(dir_template);
    if (!dir_path) {
        fprintf(stderr, "ERROR: mkdtemp failed: %s\n", strerror(errno));
        return -1;
    }

    init_test_anchor_entries(entries);

    memset(&layer, 0, sizeof(layer));
    layer.packed_weights = packed_weights;
    layer.packed_weight_bytes = sizeof(packed_weights);
    layer.scales = scales;
    layer.scale_count = 2u;
    layer.scale_bytes = sizeof(scales);
    layer.scale_dtype = SAFETENSORS_F32;
    layer.rows = 2u;
    layer.cols = 4u;
    layer.scale_group_size = 4u;
    layer.groups_per_row = 1u;
    layer.integrity = TERNARY_IO_INTEGRITY_CRC32;
    layer.anchor_entries = entries;
    layer.anchor_count = 2u;
    layer.anchor_metadata.saliency_mode = TERNARY_ANCHOR_SALIENCY_HESSIAN_ONLY;
    layer.anchor_metadata.budget_ppm = 5000u;
    layer.anchor_metadata.saliency_cutoff = 0.5f;
    layer.anchor_metadata.anchor_value_rms = 1.5f;
    layer.anchor_metadata.bulk_gamma_mean = 0.25f;
    layer.anchor_metadata.max_row_nnz = 1u;

    if (ternary_anchor_write_layer(dir_path, "layer.weight", &layer, &crc32) != 0) {
        fprintf(stderr, "ERROR: ternary_anchor_write_layer failed\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }
    if (crc32 == 0u) {
        fprintf(stderr, "ERROR: expected non-zero hybrid crc32\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    if (read_manifest_line(dir_path, manifest_line, sizeof(manifest_line)) != 0) {
        cleanup_anchor_dir(dir_path);
        return -1;
    }
    if (!strstr(manifest_line, ".anchors.safetensors") || !strstr(manifest_line, "\tanchor\t")) {
        fprintf(stderr, "ERROR: manifest does not reference safetensors anchor sidecar: %s\n", manifest_line);
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    snprintf(anchor_path, sizeof(anchor_path), "%s/%s", dir_path, "layer_weight.anchors.safetensors");
    memset(&view, 0, sizeof(view));
    if (ternary_anchor_load_for_tensor(anchor_path, "layer.weight", &view) != 0) {
        fprintf(stderr, "ERROR: ternary_anchor_load failed for safetensors sidecar\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    if (view.rows != 2u || view.cols != 4u || view.anchor_count != 2u) {
        fprintf(stderr, "ERROR: loaded safetensors anchor view shape/count mismatch\n");
        ternary_anchor_view_release(&view);
        cleanup_anchor_dir(dir_path);
        return -1;
    }
    if (view.metadata.version != TERNARY_ANCHOR_VERSION || view.metadata.budget_ppm != 5000u) {
        fprintf(stderr, "ERROR: loaded safetensors anchor metadata mismatch\n");
        ternary_anchor_view_release(&view);
        cleanup_anchor_dir(dir_path);
        return -1;
    }
    if (view.entries[0].row != 0u || view.entries[0].col != 1u ||
        view.entries[1].row != 1u || view.entries[1].col != 3u) {
        fprintf(stderr, "ERROR: loaded safetensors anchor entries mismatch\n");
        ternary_anchor_view_release(&view);
        cleanup_anchor_dir(dir_path);
        return -1;
    }
    if (!view.row_offsets || view.row_offsets[0] != 0u || view.row_offsets[1] != 1u || view.row_offsets[2] != 2u) {
        fprintf(stderr, "ERROR: loaded safetensors row offsets mismatch\n");
        ternary_anchor_view_release(&view);
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    ternary_anchor_view_release(&view);
    memset(&view, 0, sizeof(view));
    if (ternary_anchor_load_for_tensor(anchor_path, "other.weight", &view) == 0) {
        fprintf(stderr, "ERROR: safetensors sidecar unexpectedly loaded for mismatched tensor name\n");
        ternary_anchor_view_release(&view);
        cleanup_anchor_dir(dir_path);
        return -1;
    }
    cleanup_anchor_dir(dir_path);
    return 0;
}

static int write_legacy_anchor_file(const char *path, const ternary_anchor_entry_t *entries, uint32_t anchor_count)
{
    ternary_anchor_metadata_t meta;

    init_test_anchor_metadata(&meta, anchor_count);
    meta.version = TERNARY_ANCHOR_LEGACY_VERSION;
    meta.crc32 = io_crc32_update(0u, entries, anchor_count * sizeof(*entries));

    return write_legacy_anchor_file_with_metadata(path, &meta, entries, anchor_count);
}

static int test_legacy_anchor_binary_compat(void)
{
    char dir_template[] = "/tmp/sapphire_anchor_legacyXXXXXX";
    char *dir_path = NULL;
    char legacy_path[512];
    ternary_anchor_entry_t entries[2];
    ternary_anchor_view_t view;

    dir_path = mkdtemp(dir_template);
    if (!dir_path) {
        fprintf(stderr, "ERROR: mkdtemp failed: %s\n", strerror(errno));
        return -1;
    }

    memset(entries, 0, sizeof(entries));
    entries[0].row = 0u;
    entries[0].col = 0u;
    entries[0].value_bf16 = 0x3f00u;
    entries[1].row = 1u;
    entries[1].col = 2u;
    entries[1].value_bf16 = 0x4080u;

    snprintf(legacy_path, sizeof(legacy_path), "%s/%s", dir_path, "legacy.anchors.bin");
    if (write_legacy_anchor_file(legacy_path, entries, 2u) != 0) {
        fprintf(stderr, "ERROR: failed to create legacy anchor payload\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    memset(&view, 0, sizeof(view));
    if (ternary_anchor_load(legacy_path, &view) != 0) {
        fprintf(stderr, "ERROR: ternary_anchor_load failed for legacy payload\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }
    if (view.metadata.version != TERNARY_ANCHOR_LEGACY_VERSION ||
        view.anchor_count != 2u || view.entries[1].col != 2u) {
        fprintf(stderr, "ERROR: loaded legacy anchor payload mismatch\n");
        ternary_anchor_view_release(&view);
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    ternary_anchor_view_release(&view);
    cleanup_anchor_dir(dir_path);
    return 0;
}

static int test_safetensors_malformed_artifacts(void)
{
    char dir_template[] = "/tmp/sapphire_anchor_malformed_stXXXXXX";
    char *dir_path = NULL;
    char anchor_path[512];
    ternary_anchor_entry_t entries[2];
    ternary_anchor_entry_t unsorted_entries[2];
    ternary_anchor_metadata_t meta;

    dir_path = mkdtemp(dir_template);
    if (!dir_path) {
        fprintf(stderr, "ERROR: mkdtemp failed: %s\n", strerror(errno));
        return -1;
    }

    init_test_anchor_entries(entries);
    snprintf(anchor_path, sizeof(anchor_path), "%s/%s", dir_path, "layer_weight.anchors.safetensors");

    init_test_anchor_metadata(&meta, 2u);
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_anchor_safetensors_file(anchor_path,
                                      "layer.weight",
                                      &meta,
                                      2u,
                                      sizeof(entries),
                                      entries,
                                      sizeof(entries) - 1u) != 0 ||
        expect_anchor_load_failure(anchor_path, "layer.weight") != 0) {
        fprintf(stderr, "ERROR: truncated safetensors sidecar did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 2u);
    meta.version = TERNARY_ANCHOR_VERSION + 1u;
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_anchor_safetensors_file(anchor_path,
                                      "layer.weight",
                                      &meta,
                                      2u,
                                      sizeof(entries),
                                      entries,
                                      sizeof(entries)) != 0 ||
        expect_anchor_load_failure(anchor_path, "layer.weight") != 0) {
        fprintf(stderr, "ERROR: bad safetensors version did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 2u);
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries)) ^ 0x1u;
    if (write_anchor_safetensors_file(anchor_path,
                                      "layer.weight",
                                      &meta,
                                      2u,
                                      sizeof(entries),
                                      entries,
                                      sizeof(entries)) != 0 ||
        expect_anchor_load_failure(anchor_path, "layer.weight") != 0) {
        fprintf(stderr, "ERROR: bad safetensors checksum did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 2u);
    meta.rows = 1u;
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_anchor_safetensors_file(anchor_path,
                                      "layer.weight",
                                      &meta,
                                      2u,
                                      sizeof(entries),
                                      entries,
                                      sizeof(entries)) != 0 ||
        expect_anchor_load_failure(anchor_path, "layer.weight") != 0) {
        fprintf(stderr, "ERROR: safetensors row mismatch did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 2u);
    meta.cols = 3u;
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_anchor_safetensors_file(anchor_path,
                                      "layer.weight",
                                      &meta,
                                      2u,
                                      sizeof(entries),
                                      entries,
                                      sizeof(entries)) != 0 ||
        expect_anchor_load_failure(anchor_path, "layer.weight") != 0) {
        fprintf(stderr, "ERROR: safetensors col mismatch did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 3u);
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_anchor_safetensors_file(anchor_path,
                                      "layer.weight",
                                      &meta,
                                      2u,
                                      sizeof(entries),
                                      entries,
                                      sizeof(entries)) != 0 ||
        expect_anchor_load_failure(anchor_path, "layer.weight") != 0) {
        fprintf(stderr, "ERROR: safetensors anchor_count mismatch did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    unsorted_entries[0] = entries[1];
    unsorted_entries[1] = entries[0];
    init_test_anchor_metadata(&meta, 2u);
    meta.crc32 = io_crc32_update(0u, unsorted_entries, sizeof(unsorted_entries));
    if (write_anchor_safetensors_file(anchor_path,
                                      "layer.weight",
                                      &meta,
                                      2u,
                                      sizeof(unsorted_entries),
                                      unsorted_entries,
                                      sizeof(unsorted_entries)) != 0 ||
        expect_anchor_load_failure(anchor_path, "layer.weight") != 0) {
        fprintf(stderr, "ERROR: unsorted safetensors entries did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    cleanup_anchor_dir(dir_path);
    return 0;
}

static int test_legacy_malformed_artifacts(void)
{
    char dir_template[] = "/tmp/sapphire_anchor_malformed_legacyXXXXXX";
    char *dir_path = NULL;
    char legacy_path[512];
    ternary_anchor_entry_t entries[2];
    ternary_anchor_entry_t unsorted_entries[2];
    ternary_anchor_metadata_t meta;

    dir_path = mkdtemp(dir_template);
    if (!dir_path) {
        fprintf(stderr, "ERROR: mkdtemp failed: %s\n", strerror(errno));
        return -1;
    }

    init_test_anchor_entries(entries);
    snprintf(legacy_path, sizeof(legacy_path), "%s/%s", dir_path, "legacy.anchors.bin");

    init_test_anchor_metadata(&meta, 2u);
    meta.version = TERNARY_ANCHOR_LEGACY_VERSION;
    meta.magic ^= 0x1u;
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_legacy_anchor_file_with_metadata(legacy_path, &meta, entries, 2u) != 0 ||
        expect_anchor_load_failure(legacy_path, NULL) != 0) {
        fprintf(stderr, "ERROR: bad legacy magic did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 2u);
    meta.version = TERNARY_ANCHOR_LEGACY_VERSION + 1u;
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_legacy_anchor_file_with_metadata(legacy_path, &meta, entries, 2u) != 0 ||
        expect_anchor_load_failure(legacy_path, NULL) != 0) {
        fprintf(stderr, "ERROR: bad legacy version did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 2u);
    meta.version = TERNARY_ANCHOR_LEGACY_VERSION;
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries)) ^ 0x1u;
    if (write_legacy_anchor_file_with_metadata(legacy_path, &meta, entries, 2u) != 0 ||
        expect_anchor_load_failure(legacy_path, NULL) != 0) {
        fprintf(stderr, "ERROR: bad legacy checksum did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 2u);
    meta.version = TERNARY_ANCHOR_LEGACY_VERSION;
    meta.rows = 1u;
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_legacy_anchor_file_with_metadata(legacy_path, &meta, entries, 2u) != 0 ||
        expect_anchor_load_failure(legacy_path, NULL) != 0) {
        fprintf(stderr, "ERROR: legacy row mismatch did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    init_test_anchor_metadata(&meta, 2u);
    meta.version = TERNARY_ANCHOR_LEGACY_VERSION;
    meta.cols = 3u;
    meta.crc32 = io_crc32_update(0u, entries, sizeof(entries));
    if (write_legacy_anchor_file_with_metadata(legacy_path, &meta, entries, 2u) != 0 ||
        expect_anchor_load_failure(legacy_path, NULL) != 0) {
        fprintf(stderr, "ERROR: legacy col mismatch did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    unsorted_entries[0] = entries[1];
    unsorted_entries[1] = entries[0];
    init_test_anchor_metadata(&meta, 2u);
    meta.version = TERNARY_ANCHOR_LEGACY_VERSION;
    meta.crc32 = io_crc32_update(0u, unsorted_entries, sizeof(unsorted_entries));
    if (write_legacy_anchor_file_with_metadata(legacy_path, &meta, unsorted_entries, 2u) != 0 ||
        expect_anchor_load_failure(legacy_path, NULL) != 0) {
        fprintf(stderr, "ERROR: unsorted legacy entries did not fail cleanly\n");
        cleanup_anchor_dir(dir_path);
        return -1;
    }

    cleanup_anchor_dir(dir_path);
    return 0;
}

static int test_anchor_validate_rejects_invalid_row_offsets(void)
{
    ternary_anchor_entry_t entries[2];
    uint32_t row_offsets[3] = {0u, 2u, 1u};
    ternary_anchor_view_t view;

    init_test_anchor_entries(entries);
    memset(&view, 0, sizeof(view));
    view.entries = entries;
    view.row_offsets = row_offsets;
    view.anchor_count = 2u;
    view.rows = 2u;
    view.cols = 4u;

    if (ternary_anchor_validate(&view) == 0) {
        fprintf(stderr, "ERROR: invalid row offsets unexpectedly validated\n");
        return -1;
    }
    return 0;
}

int main(void)
{
    if (test_anchor_write_and_load_safetensors() != 0) {
        fprintf(stderr, "FAIL: test_anchor_write_and_load_safetensors\n");
        return 1;
    }
    if (test_legacy_anchor_binary_compat() != 0) {
        fprintf(stderr, "FAIL: test_legacy_anchor_binary_compat\n");
        return 1;
    }
    if (test_safetensors_malformed_artifacts() != 0) {
        fprintf(stderr, "FAIL: test_safetensors_malformed_artifacts\n");
        return 1;
    }
    if (test_legacy_malformed_artifacts() != 0) {
        fprintf(stderr, "FAIL: test_legacy_malformed_artifacts\n");
        return 1;
    }
    if (test_anchor_validate_rejects_invalid_row_offsets() != 0) {
        fprintf(stderr, "FAIL: test_anchor_validate_rejects_invalid_row_offsets\n");
        return 1;
    }

    printf("PASS: test_ternary_anchor_io\n");
    return 0;
}
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../include/ternary_telemetry.h"

static char *read_file_text(const char *path)
{
    FILE *file = NULL;
    char *buffer = NULL;
    long size = 0;

    file = fopen(path, "rb");
    assert(file != NULL);
    assert(fseek(file, 0, SEEK_END) == 0);
    size = ftell(file);
    assert(size >= 0);
    assert(fseek(file, 0, SEEK_SET) == 0);

    buffer = (char *)malloc((size_t)size + 1u);
    assert(buffer != NULL);
    assert(fread(buffer, 1u, (size_t)size, file) == (size_t)size);
    buffer[size] = '\0';

    fclose(file);
    return buffer;
}

static void test_spatial_telemetry_writes_expected_records(void)
{
    char path[] = "/tmp/test_ternary_telemetryXXXXXX";
    ternary_telemetry_writer_t writer;
    uint32_t teacher_counts[3] = {1u, 2u, 3u};
    uint32_t student_counts[3] = {3u, 2u, 1u};
    uint32_t student_bulk_counts[3] = {0u, 4u, 0u};
    char *text = NULL;
    int fd = mkstemp(path);

    assert(fd >= 0);
    close(fd);

    assert(ternary_telemetry_writer_init(&writer, path) == 0);
    assert(telemetry_dump_spatial_snapshot_meta(&writer,
                                                &(ternary_spatial_telemetry_meta_t){
                                                    .tensor_name = "model.layers.0.self_attn.q_proj.weight",
                                                    .config_hash = 1234u,
                                                    .layer_idx = 0u,
                                                    .resume_step_idx = 0u,
                                                    .step_idx = 2u,
                                                    .tape_hash = 99u,
                                                    .student_checkpoint_hash = 77u,
                                                    .rows = 4u,
                                                    .cols = 8u,
                                                    .scale_group_size = 4u,
                                                    .groups_per_row = 2u,
                                                    .row_bucket_size = 2u,
                                                    .row_bucket_count = 2u,
                                                    .hessian_proxy_source = 2u,
                                                    .use_anchor_mode = 1u,
                                                    .anchor_count = 1u,
                                                    .histogram_bin_count = 3u,
                                                    .histogram_min = -0.5f,
                                                    .histogram_max = 0.5f,
                                                    .effective_learning_rate = 0.03f,
                                                    .effective_hessian_scale = 0.25f,
                                                    .hessian_proxy_cap = 6.0f
                                                }) == 0);
    assert(telemetry_dump_spatial_snapshot_block(&writer,
                                                 &(ternary_spatial_telemetry_block_t){
                                                     .config_hash = 1234u,
                                                     .layer_idx = 0u,
                                                     .step_idx = 2u,
                                                     .row_bucket_idx = 0u,
                                                     .group_idx = 1u,
                                                     .row_start = 0u,
                                                     .row_end = 2u,
                                                     .col_start = 4u,
                                                     .col_end = 8u,
                                                     .gamma_mean = 0.125f,
                                                     .gamma_min = 0.1f,
                                                     .gamma_max = 0.15f,
                                                     .hessian_group_mean = 1.75f,
                                                     .hessian_group_max = 3.5f,
                                                     .block_weight_mse = 0.02f,
                                                     .block_hessian_error = 0.08f,
                                                     .p_zero_fraction = 0.75f,
                                                     .anchor_fraction = 0.125f
                                                 }) == 0);
    assert(telemetry_dump_spatial_snapshot_histogram(&writer,
                                                     &(ternary_spatial_telemetry_histogram_t){
                                                         .tensor_name = "model.layers.0.self_attn.q_proj.weight",
                                                         .config_hash = 1234u,
                                                         .layer_idx = 0u,
                                                         .step_idx = 2u,
                                                         .histogram_bin_count = 3u,
                                                         .histogram_min = -0.5f,
                                                         .histogram_max = 0.5f,
                                                         .teacher_counts = teacher_counts,
                                                         .student_counts = student_counts,
                                                         .student_bulk_counts = student_bulk_counts
                                                     }) == 0);
    ternary_telemetry_writer_close(&writer);

    text = read_file_text(path);
    assert(strstr(text, "\"record_type\":\"spatial_snapshot_meta\"") != NULL);
    assert(strstr(text, "\"tensor_name\":\"model.layers.0.self_attn.q_proj.weight\"") != NULL);
    assert(strstr(text, "\"row_bucket_size\":2") != NULL);
    assert(strstr(text, "\"record_type\":\"spatial_snapshot_block\"") != NULL);
    assert(strstr(text, "\"p_zero_fraction\":0.75") != NULL);
    assert(strstr(text, "\"record_type\":\"spatial_snapshot_histogram\"") != NULL);
    assert(strstr(text, "\"teacher_counts\":[1,2,3]") != NULL);
    assert(strstr(text, "\"student_bulk_counts\":[0,4,0]") != NULL);

    free(text);
    unlink(path);
}

int main(void)
{
    test_spatial_telemetry_writes_expected_records();
    printf("PASS: test_ternary_telemetry\n");
    return 0;
}
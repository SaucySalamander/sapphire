#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ternary_checkpoint.h"

static void cleanup_checkpoint_dir(const char *dir_path)
{
    char path[1024];

    if (!dir_path) {
        return;
    }

    snprintf(path, sizeof(path), "%s/%s", dir_path, "checkpoint-a.tsv");
    unlink(path);
    snprintf(path, sizeof(path), "%s/%s", dir_path, "checkpoint-b.tsv");
    unlink(path);
    rmdir(dir_path);
}

static void init_checkpoint(ternary_student_update_checkpoint_t *checkpoint)
{
    memset(checkpoint, 0, sizeof(*checkpoint));
    checkpoint->schema_version = TERNARY_STUDENT_CHECKPOINT_VERSION;
    checkpoint->config_hash = 0x1234abcdU;
    checkpoint->total_layer_count = 18u;
    checkpoint->next_layer_index = 7u;
    checkpoint->last_completed_layer = 6u;
    checkpoint->converted_tensor_count = 91u;
    checkpoint->checkpoint_every_n_layers = 1u;
    checkpoint->validate_every_n = 2u;
    checkpoint->alignment_manifest_crc32 = 0x0badc0deU;
    checkpoint->alignment_tape_provenance_hash = 0x10203040U;
    checkpoint->hessian_sidecar_crc32 = 0x55667788U;
    checkpoint->use_anchor_mode = 1u;
    checkpoint->anchor_budget_ppm = 1000u;
    checkpoint->anchor_saliency_mode = 2u;

    snprintf(checkpoint->model_name, sizeof(checkpoint->model_name), "%s", "gemma-3-270m-it");
    snprintf(checkpoint->teacher_model_name, sizeof(checkpoint->teacher_model_name), "%s", "gemma-3-27b-it");
    snprintf(checkpoint->output_dir, sizeof(checkpoint->output_dir), "%s", "/tmp/out-hybrid");
    snprintf(checkpoint->activation_tape_path, sizeof(checkpoint->activation_tape_path), "%s", "./data/aligned.tape");
    snprintf(checkpoint->hessian_sidecar_path, sizeof(checkpoint->hessian_sidecar_path), "%s", "./data/teacher.hsc");
    snprintf(checkpoint->alignment_manifest_path, sizeof(checkpoint->alignment_manifest_path), "%s", "./alignment.tsv");
    snprintf(checkpoint->alignment_tape_path, sizeof(checkpoint->alignment_tape_path), "%s", "./alignment.tape");
    snprintf(checkpoint->calibration_corpus_path,
             sizeof(checkpoint->calibration_corpus_path),
             "%s",
             "./corpora/calibration.txt");
    snprintf(checkpoint->calibration_corpus_manifest_path,
             sizeof(checkpoint->calibration_corpus_manifest_path),
             "%s",
             "./configs/corpus/high_signal_calib_manifest.csv");
    snprintf(checkpoint->validation_corpus_path,
             sizeof(checkpoint->validation_corpus_path),
             "%s",
             "./corpora/validation.txt");
    snprintf(checkpoint->validation_corpus_manifest_path,
             sizeof(checkpoint->validation_corpus_manifest_path),
             "%s",
             "./configs/corpus/high_signal_validation_manifest.csv");
}

static void test_checkpoint_roundtrip_anchor_policy(void)
{
    char dir_template[] = "/tmp/sapphire_checkpoint_roundtripXXXXXX";
    char *dir_path = NULL;
    char checkpoint_path[1024];
    ternary_student_update_checkpoint_t checkpoint;
    ternary_student_update_checkpoint_t loaded;
    uint32_t file_crc32 = 0u;

    printf("TEST: ternary checkpoint roundtrip preserves hybrid resume fields\n");

    dir_path = mkdtemp(dir_template);
    assert(dir_path != NULL);
    snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/%s", dir_path, "checkpoint-a.tsv");

    init_checkpoint(&checkpoint);
    assert(ternary_student_checkpoint_write_atomic(checkpoint_path, &checkpoint) == 0);
    assert(ternary_student_checkpoint_compute_file_crc32(checkpoint_path, &file_crc32) == 0);
    assert(file_crc32 != 0u);

    memset(&loaded, 0, sizeof(loaded));
    assert(ternary_student_checkpoint_load(checkpoint_path, &loaded) == 0);
    assert(loaded.schema_version == checkpoint.schema_version);
    assert(loaded.config_hash == checkpoint.config_hash);
    assert(loaded.total_layer_count == checkpoint.total_layer_count);
    assert(loaded.next_layer_index == checkpoint.next_layer_index);
    assert(loaded.converted_tensor_count == checkpoint.converted_tensor_count);
    assert(loaded.use_anchor_mode == checkpoint.use_anchor_mode);
    assert(loaded.anchor_budget_ppm == checkpoint.anchor_budget_ppm);
    assert(loaded.anchor_saliency_mode == checkpoint.anchor_saliency_mode);
    assert(strcmp(loaded.model_name, checkpoint.model_name) == 0);
    assert(strcmp(loaded.output_dir, checkpoint.output_dir) == 0);
    assert(strcmp(loaded.activation_tape_path, checkpoint.activation_tape_path) == 0);
    printf("  ✓ Hybrid checkpoint fields survive atomic write/load\n");

    cleanup_checkpoint_dir(dir_path);
}

static void test_checkpoint_crc_changes_with_anchor_policy(void)
{
    char dir_template[] = "/tmp/sapphire_checkpoint_policyXXXXXX";
    char *dir_path = NULL;
    char checkpoint_a_path[1024];
    char checkpoint_b_path[1024];
    ternary_student_update_checkpoint_t checkpoint_a;
    ternary_student_update_checkpoint_t checkpoint_b;
    uint32_t crc32_a = 0u;
    uint32_t crc32_b = 0u;

    printf("TEST: ternary checkpoint file identity changes with anchor policy\n");

    dir_path = mkdtemp(dir_template);
    assert(dir_path != NULL);
    snprintf(checkpoint_a_path, sizeof(checkpoint_a_path), "%s/%s", dir_path, "checkpoint-a.tsv");
    snprintf(checkpoint_b_path, sizeof(checkpoint_b_path), "%s/%s", dir_path, "checkpoint-b.tsv");

    init_checkpoint(&checkpoint_a);
    checkpoint_b = checkpoint_a;
    checkpoint_b.anchor_budget_ppm = 2000u;
    checkpoint_b.anchor_saliency_mode = 1u;

    assert(ternary_student_checkpoint_write_atomic(checkpoint_a_path, &checkpoint_a) == 0);
    assert(ternary_student_checkpoint_write_atomic(checkpoint_b_path, &checkpoint_b) == 0);
    assert(ternary_student_checkpoint_compute_file_crc32(checkpoint_a_path, &crc32_a) == 0);
    assert(ternary_student_checkpoint_compute_file_crc32(checkpoint_b_path, &crc32_b) == 0);
    assert(crc32_a != 0u);
    assert(crc32_b != 0u);
    assert(crc32_a != crc32_b);
    printf("  ✓ Hybrid anchor policy changes checkpoint file identity\n");

    cleanup_checkpoint_dir(dir_path);
}

int main(void)
{
    test_checkpoint_roundtrip_anchor_policy();
    test_checkpoint_crc_changes_with_anchor_policy();
    printf("PASS: test_ternary_checkpoint\n");
    return 0;
}
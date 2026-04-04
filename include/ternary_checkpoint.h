/*
 * @file ternary_checkpoint.h
 * @brief Atomic checkpoint I/O for the student update loop.
 */

#ifndef TERNARY_CHECKPOINT_H
#define TERNARY_CHECKPOINT_H

#include <stdint.h>

#include "activation_tape.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TERNARY_STUDENT_CHECKPOINT_VERSION 1u
#define TERNARY_STUDENT_CHECKPOINT_MODEL_MAX 128u
#define TERNARY_STUDENT_CHECKPOINT_PATH_MAX 1024u

typedef struct {
    uint32_t schema_version;
    uint32_t config_hash;
    uint32_t total_layer_count;
    uint32_t next_layer_index;
    uint32_t last_completed_layer;
    uint32_t converted_tensor_count;
    uint32_t checkpoint_every_n_layers;
    uint32_t validate_every_n;
    uint32_t alignment_manifest_crc32;
    uint32_t alignment_tape_provenance_hash;
    char model_name[TERNARY_STUDENT_CHECKPOINT_MODEL_MAX];
    char teacher_model_name[TERNARY_STUDENT_CHECKPOINT_MODEL_MAX];
    char output_dir[TERNARY_STUDENT_CHECKPOINT_PATH_MAX];
    char activation_tape_path[TERNARY_STUDENT_CHECKPOINT_PATH_MAX];
    char alignment_manifest_path[TERNARY_STUDENT_CHECKPOINT_PATH_MAX];
    char alignment_tape_path[TERNARY_STUDENT_CHECKPOINT_PATH_MAX];
    char calibration_corpus_path[TERNARY_STUDENT_CHECKPOINT_PATH_MAX];
    char calibration_corpus_manifest_path[TERNARY_STUDENT_CHECKPOINT_PATH_MAX];
    char validation_corpus_path[TERNARY_STUDENT_CHECKPOINT_PATH_MAX];
    char validation_corpus_manifest_path[TERNARY_STUDENT_CHECKPOINT_PATH_MAX];
} ternary_student_update_checkpoint_t;

int ternary_student_checkpoint_write_atomic(const char *checkpoint_path,
                                            const ternary_student_update_checkpoint_t *checkpoint);

/* Returns 1 when the checkpoint file does not exist, 0 on success, -1 on error. */
int ternary_student_checkpoint_load(const char *checkpoint_path,
                                    ternary_student_update_checkpoint_t *out_checkpoint);

int ternary_student_checkpoint_compute_manifest_crc32(const char *manifest_path,
                                                      uint32_t *out_crc32);

int ternary_student_checkpoint_compute_tape_provenance_hash(const ternary_student_update_checkpoint_t *checkpoint,
                                                            const activation_tape_t *alignment_tape,
                                                            uint32_t *out_hash);

int ternary_student_checkpoint_validate_alignment(const ternary_student_update_checkpoint_t *checkpoint,
                                                  const activation_tape_t *alignment_tape);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_CHECKPOINT_H */
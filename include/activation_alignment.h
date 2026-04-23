/*
 * @file activation_alignment.h
 * @brief Deterministic teacher-to-student activation alignment helpers.
 */

#ifndef ACTIVATION_ALIGNMENT_H
#define ACTIVATION_ALIGNMENT_H

#include <stddef.h>
#include <stdint.h>

#include "activation_tape.h"
#include "model_spec.h"
#include "transformer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ACTIVATION_ALIGNMENT_DEPTH_BUCKET = 0,
    ACTIVATION_ALIGNMENT_DEPTH_REPEAT = 1
} activation_alignment_depth_strategy_t;

typedef enum {
    ACTIVATION_ALIGNMENT_WIDTH_BLOCK_REPLICATION = 0,
    ACTIVATION_ALIGNMENT_WIDTH_INTERPOLATION = 1,
    ACTIVATION_ALIGNMENT_WIDTH_AUTO = 2
} activation_alignment_width_strategy_t;

typedef struct {
    const model_spec_t *teacher_spec;
    const model_spec_t *student_spec;
    const activation_tape_t *teacher_tape;
    const char *structural_map_path;
    activation_alignment_depth_strategy_t depth_strategy;
    activation_alignment_width_strategy_t width_strategy;
} activation_alignment_request_t;

typedef struct {
    uint32_t teacher_layer_idx;
    uint32_t student_layer_idx;
    capture_target_t target;
    uint32_t teacher_entry_idx;
    uint32_t student_entry_idx;
    uint32_t source_dim;
    uint32_t target_dim;
    uint32_t sample_count;
    uint32_t alias_of_entry;
    uint32_t teacher_layer_type;
    uint32_t student_layer_type;
    char teacher_tensor_name[TAPE_TENSOR_NAME_MAX];
    char student_tensor_name[TAPE_TENSOR_NAME_MAX];
} activation_alignment_entry_t;

typedef struct activation_alignment_manifest {
    char teacher_model_id[128];
    char student_model_id[128];
    char teacher_prefix[128];
    char student_prefix[128];
    uint32_t teacher_layer_count;
    uint32_t student_layer_count;
    uint32_t sample_count;
    activation_alignment_depth_strategy_t depth_strategy;
    activation_alignment_width_strategy_t width_strategy;
    size_t entry_count;
    activation_alignment_entry_t *entries;
} activation_alignment_manifest_t;

int activation_alignment_build_manifest(activation_alignment_manifest_t *manifest,
                                        const activation_alignment_request_t *request);
void activation_alignment_manifest_free(activation_alignment_manifest_t *manifest);
int activation_alignment_write_manifest(const activation_alignment_manifest_t *manifest,
                                        const char *path);
int activation_alignment_write_aligned_tape(const activation_alignment_manifest_t *manifest,
                                            const activation_alignment_request_t *request,
                                            const char *path);
int activation_alignment_prepare_artifacts(const activation_alignment_request_t *request,
                                           const char *output_dir,
                                           char **out_tape_path,
                                           char **out_manifest_path);

#ifdef __cplusplus
}
#endif

#endif /* ACTIVATION_ALIGNMENT_H */
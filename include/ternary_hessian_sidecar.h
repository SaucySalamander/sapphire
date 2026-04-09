/*
 * @file ternary_hessian_sidecar.h
 * @brief Mmap-backed diagonal curvature sidecar for ternary calibration.
 */

#ifndef TERNARY_HESSIAN_SIDECAR_H
#define TERNARY_HESSIAN_SIDECAR_H

#include <stddef.h>
#include <stdint.h>

#include "ternary_hessian_proxy.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TERNARY_HESSIAN_SIDECAR_MAGIC            0x48534331u  /* "HSC1" */
#define TERNARY_HESSIAN_SIDECAR_VERSION          1u
#define TERNARY_HESSIAN_SIDECAR_TENSOR_NAME_MAX  256u
#define TERNARY_HESSIAN_SIDECAR_TEACHER_MODEL_MAX 128u
#define TERNARY_HESSIAN_SIDECAR_NO_ALIAS         UINT32_MAX

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t entry_count;
    uint32_t sample_count;
    uint32_t tape_crc32;
    uint32_t manifest_entry_size;
    char teacher_model_name[TERNARY_HESSIAN_SIDECAR_TEACHER_MODEL_MAX];
    uint64_t data_section_offset;
    uint64_t data_section_size;
    uint32_t reserved;
    uint32_t crc32;
} ternary_hessian_sidecar_header_t;

typedef struct {
    char tensor_name[TERNARY_HESSIAN_SIDECAR_TENSOR_NAME_MAX];
    uint32_t vector_dim;
    uint32_t sample_count;
    float mean;
    float max;
    uint64_t data_offset;
    uint64_t data_bytes;
    uint32_t alias_of_entry;
    uint32_t layer_type;
    uint32_t _pad[2];
} ternary_hessian_sidecar_entry_t;

typedef struct ternary_hessian_sidecar_t ternary_hessian_sidecar_t;

ternary_hessian_sidecar_t *ternary_hessian_sidecar_open(const char *sidecar_path);

const ternary_hessian_sidecar_header_t *ternary_hessian_sidecar_header(const ternary_hessian_sidecar_t *sidecar);

uint32_t ternary_hessian_sidecar_crc32(const ternary_hessian_sidecar_t *sidecar);

const char *ternary_hessian_sidecar_teacher_model_name(const ternary_hessian_sidecar_t *sidecar);

uint32_t ternary_hessian_sidecar_tape_crc32(const ternary_hessian_sidecar_t *sidecar);

int ternary_hessian_sidecar_entry_index(const ternary_hessian_sidecar_t *sidecar,
                                       const char *tensor_name);

const ternary_hessian_sidecar_entry_t *ternary_hessian_sidecar_entry(const ternary_hessian_sidecar_t *sidecar,
                                                                     const char *tensor_name);

uint32_t ternary_hessian_sidecar_vector_dim(const ternary_hessian_sidecar_t *sidecar,
                                           const char *tensor_name);

int ternary_hessian_sidecar_sample_count(const ternary_hessian_sidecar_t *sidecar);

const float *ternary_hessian_sidecar_diagonal(const ternary_hessian_sidecar_t *sidecar,
                                              const char *tensor_name);

int ternary_hessian_sidecar_stats(const ternary_hessian_sidecar_t *sidecar,
                                  const char *tensor_name,
                                  ternary_hessian_proxy_stats_t *out_stats);

void ternary_hessian_sidecar_close(ternary_hessian_sidecar_t *sidecar);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_HESSIAN_SIDECAR_H */
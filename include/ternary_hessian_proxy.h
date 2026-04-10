/**
 * @file ternary_hessian_proxy.h
 * @brief Diagonal curvature proxies for ternary STE calibration.
 */

#ifndef TERNARY_HESSIAN_PROXY_H
#define TERNARY_HESSIAN_PROXY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TERNARY_HESSIAN_PROXY_SOURCE_NONE = 0,
    TERNARY_HESSIAN_PROXY_SOURCE_ACTIVATION_DIAGONAL = 1,
    TERNARY_HESSIAN_PROXY_SOURCE_EXTERNAL_SIDECAR = 2
} ternary_hessian_proxy_source_t;

typedef struct {
    float mean;
    float max;
} ternary_hessian_proxy_stats_t;

typedef struct {
    int valid;
    int tape_entry_idx;
    int sample_count;
    uint32_t cols;
    float *diagonal;
    ternary_hessian_proxy_stats_t stats;
} ternary_hessian_proxy_cache_t;

typedef struct {
    const float *calibration_vectors;
    int sample_count;
    uint32_t cols;
    float floor;
    float strength;
} ternary_hessian_proxy_build_request_t;

int ternary_hessian_proxy_build_diagonal(const ternary_hessian_proxy_build_request_t *request,
                                         float *out_proxy,
                                         ternary_hessian_proxy_stats_t *out_stats);

int ternary_hessian_proxy_finalize_diagonal(float *inout_proxy,
                                            uint32_t cols,
                                            float floor,
                                            float strength,
                                            ternary_hessian_proxy_stats_t *out_stats);

void ternary_hessian_proxy_cache_release(ternary_hessian_proxy_cache_t *cache);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_HESSIAN_PROXY_H */

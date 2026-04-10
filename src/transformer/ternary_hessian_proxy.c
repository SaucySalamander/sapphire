/**
 * @file ternary_hessian_proxy.c
 * @brief Diagonal curvature proxies for ternary STE calibration.
 */

#include "ternary_hessian_proxy.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int ternary_hessian_proxy_finalize_diagonal(float *inout_proxy,
                                            uint32_t cols,
                                            float floor,
                                            float strength,
                                            ternary_hessian_proxy_stats_t *out_stats)
{
    double mean_second_moment = 0.0;
    float proxy_mean = 0.0f;
    float proxy_max = 0.0f;

    if (!inout_proxy || cols == 0u) {
        return -1;
    }

    if (floor < 0.0f) {
        floor = 0.0f;
    }
    if (strength < 0.0f) {
        strength = 0.0f;
    }

    for (uint32_t col = 0; col < cols; ++col) {
        mean_second_moment += (double)inout_proxy[col];
    }

    mean_second_moment /= (double)cols;
    if (mean_second_moment <= 1e-12) {
        for (uint32_t col = 0; col < cols; ++col) {
            inout_proxy[col] = 1.0f;
        }
        proxy_mean = 1.0f;
        proxy_max = 1.0f;
    } else {
        for (uint32_t col = 0; col < cols; ++col) {
            float normalized = (float)((double)inout_proxy[col] / mean_second_moment);
            float adjusted = 1.0f + strength * (normalized - 1.0f);

            if (adjusted < floor) {
                adjusted = floor;
            }
            inout_proxy[col] = adjusted;
            proxy_mean += adjusted;
            if (adjusted > proxy_max) {
                proxy_max = adjusted;
            }
        }
        proxy_mean /= (float)cols;
    }

    if (out_stats) {
        out_stats->mean = proxy_mean;
        out_stats->max = proxy_max;
    }

    return 0;
}

int ternary_hessian_proxy_build_diagonal(const ternary_hessian_proxy_build_request_t *request,
                                         float *out_proxy,
                                         ternary_hessian_proxy_stats_t *out_stats)
{
    const float *calibration_vectors = NULL;
    int sample_count = 0;
    uint32_t cols = 0u;
    float floor = 0.0f;
    float strength = 0.0f;

    if (!request || !out_proxy) {
        return -1;
    }

    calibration_vectors = request->calibration_vectors;
    sample_count = request->sample_count;
    cols = request->cols;
    floor = request->floor;
    strength = request->strength;

    if (!calibration_vectors || sample_count <= 0 || cols == 0u) {
        return -1;
    }

    if (floor < 0.0f) {
        floor = 0.0f;
    }
    if (strength < 0.0f) {
        strength = 0.0f;
    }

    memset(out_proxy, 0, (size_t)cols * sizeof(float));
    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        const float *vector = calibration_vectors + (size_t)sample_idx * cols;

        for (uint32_t col = 0; col < cols; ++col) {
            float value = vector[col];

            out_proxy[col] += value * value;
        }
    }

    for (uint32_t col = 0; col < cols; ++col) {
        out_proxy[col] /= (float)sample_count;
    }

    return ternary_hessian_proxy_finalize_diagonal(out_proxy,
                                                   cols,
                                                   floor,
                                                   strength,
                                                   out_stats);
}

void ternary_hessian_proxy_cache_release(ternary_hessian_proxy_cache_t *cache)
{
    if (!cache) {
        return;
    }

    free(cache->diagonal);
    memset(cache, 0, sizeof(*cache));
}

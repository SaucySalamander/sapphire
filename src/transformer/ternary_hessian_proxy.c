/**
 * @file ternary_hessian_proxy.c
 * @brief Diagonal curvature proxies for ternary STE calibration.
 */

#include "ternary_hessian_proxy.h"

#include "kernels.h"

#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define HESSIAN_PROXY_PARALLEL_CHUNK_COLS 256u

typedef struct {
    const float *calibration_vectors;
    float *out_proxy;
    int sample_count;
    uint32_t cols;
    uint32_t chunk_cols;
} hessian_proxy_parallel_task_t;

static double hessian_proxy_hsum_m256d(__m256d value)
{
    double lanes[4];

    _mm256_storeu_pd(lanes, value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

static float hessian_proxy_hsum_m256(__m256 value)
{
    float lanes[8];

    _mm256_storeu_ps(lanes, value);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3] +
           lanes[4] + lanes[5] + lanes[6] + lanes[7];
}

static float hessian_proxy_hmax_m256(__m256 value)
{
    float lanes[8];
    float max_value = 0.0f;

    _mm256_storeu_ps(lanes, value);
    max_value = lanes[0];
    for (int lane = 1; lane < 8; ++lane) {
        if (lanes[lane] > max_value) {
            max_value = lanes[lane];
        }
    }

    return max_value;
}

static void hessian_proxy_accumulate_columns(float *out_proxy,
                                             const float *calibration_vectors,
                                             int sample_count,
                                             uint32_t cols,
                                             uint32_t col_begin,
                                             uint32_t col_end)
{
    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        const float *vector = calibration_vectors + (size_t)sample_idx * cols + col_begin;
        float *proxy = out_proxy + col_begin;
        uint32_t offset = 0u;
        uint32_t width = col_end - col_begin;

        for (; offset + 8u <= width; offset += 8u) {
            __m256 values = _mm256_loadu_ps(vector + offset);
            __m256 accum = _mm256_loadu_ps(proxy + offset);

            accum = _mm256_fmadd_ps(values, values, accum);
            _mm256_storeu_ps(proxy + offset, accum);
        }

        for (; offset < width; ++offset) {
            float value = vector[offset];

            proxy[offset] += value * value;
        }
    }
}

static void hessian_proxy_accumulate_parallel(void *arg, int idx)
{
    hessian_proxy_parallel_task_t *task = (hessian_proxy_parallel_task_t *)arg;
    uint32_t col_begin = 0u;
    uint32_t col_end = 0u;

    if (!task || idx < 0) {
        return;
    }

    col_begin = (uint32_t)idx * task->chunk_cols;
    if (col_begin >= task->cols) {
        return;
    }
    col_end = col_begin + task->chunk_cols;
    if (col_end > task->cols) {
        col_end = task->cols;
    }

    hessian_proxy_accumulate_columns(task->out_proxy,
                                     task->calibration_vectors,
                                     task->sample_count,
                                     task->cols,
                                     col_begin,
                                     col_end);
}

static void hessian_proxy_scale_by_sample_count(float *out_proxy,
                                                uint32_t cols,
                                                int sample_count)
{
    const __m256 inv_sample_count = _mm256_set1_ps(1.0f / (float)sample_count);
    uint32_t col = 0u;

    for (; col + 8u <= cols; col += 8u) {
        __m256 values = _mm256_loadu_ps(out_proxy + col);

        values = _mm256_mul_ps(values, inv_sample_count);
        _mm256_storeu_ps(out_proxy + col, values);
    }

    for (; col < cols; ++col) {
        out_proxy[col] /= (float)sample_count;
    }
}

int ternary_hessian_proxy_finalize_diagonal(float *inout_proxy,
                                            uint32_t cols,
                                            float floor,
                                            float strength,
                                            ternary_hessian_proxy_stats_t *out_stats)
{
    double mean_second_moment = 0.0;
    float proxy_mean = 0.0f;
    float proxy_max = 0.0f;
    uint32_t col = 0u;

    if (!inout_proxy || cols == 0u) {
        return -1;
    }

    if (floor < 0.0f) {
        floor = 0.0f;
    }
    if (strength < 0.0f) {
        strength = 0.0f;
    }

    {
        __m256d sum_lo = _mm256_setzero_pd();
        __m256d sum_hi = _mm256_setzero_pd();

        for (; col + 8u <= cols; col += 8u) {
            __m256 values = _mm256_loadu_ps(inout_proxy + col);
            __m128 low = _mm256_castps256_ps128(values);
            __m128 high = _mm256_extractf128_ps(values, 1);

            sum_lo = _mm256_add_pd(sum_lo, _mm256_cvtps_pd(low));
            sum_hi = _mm256_add_pd(sum_hi, _mm256_cvtps_pd(high));
        }
        mean_second_moment = hessian_proxy_hsum_m256d(sum_lo) + hessian_proxy_hsum_m256d(sum_hi);
    }

    for (; col < cols; ++col) {
        mean_second_moment += (double)inout_proxy[col];
    }

    mean_second_moment /= (double)cols;
    if (mean_second_moment <= 1e-12) {
        const __m256 ones = _mm256_set1_ps(1.0f);

        for (col = 0u; col + 8u <= cols; col += 8u) {
            _mm256_storeu_ps(inout_proxy + col, ones);
        }
        for (; col < cols; ++col) {
            inout_proxy[col] = 1.0f;
        }
        proxy_mean = 1.0f;
        proxy_max = 1.0f;
    } else {
        const __m256 inv_mean = _mm256_set1_ps((float)(1.0 / mean_second_moment));
        const __m256 strength_vec = _mm256_set1_ps(strength);
        const __m256 one_vec = _mm256_set1_ps(1.0f);
        const __m256 floor_vec = _mm256_set1_ps(floor);
        __m256 proxy_mean_vec = _mm256_setzero_ps();
        __m256 proxy_max_vec = _mm256_setzero_ps();

        for (col = 0u; col + 8u <= cols; col += 8u) {
            __m256 values = _mm256_loadu_ps(inout_proxy + col);
            __m256 normalized = _mm256_mul_ps(values, inv_mean);
            __m256 adjusted = _mm256_fmadd_ps(strength_vec,
                                              _mm256_sub_ps(normalized, one_vec),
                                              one_vec);

            adjusted = _mm256_max_ps(adjusted, floor_vec);
            _mm256_storeu_ps(inout_proxy + col, adjusted);
            proxy_mean_vec = _mm256_add_ps(proxy_mean_vec, adjusted);
            proxy_max_vec = _mm256_max_ps(proxy_max_vec, adjusted);
        }

        proxy_mean = hessian_proxy_hsum_m256(proxy_mean_vec);
        proxy_max = hessian_proxy_hmax_m256(proxy_max_vec);
        for (; col < cols; ++col) {
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
    kernel_context_t *parallel_ctx = NULL;
    hessian_proxy_parallel_task_t task;
    int sample_count = 0;
    uint32_t cols = 0u;
    int chunk_count = 0;
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
    parallel_ctx = request->parallel_ctx;

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

    chunk_count = (int)((cols + (HESSIAN_PROXY_PARALLEL_CHUNK_COLS - 1u)) /
                        HESSIAN_PROXY_PARALLEL_CHUNK_COLS);
    if (parallel_ctx && sample_count > 1 && chunk_count > 1) {
        memset(&task, 0, sizeof(task));
        task.calibration_vectors = calibration_vectors;
        task.out_proxy = out_proxy;
        task.sample_count = sample_count;
        task.cols = cols;
        task.chunk_cols = HESSIAN_PROXY_PARALLEL_CHUNK_COLS;
        if (kernel_parallel_for(parallel_ctx,
                                hessian_proxy_accumulate_parallel,
                                &task,
                                chunk_count) != 0) {
            hessian_proxy_accumulate_columns(out_proxy,
                                             calibration_vectors,
                                             sample_count,
                                             cols,
                                             0u,
                                             cols);
        }
    } else {
        hessian_proxy_accumulate_columns(out_proxy,
                                         calibration_vectors,
                                         sample_count,
                                         cols,
                                         0u,
                                         cols);
    }

    hessian_proxy_scale_by_sample_count(out_proxy, cols, sample_count);

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

/*
 * @file ternary_telemetry.c
 * @brief JSONL telemetry writer for ternary experiments.
 */

#include "ternary_telemetry.h"

#include "log.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int telemetry_copy_path(char *dst, size_t dst_size, const char *src)
{
    size_t path_len = 0u;

    if (!dst || dst_size == 0u || !src) {
        return -1;
    }

    path_len = strlen(src);
    if (path_len == 0u || path_len >= dst_size) {
        return -1;
    }

    memcpy(dst, src, path_len + 1u);
    return 0;
}

static const char *telemetry_format_float(char *buffer, size_t buffer_size, float value)
{
    int written = 0;

    if (!buffer || buffer_size == 0u) {
        return "null";
    }
    if (!isfinite(value)) {
        return "null";
    }

    written = snprintf(buffer, buffer_size, "%.6f", (double)value);
    if (written < 0 || (size_t)written >= buffer_size) {
        return "null";
    }
    return buffer;
}

int ternary_telemetry_writer_init(ternary_telemetry_writer_t *writer,
                                  const char *telemetry_path)
{
    if (!writer || !telemetry_path) {
        return -1;
    }

    memset(writer, 0, sizeof(*writer));
    if (telemetry_copy_path(writer->path, sizeof(writer->path), telemetry_path) != 0) {
        LOG_ERROR("telemetry: invalid path: %s", telemetry_path);
        return -1;
    }

    writer->stream = fopen(writer->path, "a");
    if (!writer->stream) {
        LOG_ERROR("telemetry: failed to open %s", writer->path);
        return -1;
    }

    LOG_INFO("Telemetry file: %s", writer->path);
    return 0;
}

void ternary_telemetry_writer_close(ternary_telemetry_writer_t *writer)
{
    if (!writer) {
        return;
    }

    if (writer->stream) {
        fclose(writer->stream);
    }
    memset(writer, 0, sizeof(*writer));
}

int telemetry_dump_step(ternary_telemetry_writer_t *writer,
                        const ternary_telemetry_t *telemetry)
{
    char line[512];
    char mse_loss_buf[32];
    char grad_norm_buf[32];
    char p_neg1_buf[32];
    char p_zero_buf[32];
    char p_pos1_buf[32];
    char gamma_scale_buf[32];
    char io_ms_buf[32];
    char compute_ms_buf[32];
    const char *mse_loss_text = NULL;
    const char *grad_norm_text = NULL;
    const char *p_neg1_text = NULL;
    const char *p_zero_text = NULL;
    const char *p_pos1_text = NULL;
    const char *gamma_scale_text = NULL;
    const char *io_ms_text = NULL;
    const char *compute_ms_text = NULL;
    int written = 0;
    size_t bytes_written = 0u;

    if (!writer || !writer->stream || !telemetry) {
        return -1;
    }

    mse_loss_text = telemetry_format_float(mse_loss_buf, sizeof(mse_loss_buf), telemetry->mse_loss);
    grad_norm_text = telemetry_format_float(grad_norm_buf, sizeof(grad_norm_buf), telemetry->grad_norm);
    p_neg1_text = telemetry_format_float(p_neg1_buf, sizeof(p_neg1_buf), telemetry->p_neg1);
    p_zero_text = telemetry_format_float(p_zero_buf, sizeof(p_zero_buf), telemetry->p_zero);
    p_pos1_text = telemetry_format_float(p_pos1_buf, sizeof(p_pos1_buf), telemetry->p_pos1);
    gamma_scale_text = telemetry_format_float(gamma_scale_buf, sizeof(gamma_scale_buf), telemetry->gamma_scale);
    io_ms_text = telemetry_format_float(io_ms_buf, sizeof(io_ms_buf), telemetry->io_ms);
    compute_ms_text = telemetry_format_float(compute_ms_buf, sizeof(compute_ms_buf), telemetry->compute_ms);

    written = snprintf(line,
                       sizeof(line),
                       "{\"config_hash\":%u,\"layer_idx\":%u,\"step_idx\":%u,\"mse_loss\":%s,\"grad_norm\":%s,\"p_neg1\":%s,\"p_zero\":%s,\"p_pos1\":%s,\"gamma_scale\":%s,\"io_ms\":%s,\"compute_ms\":%s}\n",
                       telemetry->config_hash,
                       telemetry->layer_idx,
                       telemetry->step_idx,
                       mse_loss_text,
                       grad_norm_text,
                       p_neg1_text,
                       p_zero_text,
                       p_pos1_text,
                       gamma_scale_text,
                       io_ms_text,
                       compute_ms_text);
    if (written < 0 || (size_t)written >= sizeof(line)) {
        LOG_ERROR("telemetry: line too long for %s", writer->path);
        return -1;
    }

    bytes_written = fwrite(line, 1u, (size_t)written, writer->stream);
    if (bytes_written != (size_t)written) {
        LOG_ERROR("telemetry: write failed for %s", writer->path);
        return -1;
    }

    return 0;
}
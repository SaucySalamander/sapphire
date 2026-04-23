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

    written = snprintf(buffer, buffer_size, "%.9g", (double)value);
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

void ternary_telemetry_print_pass_stdout(const ternary_telemetry_t *telemetry)
{
    if (!telemetry) {
        return;
    }

    if (telemetry->use_anchor_mode) {
        printf("calibration_pass layer_idx=%u gamma_scale=%.9g bulk_gamma_mean=%.9g anchors=%u reconstruction_mse=%.9g raw_grad_norm=%.9g clipped_grad_norm=%.9g clip_scale=%.9g latent_saturation=%.9g hessian_active_max=%.9g\n",
               telemetry->layer_idx,
               (double)telemetry->gamma_scale,
               (double)telemetry->bulk_gamma_mean,
               telemetry->anchor_count,
               (double)telemetry->mse_loss,
               (double)telemetry->raw_grad_norm,
               (double)telemetry->clipped_grad_norm,
               (double)telemetry->clip_scale,
               (double)telemetry->latent_saturation,
               (double)telemetry->hessian_proxy_active_max);
    } else {
        printf("calibration_pass layer_idx=%u gamma_scale=%.9g reconstruction_mse=%.9g raw_grad_norm=%.9g clipped_grad_norm=%.9g clip_scale=%.9g latent_saturation=%.9g hessian_active_max=%.9g\n",
               telemetry->layer_idx,
               (double)telemetry->gamma_scale,
               (double)telemetry->mse_loss,
               (double)telemetry->raw_grad_norm,
               (double)telemetry->clipped_grad_norm,
               (double)telemetry->clip_scale,
               (double)telemetry->latent_saturation,
               (double)telemetry->hessian_proxy_active_max);
    }
    fflush(stdout);
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

static int telemetry_write_line(ternary_telemetry_writer_t *writer,
                                const char *line,
                                size_t line_len,
                                const char *kind)
{
    size_t bytes_written = 0u;

    if (!writer || !writer->stream || !line || !kind) {
        return -1;
    }

    bytes_written = fwrite(line, 1u, line_len, writer->stream);
    if (bytes_written != line_len) {
        LOG_ERROR("telemetry: %s write failed for %s", kind, writer->path);
        return -1;
    }
    if (fflush(writer->stream) != 0) {
        LOG_ERROR("telemetry: %s flush failed for %s", kind, writer->path);
        return -1;
    }

    return 0;
}

static int telemetry_format_step_line(const ternary_telemetry_t *telemetry,
                                      char *line,
                                      size_t line_size)
{
    char mse_loss_buf[32];
    char grad_norm_buf[32];
    char raw_grad_norm_buf[32];
    char clipped_grad_norm_buf[32];
    char clip_scale_buf[32];
    char latent_saturation_buf[32];
    char p_neg1_buf[32];
    char p_zero_buf[32];
    char p_pos1_buf[32];
    char gamma_scale_buf[32];
    char gamma_scale_min_buf[32];
    char gamma_scale_max_buf[32];
    char gamma_floor_fraction_buf[32];
    char hessian_proxy_mean_buf[32];
    char hessian_proxy_max_buf[32];
    char hessian_proxy_active_max_buf[32];
    char effective_learning_rate_buf[32];
    char effective_hessian_scale_buf[32];
    char io_ms_buf[32];
    char compute_ms_buf[32];
    char anchor_saliency_cutoff_buf[32];
    char anchor_value_rms_buf[32];
    char bulk_gamma_mean_buf[32];
    char anchor_contribution_norm_buf[32];
    char bulk_contribution_norm_buf[32];
    int written = 0;

    if (!telemetry || !line || line_size == 0u) {
        return -1;
    }

    written = snprintf(line,
                       line_size,
                       "{\"record_type\":\"calibration_step\",\"config_hash\":%u,\"resume_step_idx\":%u,\"layer_idx\":%u,\"step_idx\":%u,\"tape_hash\":%u,\"student_checkpoint_hash\":%u,\"mse_loss\":%s,\"grad_norm\":%s,\"raw_grad_norm\":%s,\"clipped_grad_norm\":%s,\"clip_scale\":%s,\"latent_saturation\":%s,\"p_neg1\":%s,\"p_zero\":%s,\"p_pos1\":%s,\"gamma_scale\":%s,\"gamma_scale_min\":%s,\"gamma_scale_max\":%s,\"gamma_floor_fraction\":%s,\"hessian_proxy_mean\":%s,\"hessian_proxy_max\":%s,\"hessian_proxy_active_max\":%s,\"hessian_proxy_source\":%u,\"effective_learning_rate\":%s,\"effective_hessian_scale\":%s,\"io_ms\":%s,\"compute_ms\":%s,\"use_anchor_mode\":%u,\"anchor_count\":%u,\"anchor_budget_ppm\":%u,\"anchor_saliency_mode\":%u,\"anchor_saliency_cutoff\":%s,\"anchor_value_rms\":%s,\"bulk_gamma_mean\":%s,\"anchor_contribution_norm\":%s,\"bulk_contribution_norm\":%s}\n",
                       telemetry->config_hash,
                       telemetry->resume_step_idx,
                       telemetry->layer_idx,
                       telemetry->step_idx,
                       telemetry->tape_hash,
                       telemetry->student_checkpoint_hash,
                       telemetry_format_float(mse_loss_buf, sizeof(mse_loss_buf), telemetry->mse_loss),
                       telemetry_format_float(grad_norm_buf, sizeof(grad_norm_buf), telemetry->grad_norm),
                       telemetry_format_float(raw_grad_norm_buf, sizeof(raw_grad_norm_buf), telemetry->raw_grad_norm),
                       telemetry_format_float(clipped_grad_norm_buf, sizeof(clipped_grad_norm_buf), telemetry->clipped_grad_norm),
                       telemetry_format_float(clip_scale_buf, sizeof(clip_scale_buf), telemetry->clip_scale),
                       telemetry_format_float(latent_saturation_buf, sizeof(latent_saturation_buf), telemetry->latent_saturation),
                       telemetry_format_float(p_neg1_buf, sizeof(p_neg1_buf), telemetry->p_neg1),
                       telemetry_format_float(p_zero_buf, sizeof(p_zero_buf), telemetry->p_zero),
                       telemetry_format_float(p_pos1_buf, sizeof(p_pos1_buf), telemetry->p_pos1),
                       telemetry_format_float(gamma_scale_buf, sizeof(gamma_scale_buf), telemetry->gamma_scale),
                       telemetry_format_float(gamma_scale_min_buf, sizeof(gamma_scale_min_buf), telemetry->gamma_scale_min),
                       telemetry_format_float(gamma_scale_max_buf, sizeof(gamma_scale_max_buf), telemetry->gamma_scale_max),
                       telemetry_format_float(gamma_floor_fraction_buf, sizeof(gamma_floor_fraction_buf), telemetry->gamma_floor_fraction),
                       telemetry_format_float(hessian_proxy_mean_buf, sizeof(hessian_proxy_mean_buf), telemetry->hessian_proxy_mean),
                       telemetry_format_float(hessian_proxy_max_buf, sizeof(hessian_proxy_max_buf), telemetry->hessian_proxy_max),
                       telemetry_format_float(hessian_proxy_active_max_buf,
                                              sizeof(hessian_proxy_active_max_buf),
                                              telemetry->hessian_proxy_active_max),
                       telemetry->hessian_proxy_source,
                       telemetry_format_float(effective_learning_rate_buf,
                                              sizeof(effective_learning_rate_buf),
                                              telemetry->effective_learning_rate),
                       telemetry_format_float(effective_hessian_scale_buf,
                                              sizeof(effective_hessian_scale_buf),
                                              telemetry->effective_hessian_scale),
                       telemetry_format_float(io_ms_buf, sizeof(io_ms_buf), telemetry->io_ms),
                       telemetry_format_float(compute_ms_buf, sizeof(compute_ms_buf), telemetry->compute_ms),
                       telemetry->use_anchor_mode,
                       telemetry->anchor_count,
                       telemetry->anchor_budget_ppm,
                       telemetry->anchor_saliency_mode,
                       telemetry_format_float(anchor_saliency_cutoff_buf,
                                              sizeof(anchor_saliency_cutoff_buf),
                                              telemetry->anchor_saliency_cutoff),
                       telemetry_format_float(anchor_value_rms_buf,
                                              sizeof(anchor_value_rms_buf),
                                              telemetry->anchor_value_rms),
                       telemetry_format_float(bulk_gamma_mean_buf,
                                              sizeof(bulk_gamma_mean_buf),
                                              telemetry->bulk_gamma_mean),
                       telemetry_format_float(anchor_contribution_norm_buf,
                                              sizeof(anchor_contribution_norm_buf),
                                              telemetry->anchor_contribution_norm),
                       telemetry_format_float(bulk_contribution_norm_buf,
                                              sizeof(bulk_contribution_norm_buf),
                                              telemetry->bulk_contribution_norm));
    if (written < 0 || (size_t)written >= line_size) {
        return -1;
    }

    return written;
}

int telemetry_dump_step(ternary_telemetry_writer_t *writer,
                        const ternary_telemetry_t *telemetry)
{
    char line[1536];
    int written = 0;

    if (!writer || !writer->stream || !telemetry) {
        return -1;
    }

    written = telemetry_format_step_line(telemetry, line, sizeof(line));
    if (written < 0) {
        LOG_ERROR("telemetry: line too long for %s", writer->path);
        return -1;
    }

    return telemetry_write_line(writer, line, (size_t)written, "step");
}

int telemetry_dump_validation_checkpoint(ternary_telemetry_writer_t *writer,
                                         const ternary_validation_telemetry_t *telemetry)
{
    char line[1024];
    char baseline_mean_nll_buf[32];
    char current_mean_nll_buf[32];
    char baseline_ppl_buf[32];
    char current_ppl_buf[32];
    char mean_kl_buf[32];
    char max_kl_buf[32];
    char top1_agreement_buf[32];
    const char *baseline_mean_nll_text = NULL;
    const char *current_mean_nll_text = NULL;
    const char *baseline_ppl_text = NULL;
    const char *current_ppl_text = NULL;
    const char *mean_kl_text = NULL;
    const char *max_kl_text = NULL;
    const char *top1_agreement_text = NULL;
    int written = 0;

    if (!writer || !writer->stream || !telemetry || !telemetry->tensor_name) {
        return -1;
    }

    baseline_mean_nll_text = telemetry_format_float(baseline_mean_nll_buf,
                                                    sizeof(baseline_mean_nll_buf),
                                                    telemetry->baseline_mean_nll);
    current_mean_nll_text = telemetry_format_float(current_mean_nll_buf,
                                                   sizeof(current_mean_nll_buf),
                                                   telemetry->current_mean_nll);
    baseline_ppl_text = telemetry_format_float(baseline_ppl_buf,
                                               sizeof(baseline_ppl_buf),
                                               expf(telemetry->baseline_mean_nll));
    current_ppl_text = telemetry_format_float(current_ppl_buf,
                                              sizeof(current_ppl_buf),
                                              expf(telemetry->current_mean_nll));
    mean_kl_text = telemetry_format_float(mean_kl_buf, sizeof(mean_kl_buf), telemetry->mean_kl);
    max_kl_text = telemetry_format_float(max_kl_buf, sizeof(max_kl_buf), telemetry->max_kl);
    top1_agreement_text = telemetry_format_float(top1_agreement_buf,
                                                 sizeof(top1_agreement_buf),
                                                 telemetry->top1_agreement);

    written = snprintf(line,
                       sizeof(line),
                       "{\"record_type\":\"validation_checkpoint\",\"converted_count\":%d,\"tensor_name\":\"%s\",\"crc32\":%u,\"baseline_mean_nll\":%s,\"current_mean_nll\":%s,\"baseline_ppl\":%s,\"current_ppl\":%s,\"mean_kl\":%s,\"max_kl\":%s,\"top1_agreement\":%s,\"sample_count\":%d}\n",
                       telemetry->converted_count,
                       telemetry->tensor_name,
                       telemetry->crc32,
                       baseline_mean_nll_text,
                       current_mean_nll_text,
                       baseline_ppl_text,
                       current_ppl_text,
                       mean_kl_text,
                       max_kl_text,
                       top1_agreement_text,
                       telemetry->sample_count);
    if (written < 0 || (size_t)written >= sizeof(line)) {
        LOG_ERROR("telemetry: validation line too long for %s", writer->path);
        return -1;
    }

    return telemetry_write_line(writer, line, (size_t)written, "validation");
}

int telemetry_dump_spatial_snapshot_meta(ternary_telemetry_writer_t *writer,
                                         const ternary_spatial_telemetry_meta_t *telemetry)
{
    char line[1536];
    char histogram_min_buf[32];
    char histogram_max_buf[32];
    char effective_learning_rate_buf[32];
    char effective_hessian_scale_buf[32];
    char hessian_proxy_cap_buf[32];
    int written = 0;

    if (!writer || !writer->stream || !telemetry || !telemetry->tensor_name) {
        return -1;
    }

    written = snprintf(line,
                       sizeof(line),
                       "{\"record_type\":\"spatial_snapshot_meta\",\"tensor_name\":\"%s\",\"config_hash\":%u,\"layer_idx\":%u,\"resume_step_idx\":%u,\"step_idx\":%u,\"tape_hash\":%u,\"student_checkpoint_hash\":%u,\"rows\":%u,\"cols\":%u,\"scale_group_size\":%u,\"groups_per_row\":%u,\"row_bucket_size\":%u,\"row_bucket_count\":%u,\"hessian_proxy_source\":%u,\"use_anchor_mode\":%u,\"anchor_count\":%u,\"histogram_bin_count\":%u,\"histogram_min\":%s,\"histogram_max\":%s,\"effective_learning_rate\":%s,\"effective_hessian_scale\":%s,\"hessian_proxy_cap\":%s}\n",
                       telemetry->tensor_name,
                       telemetry->config_hash,
                       telemetry->layer_idx,
                       telemetry->resume_step_idx,
                       telemetry->step_idx,
                       telemetry->tape_hash,
                       telemetry->student_checkpoint_hash,
                       telemetry->rows,
                       telemetry->cols,
                       telemetry->scale_group_size,
                       telemetry->groups_per_row,
                       telemetry->row_bucket_size,
                       telemetry->row_bucket_count,
                       telemetry->hessian_proxy_source,
                       telemetry->use_anchor_mode,
                       telemetry->anchor_count,
                       telemetry->histogram_bin_count,
                       telemetry_format_float(histogram_min_buf, sizeof(histogram_min_buf), telemetry->histogram_min),
                       telemetry_format_float(histogram_max_buf, sizeof(histogram_max_buf), telemetry->histogram_max),
                       telemetry_format_float(effective_learning_rate_buf,
                                              sizeof(effective_learning_rate_buf),
                                              telemetry->effective_learning_rate),
                       telemetry_format_float(effective_hessian_scale_buf,
                                              sizeof(effective_hessian_scale_buf),
                                              telemetry->effective_hessian_scale),
                       telemetry_format_float(hessian_proxy_cap_buf,
                                              sizeof(hessian_proxy_cap_buf),
                                              telemetry->hessian_proxy_cap));
    if (written < 0 || (size_t)written >= sizeof(line)) {
        LOG_ERROR("telemetry: spatial meta line too long for %s", writer->path);
        return -1;
    }

    return telemetry_write_line(writer, line, (size_t)written, "spatial-meta");
}

int telemetry_dump_spatial_snapshot_block(ternary_telemetry_writer_t *writer,
                                          const ternary_spatial_telemetry_block_t *telemetry)
{
    char line[1024];
    char gamma_mean_buf[32];
    char gamma_min_buf[32];
    char gamma_max_buf[32];
    char hessian_group_mean_buf[32];
    char hessian_group_max_buf[32];
    char block_weight_mse_buf[32];
    char block_hessian_error_buf[32];
    char p_zero_fraction_buf[32];
    char anchor_fraction_buf[32];
    int written = 0;

    if (!writer || !writer->stream || !telemetry) {
        return -1;
    }

    written = snprintf(line,
                       sizeof(line),
                       "{\"record_type\":\"spatial_snapshot_block\",\"config_hash\":%u,\"layer_idx\":%u,\"step_idx\":%u,\"row_bucket_idx\":%u,\"group_idx\":%u,\"row_start\":%u,\"row_end\":%u,\"col_start\":%u,\"col_end\":%u,\"gamma_mean\":%s,\"gamma_min\":%s,\"gamma_max\":%s,\"hessian_group_mean\":%s,\"hessian_group_max\":%s,\"block_weight_mse\":%s,\"block_hessian_error\":%s,\"p_zero_fraction\":%s,\"anchor_fraction\":%s}\n",
                       telemetry->config_hash,
                       telemetry->layer_idx,
                       telemetry->step_idx,
                       telemetry->row_bucket_idx,
                       telemetry->group_idx,
                       telemetry->row_start,
                       telemetry->row_end,
                       telemetry->col_start,
                       telemetry->col_end,
                       telemetry_format_float(gamma_mean_buf, sizeof(gamma_mean_buf), telemetry->gamma_mean),
                       telemetry_format_float(gamma_min_buf, sizeof(gamma_min_buf), telemetry->gamma_min),
                       telemetry_format_float(gamma_max_buf, sizeof(gamma_max_buf), telemetry->gamma_max),
                       telemetry_format_float(hessian_group_mean_buf,
                                              sizeof(hessian_group_mean_buf),
                                              telemetry->hessian_group_mean),
                       telemetry_format_float(hessian_group_max_buf,
                                              sizeof(hessian_group_max_buf),
                                              telemetry->hessian_group_max),
                       telemetry_format_float(block_weight_mse_buf,
                                              sizeof(block_weight_mse_buf),
                                              telemetry->block_weight_mse),
                       telemetry_format_float(block_hessian_error_buf,
                                              sizeof(block_hessian_error_buf),
                                              telemetry->block_hessian_error),
                       telemetry_format_float(p_zero_fraction_buf,
                                              sizeof(p_zero_fraction_buf),
                                              telemetry->p_zero_fraction),
                       telemetry_format_float(anchor_fraction_buf,
                                              sizeof(anchor_fraction_buf),
                                              telemetry->anchor_fraction));
    if (written < 0 || (size_t)written >= sizeof(line)) {
        LOG_ERROR("telemetry: spatial block line too long for %s", writer->path);
        return -1;
    }

    return telemetry_write_line(writer, line, (size_t)written, "spatial-block");
}

static int telemetry_write_json_u32_array(FILE *stream,
                                          const uint32_t *values,
                                          uint32_t count)
{
    if (!stream) {
        return -1;
    }
    if (fputc('[', stream) == EOF) {
        return -1;
    }
    for (uint32_t index = 0u; index < count; ++index) {
        if (index > 0u && fputc(',', stream) == EOF) {
            return -1;
        }
        if (fprintf(stream, "%u", values ? values[index] : 0u) < 0) {
            return -1;
        }
    }
    if (fputc(']', stream) == EOF) {
        return -1;
    }
    return 0;
}

int telemetry_dump_spatial_snapshot_histogram(ternary_telemetry_writer_t *writer,
                                              const ternary_spatial_telemetry_histogram_t *telemetry)
{
    FILE *stream = NULL;
    char histogram_min_buf[32];
    char histogram_max_buf[32];

    if (!writer || !writer->stream || !telemetry || !telemetry->tensor_name ||
        !telemetry->teacher_counts || !telemetry->student_counts) {
        return -1;
    }

    stream = writer->stream;
    if (fprintf(stream,
                "{\"record_type\":\"spatial_snapshot_histogram\",\"tensor_name\":\"%s\",\"config_hash\":%u,\"layer_idx\":%u,\"step_idx\":%u,\"histogram_bin_count\":%u,\"histogram_min\":%s,\"histogram_max\":%s,\"teacher_counts\":",
                telemetry->tensor_name,
                telemetry->config_hash,
                telemetry->layer_idx,
                telemetry->step_idx,
                telemetry->histogram_bin_count,
                telemetry_format_float(histogram_min_buf, sizeof(histogram_min_buf), telemetry->histogram_min),
                telemetry_format_float(histogram_max_buf, sizeof(histogram_max_buf), telemetry->histogram_max)) < 0) {
        LOG_ERROR("telemetry: spatial histogram write failed for %s", writer->path);
        return -1;
    }
    if (telemetry_write_json_u32_array(stream,
                                       telemetry->teacher_counts,
                                       telemetry->histogram_bin_count) != 0 ||
        fprintf(stream, ",\"student_counts\":") < 0 ||
        telemetry_write_json_u32_array(stream,
                                       telemetry->student_counts,
                                       telemetry->histogram_bin_count) != 0) {
        LOG_ERROR("telemetry: spatial histogram write failed for %s", writer->path);
        return -1;
    }
    if (fprintf(stream, ",\"student_bulk_counts\":") < 0) {
        LOG_ERROR("telemetry: spatial histogram write failed for %s", writer->path);
        return -1;
    }
    if (telemetry->student_bulk_counts) {
        if (telemetry_write_json_u32_array(stream,
                                           telemetry->student_bulk_counts,
                                           telemetry->histogram_bin_count) != 0) {
            LOG_ERROR("telemetry: spatial histogram write failed for %s", writer->path);
            return -1;
        }
    } else if (fprintf(stream, "null") < 0) {
        LOG_ERROR("telemetry: spatial histogram write failed for %s", writer->path);
        return -1;
    }
    if (fprintf(stream, "}\n") < 0 || fflush(stream) != 0) {
        LOG_ERROR("telemetry: spatial histogram flush failed for %s", writer->path);
        return -1;
    }

    return 0;
}
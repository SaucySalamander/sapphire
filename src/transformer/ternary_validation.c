/**
 * @file ternary_validation.c
 * @brief Full-model validation checkpoints for ternary conversion.
 */

#include "ternary_validation.h"

#include "gemma3_config.h"
#include "kernels.h"
#include "llm_model.h"
#include "log.h"
#include "tensor.h"
#include "ternary_io.h"
#include "tokenizer.h"
#include "tracy_profile.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ternary_validation_patch {
    char tensor_name[256];
    tensor_t **slot;
    tensor_t *proxy_tensor;
    safetensors_file_t *backing_file;
    safetensors_file_t *anchor_backing_file;
    uint32_t *anchor_row_offsets;
    uint32_t crc32;
};

static int validation_tracy_end_status(sapphire_tracy_zone_t *zone, int status)
{
    sapphire_tracy_zone_end(zone);
    return status;
}

static void validation_tracy_zone_text_if_present(const sapphire_tracy_zone_t *zone,
                                                  const char *text)
{
    if (!zone || !text || text[0] == '\0') {
        return;
    }

    sapphire_tracy_zone_text(zone, text, strlen(text));
}

static uint16_t validation_f32_to_bf16(float value)
{
    union {
        uint32_t u32;
        float f32;
    } bits;

    bits.f32 = value;
    return (uint16_t)(bits.u32 >> 16u);
}

static void *validation_memdup(const void *src, size_t size)
{
    void *copy = NULL;

    if (!src || size == 0u) {
        return NULL;
    }

    copy = malloc(size);
    if (!copy) {
        return NULL;
    }

    memcpy(copy, src, size);
    return copy;
}

static tensor_t *build_proxy_tensor_from_bf16_words(const uint16_t *weights,
                                                    uint32_t rows,
                                                    uint32_t cols)
{
    int shape[2] = { 0, 0 };
    tensor_t *proxy = NULL;
    uint16_t *data = NULL;
    size_t weight_count = 0u;

    if (!weights || rows == 0u || cols == 0u) {
        return NULL;
    }

    shape[0] = (int)rows;
    shape[1] = (int)cols;
    proxy = tensor_create(2, shape, DTYPE_BF16);
    if (!proxy) {
        return NULL;
    }

    data = (uint16_t *)tensor_data_mutable(proxy);
    if (!data) {
        tensor_release(proxy);
        return NULL;
    }

    weight_count = (size_t)rows * cols;
    memcpy(data, weights, weight_count * sizeof(*data));
    return proxy;
}

static tensor_t *build_proxy_tensor_from_dense(const float *weights,
                                               uint32_t rows,
                                               uint32_t cols)
{
    int shape[2] = { 0, 0 };
    tensor_t *proxy = NULL;
    uint16_t *data = NULL;
    size_t element_count = 0u;

    if (!weights || rows == 0u || cols == 0u) {
        return NULL;
    }

    shape[0] = (int)rows;
    shape[1] = (int)cols;
    proxy = tensor_create(2, shape, DTYPE_BF16);
    if (!proxy) {
        return NULL;
    }

    data = (uint16_t *)tensor_data_mutable(proxy);
    if (!data) {
        tensor_release(proxy);
        return NULL;
    }

    element_count = (size_t)rows * cols;
    for (size_t idx = 0u; idx < element_count; ++idx) {
        data[idx] = validation_f32_to_bf16(weights[idx]);
    }

    return proxy;
}

static tensor_t *build_proxy_tensor_from_bf16(const uint16_t *weights,
                                              uint32_t rows,
                                              uint32_t cols)
{
    return build_proxy_tensor_from_bf16_words(weights, rows, cols);
}

static tensor_t *build_proxy_tensor_from_ternary(const ternary_calibration_result_t *result) {
    tensor_t *proxy = NULL;
    uint8_t *packed_weights = NULL;
    float *scales = NULL;
    ternary_anchor_entry_t *anchor_entries = NULL;
    uint32_t *anchor_row_offsets = NULL;
    tensor_ternary_payload_t ternary_payload;
    tensor_hybrid_payload_t hybrid_payload;

    if (!result || !result->packed_weights || !result->scales || result->rows == 0u || result->cols == 0u) {
        return NULL;
    }

    packed_weights = (uint8_t *)validation_memdup(result->packed_weights, result->packed_weight_bytes);
    scales = (float *)validation_memdup(result->scales, result->scale_count * sizeof(*scales));
    if (!packed_weights || !scales) {
        goto cleanup;
    }

    memset(&ternary_payload, 0, sizeof(ternary_payload));
    ternary_payload.packed_weights = packed_weights;
    ternary_payload.packed_weight_bytes = result->packed_weight_bytes;
    ternary_payload.scales = scales;
    ternary_payload.scale_count = result->scale_count;
    ternary_payload.scale_group_size = result->scale_group_size;

    if (result->anchor_count > 0u) {
        if (!result->anchor_entries || !result->anchor_row_offsets) {
            goto cleanup;
        }
        anchor_entries = (ternary_anchor_entry_t *)validation_memdup(result->anchor_entries,
                                                                     result->anchor_count * sizeof(*anchor_entries));
        anchor_row_offsets = (uint32_t *)validation_memdup(result->anchor_row_offsets,
                                                           (result->rows + 1u) * sizeof(*anchor_row_offsets));
        if (!anchor_entries || !anchor_row_offsets) {
            goto cleanup;
        }

        memset(&hybrid_payload, 0, sizeof(hybrid_payload));
        hybrid_payload.bulk = ternary_payload;
        hybrid_payload.anchor_entries = anchor_entries;
        hybrid_payload.anchor_row_offsets = anchor_row_offsets;
        hybrid_payload.anchor_count = result->anchor_count;
        hybrid_payload.owns_anchor_memory = 1;
        proxy = tensor_create_hybrid_view(result->rows, result->cols, &hybrid_payload, 0);
    } else {
        proxy = tensor_create_ternary_view(result->rows, result->cols, &ternary_payload, 0);
    }

    if (proxy) {
        packed_weights = NULL;
        scales = NULL;
        anchor_entries = NULL;
        anchor_row_offsets = NULL;
    }

cleanup:
    free(packed_weights);
    free(scales);
    free(anchor_entries);
    free(anchor_row_offsets);
    return proxy;
}

static tensor_t *build_proxy_tensor_from_payload(const ternary_layer_payload_t *payload,
                                                 const ternary_anchor_view_t *anchor_view)
{
    tensor_t *proxy = NULL;
    uint8_t *packed_weights = NULL;
    float *scales = NULL;
    ternary_anchor_entry_t *anchor_entries = NULL;
    uint32_t *anchor_row_offsets = NULL;
    tensor_ternary_payload_t ternary_payload;
    tensor_hybrid_payload_t hybrid_payload;

    if (!payload || !payload->packed_weights || !payload->scales || payload->rows == 0u || payload->cols == 0u) {
        return NULL;
    }

    packed_weights = (uint8_t *)validation_memdup(payload->packed_weights, payload->packed_weight_bytes);
    scales = (float *)validation_memdup(payload->scales, payload->scale_bytes);
    if (!packed_weights || !scales) {
        goto cleanup;
    }

    memset(&ternary_payload, 0, sizeof(ternary_payload));
    ternary_payload.packed_weights = packed_weights;
    ternary_payload.packed_weight_bytes = payload->packed_weight_bytes;
    ternary_payload.scales = scales;
    ternary_payload.scale_count = payload->scale_count;
    ternary_payload.scale_group_size = payload->scale_group_size;

    if (anchor_view && anchor_view->anchor_count > 0u) {
        if (anchor_view->rows != payload->rows || anchor_view->cols != payload->cols) {
            goto cleanup;
        }
        if (!anchor_view->entries || !anchor_view->row_offsets) {
            goto cleanup;
        }

        anchor_entries = (ternary_anchor_entry_t *)validation_memdup(anchor_view->entries,
                                                                     anchor_view->anchor_count * sizeof(*anchor_entries));
        anchor_row_offsets = (uint32_t *)validation_memdup(anchor_view->row_offsets,
                                                           (anchor_view->rows + 1u) * sizeof(*anchor_row_offsets));
        if (!anchor_entries || !anchor_row_offsets) {
            goto cleanup;
        }

        memset(&hybrid_payload, 0, sizeof(hybrid_payload));
        hybrid_payload.bulk = ternary_payload;
        hybrid_payload.anchor_entries = anchor_entries;
        hybrid_payload.anchor_row_offsets = anchor_row_offsets;
        hybrid_payload.anchor_count = anchor_view->anchor_count;
        hybrid_payload.owns_anchor_memory = 1;
        proxy = tensor_create_hybrid_view(payload->rows, payload->cols, &hybrid_payload, 0);
    } else {
        proxy = tensor_create_ternary_view(payload->rows, payload->cols, &ternary_payload, 0);
    }

    if (proxy) {
        packed_weights = NULL;
        scales = NULL;
        anchor_entries = NULL;
        anchor_row_offsets = NULL;
    }

cleanup:
    free(packed_weights);
    free(scales);
    free(anchor_entries);
    free(anchor_row_offsets);
    return proxy;
}

static int validation_append_telemetry_checkpoint(ternary_validation_state_t *state,
                                                  const ternary_validation_checkpoint_t *checkpoint)
{
    ternary_validation_telemetry_t telemetry;

    if (!state || !checkpoint || !state->telemetry_enabled) {
        return 0;
    }

    memset(&telemetry, 0, sizeof(telemetry));
    telemetry.converted_count = checkpoint->converted_count;
    telemetry.tensor_name = checkpoint->tensor_name;
    telemetry.crc32 = checkpoint->crc32;
    telemetry.baseline_mean_nll = checkpoint->baseline_mean_nll;
    telemetry.current_mean_nll = checkpoint->current_mean_nll;
    telemetry.mean_kl = checkpoint->mean_kl;
    telemetry.max_kl = checkpoint->max_kl;
    telemetry.top1_agreement = checkpoint->top1_agreement;
    telemetry.sample_count = checkpoint->sample_count;
    return telemetry_dump_validation_checkpoint(&state->telemetry_writer, &telemetry);
}

static int token_argmax(const float *logits, int vocab_size) {
    int best_idx = 0;
    float best_val = logits[0];

    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static float negative_log_prob_from_logits(const float *logits, int vocab_size, int target_token) {
    float max_logit = -FLT_MAX;
    float sum_exp = 0.0f;

    if (!logits || vocab_size <= 0 || target_token < 0 || target_token >= vocab_size) {
        return 0.0f;
    }

    for (int i = 0; i < vocab_size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    for (int i = 0; i < vocab_size; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }

    if (sum_exp <= 1e-30f) {
        return 0.0f;
    }
    return logf(sum_exp) + max_logit - logits[target_token];
}

static int collect_prompt_metrics(inference_context_t *ctx,
                                  const char *text,
                                  float *out_final_logits,
                                  float *out_mean_nll,
                                  int *out_top1) {
    const gemma3_270m_config_t *config = NULL;
    int *tokens = NULL;
    int token_count = 0;
    const int max_tokens = 1024;
    float nll_sum = 0.0f;

    if (!ctx || !ctx->spec || !ctx->session || !ctx->tokenizer || !text ||
        !out_final_logits || !out_mean_nll || !out_top1) {
        return -1;
    }

    config = (gemma3_270m_config_t *)ctx->spec->variant_config;
    if (!config) {
        return -1;
    }

    tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) {
        return -1;
    }

    token_count = build_gemma3_prompt(ctx->spec, text, tokens, max_tokens);
    if (token_count <= 0) {
        token_count = tokenize(ctx->tokenizer, text, tokens, max_tokens);
    }
    if (token_count <= 0) {
        free(tokens);
        return -1;
    }

    inference_session_reset(ctx->session);
    for (int i = 0; i < token_count; ++i) {
        inference_forward(ctx->session, tokens[i], i, out_final_logits);
        if (i + 1 < token_count) {
            nll_sum += negative_log_prob_from_logits(out_final_logits,
                                                     config->vocab_size,
                                                     tokens[i + 1]);
        }
    }

    *out_mean_nll = (token_count > 1) ? (nll_sum / (float)(token_count - 1)) : 0.0f;
    *out_top1 = token_argmax(out_final_logits, config->vocab_size);
    free(tokens);
    return 0;
}

static int validation_collect_prompt_metrics(ternary_validation_state_t *state,
                                             const char *text,
                                             float *out_final_logits,
                                             float *out_mean_nll,
                                             int *out_top1)
{
    int rc = -1;

    if (!state || !state->ctx || !state->ctx->session) {
        return -1;
    }

    if (state->config.student_down_proj_input_rmsnorm) {
        inference_session_set_ffn_down_proj_input_rmsnorm_override(state->ctx->session, 1);
    }
    rc = collect_prompt_metrics(state->ctx,
                                text,
                                out_final_logits,
                                out_mean_nll,
                                out_top1);
    if (state->config.student_down_proj_input_rmsnorm) {
        inference_session_clear_ffn_down_proj_input_rmsnorm_override(state->ctx->session);
    }

    return rc;
}

static float compute_logits_kl(float *baseline_probs,
                               float *current_probs,
                               const float *baseline_logits,
                               const float *current_logits,
                               int vocab_size) {
    float kl = 0.0f;

    memcpy(baseline_probs, baseline_logits, (size_t)vocab_size * sizeof(float));
    memcpy(current_probs, current_logits, (size_t)vocab_size * sizeof(float));
    softmax(baseline_probs, vocab_size);
    softmax(current_probs, vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        float p = baseline_probs[i];
        float q = current_probs[i];
        if (p > 1e-12f && q > 1e-12f) {
            kl += p * logf(p / q);
        }
    }
    return kl;
}

static tensor_t **resolve_tensor_slot(llm_model_t *model,
                                      const model_spec_t *spec,
                                      const char *tensor_name) {
    int layer_idx = -1;
    const char *field_name = NULL;
    char *endptr = NULL;

    if (!model || !spec || !spec->tensor_map || !tensor_name) {
        return NULL;
    }

    for (int i = 0; i < spec->tensor_map_size; ++i) {
        const tensor_map_entry_t *entry = &spec->tensor_map[i];
        if (!entry->hf_name || strcmp(entry->hf_name, tensor_name) != 0) {
            continue;
        }
        field_name = entry->field_name;
        if (entry->internal_key && strncmp(entry->internal_key, "blk.", 4) == 0) {
            layer_idx = (int)strtol(entry->internal_key + 4, &endptr, 10);
            if (endptr == entry->internal_key + 4) {
                layer_idx = -1;
            }
        }
        break;
    }

    if (!field_name) return NULL;
    if (layer_idx < 0) return NULL;

    if (strcmp(field_name, "q_proj_weight") == 0) return &model->layers[layer_idx].q_proj_weight;
    if (strcmp(field_name, "k_proj_weight") == 0) return &model->layers[layer_idx].k_proj_weight;
    if (strcmp(field_name, "v_proj_weight") == 0) return &model->layers[layer_idx].v_proj_weight;
    if (strcmp(field_name, "out_proj_weight") == 0) return &model->layers[layer_idx].out_proj_weight;
    if (strcmp(field_name, "gate_proj_weight") == 0) return &model->layers[layer_idx].gate_proj_weight;
    if (strcmp(field_name, "up_proj_weight") == 0) return &model->layers[layer_idx].up_proj_weight;
    if (strcmp(field_name, "down_proj_weight") == 0) return &model->layers[layer_idx].down_proj_weight;
    return NULL;
}

static int run_checkpoint(ternary_validation_state_t *state, int converted_count);
static int store_validation_patch_record(ternary_validation_state_t *state,
                                         ternary_validation_patch_record_t *record);

static int make_proxy_record_from_tensor(inference_context_t *ctx,
                                         const char *tensor_name,
                                         tensor_t *proxy_tensor,
                                         uint32_t crc32,
                                         ternary_validation_patch_record_t *out_record)
{
    llm_model_t *model = NULL;
    tensor_t **slot = NULL;

    if (!ctx || !tensor_name || !proxy_tensor || !out_record || !ctx->spec || !ctx->spec->llm_model) {
        tensor_release(proxy_tensor);
        return -1;
    }

    model = (llm_model_t *)ctx->spec->llm_model;
    slot = resolve_tensor_slot(model, ctx->spec, tensor_name);
    if (!slot || !*slot) {
        tensor_release(proxy_tensor);
        return 0;
    }

    memset(out_record, 0, sizeof(*out_record));
    memcpy(out_record->tensor_name, tensor_name, strlen(tensor_name) + 1u);
    out_record->slot = slot;
    out_record->original_tensor = *slot;
    out_record->proxy_tensor = proxy_tensor;
    out_record->crc32 = crc32;
    *slot = proxy_tensor;
    return 1;
}

static int make_proxy_record_from_payload(inference_context_t *ctx,
                                          const char *tensor_name,
                                          const ternary_layer_payload_t *payload,
                                          const ternary_anchor_view_t *anchor_view,
                                          uint32_t crc32,
                                          ternary_validation_patch_record_t *out_record)
{
    tensor_t *proxy_tensor = NULL;

    if (!ctx || !tensor_name || !payload || !out_record || !ctx->spec || !ctx->spec->llm_model) {
        return -1;
    }

    proxy_tensor = build_proxy_tensor_from_payload(payload, anchor_view);
    if (!proxy_tensor) {
        return -1;
    }

    return make_proxy_record_from_tensor(ctx, tensor_name, proxy_tensor, crc32, out_record);
}

static int make_proxy_record(inference_context_t *ctx,
                             const char *tensor_name,
                             const ternary_calibration_result_t *result,
                             uint32_t crc32,
                             ternary_validation_patch_record_t *out_record)
{
    tensor_t *proxy_tensor = NULL;

    if (!ctx || !tensor_name || !result || !out_record || !ctx->spec || !ctx->spec->llm_model) {
        return -1;
    }

    proxy_tensor = build_proxy_tensor_from_ternary(result);
    if (!proxy_tensor) {
        return -1;
    }

    return make_proxy_record_from_tensor(ctx, tensor_name, proxy_tensor, crc32, out_record);
}

static int make_proxy_record_from_dense(inference_context_t *ctx,
                                        const char *tensor_name,
                                        const ternary_validation_dense_view_t *view,
                                        uint32_t crc32,
                                        ternary_validation_patch_record_t *out_record)
{
    tensor_t *proxy_tensor = NULL;

    if (!ctx || !tensor_name || !view || !view->weights || !out_record || !ctx->spec || !ctx->spec->llm_model) {
        return -1;
    }

    proxy_tensor = build_proxy_tensor_from_dense(view->weights, view->rows, view->cols);
    if (!proxy_tensor) {
        return -1;
    }

    return make_proxy_record_from_tensor(ctx, tensor_name, proxy_tensor, crc32, out_record);
}

static int make_proxy_record_from_bf16(inference_context_t *ctx,
                                       const char *tensor_name,
                                       const ternary_validation_bf16_view_t *view,
                                       uint32_t crc32,
                                       ternary_validation_patch_record_t *out_record)
{
    tensor_t *proxy_tensor = NULL;

    if (!ctx || !tensor_name || !view || !view->weights || !out_record || !ctx->spec || !ctx->spec->llm_model) {
        return -1;
    }

    proxy_tensor = build_proxy_tensor_from_bf16(view->weights, view->rows, view->cols);
    if (!proxy_tensor) {
        return -1;
    }

    return make_proxy_record_from_tensor(ctx, tensor_name, proxy_tensor, crc32, out_record);
}

static void validation_release_record_resources(ternary_validation_patch_record_t *record)
{
    if (!record) {
        return;
    }

    tensor_release(record->proxy_tensor);
    record->proxy_tensor = NULL;

    if (record->backing_file) {
        safetensors_close(record->backing_file);
        record->backing_file = NULL;
    }
    if (record->anchor_backing_file) {
        safetensors_close(record->anchor_backing_file);
        record->anchor_backing_file = NULL;
    }
    free(record->anchor_row_offsets);
    record->anchor_row_offsets = NULL;
}

static void validation_release_patch_resources(ternary_validation_patch_t *patch)
{
    if (!patch) {
        return;
    }

    tensor_release(patch->proxy_tensor);
    patch->proxy_tensor = NULL;

    if (patch->backing_file) {
        safetensors_close(patch->backing_file);
        patch->backing_file = NULL;
    }
    if (patch->anchor_backing_file) {
        safetensors_close(patch->anchor_backing_file);
        patch->anchor_backing_file = NULL;
    }
    free(patch->anchor_row_offsets);
    patch->anchor_row_offsets = NULL;
}

static int validation_build_anchor_row_offsets(const ternary_anchor_entry_t *entries,
                                               uint32_t anchor_count,
                                               uint32_t rows,
                                               uint32_t **out_row_offsets)
{
    uint32_t *row_offsets = NULL;

    if (!out_row_offsets) {
        return -1;
    }
    *out_row_offsets = NULL;
    if (anchor_count == 0u) {
        return 0;
    }
    if (!entries || rows == 0u) {
        return -1;
    }

    row_offsets = (uint32_t *)calloc((size_t)rows + 1u, sizeof(uint32_t));
    if (!row_offsets) {
        return -1;
    }

    for (uint32_t idx = 0u; idx < anchor_count; ++idx) {
        if (entries[idx].row >= rows) {
            free(row_offsets);
            return -1;
        }
        row_offsets[entries[idx].row + 1u]++;
    }

    for (uint32_t row = 1u; row <= rows; ++row) {
        row_offsets[row] += row_offsets[row - 1u];
    }

    *out_row_offsets = row_offsets;
    return 0;
}

static int make_proxy_record_from_ternary_file(inference_context_t *ctx,
                                               const char *tensor_name,
                                               const char *layer_path,
                                               const ternary_proxy_spec_t *spec,
                                               ternary_validation_patch_record_t *out_record)
{
    safetensors_file_t *st = NULL;
    tensor_t *proxy_tensor = NULL;
    int rc = -1;

    if (!ctx || !tensor_name || !layer_path || !spec || !out_record ||
        !ctx->spec || !ctx->spec->llm_model) {
        return -1;
    }

    st = safetensors_open(layer_path);
    if (!st) {
        return -1;
    }

    proxy_tensor = safetensors_create_ternary_tensor_ref(st, tensor_name,
                                                         spec->rows, spec->cols,
                                                         spec->packed_weight_bytes, spec->crc32);
    if (!proxy_tensor) {
        safetensors_close(st);
        return -1;
    }

    rc = make_proxy_record_from_tensor(ctx, tensor_name, proxy_tensor, spec->crc32, out_record);
    if (rc > 0) {
        out_record->backing_file = st;
        return rc;
    }

    safetensors_close(st);
    return rc;
}

static int make_proxy_record_from_bf16_file(inference_context_t *ctx,
                                            const char *tensor_name,
                                            const char *layer_path,
                                            uint32_t crc32,
                                            ternary_validation_patch_record_t *out_record)
{
    safetensors_file_t *st = NULL;
    const safetensors_tensor_meta_t *meta = NULL;
    tensor_t *proxy_tensor = NULL;
    int rc = -1;

    if (!ctx || !tensor_name || !layer_path || !out_record || !ctx->spec || !ctx->spec->llm_model) {
        return -1;
    }

    st = safetensors_open(layer_path);
    if (!st) {
        return -1;
    }

    meta = safetensors_get_tensor_by_name(st, tensor_name);
    if (!meta) {
        safetensors_close(st);
        return -1;
    }

    proxy_tensor = safetensors_create_tensor_ref(st, meta);
    if (!proxy_tensor) {
        safetensors_close(st);
        return -1;
    }

    rc = make_proxy_record_from_tensor(ctx, tensor_name, proxy_tensor, crc32, out_record);
    if (rc > 0) {
        out_record->backing_file = st;
        return rc;
    }

    safetensors_close(st);
    return rc;
}

/* Reads all scalar metadata fields from an anchor safetensors file.
 * Returns 0 on success, -1 if any field is absent or has an unexpected value. */
typedef struct {
    uint32_t rows, cols, anchor_count;
    uint32_t scale_group_size, groups_per_row, entry_crc32;
} anchor_file_meta_t;

static int read_anchor_file_meta(const safetensors_file_t *f, anchor_file_meta_t *out)
{
    uint32_t format_version = 0u;
    char value_dtype[32];

    if (!safetensors_metadata_get_u32(f, "format_version", &format_version) ||
        format_version != TERNARY_ANCHOR_VERSION ||
        !safetensors_metadata_get_u32(f, "rows",            &out->rows) ||
        !safetensors_metadata_get_u32(f, "cols",            &out->cols) ||
        !safetensors_metadata_get_u32(f, "anchor_count",    &out->anchor_count) ||
        !safetensors_metadata_get_u32(f, "scale_group_size",&out->scale_group_size) ||
        !safetensors_metadata_get_u32(f, "groups_per_row",  &out->groups_per_row) ||
        !safetensors_metadata_get_u32(f, "entry_crc32",     &out->entry_crc32) ||
        !safetensors_metadata_get_string(f, "value_dtype", value_dtype, sizeof(value_dtype)) ||
        strcmp(value_dtype, "BF16") != 0) {
        return -1;
    }
    return 0;
}

/* Validates anchor-file metadata against *spec and *bulk_view, then loads and
 * CRC-checks the anchor entries and builds the per-row offset table.
 * anchor_file must already be open; this helper does not close it on failure.
 * On success *out_entries points into the mmap'd file data (caller does not
 * own it), and *out_row_offsets is heap-allocated (caller must free it).     */
static int load_and_validate_anchor_entries(const safetensors_file_t *anchor_file,
                                            const char *tensor_name,
                                            const hybrid_proxy_spec_t *spec,
                                            const tensor_ternary_view_t *bulk_view,
                                            const ternary_anchor_entry_t **out_entries,
                                            uint32_t **out_row_offsets)
{
    const safetensors_tensor_meta_t *entries_meta = NULL;
    anchor_file_meta_t meta;
    char entries_name[320];
    size_t entry_bytes = 0u;

    *out_entries    = NULL;
    *out_row_offsets = NULL;

    memset(&meta, 0, sizeof(meta));
    if (read_anchor_file_meta(anchor_file, &meta) != 0) {
        return -1;
    }
    if (meta.rows != spec->rows || meta.cols != spec->cols ||
        meta.anchor_count != spec->anchor_count) {
        return -1;
    }
    if ((meta.scale_group_size != 0u && meta.scale_group_size != bulk_view->scale_group_size) ||
        (meta.groups_per_row != 0u && meta.groups_per_row != bulk_view->groups_per_row)) {
        return -1;
    }

    if (snprintf(entries_name, sizeof(entries_name), "%s.anchor_entries", tensor_name) < 0 ||
        strlen(entries_name) >= sizeof(entries_name)) {
        return -1;
    }
    entries_meta = safetensors_get_tensor_by_name(anchor_file, entries_name);
    if (!entries_meta || entries_meta->dtype != SAFETENSORS_U16 || entries_meta->ndim != 2 ||
        entries_meta->shape[0] != meta.anchor_count || entries_meta->shape[1] != 4u) {
        return -1;
    }
    entry_bytes = (size_t)meta.anchor_count * sizeof(ternary_anchor_entry_t);
    if (entries_meta->size_bytes != entry_bytes) {
        return -1;
    }

    if (meta.anchor_count > 0u) {
        const void *anchor_words = safetensors_data_ptr(anchor_file, entries_meta);
        uint32_t actual_crc32 = 0u;

        if (!anchor_words) {
            return -1;
        }
        *out_entries = (const ternary_anchor_entry_t *)anchor_words;
        actual_crc32 = io_crc32_update(0u, anchor_words, entry_bytes);
        if (meta.entry_crc32 != 0u && actual_crc32 != meta.entry_crc32) {
            *out_entries = NULL;
            return -1;
        }
        if (validation_build_anchor_row_offsets(*out_entries, meta.anchor_count,
                                                spec->rows, out_row_offsets) != 0) {
            *out_entries    = NULL;
            free(*out_row_offsets);
            *out_row_offsets = NULL;
            return -1;
        }
    }
    return 0;
}

static int make_proxy_record_from_hybrid_files(inference_context_t *ctx,
                                               const char *tensor_name,
                                               const char *bulk_path,
                                               const char *anchor_path,
                                               const hybrid_proxy_spec_t *spec,
                                               ternary_validation_patch_record_t *out_record)
{
    safetensors_file_t *bulk_file = NULL;
    safetensors_file_t *anchor_file = NULL;
    tensor_t *bulk_tensor = NULL;
    tensor_t *proxy_tensor = NULL;
    const tensor_ternary_view_t *bulk_view = NULL;
    const ternary_anchor_entry_t *anchor_entries = NULL;
    tensor_hybrid_payload_t payload;
    uint32_t *row_offsets = NULL;
    int rc = -1;

    if (!ctx || !tensor_name || !bulk_path || !anchor_path || !spec || !out_record ||
        !ctx->spec || !ctx->spec->llm_model) {
        return -1;
    }

    bulk_file = safetensors_open(bulk_path);
    if (!bulk_file) {
        return -1;
    }
    bulk_tensor = safetensors_create_ternary_tensor_ref(bulk_file, tensor_name,
                                                        spec->rows, spec->cols,
                                                        spec->packed_weight_bytes, spec->crc32);
    if (!bulk_tensor) {
        safetensors_close(bulk_file);
        return -1;
    }
    bulk_view = tensor_data_ternary(bulk_tensor);
    if (!bulk_view) {
        tensor_release(bulk_tensor);
        safetensors_close(bulk_file);
        return -1;
    }

    anchor_file = safetensors_open(anchor_path);
    if (!anchor_file) {
        tensor_release(bulk_tensor);
        safetensors_close(bulk_file);
        return -1;
    }
    if (load_and_validate_anchor_entries(anchor_file, tensor_name, spec, bulk_view,
                                         &anchor_entries, &row_offsets) != 0) {
        tensor_release(bulk_tensor);
        safetensors_close(anchor_file);
        safetensors_close(bulk_file);
        return -1;
    }

    memset(&payload, 0, sizeof(payload));
    payload.bulk.packed_weights      = bulk_view->packed_weights;
    payload.bulk.packed_weight_bytes = bulk_view->packed_weight_bytes;
    payload.bulk.scales              = bulk_view->scales;
    payload.bulk.scale_count         = bulk_view->scale_count;
    payload.bulk.scale_group_size    = bulk_view->scale_group_size;
    payload.anchor_entries           = anchor_entries;
    payload.anchor_row_offsets       = row_offsets;
    payload.anchor_count             = spec->anchor_count;
    payload.owns_anchor_memory       = 0;

    proxy_tensor = tensor_create_hybrid_view(spec->rows, spec->cols, &payload, 1);
    tensor_release(bulk_tensor);
    bulk_tensor = NULL;
    if (!proxy_tensor) {
        safetensors_close(anchor_file);
        safetensors_close(bulk_file);
        free(row_offsets);
        return -1;
    }

    rc = make_proxy_record_from_tensor(ctx, tensor_name, proxy_tensor, spec->crc32, out_record);
    if (rc > 0) {
        out_record->backing_file        = bulk_file;
        out_record->anchor_backing_file = anchor_file;
        out_record->anchor_row_offsets  = row_offsets;
        return rc;
    }

    safetensors_close(anchor_file);
    safetensors_close(bulk_file);
    free(row_offsets);
    return rc;
}

static int finish_proxy_application(ternary_validation_state_t *state,
                                    const char *tensor_name,
                                    uint32_t crc32,
                                    int converted_count,
                                    int make_record_rc,
                                    ternary_validation_patch_record_t *record)
{
    if (!state || !record) {
        return -1;
    }
    if (make_record_rc <= 0) {
        return make_record_rc;
    }

    if (store_validation_patch_record(state, record) != 0) {
        if (record->slot) {
            *record->slot = record->original_tensor;
        }
        validation_release_record_resources(record);
        return -1;
    }

    tensor_release(record->original_tensor);
    record->original_tensor = NULL;

    memcpy(state->last_tensor_name, tensor_name, strlen(tensor_name) + 1u);
    state->last_crc32 = crc32;

    if (state->config.validate_every_n > 0 &&
        converted_count > 0 &&
        (converted_count % state->config.validate_every_n) == 0) {
        return run_checkpoint(state, converted_count);
    }
    return 0;
}

int ternary_validation_capture_proxy_record(inference_context_t *ctx,
                                            const char *tensor_name,
                                            const ternary_calibration_result_t *result,
                                            ternary_validation_patch_record_t *out_record)
{
    return make_proxy_record(ctx, tensor_name, result, 0u, out_record) > 0 ? 0 : -1;
}

static ternary_validation_patch_t *find_patch_for_tensor(ternary_validation_state_t *state,
                                                         const char *tensor_name)
{
    if (!state || !tensor_name) {
        return NULL;
    }

    for (int patch_idx = 0; patch_idx < state->patch_count; ++patch_idx) {
        if (strcmp(state->patches[patch_idx].tensor_name, tensor_name) == 0) {
            return &state->patches[patch_idx];
        }
    }

    return NULL;
}

static int store_validation_patch_record(ternary_validation_state_t *state,
                                         ternary_validation_patch_record_t *record)
{
    ternary_validation_patch_t *patch = NULL;

    if (!state || !record) {
        return -1;
    }

    patch = find_patch_for_tensor(state, record->tensor_name);
    if (patch) {
        validation_release_patch_resources(patch);
        patch->slot = record->slot;
        patch->proxy_tensor = record->proxy_tensor;
        patch->backing_file = record->backing_file;
        patch->anchor_backing_file = record->anchor_backing_file;
        patch->anchor_row_offsets = record->anchor_row_offsets;
        patch->crc32 = record->crc32;
        record->original_tensor = NULL;
        return 0;
    }

    if (state->patch_count >= state->patch_capacity) {
        LOG_ERROR("Validation patch table exhausted");
        return -1;
    }

    patch = &state->patches[state->patch_count++];
    memset(patch, 0, sizeof(*patch));
    memcpy(patch->tensor_name, record->tensor_name, strlen(record->tensor_name) + 1u);
    patch->slot = record->slot;
    patch->proxy_tensor = record->proxy_tensor;
    patch->backing_file = record->backing_file;
    patch->anchor_backing_file = record->anchor_backing_file;
    patch->anchor_row_offsets = record->anchor_row_offsets;
    patch->crc32 = record->crc32;
    return 0;
}

int ternary_validation_apply_proxy_from_payload(ternary_validation_state_t *state,
                                                const char *tensor_name,
                                                const ternary_layer_payload_t *payload,
                                                uint32_t crc32,
                                                int converted_count)
{
    ternary_validation_patch_record_t record;
    int rc = 0;

    if (!state || !tensor_name || !payload || !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    rc = make_proxy_record_from_payload(state->ctx, tensor_name, payload, NULL, crc32, &record);
    return finish_proxy_application(state, tensor_name, crc32, converted_count, rc, &record);
}

int ternary_validation_apply_proxy_from_hybrid_payload(ternary_validation_state_t *state,
                                                       const char *tensor_name,
                                                       const ternary_layer_payload_t *payload,
                                                       const ternary_anchor_view_t *anchor_view,
                                                       uint32_t crc32,
                                                       int converted_count)
{
    ternary_validation_patch_record_t record;
    int rc = 0;

    if (!state || !tensor_name || !payload || !anchor_view ||
        !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    rc = make_proxy_record_from_payload(state->ctx, tensor_name, payload, anchor_view, crc32, &record);
    return finish_proxy_application(state, tensor_name, crc32, converted_count, rc, &record);
}

int ternary_validation_apply_proxy_from_dense(ternary_validation_state_t *state,
                                              const char *tensor_name,
                                              const ternary_validation_dense_view_t *view,
                                              uint32_t crc32,
                                              int converted_count)
{
    ternary_validation_patch_record_t record;
    int rc = 0;

    if (!state || !tensor_name || !view || !view->weights ||
        !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    rc = make_proxy_record_from_dense(state->ctx, tensor_name, view, crc32, &record);
    return finish_proxy_application(state, tensor_name, crc32, converted_count, rc, &record);
}

int ternary_validation_apply_proxy_from_bf16(ternary_validation_state_t *state,
                                             const char *tensor_name,
                                             const ternary_validation_bf16_view_t *view,
                                             uint32_t crc32,
                                             int converted_count)
{
    ternary_validation_patch_record_t record;
    int rc = 0;

    if (!state || !tensor_name || !view || !view->weights ||
        !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    rc = make_proxy_record_from_bf16(state->ctx, tensor_name, view, crc32, &record);
    return finish_proxy_application(state, tensor_name, crc32, converted_count, rc, &record);
}

int ternary_validation_apply_proxy_from_ternary_file(ternary_validation_state_t *state,
                                                     const char *tensor_name,
                                                     const char *layer_path,
                                                     const ternary_proxy_spec_t *spec,
                                                     int converted_count)
{
    ternary_validation_patch_record_t record;
    int rc = 0;

    if (!state || !tensor_name || !layer_path || !spec ||
        !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    rc = make_proxy_record_from_ternary_file(state->ctx, tensor_name, layer_path, spec, &record);
    return finish_proxy_application(state, tensor_name, spec->crc32, converted_count, rc, &record);
}

int ternary_validation_apply_proxy_from_hybrid_files(ternary_validation_state_t *state,
                                                     const char *tensor_name,
                                                     const char *bulk_path,
                                                     const char *anchor_path,
                                                     const hybrid_proxy_spec_t *spec,
                                                     int converted_count)
{
    ternary_validation_patch_record_t record;
    int rc = 0;

    if (!state || !tensor_name || !bulk_path || !anchor_path || !spec ||
        !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    rc = make_proxy_record_from_hybrid_files(state->ctx, tensor_name,
                                             bulk_path, anchor_path,
                                             spec, &record);
    return finish_proxy_application(state, tensor_name, spec->crc32, converted_count, rc, &record);
}

int ternary_validation_apply_proxy_from_bf16_file(ternary_validation_state_t *state,
                                                  const char *tensor_name,
                                                  const char *layer_path,
                                                  uint32_t crc32,
                                                  int converted_count)
{
    ternary_validation_patch_record_t record;
    int rc = 0;

    if (!state || !tensor_name || !layer_path ||
        !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    rc = make_proxy_record_from_bf16_file(state->ctx, tensor_name, layer_path, crc32, &record);
    return finish_proxy_application(state, tensor_name, crc32, converted_count, rc, &record);
}

int ternary_validation_adopt_patch_records(ternary_validation_state_t *state,
                                           const ternary_validation_patch_record_t *records,
                                           int record_count)
{
    if (!state || !records || record_count < 0) {
        return -1;
    }
    if (state->patch_count + record_count > state->patch_capacity) {
        LOG_ERROR("Validation patch table exhausted while adopting resume patches");
        return -1;
    }

    for (int i = 0; i < record_count; ++i) {
        ternary_validation_patch_t *patch = &state->patches[state->patch_count++];

        memset(patch, 0, sizeof(*patch));
        memcpy(patch->tensor_name, records[i].tensor_name, strlen(records[i].tensor_name) + 1u);
        patch->slot = records[i].slot;
        patch->proxy_tensor = records[i].proxy_tensor;
        patch->backing_file = records[i].backing_file;
        patch->anchor_backing_file = records[i].anchor_backing_file;
        patch->anchor_row_offsets = records[i].anchor_row_offsets;
        patch->crc32 = records[i].crc32;
    }

    return 0;
}

static int run_checkpoint(ternary_validation_state_t *state, int converted_count) {
    const gemma3_270m_config_t *config = NULL;
    ternary_validation_checkpoint_t checkpoint;
    float mean_kl = 0.0f;
    float max_kl = 0.0f;
    float current_mean_nll = 0.0f;
    float baseline_mean_nll = 0.0f;
    float top1_hits = 0.0f;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "run_validation_checkpoint");

    if (!state || !state->ctx || !state->ctx->spec || !state->config.output_dir ||
        state->config.sample_count <= 0 || state->last_tensor_name[0] == '\0') {
        return validation_tracy_end_status(&tracy_zone, 0);
    }

    validation_tracy_zone_text_if_present(&tracy_zone, state->last_tensor_name);
    sapphire_tracy_plot_i64("validation.converted_tensors", (int64_t)converted_count);

    config = (gemma3_270m_config_t *)state->ctx->spec->variant_config;
    if (!config) {
        return validation_tracy_end_status(&tracy_zone, -1);
    }

    for (int s = 0; s < state->config.sample_count; ++s) {
        float sample_mean_nll = 0.0f;
        int sample_top1 = -1;
        float kl = 0.0f;

        if (validation_collect_prompt_metrics(state,
                                              state->config.sample_texts[s],
                                              state->current_logits + (size_t)s * config->vocab_size,
                                              &sample_mean_nll,
                                              &sample_top1) != 0) {
            return validation_tracy_end_status(&tracy_zone, -1);
        }

        kl = compute_logits_kl(state->baseline_probs,
                               state->current_probs,
                               state->baseline_logits + (size_t)s * config->vocab_size,
                               state->current_logits + (size_t)s * config->vocab_size,
                               config->vocab_size);
        mean_kl += kl;
        if (kl > max_kl) {
            max_kl = kl;
        }
        current_mean_nll += sample_mean_nll;
        baseline_mean_nll += state->baseline_mean_nll[s];
        if (sample_top1 == state->baseline_top1[s]) {
            top1_hits += 1.0f;
        }
    }

    mean_kl /= (float)state->config.sample_count;
    current_mean_nll /= (float)state->config.sample_count;
    baseline_mean_nll /= (float)state->config.sample_count;
    top1_hits /= (float)state->config.sample_count;

    memset(&checkpoint, 0, sizeof(checkpoint));
    checkpoint.converted_count = converted_count;
    checkpoint.tensor_name = state->last_tensor_name;
    checkpoint.crc32 = state->last_crc32;
    checkpoint.baseline_mean_nll = baseline_mean_nll;
    checkpoint.current_mean_nll = current_mean_nll;
    checkpoint.mean_kl = mean_kl;
    checkpoint.max_kl = max_kl;
    checkpoint.top1_agreement = top1_hits;
    checkpoint.sample_count = state->config.sample_count;

    if (io_append_validation_checkpoint(state->config.output_dir, &checkpoint) != 0) {
        return validation_tracy_end_status(&tracy_zone, -1);
    }
    if (validation_append_telemetry_checkpoint(state, &checkpoint) != 0) {
        LOG_WARN("validation telemetry: failed to append checkpoint for %s", state->last_tensor_name);
    }

    sapphire_tracy_plot_f64("validation.mean_kl", mean_kl);
    sapphire_tracy_plot_f64("validation.max_kl", max_kl);
    sapphire_tracy_plot_f64("validation.top1_agreement", top1_hits);
    sapphire_tracy_plot_f64("validation.current_mean_nll", current_mean_nll);
    sapphire_tracy_plot_f64("validation.baseline_mean_nll", baseline_mean_nll);

    state->last_reported_count = converted_count;
    LOG_INFO("Validation checkpoint: converted=%d tensor=%s mean_kl=%.6f top1=%.3f ppl=%.3f",
             converted_count,
             state->last_tensor_name,
             mean_kl,
             top1_hits,
             expf(current_mean_nll));
    return validation_tracy_end_status(&tracy_zone, 0);
}

int ternary_validation_init(ternary_validation_state_t *state,
                            const ternary_validation_config_t *config,
                            inference_context_t *ctx) {
    const gemma3_270m_config_t *model_config = NULL;
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "ternary_validation_init");

    if (!state || !config || !ctx || !ctx->spec || !ctx->session || !ctx->tokenizer) {
        return validation_tracy_end_status(&tracy_zone, -1);
    }

    memset(state, 0, sizeof(*state));
    state->config = *config;
    state->ctx = ctx;
    if (config->telemetry_path && config->telemetry_path[0] != '\0') {
        if (ternary_telemetry_writer_init(&state->telemetry_writer, config->telemetry_path) == 0) {
            state->telemetry_enabled = 1;
        } else {
            LOG_WARN("validation telemetry: disabled for %s", config->telemetry_path);
        }
    }
    validation_tracy_zone_text_if_present(&tracy_zone, config->output_dir);
    sapphire_tracy_plot_i64("validation.sample_count", (int64_t)config->sample_count);
    sapphire_tracy_plot_i64("validation.validate_every_n", (int64_t)config->validate_every_n);
    model_config = (gemma3_270m_config_t *)ctx->spec->variant_config;
    if (!model_config || config->sample_count <= 0) {
        return validation_tracy_end_status(&tracy_zone, 0);
    }

    state->baseline_logits = (float *)malloc((size_t)config->sample_count * model_config->vocab_size * sizeof(float));
    state->current_logits = (float *)malloc((size_t)config->sample_count * model_config->vocab_size * sizeof(float));
    state->baseline_probs = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    state->current_probs = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    state->baseline_mean_nll = (float *)malloc((size_t)config->sample_count * sizeof(float));
    state->baseline_top1 = (int *)malloc((size_t)config->sample_count * sizeof(int));
    state->patch_capacity = ctx->spec->tensor_map_size;
    state->patches = (ternary_validation_patch_t *)calloc((size_t)state->patch_capacity,
                                                          sizeof(ternary_validation_patch_t));
    if (!state->baseline_logits || !state->current_logits || !state->baseline_probs || !state->current_probs ||
        !state->baseline_mean_nll || !state->baseline_top1 || !state->patches) {
        ternary_validation_destroy(state);
        return validation_tracy_end_status(&tracy_zone, -1);
    }

    for (int s = 0; s < config->sample_count; ++s) {
        if (validation_collect_prompt_metrics(state,
                                              config->sample_texts[s],
                                              state->baseline_logits + (size_t)s * model_config->vocab_size,
                                              &state->baseline_mean_nll[s],
                                              &state->baseline_top1[s]) != 0) {
            ternary_validation_destroy(state);
            return validation_tracy_end_status(&tracy_zone, -1);
        }
    }

    LOG_INFO("Validation baseline initialized: prompts=%d validate_every=%d",
             config->sample_count,
             config->validate_every_n);

    LOG_INFO("Validation baseline initialized: prompts=%d validate_every=%d",
             config->sample_count,
             config->validate_every_n);
    return validation_tracy_end_status(&tracy_zone, 0);
}

int ternary_validation_apply_proxy(ternary_validation_state_t *state,
                                   const char *tensor_name,
                                   const ternary_calibration_result_t *result,
                                   uint32_t crc32,
                                   int converted_count) {
    ternary_validation_patch_record_t record;
    int rc = 0;

    if (!state || !tensor_name || !result || !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    rc = make_proxy_record(state->ctx, tensor_name, result, crc32, &record);
    return finish_proxy_application(state, tensor_name, crc32, converted_count, rc, &record);
}

int ternary_validation_finish(ternary_validation_state_t *state,
                              int converted_count) {
    SAPPHIRE_TRACY_ZONE_SCOPE(tracy_zone, "ternary_validation_finish");

    if (!state || state->config.sample_count <= 0 || converted_count <= 0) {
        return validation_tracy_end_status(&tracy_zone, 0);
    }
    sapphire_tracy_plot_i64("validation.converted_tensors", (int64_t)converted_count);
    if (state->last_reported_count == converted_count) {
        return validation_tracy_end_status(&tracy_zone, 0);
    }
    return validation_tracy_end_status(&tracy_zone, run_checkpoint(state, converted_count));
}

void ternary_validation_destroy(ternary_validation_state_t *state) {
    if (!state) {
        return;
    }

    if (state->telemetry_enabled) {
        ternary_telemetry_writer_close(&state->telemetry_writer);
    }

    if (state->patches) {
        for (int i = 0; i < state->patch_count; ++i) {
            ternary_validation_patch_t *patch = &state->patches[i];
            /* Null the model slot before releasing the proxy. The original
             * tensor was already released in finish_proxy_application, so
             * we cannot restore it. Setting the slot to NULL prevents
             * llm_model_destroy from calling tensor_release on the freed
             * proxy pointer after we release it below. */
            if (patch->slot) {
                *patch->slot = NULL;
                patch->slot = NULL;
            }
            validation_release_patch_resources(patch);
        }
    }

    free(state->patches);
    free(state->baseline_logits);
    free(state->current_logits);
    free(state->baseline_probs);
    free(state->current_probs);
    free(state->baseline_mean_nll);
    free(state->baseline_top1);
    memset(state, 0, sizeof(*state));
}
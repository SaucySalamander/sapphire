/*
 * @file ternary_hessian_oracle.c
 * @brief Native Vulkan-side Hessian sidecar recording workflow.
 */

#include "ternary_hessian_oracle.h"

#include "backend_vulkan.h"
#include "log.h"
#include "ternary_hessian_sidecar.h"
#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

size_t ternary_hessian_oracle_capture_bytes(const gemma3_270m_config_t *cfg)
{
    uint64_t per_layer_bytes = 0u;

    if (!cfg || cfg->num_hidden_layers <= 0 || cfg->hidden_size <= 0 ||
        cfg->num_attention_heads <= 0 || cfg->head_dim <= 0 ||
        cfg->intermediate_size <= 0) {
        return 0u;
    }

    per_layer_bytes += (uint64_t)cfg->hidden_size * sizeof(float);
    per_layer_bytes += (uint64_t)(cfg->num_attention_heads * cfg->head_dim) * sizeof(float);
    per_layer_bytes += (uint64_t)cfg->hidden_size * sizeof(float);
    per_layer_bytes += (uint64_t)cfg->intermediate_size * sizeof(float);
    return (size_t)((uint64_t)cfg->num_hidden_layers * per_layer_bytes);
}

int ternary_hessian_oracle_capture_view(const gemma3_270m_config_t *cfg,
                                        int layer_idx,
                                        capture_target_t target,
                                        size_t *out_offset_bytes,
                                        uint32_t *out_vector_dim)
{
    uint64_t d_model_bytes = 0u;
    uint64_t d_inner_bytes = 0u;
    uint64_t d_ff_bytes = 0u;
    uint64_t layer_base = 0u;
    uint32_t d_inner = 0u;

    if (!cfg || !out_offset_bytes || !out_vector_dim ||
        layer_idx < 0 || layer_idx >= cfg->num_hidden_layers) {
        return -1;
    }

    d_inner = (uint32_t)(cfg->num_attention_heads * cfg->head_dim);
    d_model_bytes = (uint64_t)cfg->hidden_size * sizeof(float);
    d_inner_bytes = (uint64_t)d_inner * sizeof(float);
    d_ff_bytes = (uint64_t)cfg->intermediate_size * sizeof(float);
    layer_base = (uint64_t)layer_idx * (d_model_bytes + d_inner_bytes + d_model_bytes + d_ff_bytes);

    switch (target) {
    case CAPTURE_TARGET_QKV_INPUT:
        *out_offset_bytes = (size_t)layer_base;
        *out_vector_dim = (uint32_t)cfg->hidden_size;
        return 0;
    case CAPTURE_TARGET_OUT_INPUT:
        *out_offset_bytes = (size_t)(layer_base + d_model_bytes);
        *out_vector_dim = d_inner;
        return 0;
    case CAPTURE_TARGET_FFN_INPUT:
        *out_offset_bytes = (size_t)(layer_base + d_model_bytes + d_inner_bytes);
        *out_vector_dim = (uint32_t)cfg->hidden_size;
        return 0;
    case CAPTURE_TARGET_DOWN_INPUT:
        *out_offset_bytes = (size_t)(layer_base + d_model_bytes + d_inner_bytes + d_model_bytes);
        *out_vector_dim = (uint32_t)cfg->intermediate_size;
        return 0;
    default:
        return -1;
    }
}

static int oracle_parse_capture_target(const char *tensor_name,
                                       int *out_layer_idx,
                                       capture_target_t *out_target)
{
    static const char *capture_prefixes[] = {
        "model.layers.",
        "language_model.model.layers.",
        NULL
    };
    const char *cursor = NULL;
    char *endptr = NULL;
    long parsed_layer = 0;

    if (!tensor_name || !out_layer_idx || !out_target) {
        return -1;
    }

    for (int prefix_idx = 0; capture_prefixes[prefix_idx] != NULL; ++prefix_idx) {
        size_t prefix_len = strlen(capture_prefixes[prefix_idx]);

        if (strncmp(tensor_name, capture_prefixes[prefix_idx], prefix_len) == 0) {
            cursor = tensor_name + prefix_len;
            break;
        }
    }
    if (!cursor) {
        return -1;
    }

    parsed_layer = strtol(cursor, &endptr, 10);
    if (endptr == cursor || parsed_layer < 0 || strncmp(endptr, ".", 1) != 0) {
        return -1;
    }

    if (strstr(tensor_name, ".self_attn.q_proj.weight") ||
        strstr(tensor_name, ".self_attn.k_proj.weight") ||
        strstr(tensor_name, ".self_attn.v_proj.weight")) {
        *out_target = CAPTURE_TARGET_QKV_INPUT;
    } else if (strstr(tensor_name, ".self_attn.o_proj.weight")) {
        *out_target = CAPTURE_TARGET_OUT_INPUT;
    } else if (strstr(tensor_name, ".mlp.gate_proj.weight") ||
               strstr(tensor_name, ".mlp.up_proj.weight")) {
        *out_target = CAPTURE_TARGET_FFN_INPUT;
    } else if (strstr(tensor_name, ".mlp.down_proj.weight")) {
        *out_target = CAPTURE_TARGET_DOWN_INPUT;
    } else {
        return -1;
    }

    *out_layer_idx = (int)parsed_layer;
    return 0;
}

static int oracle_build_prompt_tokens(const inference_context_t *ctx,
                                      const char *text,
                                      int *tokens,
                                      int max_tokens)
{
    int token_count = 0;

    if (!ctx || !ctx->spec || !ctx->tokenizer || !text || !tokens || max_tokens <= 0) {
        return -1;
    }

    token_count = build_gemma3_prompt(ctx->spec, text, tokens, max_tokens);
    if (token_count <= 0) {
        token_count = tokenize(ctx->tokenizer, text, tokens, max_tokens);
    }

    return token_count;
}

int ternary_hessian_oracle_write_sidecar(const ternary_hessian_oracle_write_request_t *request)
{
    ternary_hessian_sidecar_write_entry_t *write_entries = NULL;
    uint32_t sidecar_crc32 = 0u;
    int rc = -1;

    if (!request || !request->output_path || !request->teacher_model_name ||
        !request->config || !request->manifest || request->entry_count == 0u ||
        request->sample_count == 0u || !request->diagonal_capture_buffer) {
        return -1;
    }

    write_entries = (ternary_hessian_sidecar_write_entry_t *)calloc((size_t)request->entry_count,
                                                                    sizeof(*write_entries));
    if (!write_entries) {
        return -1;
    }

    for (uint32_t entry_idx = 0; entry_idx < request->entry_count; ++entry_idx) {
        const tape_manifest_entry_t *entry = &request->manifest[entry_idx];
        ternary_hessian_sidecar_write_entry_t *write_entry = &write_entries[entry_idx];
        capture_target_t target = CAPTURE_TARGET_UNKNOWN;
        size_t capture_offset = 0u;
        uint32_t vector_dim = 0u;
        int layer_idx = 0;

        if (!entry->tensor_name[0]) {
            rc = -1;
            goto cleanup;
        }
        if (oracle_parse_capture_target(entry->tensor_name, &layer_idx, &target) != 0) {
            LOG_ERROR("hessian oracle: failed to parse tensor name %s", entry->tensor_name);
            rc = -1;
            goto cleanup;
        }
        if (ternary_hessian_oracle_capture_view(request->config,
                                                layer_idx,
                                                target,
                                                &capture_offset,
                                                &vector_dim) != 0 ||
            vector_dim != entry->vector_dim) {
            LOG_ERROR("hessian oracle: manifest dimension mismatch for %s", entry->tensor_name);
            rc = -1;
            goto cleanup;
        }

        memset(write_entry, 0, sizeof(*write_entry));
        write_entry->tensor_name = entry->tensor_name;
        write_entry->vector_dim = entry->vector_dim;
        write_entry->sample_count = entry->sample_count;
        write_entry->alias_of_entry = (entry->alias_of_entry == TAPE_NO_ALIAS)
            ? TERNARY_HESSIAN_SIDECAR_NO_ALIAS
            : entry->alias_of_entry;
        write_entry->layer_type = entry->layer_type;
        if (write_entry->alias_of_entry == TERNARY_HESSIAN_SIDECAR_NO_ALIAS) {
            write_entry->diagonal = request->diagonal_capture_buffer + (capture_offset / sizeof(float));
        }
    }

    {
        const ternary_hessian_sidecar_write_config_t write_config = {
            .teacher_model_name = request->teacher_model_name,
            .tape_crc32 = request->tape_crc32,
            .sample_count = request->sample_count
        };

        if (ternary_hessian_sidecar_write(request->output_path,
                                          &write_config,
                                          write_entries,
                                          request->entry_count,
                                          &sidecar_crc32) != 0) {
            LOG_ERROR("hessian oracle: failed to write sidecar %s", request->output_path);
            rc = -1;
            goto cleanup;
        }
    }

    LOG_INFO("hessian oracle: wrote %s (samples=%u entries=%u crc32=%08x)",
             request->output_path,
             request->sample_count,
             request->entry_count,
             sidecar_crc32);
    rc = 0;

cleanup:
    free(write_entries);
    return rc;
}

int ternary_record_hessian_sidecar_vulkan(const char *output_path,
                                          inference_context_t *ctx,
                                          const calibration_corpus_t *corpus,
                                          const activation_tape_t *tape,
                                          float hessian_proxy_strength,
                                          float hessian_proxy_floor)
{
    const gemma3_270m_config_t *cfg = NULL;
    tape_manifest_entry_t *manifest_copy = NULL;
    float *oracle_buffer = NULL;
    int *tokens = NULL;
    size_t capture_bytes = 0u;
    uint32_t entry_count = 0u;
    int sample_count = 0;
    int rc = -1;
    int oracle_began = 0;

    if (!output_path || !ctx || !ctx->session || !ctx->spec || !ctx->tokenizer ||
        !corpus || !tape) {
        return -1;
    }
    if (!ctx->session->backend || ctx->session->backend->type != SAPPHIRE_BACKEND_TYPE_VULKAN) {
        LOG_ERROR("hessian oracle: Vulkan backend is required; set SAPPHIRE_BACKEND=vulkan");
        return -1;
    }

    cfg = (const gemma3_270m_config_t *)ctx->spec->variant_config;
    sample_count = activation_tape_sample_count(tape);
    if (!cfg || sample_count <= 0 || corpus->sample_count != sample_count) {
        LOG_ERROR("hessian oracle: corpus/tape sample count mismatch");
        return -1;
    }
    if (activation_tape_header(tape)->hidden_size != (uint32_t)cfg->hidden_size) {
        LOG_ERROR("hessian oracle: activation tape hidden-size mismatch");
        return -1;
    }

    capture_bytes = ternary_hessian_oracle_capture_bytes(cfg);
    entry_count = activation_tape_entry_count(tape);
    if (capture_bytes == 0u || entry_count == 0u || ctx->context_len <= 0) {
        LOG_ERROR("hessian oracle: invalid capture layout");
        return -1;
    }

    oracle_buffer = (float *)malloc(capture_bytes);
    tokens = (int *)malloc((size_t)ctx->context_len * sizeof(int));
    manifest_copy = (tape_manifest_entry_t *)calloc((size_t)entry_count, sizeof(*manifest_copy));
    if (!oracle_buffer || !tokens || !manifest_copy) {
        LOG_ERROR("hessian oracle: allocation failed");
        goto cleanup;
    }

    if (backend_vulkan_begin_hessian_oracle_capture(ctx->session) != 0) {
        LOG_ERROR("hessian oracle: failed to initialize Vulkan oracle capture state");
        goto cleanup;
    }
    oracle_began = 1;

    for (uint32_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
        const tape_manifest_entry_t *entry = activation_tape_entry(tape, entry_idx);

        if (!entry) {
            LOG_ERROR("hessian oracle: failed to load tape manifest entry %u", entry_idx);
            goto cleanup;
        }
        manifest_copy[entry_idx] = *entry;
    }

    for (int sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
        int token_count = oracle_build_prompt_tokens(ctx,
                                                     corpus->samples[sample_idx],
                                                     tokens,
                                                     ctx->context_len);

        if (token_count <= 0) {
            LOG_ERROR("hessian oracle: tokenization failed at sample %d", sample_idx);
            goto cleanup;
        }
        if (backend_vulkan_capture_last_token_activations(ctx->session,
                                                          tokens,
                                                          token_count,
                                                          NULL,
                                                          0u) != 0) {
            LOG_ERROR("hessian oracle: Vulkan capture failed at sample %d", sample_idx);
            goto cleanup;
        }
    }

    {
        const backend_vulkan_oracle_finish_config_t finish_config = {
            .sample_count = (uint32_t)sample_count,
            .strength = hessian_proxy_strength,
            .floor = hessian_proxy_floor,
            .out_buffer = oracle_buffer,
            .out_size = capture_bytes
        };

        if (backend_vulkan_finish_hessian_oracle_capture(ctx->session, &finish_config) != 0) {
            LOG_ERROR("hessian oracle: failed to finalize Vulkan oracle capture");
            goto cleanup;
        }
    }
    oracle_began = 0;

    {
        const ternary_hessian_oracle_write_request_t write_request = {
            .output_path = output_path,
            .teacher_model_name = ctx->spec->model_id,
            .config = cfg,
            .manifest = manifest_copy,
            .entry_count = entry_count,
            .tape_crc32 = activation_tape_crc32(tape),
            .sample_count = (uint32_t)sample_count,
            .diagonal_capture_buffer = oracle_buffer
        };

        if (ternary_hessian_oracle_write_sidecar(&write_request) != 0) {
            goto cleanup;
        }
    }

    rc = 0;

cleanup:
    if (oracle_began) {
        backend_vulkan_abort_hessian_oracle_capture(ctx->session);
    }
    free(manifest_copy);
    free(tokens);
    free(oracle_buffer);
    return rc;
}
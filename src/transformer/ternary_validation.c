/**
 * @file ternary_validation.c
 * @brief Full-model validation checkpoints for ternary conversion.
 */

#include "ternary_validation.h"

#include "gemma3_270m_config.h"
#include "kernels.h"
#include "llm_model.h"
#include "log.h"
#include "tensor.h"
#include "ternary_io.h"
#include "tokenizer.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct ternary_validation_patch {
    char tensor_name[256];
    tensor_t **slot;
    tensor_t *original_tensor;
    tensor_t *proxy_tensor;
    uint32_t crc32;
};

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

static tensor_t *build_proxy_tensor_from_ternary(const ternary_calibration_result_t *result) {
    int shape[2] = { 0, 0 };
    tensor_t *proxy = NULL;
    float *data = NULL;

    if (!result || !result->ternary_weights || !result->scales) {
        return NULL;
    }

    shape[0] = (int)result->rows;
    shape[1] = (int)result->cols;

    proxy = tensor_create(2, shape, DTYPE_F32);
    if (!proxy) {
        return NULL;
    }
    data = tensor_data_f32(proxy);
    if (!data) {
        tensor_release(proxy);
        return NULL;
    }

    for (uint32_t r = 0; r < result->rows; ++r) {
        size_t row_base = (size_t)r * result->cols;
        float scale = result->scales[r];
        for (uint32_t c = 0; c < result->cols; ++c) {
            data[row_base + c] = (float)result->ternary_weights[row_base + c] * scale;
        }
    }

    return proxy;
}

static int run_checkpoint(ternary_validation_state_t *state, int converted_count) {
    const gemma3_270m_config_t *config = NULL;
    ternary_validation_checkpoint_t checkpoint;
    float mean_kl = 0.0f;
    float max_kl = 0.0f;
    float current_mean_nll = 0.0f;
    float baseline_mean_nll = 0.0f;
    float top1_hits = 0.0f;

    if (!state || !state->ctx || !state->ctx->spec || !state->config.output_dir ||
        state->config.sample_count <= 0 || state->last_tensor_name[0] == '\0') {
        return 0;
    }

    config = (gemma3_270m_config_t *)state->ctx->spec->variant_config;
    if (!config) {
        return -1;
    }

    for (int s = 0; s < state->config.sample_count; ++s) {
        float sample_mean_nll = 0.0f;
        int sample_top1 = -1;
        float kl = 0.0f;

        if (collect_prompt_metrics(state->ctx,
                                   state->config.sample_texts[s],
                                   state->current_logits + (size_t)s * config->vocab_size,
                                   &sample_mean_nll,
                                   &sample_top1) != 0) {
            return -1;
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
        return -1;
    }

    state->last_reported_count = converted_count;
    LOG_INFO("Validation checkpoint: converted=%d tensor=%s mean_kl=%.6f top1=%.3f ppl=%.3f",
             converted_count,
             state->last_tensor_name,
             mean_kl,
             top1_hits,
             expf(current_mean_nll));
    return 0;
}

int ternary_validation_init(ternary_validation_state_t *state,
                            const ternary_validation_config_t *config,
                            inference_context_t *ctx) {
    const gemma3_270m_config_t *model_config = NULL;

    if (!state || !config || !ctx || !ctx->spec || !ctx->session || !ctx->tokenizer) {
        return -1;
    }

    memset(state, 0, sizeof(*state));
    state->config = *config;
    state->ctx = ctx;
    model_config = (gemma3_270m_config_t *)ctx->spec->variant_config;
    if (!model_config || config->sample_count <= 0) {
        return 0;
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
        return -1;
    }

    for (int s = 0; s < config->sample_count; ++s) {
        if (collect_prompt_metrics(ctx,
                                   config->sample_texts[s],
                                   state->baseline_logits + (size_t)s * model_config->vocab_size,
                                   &state->baseline_mean_nll[s],
                                   &state->baseline_top1[s]) != 0) {
            ternary_validation_destroy(state);
            return -1;
        }
    }

    LOG_INFO("Validation baseline initialized: prompts=%d validate_every=%d",
             config->sample_count,
             config->validate_every_n);
    return 0;
}

int ternary_validation_apply_proxy(ternary_validation_state_t *state,
                                   const char *tensor_name,
                                   const ternary_calibration_result_t *result,
                                   uint32_t crc32,
                                   int converted_count) {
    llm_model_t *model = NULL;
    tensor_t **slot = NULL;
    tensor_t *proxy_tensor = NULL;
    ternary_validation_patch_t *patch = NULL;

    if (!state || !tensor_name || !result || !state->ctx || !state->ctx->spec || !state->ctx->spec->llm_model) {
        return -1;
    }
    if (state->config.sample_count <= 0) {
        return 0;
    }

    model = (llm_model_t *)state->ctx->spec->llm_model;
    slot = resolve_tensor_slot(model, state->ctx->spec, tensor_name);
    if (!slot || !*slot) {
        return 0;
    }
    if (state->patch_count >= state->patch_capacity) {
        LOG_ERROR("Validation patch table exhausted");
        return -1;
    }

    proxy_tensor = build_proxy_tensor_from_ternary(result);
    if (!proxy_tensor) {
        return -1;
    }

    patch = &state->patches[state->patch_count++];
    memset(patch, 0, sizeof(*patch));
    memcpy(patch->tensor_name, tensor_name, strlen(tensor_name) + 1u);
    patch->slot = slot;
    patch->original_tensor = *slot;
    patch->proxy_tensor = proxy_tensor;
    patch->crc32 = crc32;
    *slot = proxy_tensor;

    memcpy(state->last_tensor_name, tensor_name, strlen(tensor_name) + 1u);
    state->last_crc32 = crc32;

    if (state->config.validate_every_n > 0 &&
        converted_count > 0 &&
        (converted_count % state->config.validate_every_n) == 0) {
        return run_checkpoint(state, converted_count);
    }
    return 0;
}

int ternary_validation_finish(ternary_validation_state_t *state,
                              int converted_count) {
    if (!state || state->config.sample_count <= 0 || converted_count <= 0) {
        return 0;
    }
    if (state->last_reported_count == converted_count) {
        return 0;
    }
    return run_checkpoint(state, converted_count);
}

void ternary_validation_destroy(ternary_validation_state_t *state) {
    if (!state) {
        return;
    }

    for (int i = state->patch_count - 1; i >= 0; --i) {
        ternary_validation_patch_t *patch = &state->patches[i];
        if (patch->slot) {
            *patch->slot = patch->original_tensor;
        }
        tensor_release(patch->proxy_tensor);
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
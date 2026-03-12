/**
 * @file ternary_calibration.c
 * @brief Prompt 2 STE calibration shell for ternary conversion.
 */

#include "ternary_calibration.h"

#include "log.h"
#include "llm_model.h"
#include "inference.h"
#include "model_spec.h"
#include "tokenizer.h"
#include "transformer.h"
#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TERNARY_SYMBOL_ZERO     0u
#define TERNARY_SYMBOL_POSITIVE 1u
#define TERNARY_SYMBOL_NEGATIVE 2u
#define TERNARY_PACK_WIDTH      4u

static const char *g_calibration_docs[] = {
    "A careful systems design balances latency, memory locality, and correctness under constrained hardware budgets.",
    "Educational prose with precise explanations improves retention when each paragraph introduces one concept at a time.",
    "Static descriptor tables reduce runtime churn and make command recording deterministic for high-throughput inference.",
    "for (int i = 0; i < n; ++i) { acc += weights[i] * input[i]; }",
    "VkBufferMemoryBarrier protects only the buffers that actually participate in a read-after-write dependency.",
    "Given three switches and one lamp, determine the state transitions that isolate which switch controls the lamp.",
    "If every valid path must satisfy two independent constraints, enumerate the cases and eliminate contradictions.",
    "A C18 codebase benefits from explicit ownership, manual allocation checks, and small helper functions with narrow scope."
};

#define CALIB_DOC_COUNT ((int)(sizeof(g_calibration_docs) / sizeof(g_calibration_docs[0])))

static float bf16_to_f32_scalar(uint16_t value) {
    union {
        uint32_t u32;
        float f32;
    } bits;
    bits.u32 = ((uint32_t)value) << 16;
    return bits.f32;
}

static transformer_ste_config_t default_ste_config(void) {
    transformer_ste_config_t config;
    config.ste_steps = 3;
    config.learning_rate = 0.03f;
    config.zero_threshold = 0.05f;
    config.momentum = 0.85f;
    config.regularization_strength = 0.01f;
    config.clip_value = 1.0f;
    config.calibration_samples = 4;
    config.kl_weight = 0.05f;
    return config;
}

static size_t ternary_packed_bytes(uint32_t rows, uint32_t cols) {
    size_t packed_cols = ((size_t)cols + (TERNARY_PACK_WIDTH - 1u)) / TERNARY_PACK_WIDTH;
    return (size_t)rows * packed_cols;
}

static uint8_t ternary_symbol_from_value(int8_t value) {
    if (value > 0) return TERNARY_SYMBOL_POSITIVE;
    if (value < 0) return TERNARY_SYMBOL_NEGATIVE;
    return TERNARY_SYMBOL_ZERO;
}

static uint32_t mix_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

static void build_feature_hashed_vector(const int *tokens,
                                        int token_count,
                                        uint32_t cols,
                                        float *out_vector) {
    float norm = 0.0f;

    memset(out_vector, 0, (size_t)cols * sizeof(float));
    if (!tokens || token_count <= 0 || cols == 0) {
        return;
    }

    for (int t = 0; t < token_count; ++t) {
        uint32_t base = mix_u32((uint32_t)tokens[t] ^ ((uint32_t)(t + 1) * 2654435761u));
        for (uint32_t proj = 0; proj < 3u; ++proj) {
            uint32_t state = mix_u32(base ^ ((proj + 1u) * 2246822519u));
            uint32_t idx = state % cols;
            float sign = (state & 1u) ? 1.0f : -1.0f;
            out_vector[idx] += sign / sqrtf((float)(proj + 1u));
        }
        if (t > 0) {
            uint32_t pair_state = mix_u32((uint32_t)tokens[t - 1] * 1315423911u ^ (uint32_t)tokens[t]);
            uint32_t pair_idx = pair_state % cols;
            out_vector[pair_idx] += (pair_state & 2u) ? 0.5f : -0.5f;
        }
    }

    for (uint32_t c = 0; c < cols; ++c) {
        norm += out_vector[c] * out_vector[c];
    }
    if (norm > 1e-12f) {
        float inv_norm = 1.0f / sqrtf(norm);
        for (uint32_t c = 0; c < cols; ++c) {
            out_vector[c] *= inv_norm;
        }
    }
}

static int build_tokenized_prompt_vector(const ternary_calibration_corpus_t *corpus,
                                         const char *text,
                                         uint32_t cols,
                                         float *out_vector) {
    int token_count = 0;
    int *tokens = NULL;
    const int max_tokens = 1024;

    if (!corpus || !corpus->tokenizer || !corpus->model_spec || !text || !out_vector) {
        return -1;
    }

    tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) {
        LOG_ERROR("build_tokenized_prompt_vector: token buffer allocation failed");
        return -1;
    }

    token_count = build_gemma3_prompt(corpus->model_spec, text, tokens, max_tokens);
    if (token_count <= 0) {
        token_count = tokenize(corpus->tokenizer, text, tokens, max_tokens);
    }
    if (token_count <= 0) {
        LOG_WARN("build_tokenized_prompt_vector: tokenization failed for a calibration sample");
        free(tokens);
        return -1;
    }

    build_feature_hashed_vector(tokens, token_count, cols, out_vector);
    free(tokens);
    return 0;
}

static float calibration_value_from_text(const char *text, uint32_t sample_idx, uint32_t dim_idx) {
    uint32_t state = 2166136261u ^ (sample_idx * 16777619u) ^ (dim_idx * 374761393u);
    size_t len = 0;
    float acc = 0.0f;

    if (!text) {
        return 0.0f;
    }

    len = strlen(text);
    if (len == 0) {
        return 0.0f;
    }

    for (size_t k = 0; k < 4; ++k) {
        unsigned char ch = (unsigned char)text[(dim_idx + sample_idx * 13u + (uint32_t)k * 17u) % len];
        state = mix_u32(state ^ (uint32_t)ch ^ ((uint32_t)k << 24));
        acc += ((float)(int)(ch % 31u) - 15.0f) / 15.0f;
    }

    acc += ((float)(state & 1023u) / 511.5f) - 1.0f;
    return acc * 0.5f;
}

static float* build_fallback_vectors(uint32_t cols, int sample_count) {
    float *vectors = NULL;

    if (cols == 0 || sample_count <= 0) {
        return NULL;
    }

    vectors = (float *)malloc((size_t)sample_count * cols * sizeof(float));
    if (!vectors) {
        LOG_ERROR("build_calibration_vectors: allocation failed");
        return NULL;
    }

    for (int s = 0; s < sample_count; ++s) {
        const char *doc = g_calibration_docs[s % CALIB_DOC_COUNT];
        float norm = 0.0f;
        size_t base = (size_t)s * cols;

        for (uint32_t c = 0; c < cols; ++c) {
            float value = calibration_value_from_text(doc, (uint32_t)s, c);
            vectors[base + c] = value;
            norm += value * value;
        }

        if (norm > 1e-12f) {
            float inv_norm = 1.0f / sqrtf(norm);
            for (uint32_t c = 0; c < cols; ++c) {
                vectors[base + c] *= inv_norm;
            }
        }
    }

    return vectors;
}

static float* build_calibration_vectors(uint32_t cols,
                                        int sample_count,
                                        const ternary_calibration_corpus_t *corpus) {
    float *vectors = NULL;
    int used_activation_replay = 0;

    if (cols == 0 || sample_count <= 0) {
        return NULL;
    }

    if (!corpus || !corpus->sample_texts || corpus->sample_count <= 0 ||
        !corpus->tokenizer || !corpus->model_spec) {
        return build_fallback_vectors(cols, sample_count);
    }

    vectors = (float *)malloc((size_t)sample_count * cols * sizeof(float));
    if (!vectors) {
        LOG_ERROR("build_calibration_vectors: allocation failed");
        return NULL;
    }

    for (int s = 0; s < sample_count; ++s) {
        const char *sample_text = corpus->sample_texts[s % corpus->sample_count];
        float *dst = vectors + (size_t)s * cols;
        if (corpus->session && corpus->tensor_name && corpus->tensor_name[0] != '\0') {
            transformer_activation_capture_request_t activation_request;

            memset(&activation_request, 0, sizeof(activation_request));
            activation_request.spec = corpus->model_spec;
            activation_request.text = sample_text;
            activation_request.tensor_name = corpus->tensor_name;
            activation_request.out_vector = dst;
            activation_request.out_dim = cols;

            if (sapphire_collect_tensor_activation(corpus->session,
                                                  corpus->tokenizer,
                                                  &activation_request) == 0) {
                used_activation_replay = 1;
                continue;
            }
        }
        if (build_tokenized_prompt_vector(corpus, sample_text, cols, dst) != 0) {
            free(vectors);
            return build_fallback_vectors(cols, sample_count);
        }
    }

    if (used_activation_replay) {
        LOG_INFO("Built activation-replay calibration vectors: requested=%d available_samples=%d tensor=%s",
                 sample_count,
                 corpus->sample_count,
                 corpus->tensor_name ? corpus->tensor_name : "<unknown>");
    } else {
        LOG_INFO("Built tokenized calibration vectors: requested=%d available_samples=%d",
                 sample_count, corpus->sample_count);
    }
    return vectors;
}

static void quantize_row_ternary(const float *latent_row,
                                 uint32_t cols,
                                 float zero_threshold,
                                 float *out_scale,
                                 int8_t *out_ternary_row) {
    float max_abs = 0.0f;
    float scale = 1.0f;
    float threshold = 0.0f;

    for (uint32_t c = 0; c < cols; ++c) {
        float magnitude = fabsf(latent_row[c]);
        if (magnitude > max_abs) {
            max_abs = magnitude;
        }
    }

    if (max_abs > 1e-12f) {
        scale = max_abs;
    }
    threshold = scale * zero_threshold;

    for (uint32_t c = 0; c < cols; ++c) {
        float value = latent_row[c];
        if (value > threshold) {
            out_ternary_row[c] = 1;
        } else if (value < -threshold) {
            out_ternary_row[c] = -1;
        } else {
            out_ternary_row[c] = 0;
        }
    }

    *out_scale = scale;
}

static float dot_row(const float *row, const float *vec, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
        acc += row[c] * vec[c];
    }
    return acc;
}

static float dot_row_ternary(const int8_t *row, float scale, const float *vec, uint32_t cols) {
    float acc = 0.0f;
    for (uint32_t c = 0; c < cols; ++c) {
        acc += ((float)row[c] * scale) * vec[c];
    }
    return acc;
}

static float ternary_regularizer_grad(float value, int8_t ternary_value) {
    if (ternary_value > 0) return value - 1.0f;
    if (ternary_value < 0) return value + 1.0f;
    return value;
}

typedef struct {
    uint32_t cols;
    const float *calibration_vectors;
    const float *sample_weights;
    int sample_count;
    float learning_rate;
    float momentum;
    float regularization_strength;
    float clip_value;
} ste_row_update_context_t;

static void ste_update_row(float *latent_row,
                           float *velocity_row,
                           const int8_t *ternary_row,
                           float scale,
                           const ste_row_update_context_t *context) {
    float sample_diffs[16];
    float weight_sum = 0.0f;
    uint32_t cols = context->cols;
    int sample_count = context->sample_count;

    if (sample_count > (int)(sizeof(sample_diffs) / sizeof(sample_diffs[0]))) {
        sample_count = (int)(sizeof(sample_diffs) / sizeof(sample_diffs[0]));
    }

    for (int s = 0; s < sample_count; ++s) {
        float w = context->sample_weights ? context->sample_weights[s] : 1.0f;
        if (w < 0.0f) w = 0.0f;
        weight_sum += w;
    }
    if (weight_sum <= 1e-12f) {
        weight_sum = (float)sample_count;
    }

    for (int s = 0; s < sample_count; ++s) {
        const float *vec = context->calibration_vectors + (size_t)s * cols;
        float reference = dot_row(latent_row, vec, cols);
        float proxy = dot_row_ternary(ternary_row, scale, vec, cols);
        sample_diffs[s] = proxy - reference;
    }

    for (uint32_t c = 0; c < cols; ++c) {
        float grad = context->regularization_strength *
                     ternary_regularizer_grad(latent_row[c], ternary_row[c]);

        for (int s = 0; s < sample_count; ++s) {
            const float *vec = context->calibration_vectors + (size_t)s * cols;
            float w = context->sample_weights ? context->sample_weights[s] : 1.0f;
            grad += (2.0f * w / weight_sum) * sample_diffs[s] * vec[c];
        }

        velocity_row[c] = context->momentum * velocity_row[c] - context->learning_rate * grad;
        latent_row[c] += velocity_row[c];
        if (latent_row[c] > context->clip_value) latent_row[c] = context->clip_value;
        if (latent_row[c] < -context->clip_value) latent_row[c] = -context->clip_value;
    }
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

    if (!field_name) {
        return NULL;
    }
    if (layer_idx < 0) {
        if (strcmp(field_name, "embedding_weight") == 0) return &model->embedding_weight;
        if (strcmp(field_name, "lm_head_weight") == 0 || strcmp(field_name, "lm_head") == 0) return &model->lm_head_weight;
        return NULL;
    }

    if (strcmp(field_name, "q_proj_weight") == 0) return &model->layers[layer_idx].q_proj_weight;
    if (strcmp(field_name, "k_proj_weight") == 0) return &model->layers[layer_idx].k_proj_weight;
    if (strcmp(field_name, "v_proj_weight") == 0) return &model->layers[layer_idx].v_proj_weight;
    if (strcmp(field_name, "out_proj_weight") == 0) return &model->layers[layer_idx].out_proj_weight;
    if (strcmp(field_name, "gate_proj_weight") == 0) return &model->layers[layer_idx].gate_proj_weight;
    if (strcmp(field_name, "up_proj_weight") == 0) return &model->layers[layer_idx].up_proj_weight;
    if (strcmp(field_name, "down_proj_weight") == 0) return &model->layers[layer_idx].down_proj_weight;
    return NULL;
}

static tensor_t *build_proxy_tensor_from_ternary(const int8_t *ternary,
                                                 const float *scales,
                                                 uint32_t rows,
                                                 uint32_t cols) {
    const int shape[2] = { (int)rows, (int)cols };
    tensor_t *proxy = NULL;
    float *data = NULL;

    if (!ternary || !scales || rows == 0 || cols == 0) {
        return NULL;
    }

    proxy = tensor_create(2, shape, DTYPE_F32);
    if (!proxy) {
        return NULL;
    }
    data = tensor_data_f32(proxy);
    if (!data) {
        tensor_release(proxy);
        return NULL;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        float scale = scales[r];
        size_t row_base = (size_t)r * cols;
        for (uint32_t c = 0; c < cols; ++c) {
            data[row_base + c] = (float)ternary[row_base + c] * scale;
        }
    }

    return proxy;
}

static int run_prompt_logits(inference_session_t *session,
                             sapphire_tokenizer_t *tokenizer,
                             const model_spec_t *spec,
                             const char *text,
                             float *out_logits,
                             int vocab_size) {
    int *tokens = NULL;
    int token_count = 0;
    const int max_tokens = 1024;

    if (!session || !tokenizer || !spec || !text || !out_logits || vocab_size <= 0) {
        return -1;
    }

    tokens = (int *)malloc((size_t)max_tokens * sizeof(int));
    if (!tokens) {
        return -1;
    }

    token_count = build_gemma3_prompt(spec, text, tokens, max_tokens);
    if (token_count <= 0) {
        token_count = tokenize(tokenizer, text, tokens, max_tokens);
    }
    if (token_count <= 0) {
        free(tokens);
        return -1;
    }

    inference_session_reset(session);
    inference_forward_batch(session, tokens, 0, token_count, out_logits);
    free(tokens);
    return 0;
}

static float compute_logits_kl(const float *reference_logits,
                               const float *proxy_logits,
                               float *reference_probs,
                               float *proxy_probs,
                               int vocab_size) {
    float kl = 0.0f;

    memcpy(reference_probs, reference_logits, (size_t)vocab_size * sizeof(float));
    memcpy(proxy_probs, proxy_logits, (size_t)vocab_size * sizeof(float));
    softmax(reference_probs, vocab_size);
    softmax(proxy_probs, vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        float p = reference_probs[i];
        float q = proxy_probs[i];
        if (p > 1e-12f && q > 1e-12f) {
            kl += p * logf(p / q);
        }
    }
    return kl;
}

typedef struct {
    const int8_t *ternary;
    const float *scales;
    uint32_t rows;
    uint32_t cols;
    const transformer_ste_config_t *config;
    const ternary_calibration_corpus_t *corpus;
    float *out_sample_weights;
    int sample_count;
} distillation_weight_request_t;

typedef struct {
    const ternary_calibration_corpus_t *corpus;
    tensor_t **slot;
    tensor_t *original_tensor;
    tensor_t *proxy_tensor;
    const gemma3_270m_config_t *model_config;
    float *reference_logits;
    float *proxy_logits;
    float *reference_probs;
    float *proxy_probs;
    float kl_weight;
} distillation_runtime_t;

static float compute_sample_distillation_weight(const distillation_runtime_t *runtime,
                                                const char *sample_text) {
    float kl = 0.0f;

    if (!runtime || !sample_text) {
        return 1.0f;
    }

    *runtime->slot = runtime->original_tensor;
    if (run_prompt_logits(runtime->corpus->session,
                          runtime->corpus->tokenizer,
                          runtime->corpus->model_spec,
                          sample_text,
                          runtime->reference_logits,
                          runtime->model_config->vocab_size) != 0) {
        return 1.0f;
    }

    *runtime->slot = runtime->proxy_tensor;
    if (run_prompt_logits(runtime->corpus->session,
                          runtime->corpus->tokenizer,
                          runtime->corpus->model_spec,
                          sample_text,
                          runtime->proxy_logits,
                          runtime->model_config->vocab_size) != 0) {
        *runtime->slot = runtime->original_tensor;
        return 1.0f;
    }
    *runtime->slot = runtime->original_tensor;

    kl = compute_logits_kl(runtime->reference_logits,
                           runtime->proxy_logits,
                           runtime->reference_probs,
                           runtime->proxy_probs,
                           runtime->model_config->vocab_size);
    if (kl < 0.0f) kl = 0.0f;
    if (kl > 8.0f) kl = 8.0f;
    return 1.0f + (runtime->kl_weight * kl);
}

static void compute_distillation_sample_weights(const distillation_weight_request_t *request) {
    distillation_runtime_t runtime;
    llm_model_t *model = NULL;
    tensor_t **slot = NULL;
    tensor_t *original_tensor = NULL;
    tensor_t *proxy_tensor = NULL;
    const gemma3_270m_config_t *model_config = NULL;
    float *reference_logits = NULL;
    float *proxy_logits = NULL;
    float *reference_probs = NULL;
    float *proxy_probs = NULL;
    const transformer_ste_config_t *config = request ? request->config : NULL;
    const ternary_calibration_corpus_t *corpus = request ? request->corpus : NULL;
    float *out_sample_weights = request ? request->out_sample_weights : NULL;
    int sample_count = request ? request->sample_count : 0;

    for (int s = 0; s < sample_count; ++s) {
        out_sample_weights[s] = 1.0f;
    }

    if (!config || config->kl_weight <= 0.0f || !corpus || !corpus->session ||
        !corpus->tokenizer || !corpus->model_spec || !corpus->tensor_name || sample_count <= 0) {
        return;
    }

    model = (llm_model_t *)corpus->model_spec->llm_model;
    model_config = (gemma3_270m_config_t *)corpus->model_spec->variant_config;
    if (!model || !model_config) {
        return;
    }

    slot = resolve_tensor_slot(model, corpus->model_spec, corpus->tensor_name);
    if (!slot || !*slot) {
        return;
    }
    original_tensor = *slot;
    proxy_tensor = build_proxy_tensor_from_ternary(request->ternary,
                                                   request->scales,
                                                   request->rows,
                                                   request->cols);
    if (!proxy_tensor) {
        return;
    }

    reference_logits = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    proxy_logits = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    reference_probs = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    proxy_probs = (float *)malloc((size_t)model_config->vocab_size * sizeof(float));
    if (!reference_logits || !proxy_logits || !reference_probs || !proxy_probs) {
        goto cleanup;
    }

    memset(&runtime, 0, sizeof(runtime));
    runtime.corpus = corpus;
    runtime.slot = slot;
    runtime.original_tensor = original_tensor;
    runtime.proxy_tensor = proxy_tensor;
    runtime.model_config = model_config;
    runtime.reference_logits = reference_logits;
    runtime.proxy_logits = proxy_logits;
    runtime.reference_probs = reference_probs;
    runtime.proxy_probs = proxy_probs;
    runtime.kl_weight = config->kl_weight;

    for (int s = 0; s < sample_count; ++s) {
        const char *sample_text = corpus->sample_texts[s % corpus->sample_count];
        out_sample_weights[s] = compute_sample_distillation_weight(&runtime, sample_text);
    }

    {
        float avg_weight = 0.0f;
        for (int s = 0; s < sample_count; ++s) {
            avg_weight += out_sample_weights[s];
        }
        avg_weight /= (float)sample_count;
        LOG_INFO("Computed KL distillation weights: tensor=%s samples=%d avg_weight=%.4f",
                 corpus->tensor_name,
                 sample_count,
                 (double)avg_weight);
    }

cleanup:
    if (slot) {
        *slot = original_tensor;
    }
    free(reference_logits);
    free(proxy_logits);
    free(reference_probs);
    free(proxy_probs);
    tensor_release(proxy_tensor);
}

typedef struct {
    float *latent;
    float *velocity;
    int8_t *ternary;
    float *scales;
    uint32_t rows;
    uint32_t cols;
    const transformer_ste_config_t *config;
    const float *calibration_vectors;
    const ternary_calibration_corpus_t *corpus;
} ste_calibration_context_t;

static void ste_calibrate_with_samples(const ste_calibration_context_t *context) {
    float sample_weights[16];
    ste_row_update_context_t row_context;
    distillation_weight_request_t distillation_request;

    if (context->config->calibration_samples > (int)(sizeof(sample_weights) / sizeof(sample_weights[0]))) {
        memset(sample_weights, 0, sizeof(sample_weights));
    }

    for (uint32_t r = 0; r < context->rows; ++r) {
        size_t row_base = (size_t)r * context->cols;
        quantize_row_ternary(context->latent + row_base,
                             context->cols,
                             context->config->zero_threshold,
                             &context->scales[r],
                             context->ternary + row_base);
    }

    memset(&distillation_request, 0, sizeof(distillation_request));
    distillation_request.ternary = context->ternary;
    distillation_request.scales = context->scales;
    distillation_request.rows = context->rows;
    distillation_request.cols = context->cols;
    distillation_request.config = context->config;
    distillation_request.corpus = context->corpus;
    distillation_request.out_sample_weights = sample_weights;
    distillation_request.sample_count = context->config->calibration_samples;
    compute_distillation_sample_weights(&distillation_request);

    memset(&row_context, 0, sizeof(row_context));
    row_context.cols = context->cols;
    row_context.calibration_vectors = context->calibration_vectors;
    row_context.sample_weights = sample_weights;
    row_context.sample_count = context->config->calibration_samples;
    row_context.learning_rate = context->config->learning_rate;
    row_context.momentum = context->config->momentum;
    row_context.regularization_strength = context->config->regularization_strength;
    row_context.clip_value = context->config->clip_value;

    for (uint32_t r = 0; r < context->rows; ++r) {
        size_t row_base = (size_t)r * context->cols;
        ste_update_row(context->latent + row_base,
                       context->velocity + row_base,
                       context->ternary + row_base,
                       context->scales[r],
                       &row_context);
    }
}

static int pack_ternary_2bit(const int8_t *ternary,
                             uint32_t rows,
                             uint32_t cols,
                             uint8_t *out_packed,
                             size_t packed_bytes) {
    size_t expected_bytes = ternary_packed_bytes(rows, cols);
    size_t out_idx = 0;

    if (!ternary || !out_packed || packed_bytes != expected_bytes) {
        LOG_ERROR("pack_ternary_2bit: invalid arguments");
        return -1;
    }

    memset(out_packed, 0, packed_bytes);
    for (uint32_t r = 0; r < rows; ++r) {
        size_t row_base = (size_t)r * cols;
        for (uint32_t c = 0; c < cols; c += TERNARY_PACK_WIDTH) {
            uint8_t packed = 0;
            for (uint32_t lane = 0; lane < TERNARY_PACK_WIDTH; ++lane) {
                uint32_t idx = c + lane;
                uint8_t symbol = 0;
                if (idx < cols) {
                    symbol = ternary_symbol_from_value(ternary[row_base + idx]);
                }
                packed |= (uint8_t)(symbol << (lane * 2u));
            }
            out_packed[out_idx++] = packed;
        }
    }

    return 0;
}

void transformer_free_ternary_calibration_result(ternary_calibration_result_t *result) {
    if (!result) {
        return;
    }
    free(result->latent_weights);
    free(result->ternary_weights);
    free(result->packed_weights);
    free(result->scales);
    memset(result, 0, sizeof(*result));
}

int transformer_calibrate_layer_ste(const uint16_t *bf16_weights,
                                    uint32_t rows,
                                    uint32_t cols,
                                    const transformer_ste_config_t *config,
                                    const ternary_calibration_corpus_t *corpus,
                                    ternary_calibration_result_t *out_result) {
    transformer_ste_config_t effective_config;
    ste_calibration_context_t calibration_context;
    float *velocity = NULL;
    float *calibration_vectors = NULL;
    size_t weight_count = 0;
    size_t packed_bytes = 0;

    if (!bf16_weights || !out_result || rows == 0 || cols == 0) {
        LOG_ERROR("transformer_calibrate_layer_ste: invalid arguments");
        return -1;
    }

    effective_config = config ? *config : default_ste_config();
    if (effective_config.ste_steps <= 0) effective_config.ste_steps = 1;
    if (effective_config.learning_rate <= 0.0f) effective_config.learning_rate = 0.05f;
    if (effective_config.zero_threshold < 0.0f) effective_config.zero_threshold = 0.05f;
    if (effective_config.momentum < 0.0f || effective_config.momentum >= 1.0f) effective_config.momentum = 0.85f;
    if (effective_config.regularization_strength < 0.0f) effective_config.regularization_strength = 0.01f;
    if (effective_config.clip_value <= 0.0f) effective_config.clip_value = 1.0f;
    if (effective_config.calibration_samples <= 0) effective_config.calibration_samples = 4;
    if (effective_config.calibration_samples > 16) effective_config.calibration_samples = 16;
    if (effective_config.kl_weight < 0.0f) effective_config.kl_weight = 0.0f;

    memset(out_result, 0, sizeof(*out_result));
    weight_count = (size_t)rows * cols;
    packed_bytes = ternary_packed_bytes(rows, cols);

    out_result->latent_weights = (float *)malloc(weight_count * sizeof(float));
    out_result->ternary_weights = (int8_t *)malloc(weight_count * sizeof(int8_t));
    out_result->packed_weights = (uint8_t *)malloc(packed_bytes);
    out_result->scales = (float *)malloc((size_t)rows * sizeof(float));
    if (!out_result->latent_weights || !out_result->ternary_weights ||
        !out_result->packed_weights || !out_result->scales) {
        LOG_ERROR("transformer_calibrate_layer_ste: allocation failed");
        transformer_free_ternary_calibration_result(out_result);
        return -1;
    }
    velocity = (float *)calloc(weight_count, sizeof(float));
    calibration_vectors = build_calibration_vectors(cols,
                                                    effective_config.calibration_samples,
                                                    corpus);
    if (!velocity || !calibration_vectors) {
        LOG_ERROR("transformer_calibrate_layer_ste: calibration buffer allocation failed");
        free(velocity);
        free(calibration_vectors);
        transformer_free_ternary_calibration_result(out_result);
        return -1;
    }

    for (size_t i = 0; i < weight_count; ++i) {
        out_result->latent_weights[i] = bf16_to_f32_scalar(bf16_weights[i]);
        if (out_result->latent_weights[i] > effective_config.clip_value) {
            out_result->latent_weights[i] = effective_config.clip_value;
        }
        if (out_result->latent_weights[i] < -effective_config.clip_value) {
            out_result->latent_weights[i] = -effective_config.clip_value;
        }
    }

    memset(&calibration_context, 0, sizeof(calibration_context));
    calibration_context.latent = out_result->latent_weights;
    calibration_context.velocity = velocity;
    calibration_context.ternary = out_result->ternary_weights;
    calibration_context.scales = out_result->scales;
    calibration_context.rows = rows;
    calibration_context.cols = cols;
    calibration_context.config = &effective_config;
    calibration_context.calibration_vectors = calibration_vectors;
    calibration_context.corpus = corpus;

    for (int step = 0; step < effective_config.ste_steps; ++step) {
        ste_calibrate_with_samples(&calibration_context);
    }

    for (uint32_t r = 0; r < rows; ++r) {
        size_t row_base = (size_t)r * cols;
        quantize_row_ternary(out_result->latent_weights + row_base,
                             cols,
                             effective_config.zero_threshold,
                             &out_result->scales[r],
                             out_result->ternary_weights + row_base);
    }

    if (pack_ternary_2bit(out_result->ternary_weights, rows, cols,
                          out_result->packed_weights, packed_bytes) != 0) {
        free(velocity);
        free(calibration_vectors);
        transformer_free_ternary_calibration_result(out_result);
        return -1;
    }

    out_result->weight_count = weight_count;
    out_result->packed_weight_bytes = packed_bytes;
    out_result->rows = rows;
    out_result->cols = cols;

    free(velocity);
    free(calibration_vectors);

    LOG_INFO("Calibrated ternary layer: rows=%u cols=%u steps=%d samples=%d kl_weight=%.4f packed=%zuB",
             rows,
             cols,
             effective_config.ste_steps,
             effective_config.calibration_samples,
             (double)effective_config.kl_weight,
             packed_bytes);
    return 0;
}
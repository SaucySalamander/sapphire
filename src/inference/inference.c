/**
 * @file inference.c
 * @brief Inference session and full forward pass implementation.
 */

#include "../include/inference.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  /* clock_gettime for high-resolution wall-clock timing */

#include "../include/attention.h"
#include "../include/gemma3_270m_config.h"
#include "../include/ggml_model.h"
#include "../include/kv_cache.h"
#include "../include/kv_cache_state.h"
#include "../include/layer_config_loader.h"
#include "../include/log.h"
#include "../include/model_reader.h"
#include "../include/backend_vulkan.h"
#include "../include/rope.h"
#include "../include/tensor.h"
#include "../include/kernels.h"
#include "../include/transformer.h"
#include "../include/utils.h"
#include "../include/file_reader.h"
#include "tokenizer.h"

/* Note: build_gemma3_prompt moved to tokenizer module (see include/tokenizer.h)
    to centralize tokenization / prompt construction logic. The function intelligently
    detects whether the model is IT or base and calls the appropriate builder. */

/**
 * @brief Load tokenizer from model directory or reuse cached tokenizer from spec.
 * Caches tokenizer in spec for reuse across multiple contexts.
 * 
 * @returns 0 on success (tokenizer loaded or cached), -1 on error
 */
static int load_or_reuse_tokenizer(model_spec_t* spec, const char* model_dir) {
    if (!spec || !model_dir) return -1;
    
    // Tokenizer may already be cached in spec (from previous context)
    if (spec->tokenizer_handle) {
        LOG_INFO("Reusing cached tokenizer for model %s", spec->model_id);
        return 0;
    }
    
    LOG_INFO("Loading tokenizer from %s", model_dir);
    sapphire_tokenizer_t* tk = tokenizer_load(model_dir);
    if (!tk) {
        LOG_ERROR("Failed to load tokenizer from %s", model_dir);
        return -1;
    }
    
    // Cache tokenizer in spec for reuse by future contexts
    spec->tokenizer_handle = (void*)tk;
    return 0;
}

/**
 * @brief Initialize inference context
 *
 * Creates an inference context with the specified parameters:
 * - Loads model weights from disk
 * - Detects and initializes the hardware backend (CPU or Vulkan)
 * - Allocates backend-specific buffers (scratch, KV cache, RoPE frequencies)
 * - Initializes tokenizer
 */
inference_context_t* create_inference_context(float temperature, int max_tokens, int context_len, const char* model_name) {
    // Allow max_tokens == 0 for diagnostic-only initialization runs
    if (!model_name || temperature < 0.0f || max_tokens < 0 || context_len <= 0) return NULL;

    // Get the model specification for the requested model
    model_spec_t* spec = get_model_spec(model_name);
    if (!spec) {
        LOG_ERROR("Failed to get spec for model: %s", model_name);
        return NULL;
    }

    // Construct the model directory path: ./models/{model_name}
    // construct_safe_path allocates exact size needed and validates all components
    char *model_dir = construct_safe_path("./models", model_name, NULL);
    if (!model_dir) {
        LOG_ERROR("Failed to construct model directory path");
        return NULL;
    }

    // Allocate model structure
    llm_model_t* model = (llm_model_t*)malloc(sizeof(llm_model_t));
    if (!model) {
        LOG_ERROR("Failed to allocate model structure");
        free(model_dir);
        return NULL;
    }
    memset(model, 0, sizeof(llm_model_t));

    spec->llm_model = model;

    // Trigger the loader hooks to populate model and config
    // The loader function logs its own detailed errors; we just check the return code
    if (spec->loader_hooks && spec->loader_hooks->populate_from_files) {
        int rc = spec->loader_hooks->populate_from_files(model_dir, spec);
        if (rc != 0) {
            LOG_ERROR("Failed to populate model from files");
            free(model);
            free(model_dir);
            return NULL;
        }
    } else {
        LOG_ERROR("Model spec has no loader hooks");
        free(model);
        free(model_dir);
        return NULL;
    }

    // Run postprocessing hook if available
    if (spec->loader_hooks && spec->loader_hooks->postprocess_model) {
        spec->loader_hooks->postprocess_model(spec);
    }

    inference_context_t* ctx = (inference_context_t*)malloc(sizeof(inference_context_t));
    if (!ctx) {
        LOG_ERROR("Failed to allocate inference context");
        llm_model_destroy(model);
        free(model_dir);
        return NULL;
    }
    memset(ctx, 0, sizeof(inference_context_t));

    ctx->spec = spec;
    ctx->context_len = context_len;
    ctx->max_tokens = max_tokens;
    ctx->temperature = temperature;

    // Allocate logits buffer
    // NOTE: embedding_weight shape[0] may exceed config->vocab_size (e.g. alignment
    // padding row in sharded safetensors).  We must allocate enough for all rows that
    // the lm_head gemv will write, otherwise the kernel pool overflows into adjacent
    // heap memory (heap-buffer-overflow, manifests as "double free or corruption").
    const gemma3_270m_config_t* config = (const gemma3_270m_config_t*)spec->variant_config;
    int logits_rows = config->vocab_size;
    {
        const llm_model_t *lm = (const llm_model_t *)spec->llm_model;
        if (lm && lm->embedding_weight && tensor_ndim(lm->embedding_weight) == 2) {
            const int *emb_shape = tensor_shape(lm->embedding_weight);
            if (emb_shape && emb_shape[0] > logits_rows) {
                LOG_DEBUG("logits alloc: emb rows %d > vocab_size %d, "
                          "using emb rows to prevent OOB write",
                          emb_shape[0], logits_rows);
                logits_rows = emb_shape[0];
            }
        }
    }
    ctx->logits = (float*)malloc((size_t)logits_rows * sizeof(float));
    if (!ctx->logits) {
        LOG_ERROR("Failed to allocate logits buffer");
        free(ctx);
        free(model_dir);
        return NULL;
    }

    ctx->conversation_tokens = (int*)malloc((size_t)context_len * sizeof(int));
    if (!ctx->conversation_tokens) {
        LOG_ERROR("Failed to allocate conversation token buffer");
        free(ctx->logits);
        free(ctx);
        free(model_dir);
        return NULL;
    }
    ctx->conversation_len = 0;

    // Create a single inference session for convenience (1:many supported via model_spec)
    ctx->session = inference_session_create(spec, context_len);
    if (!ctx->session) {
        LOG_ERROR("Failed to create inference session");
        free(ctx->conversation_tokens);
        free(ctx->logits);
        free(ctx);
        free(model_dir);
        return NULL;
    }

    // Load tokenizer from model directory or spec
    // Spec caches tokenizer across multiple context creations for efficiency
    if (load_or_reuse_tokenizer(spec, model_dir) != 0) {
        LOG_ERROR("Failed to initialize tokenizer");
        // Clean up allocated resources and return error
        if (ctx->session) destroy_inference_session(ctx->session);
        if (ctx->conversation_tokens) free(ctx->conversation_tokens);
        if (ctx->logits) free(ctx->logits);
        free(ctx);
        if (spec->llm_model) {
            llm_model_destroy_ex((const struct model_spec*)spec);
            spec->llm_model = NULL;
        } else {
            free(model);
        }
        free(model_dir);
        return NULL;
    }
    
    free(model_dir);  // We're done with the path
    ctx->tokenizer = (sapphire_tokenizer_t*)spec->tokenizer_handle;
    return ctx;
}

/**
 * @brief Destroy inference context
 */
void destroy_inference_context(inference_context_t* ctx) {
    if (!ctx) return;

    if (ctx->session) {
        destroy_inference_session(ctx->session);
    }

    if (ctx->logits) {
        free(ctx->logits);
    }

    if (ctx->conversation_tokens) {
        free(ctx->conversation_tokens);
    }

    // Tokenizer ownership: prefer spec-owned tokenizer. Only free ctx->tokenizer
    // if the spec did not take ownership.
    if (ctx->tokenizer) {
        if (ctx->spec && ctx->spec->tokenizer_handle) {
            // tokenizer is owned by spec; it will be freed below.
        } else {
            tokenizer_free(ctx->tokenizer);
        }
    }

    // Clean up model-owned resources (tensors and any open file handles).
    // The safetensors file handle is owned by the `llm_model_t` and is closed
    // inside `llm_model_destroy()`, so destroy the model here if present.
    if (ctx->spec && ctx->spec->llm_model) {
        llm_model_destroy_ex((const struct model_spec*)ctx->spec);
        ctx->spec->llm_model = NULL;
    }
    if (ctx->spec && ctx->spec->tokenizer_handle) {
        tokenizer_free((sapphire_tokenizer_t*)ctx->spec->tokenizer_handle);
        ctx->spec->tokenizer_handle = NULL;
    }

    free(ctx);
}

/**
 * Print one decoded token to stdout, replacing the SentencePiece
 * U+2581 (▁) prefix byte sequence (0xE2 0x96 0x81) with a space.
 */
static void print_decoded_token_chars(const char *token_str) {
    if (!token_str) return;
    int j = 0;
    if ((unsigned char)token_str[0] == 0xE2 &&
        (unsigned char)token_str[1] == 0x96 &&
        (unsigned char)token_str[2] == 0x81) {
        printf(" ");
        j = 3;
    }
    for (; token_str[j] != '\0'; j++) {
        printf("%c", token_str[j]);
    }
    fflush(stdout);
}

/**
 * CPU decode path: run forward pass, optionally scale logits by temperature,
 * run logit collapse diagnostic, optionally log first-token debug info, then
 * sample and return the next token.
 *
 * @param session        Active inference session
 * @param ctx            Inference context (holds logits, temperature)
 * @param last_token     Current token id fed as input
 * @param cur_pos        Current sequence position
 * @param generated_count Number of tokens generated so far
 * @return Sampled next token id
 */
static int cpu_forward_and_sample(inference_session_t *session,
                                  inference_context_t  *ctx,
                                  int last_token, int cur_pos,
                                  int generated_count) {
    const gemma3_270m_config_t *config =
        (const gemma3_270m_config_t *)ctx->spec->variant_config;

    inference_forward(session, last_token, cur_pos, ctx->logits);

    if (generated_count == 0 && getenv("SAPPHIRE_DEBUG_LOGITS")) {
        LOG_DEBUG("generated_count=%d, cur_pos=%d", generated_count, cur_pos);
    }

    if (ctx->temperature > 0.0f) {
        vec_scale(ctx->logits, 1.0f / ctx->temperature, config->vocab_size);
    }

    /* Logit collapse diagnostic */
    if (generated_count == 0 || generated_count % 10 == 0) {
        float min_l, max_l, rms_l;
        vec_stats(ctx->logits, config->vocab_size, &min_l, &max_l, &rms_l);
        if (rms_l < 0.5f) {
            LOG_WARN("Logit RMS Low (%.4f) - Possible Probability Collapse", rms_l);
        }
    }

    /* Debug: print argmax logit for first generated token */
    if (getenv("SAPPHIRE_DEBUG_LOGITS") && generated_count == 0) {
        LOG_DEBUG("token_106=%.6f token_818=%.6f token_236776=%.6f",
                  ctx->logits[106], ctx->logits[818], ctx->logits[236776]);
        int max_idx = 0;
        float max_logit = ctx->logits[0];
        for (int i = 1; i < config->vocab_size; ++i) {
            if (ctx->logits[i] > max_logit) {
                max_logit = ctx->logits[i];
                max_idx = i;
            }
        }
        LOG_DEBUG("argmax=token_%d with logit=%.6f", max_idx, max_logit);
    }

    return sample_temperature(ctx->logits, config->vocab_size, ctx->temperature);
}

static int is_instruction_tuned_model(const model_spec_t *spec) {
    if (!spec || !spec->model_id) return 0;
    return (strstr(spec->model_id, "-it") != NULL);
}

typedef struct {
    int *turn_tokens;
    int prompt_len;
    int total_tokens;
    int gen_start_idx;
    int last_token;
    int generated_count;
    int is_it_model;
} inference_run_state_t;

static int prepare_turn_and_prefill(inference_context_t *ctx,
                                    inference_session_t *session,
                                    const char *prompt,
                                    inference_run_state_t *st) {
    st->turn_tokens = malloc((size_t)ctx->context_len * sizeof(int));
    if (!st->turn_tokens) {
        LOG_ERROR("Failed to allocate turn token buffer");
        return -1;
    }

    if (!ctx->conversation_tokens) {
        LOG_ERROR("Conversation token buffer is not initialized");
        free(st->turn_tokens);
        st->turn_tokens = NULL;
        return -1;
    }

    st->prompt_len = build_gemma3_prompt(ctx->spec, prompt, st->turn_tokens, ctx->context_len);
    if (st->prompt_len <= 0) {
        LOG_ERROR("Failed to build prompt");
        free(st->turn_tokens);
        st->turn_tokens = NULL;
        return -1;
    }

    if (st->is_it_model && ctx->conversation_len > 0) {
        int leading_bos = 0;
        while (leading_bos < st->prompt_len && st->turn_tokens[leading_bos] == 2) {
            leading_bos++;
        }
        if (leading_bos > 0) {
            memmove(st->turn_tokens,
                    st->turn_tokens + leading_bos,
                    (size_t)(st->prompt_len - leading_bos) * sizeof(int));
            st->prompt_len -= leading_bos;
        }

        if (st->prompt_len > 0 && st->turn_tokens[0] == 105) {
            if (ctx->conversation_tokens[ctx->conversation_len - 1] != 107) {
                if (st->prompt_len >= ctx->context_len) {
                    LOG_ERROR("Not enough room to normalize continuation boundary");
                    free(st->turn_tokens);
                    st->turn_tokens = NULL;
                    return -1;
                }
                memmove(st->turn_tokens + 1,
                        st->turn_tokens,
                        (size_t)st->prompt_len * sizeof(int));
                st->turn_tokens[0] = 107;
                st->prompt_len += 1;
            }
        }
    }

    if (st->prompt_len <= 0) {
        LOG_ERROR("Prompt became empty after continuation normalization");
        free(st->turn_tokens);
        st->turn_tokens = NULL;
        return -1;
    }

    if (!st->is_it_model) {
        ctx->conversation_len = 0;
        st->total_tokens = 0;
        st->gen_start_idx = 0;
    }

    if (ctx->conversation_len + st->prompt_len >= ctx->context_len) {
        LOG_WARN("Context full (existing=%d, new=%d, max=%d); resetting session context",
                 ctx->conversation_len, st->prompt_len, ctx->context_len);
        inference_session_reset(session);
        ctx->conversation_len = 0;
        st->total_tokens = 0;
        st->gen_start_idx = 0;
    }

    if (ctx->conversation_len + st->prompt_len >= ctx->context_len) {
        LOG_ERROR("Prompt too long for available context after reset");
        free(st->turn_tokens);
        st->turn_tokens = NULL;
        return -1;
    }

    memcpy(ctx->conversation_tokens + ctx->conversation_len,
           st->turn_tokens,
           (size_t)st->prompt_len * sizeof(int));
    ctx->conversation_len += st->prompt_len;
    st->total_tokens = ctx->conversation_len;

    inference_session_reset(session);

    int serial_prefill = (getenv("SAPPHIRE_SERIAL_PREFILL") != NULL);
    int p_idx = 0;
    while (p_idx < ctx->conversation_len - 1) {
        int b_size = (ctx->conversation_len - 1) - p_idx;
        if (b_size > 32) b_size = 32;
        if (serial_prefill) b_size = 1;
        inference_forward_batch(session, ctx->conversation_tokens + p_idx, p_idx, b_size, NULL);
        p_idx += b_size;
    }

    st->last_token = ctx->conversation_tokens[ctx->conversation_len - 1];

    free(st->turn_tokens);
    st->turn_tokens = NULL;
    return 0;
}

static int run_generation_loop(inference_context_t *ctx,
                               inference_session_t *session,
                               inference_run_state_t *st) {
    while (st->total_tokens < ctx->context_len && st->generated_count < ctx->max_tokens) {
        int cur_pos = st->total_tokens - 1;
        int next_token = -1;

        if (session->backend &&
            session->backend->type == SAPPHIRE_BACKEND_TYPE_VULKAN &&
            session->backend->forward_select_batch) {
            int rc_sel = session->backend->forward_select_batch(
                session, &st->last_token, cur_pos, 1, &next_token);
            if (rc_sel != 0) {
                LOG_ERROR("Vulkan GPU token selection failed at pos=%d", cur_pos);
                return -1;
            }
        } else {
            next_token = cpu_forward_and_sample(session, ctx,
                                                st->last_token, cur_pos,
                                                st->generated_count);
        }

        if (st->total_tokens >= ctx->context_len) {
            LOG_WARN("Reached maximum context length during generation");
            break;
        }

        ctx->conversation_tokens[st->total_tokens++] = next_token;
        ctx->conversation_len = st->total_tokens;

        if (next_token == 1 || (st->is_it_model && next_token == 106)) break;

        print_decoded_token_chars(decode(ctx->tokenizer, next_token));

        st->last_token = next_token;
        st->generated_count++;
    }

    return 0;
}

static void append_assistant_terminator(inference_context_t *ctx,
                                        const inference_run_state_t *st) {
    if (st->is_it_model && ctx->conversation_len > 0 && ctx->conversation_tokens[ctx->conversation_len - 1] != 106) {
        if (ctx->conversation_len < ctx->context_len) {
            ctx->conversation_tokens[ctx->conversation_len++] = 106;
        } else {
            LOG_WARN("Context full; unable to append assistant turn terminator");
        }
    }
}

static void write_output_text(inference_context_t *ctx,
                              const inference_run_state_t *st,
                              char *output,
                              int output_size) {
    if (st->generated_count > 0) {
        detokenize(ctx->tokenizer,
                   ctx->conversation_tokens + st->gen_start_idx + st->prompt_len,
                   st->generated_count,
                   output,
                   output_size);
    } else if (output_size > 0) {
        output[0] = '\0';
    }
}

int perform_inference(inference_context_t* ctx, const char* prompt, char* output, int output_size) {
    if (!ctx || !prompt || !output) return -1;

    /* Start high-resolution wall-clock timer for this inference call */
    struct timespec __perf_start_ts;
    clock_gettime(CLOCK_MONOTONIC, &__perf_start_ts);

    inference_session_t* session = ctx->session;
    inference_run_state_t st = {
        .turn_tokens = NULL,
        .prompt_len = 0,
        .total_tokens = ctx->conversation_len,
        .gen_start_idx = ctx->conversation_len,
        .last_token = 0,
        .generated_count = 0,
        .is_it_model = is_instruction_tuned_model(ctx->spec)
    };

    if (prepare_turn_and_prefill(ctx, session, prompt, &st) != 0) {
        return -1;
    }

    LOG_INFO("Response generation starting");

    if (run_generation_loop(ctx, session, &st) != 0) {
        return -1;
    }

    append_assistant_terminator(ctx, &st);

    write_output_text(ctx, &st, output, output_size);

    /* Stop timer and record elapsed seconds in the context for callers to inspect */
    struct timespec __perf_end_ts;
    clock_gettime(CLOCK_MONOTONIC, &__perf_end_ts);
    double __elapsed = (__perf_end_ts.tv_sec - __perf_start_ts.tv_sec) +
                       (__perf_end_ts.tv_nsec - __perf_start_ts.tv_nsec) / 1e9;
    ctx->last_inference_time = __elapsed;

    LOG_INFO("perform_inference: elapsed=%.3f sec (backend=%s)", __elapsed,
             (session && session->backend) ? session->backend->name : "unknown");

    LOG_DEBUG("Generated %d tokens", st.generated_count);

    return 0;
}

int inference_session_save_state(inference_session_t *session, const char *path) {
    if (!session || !path) return -1;

    if (session->backend && session->backend->type == SAPPHIRE_BACKEND_TYPE_VULKAN) {
        return backend_vulkan_save_state(session, path);
    }

    if (!session->kv_cache) {
        LOG_ERROR("Session save failed: KV cache is unavailable");
        return -1;
    }

    return kv_cache_save_state(session->kv_cache, path);
}

int inference_session_load_state(inference_session_t *session, const char *path) {
    if (!session || !path) return -1;

    if (session->backend && session->backend->type == SAPPHIRE_BACKEND_TYPE_VULKAN) {
        return backend_vulkan_load_state(session, path);
    }

    if (!session->kv_cache) {
        LOG_ERROR("Session load failed: KV cache is unavailable");
        return -1;
    }

    return kv_cache_load_state(session->kv_cache, path);
}

int inference_context_save_state(inference_context_t *ctx, const char *path) {
    if (!ctx || !ctx->session || !path) return -1;
    if (inference_session_save_state(ctx->session, path) != 0) {
        return -1;
    }

    if (ctx->conversation_len < 0 || ctx->conversation_len > ctx->context_len) {
        LOG_ERROR("Context save failed: invalid conversation length %d (capacity=%d)",
                  ctx->conversation_len,
                  ctx->context_len);
        return -1;
    }

    if (kv_state_append_transcript(path,
                                   ctx->conversation_tokens,
                                   (uint32_t)ctx->conversation_len) != 0) {
        LOG_ERROR("Context save failed: transcript append failed (%s)", path);
        return -1;
    }

    return 0;
}

int inference_context_load_state(inference_context_t *ctx, const char *path) {
    if (!ctx || !ctx->session || !path) return -1;
    int rc = inference_session_load_state(ctx->session, path);
    if (rc != 0) return rc;

    uint32_t loaded_tokens = 0u;
    rc = kv_state_read_transcript(path,
                                  ctx->conversation_tokens,
                                  (uint32_t)ctx->context_len,
                                  &loaded_tokens);
    if (rc == 0) {
        ctx->conversation_len = (int)loaded_tokens;
    } else if (rc == 1) {
        ctx->conversation_len = 0;
        LOG_WARN("Loaded legacy KV state without transcript section: %s", path);
    } else {
        LOG_ERROR("Context load failed: transcript read failed (%s)", path);
        return -1;
    }

    return 0;
}

// Forward declarations for functions used by CPU backend


/**
 * Create an inference session.
 *
 * Creates a new session and delegates all backend-specific initialization
 * to the selected backend (CPU or Vulkan).
 *
 * @param spec Model specification with loaded model and config
 * @param max_context_len Maximum sequence length for KV cache
 * @return Allocated session with backend initialized, or NULL on failure
 */
inference_session_t* inference_session_create(model_spec_t* spec, int max_context_len) {
    if (!spec) {
        LOG_ERROR("inference_session_create requires model");
        return NULL;
    }
    LOG_DEBUG("Creating inference session for model: %s", spec->model_id);

    gemma3_270m_config_t* config = (gemma3_270m_config_t*)spec->variant_config;
    if (!config) {
        LOG_ERROR("Model config is NULL in inference_session_create");
        return NULL;
    }

    // Allocate the session structure
    inference_session_t* session = (inference_session_t*)malloc(sizeof(inference_session_t));
    if (!session) {
        LOG_ERROR("Failed to allocate inference session");
        return NULL;
    }
    memset(session, 0, sizeof(inference_session_t));

    session->model_spec = spec;
    session->num_layers = config->num_hidden_layers;
    session->layer_configs = (sapphire_layer_config_t*)malloc(session->num_layers * sizeof(sapphire_layer_config_t));
    if (!session->layer_configs) {
        LOG_ERROR("Failed to allocate layer configs array");
        free(session);
        return NULL;
    }

    // Load layer configurations (dispatch routing)
    if (layer_config_load_from_spec(spec, session->num_layers, session->layer_configs) != 0) {
        LOG_ERROR("Failed to load layer configurations");
        free(session->layer_configs);
        free(session);
        return NULL;
    }

    // Detect and select backend
    sapphire_backend_type_t backend_type = backend_detect();
    session->backend = backend_get(backend_type);
    if (!session->backend) {
        LOG_ERROR("Failed to get backend implementation for type %d", (int)backend_type);
        free(session->layer_configs);
        free(session);
        return NULL;
    }

    // Initialize backend-specific session data
    if (session->backend->session_init(session, spec, max_context_len) != 0) {
        LOG_ERROR("Backend initialization failed for type %d", (int)backend_type);
        free(session->layer_configs);
        free(session);
        return NULL;
    }

    LOG_INFO("Inference session created with %d layers, context_len=%d, backend=%s",
             config->num_hidden_layers, max_context_len, session->backend->name);

    return session;
}

/**
 * Reset KV caches for a new sequence.
 *
 * Delegates to backend->reset() to clear any sequence-specific state.
 */
void inference_session_reset(inference_session_t* session) {
    if (!session || !session->backend) return;

    session->backend->reset(session);
}

/**
 * Single-token forward pass for autoregressive generation.
 *
 * Implements the full transformer forward pipeline:
 * 1. Token embedding lookup (row slice from embedding matrix)
 * 2. Transformer layer stack (18 layers for Gemma 3) with 5:1 interleave
 *    - RMSNorm pre-attention
 *    - Multi-head self-attention with GQA (16 query, 4 KV heads)
 *    - Residual connection
 *    - RMSNorm pre-FFN
 *    - Feed-forward with GeGLU gating
 *    - Residual connection
 * 3. Final RMSNorm
 * 4. LM head (weight-tied with embeddings)
 *
 * Performance target: ~2.67ms/token at -O3 with AVX2
 */
void inference_forward(inference_session_t* session, int token_id, int token_pos, float* logits) {
    inference_forward_batch(session, &token_id, token_pos, 1, logits);
}

/**
 * Execute forward batch pass via the selected backend.
 *
 * Delegates all computation to the backend's forward_batch implementation.
 * The backend handles layer-by-layer execution (CPU) or GPU dispatch (Vulkan).
 */
void inference_forward_batch(inference_session_t* session, const int* token_ids, int start_pos, int batch_size, float* logits) {
    if (!session || !session->backend) {
        LOG_ERROR("inference_forward_batch requires session with backend");
        return;
    }

    session->backend->forward_batch(session, token_ids, start_pos, batch_size, logits);
}

/**
 * Free inference session.
 *
 * Delegates cleanup to the backend and frees shared metadata.
 */
void destroy_inference_session(inference_session_t* session) {
    if (!session) return;

    if (session->backend) {
        session->backend->session_destroy(session);
    }

    if (session->layer_configs) {
        free(session->layer_configs);
    }

    free(session);
}

/* build_gemma3_prompt moved to tokenizer module (include/tokenizer.h).
   The tokenizer now owns prompt construction and tokenization helpers.
   The smart selector function detects IT vs base model and calls the right builder. */
/**
 * @file activation_tape.c
 * @brief Single-pass activation recorder and disk-backed STE replay.
 *
 * Recording:  one forward pass per sample → all projection-input activations
 *             are written to a flat binary tape.
 * Playback:   full tape is mmap'd (MAP_NORESERVE); callers do random access
 *             by tensor name and sample index.
 */

#include "activation_tape.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "calibration_corpus.h"
#include "backend_vulkan.h"
#include "gemma3_270m_config.h"
#include "inference.h"
#include "log.h"
#include "model_spec.h"
#include "ternary_hessian_oracle.h"
#include "ternary_io.h"
#include "tokenizer.h"
#include "transformer.h"

/* -------------------------------------------------------------------------
 * Internal constants and types
 * -------------------------------------------------------------------------*/

/** Number of unique capture targets per layer. */
#define TAPE_N_UNIQUE 4

/** Number of weight types per layer (including aliased ones). */
#define TAPE_N_WT 7

/** Per-unique-target staging buffer. Indexed as layer * TAPE_N_UNIQUE + (target-1). */
typedef struct {
    float           *buffer;      /**< n_samples × vector_dim floats (pre-allocated). */
    uint32_t         vector_dim;
    int              layer_idx;
    capture_target_t target;
} tape_staging_t;

/** Static weight-type table. All 7 entries per layer for the manifest. */
typedef struct {
    const char      *suffix;       /**< HF tensor name suffix after "layers.N.". */
    capture_target_t target;       /**< Capture target for this weight type. */
    int              alias_idx;    /**< -1 = primary; else TAPE_WT index of primary. */
} tape_wt_t;

static const tape_wt_t TAPE_WT[TAPE_N_WT] = {
    {"self_attn.q_proj.weight",  CAPTURE_TARGET_QKV_INPUT,  -1},  /* 0 – primary */
    {"self_attn.k_proj.weight",  CAPTURE_TARGET_QKV_INPUT,   0},  /* 1 – alias Q */
    {"self_attn.v_proj.weight",  CAPTURE_TARGET_QKV_INPUT,   0},  /* 2 – alias Q */
    {"self_attn.o_proj.weight",  CAPTURE_TARGET_OUT_INPUT,  -1},  /* 3 – primary */
    {"mlp.gate_proj.weight",     CAPTURE_TARGET_FFN_INPUT,  -1},  /* 4 – primary */
    {"mlp.up_proj.weight",       CAPTURE_TARGET_FFN_INPUT,   4},  /* 5 – alias gate */
    {"mlp.down_proj.weight",     CAPTURE_TARGET_DOWN_INPUT, -1},  /* 6 – primary */
};

/** Shared context for manifest building and file writing. */
typedef struct {
    int                        n_unique;
    int                        n_layers;
    int                        n_samples;
    const gemma3_270m_config_t *cfg;
    const char                 *prefix;
} tape_manifest_ctx_t;

typedef struct {
    tape_file_header_t header;
    tape_manifest_entry_t *manifest;
    int entry_count;
    uint32_t file_crc32;
} tape_write_plan_t;

/** Opaque tape handle. */
struct activation_tape_t {
    int                          fd;
    void                        *mmap_ptr;
    size_t                       mmap_size;
    tape_file_header_t           header;
    const tape_manifest_entry_t *manifest;  /**< Points into mmap. */
    uint32_t                     entry_count;
};

static volatile sig_atomic_t g_activation_tape_stop_requested = 0;

void activation_tape_request_stop(void)
{
    g_activation_tape_stop_requested = 1;
}

void activation_tape_clear_stop_request(void)
{
    g_activation_tape_stop_requested = 0;
}

int activation_tape_stop_requested(void)
{
    return g_activation_tape_stop_requested != 0;
}

/* -------------------------------------------------------------------------
 * Small helpers
 * -------------------------------------------------------------------------*/

static uint32_t dim_for_target(capture_target_t t, const gemma3_270m_config_t *c)
{
    switch (t) {
    case CAPTURE_TARGET_QKV_INPUT:  return (uint32_t)c->hidden_size;
    case CAPTURE_TARGET_OUT_INPUT:  return (uint32_t)(c->num_attention_heads * c->head_dim);
    case CAPTURE_TARGET_FFN_INPUT:  return (uint32_t)c->hidden_size;
    case CAPTURE_TARGET_DOWN_INPUT: return (uint32_t)c->intermediate_size;
    default:                        return 0u;
    }
}

static int staging_idx_of(int layer, capture_target_t target)
{
    return layer * TAPE_N_UNIQUE + ((int)target - 1);
}

static const char *tape_tensor_prefix(const model_spec_t *ms)
{
    if (!ms || !ms->tensor_map) return "model.layers.";
    for (size_t i = 0; ms->tensor_map[i].hf_name; ++i) {
        if (strstr(ms->tensor_map[i].hf_name, "language_model.model.layers."))
            return "language_model.model.layers.";
        if (strstr(ms->tensor_map[i].hf_name, "model.layers."))
            return "model.layers.";
    }
    return "model.layers.";
}

static int write_all(int fd, const void *buffer, size_t size)
{
    const uint8_t *cursor = (const uint8_t *)buffer;

    while (size > 0u) {
        ssize_t written = write(fd, cursor, size);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (written == 0) {
            return -1;
        }
        cursor += (size_t)written;
        size -= (size_t)written;
    }

    return 0;
}

/* -------------------------------------------------------------------------
 * Staging allocation / free
 * -------------------------------------------------------------------------*/

static tape_staging_t *tape_alloc_staging(int n_layers, int n_samples,
                                           const gemma3_270m_config_t *cfg)
{
    int n = n_layers * TAPE_N_UNIQUE;
    tape_staging_t *st = (tape_staging_t *)calloc((size_t)n, sizeof(*st));
    if (!st) { LOG_ERROR("tape: staging array OOM"); return NULL; }

    for (int l = 0; l < n_layers; ++l) {
        for (int ti = 0; ti < TAPE_N_UNIQUE; ++ti) {
            capture_target_t t = (capture_target_t)(ti + 1);
            int idx = staging_idx_of(l, t);
            st[idx].layer_idx  = l;
            st[idx].target     = t;
            st[idx].vector_dim = dim_for_target(t, cfg);
            size_t nb = (size_t)n_samples * st[idx].vector_dim * sizeof(float);
            if (nb == 0u) {
                st[idx].buffer = NULL;
                continue;
            }
            st[idx].buffer = (float *)calloc(1, nb);
            if (!st[idx].buffer) {
                LOG_ERROR("tape: OOM at layer=%d target=%d", l, ti);
                for (int j = 0; j < idx; ++j) free(st[j].buffer);
                free(st);
                return NULL;
            }
        }
    }
    return st;
}

static void tape_free_staging(tape_staging_t *st, int n)
{
    if (!st) return;
    for (int i = 0; i < n; ++i) free(st[i].buffer);
    free(st);
}

/* -------------------------------------------------------------------------
 * Slot helpers
 * -------------------------------------------------------------------------*/

static activation_record_slot_t *tape_build_slots(const tape_staging_t *st,
                                                   int n_unique)
{
    activation_record_slot_t *slots =
        (activation_record_slot_t *)calloc((size_t)n_unique, sizeof(*slots));
    if (!slots) { LOG_ERROR("tape: slots OOM"); return NULL; }
    for (int i = 0; i < n_unique; ++i) {
        slots[i].layer_idx  = st[i].layer_idx;
        slots[i].target     = st[i].target;
        slots[i].out_dim    = st[i].vector_dim;
        slots[i].out_vector = NULL;
        slots[i].captured   = 0;
    }
    return slots;
}

static void tape_set_slot_vectors(activation_record_slot_t *slots,
                                   const tape_staging_t *st,
                                   int n_unique, int sample_idx)
{
    for (int i = 0; i < n_unique; ++i) {
        slots[i].out_vector = st[i].buffer + (size_t)sample_idx * st[i].vector_dim;
        slots[i].captured   = 0;
    }
}

/* -------------------------------------------------------------------------
 * Manifest build
 * -------------------------------------------------------------------------*/

static void tape_compute_staging_offsets(const tape_staging_t *st, int n_unique,
                                          int n_samples, uint64_t *out_offsets)
{
    uint64_t off = 0;
    for (int i = 0; i < n_unique; ++i) {
        out_offsets[i] = off;
        off += (uint64_t)st[i].vector_dim * (uint32_t)n_samples * sizeof(float);
    }
}

static void tape_build_manifest(tape_manifest_entry_t *manifest,
                                 const tape_staging_t  *st,
                                 const uint64_t        *st_offsets,
                                 const tape_manifest_ctx_t *ctx)
{
    for (int l = 0; l < ctx->n_layers; ++l) {
        uint32_t is_global = ((l % 6) == 5) ? 1u : 0u;
        for (int w = 0; w < TAPE_N_WT; ++w) {
            int eidx = l * TAPE_N_WT + w;
            tape_manifest_entry_t *e = &manifest[eidx];
            snprintf(e->tensor_name, TAPE_TENSOR_NAME_MAX,
                     "%s%d.%s", ctx->prefix, l, TAPE_WT[w].suffix);
            e->layer_type   = is_global;
            e->sample_count = (uint32_t)ctx->n_samples;
            if (TAPE_WT[w].alias_idx < 0) {
                int sidx          = staging_idx_of(l, TAPE_WT[w].target);
                e->vector_dim     = st[sidx].vector_dim;
                e->data_offset    = st_offsets[sidx];
                e->data_bytes     = (uint64_t)e->vector_dim * (uint32_t)ctx->n_samples * sizeof(float);
                e->alias_of_entry = TAPE_NO_ALIAS;
            } else {
                int primary_eidx           = l * TAPE_N_WT + TAPE_WT[w].alias_idx;
                const tape_manifest_entry_t *pe = &manifest[primary_eidx];
                e->vector_dim     = pe->vector_dim;
                e->data_offset    = pe->data_offset;
                e->data_bytes     = pe->data_bytes;
                e->alias_of_entry = (uint32_t)primary_eidx;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * File writing helpers
 * -------------------------------------------------------------------------*/

static int tape_write_header_and_manifest(int fd,
                                           const tape_file_header_t *hdr,
                                           const tape_manifest_entry_t *manifest,
                                           int n_entries)
{
    if (write_all(fd, hdr, sizeof(*hdr)) != 0) {
        LOG_ERROR("tape: header write failed: %s", strerror(errno));
        return -1;
    }
    size_t msz = (size_t)n_entries * sizeof(*manifest);
    if (write_all(fd, manifest, msz) != 0) {
        LOG_ERROR("tape: manifest write failed: %s", strerror(errno));
        return -1;
    }
    return 0;
}

static int tape_write_data_section(int fd, const tape_staging_t *st,
                                    int n_unique, int n_samples)
{
    for (int i = 0; i < n_unique; ++i) {
        size_t block = (size_t)st[i].vector_dim * (size_t)n_samples * sizeof(float);
        if (block == 0u) {
            continue;
        }
        if (write_all(fd, st[i].buffer, block) != 0) {
            LOG_ERROR("tape: data write failed at slot %d: %s", i, strerror(errno));
            return -1;
        }
    }
    return 0;
}

static void tape_write_plan_release(tape_write_plan_t *plan)
{
    if (!plan) {
        return;
    }

    free(plan->manifest);
    memset(plan, 0, sizeof(*plan));
}

static int tape_prepare_write_plan(const tape_staging_t *st,
                                   const tape_manifest_ctx_t *ctx,
                                   const model_spec_t *ms,
                                   tape_write_plan_t *out_plan)
{
    int n_entries = 0;
    uint64_t data_size = 0u;
    tape_manifest_entry_t *manifest = NULL;
    uint64_t *st_offsets = NULL;
    tape_file_header_t header;
    uint32_t file_crc32 = 0u;

    if (!st || !ctx || !ms || !out_plan) {
        return -1;
    }

    memset(out_plan, 0, sizeof(*out_plan));
    n_entries = ctx->n_layers * TAPE_N_WT;
    manifest = (tape_manifest_entry_t *)calloc((size_t)n_entries, sizeof(*manifest));
    st_offsets = (uint64_t *)calloc((size_t)ctx->n_unique, sizeof(*st_offsets));
    if (!manifest || !st_offsets) {
        LOG_ERROR("tape: write plan OOM");
        free(st_offsets);
        free(manifest);
        return -1;
    }

    tape_compute_staging_offsets(st, ctx->n_unique, ctx->n_samples, st_offsets);
    for (int i = 0; i < ctx->n_unique; ++i) {
        data_size += (uint64_t)st[i].vector_dim * (uint32_t)ctx->n_samples * sizeof(float);
    }

    tape_build_manifest(manifest, st, st_offsets, ctx);

    memset(&header, 0, sizeof(header));
    header.magic = TAPE_MAGIC;
    header.version = TAPE_VERSION;
    header.entry_count = (uint32_t)n_entries;
    header.sample_count = (uint32_t)ctx->n_samples;
    header.hidden_size = (uint32_t)ctx->cfg->hidden_size;
    header.data_section_offset = (uint64_t)sizeof(tape_file_header_t)
                               + (uint64_t)n_entries * sizeof(tape_manifest_entry_t);
    header.data_section_size = data_size;
    header.crc32 = 0u;
    header.crc32 = io_crc32_update(0u, &header, offsetof(tape_file_header_t, crc32));

    file_crc32 = io_crc32_update(0u, &header, sizeof(header));
    file_crc32 = io_crc32_update(file_crc32,
                                 manifest,
                                 (size_t)n_entries * sizeof(*manifest));
    for (int i = 0; i < ctx->n_unique; ++i) {
        size_t block = (size_t)st[i].vector_dim * (size_t)ctx->n_samples * sizeof(float);

        if (block == 0u) {
            continue;
        }
        file_crc32 = io_crc32_update(file_crc32, st[i].buffer, block);
    }

    out_plan->header = header;
    out_plan->manifest = manifest;
    out_plan->entry_count = n_entries;
    out_plan->file_crc32 = file_crc32;
    free(st_offsets);
    (void)ms;
    return 0;
}

static int tape_write_file_from_plan(const char *path,
                                     const tape_staging_t *st,
                                     int n_unique,
                                     const tape_write_plan_t *plan)
{
    int fd = -1;
    int rc = -1;

    if (!path || !st || !plan || !plan->manifest || plan->entry_count <= 0) {
        return -1;
    }

    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOG_ERROR("tape: open %s: %s", path, strerror(errno));
        return -1;
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    if (tape_write_header_and_manifest(fd,
                                       &plan->header,
                                       plan->manifest,
                                       plan->entry_count) != 0) {
        goto cleanup;
    }
    if (tape_write_data_section(fd,
                                st,
                                n_unique,
                                (int)plan->header.sample_count) != 0) {
        goto cleanup;
    }
    while (fsync(fd) != 0) {
        if (errno == EINTR) {
            continue;
        }
        LOG_ERROR("tape: fsync failed: %s", strerror(errno));
        goto cleanup;
    }

    rc = 0;
    LOG_INFO("tape: wrote %s (%d entries, %u samples, %.1f MB)",
             path,
             plan->entry_count,
             plan->header.sample_count,
             (double)plan->header.data_section_size / (1024.0 * 1024.0));

cleanup:
    if (fd >= 0) {
        if (close(fd) != 0 && errno != EINTR) {
            LOG_ERROR("tape: close failed");
            rc = -1;
        }
    }
    return rc;
}

static int tape_build_prompt_tokens(const model_spec_t *spec,
                                    sapphire_tokenizer_t *tokenizer,
                                    const char *text,
                                    int *tokens,
                                    int max_tokens)
{
    int token_count = 0;

    if (!spec || !tokenizer || !text || !tokens || max_tokens <= 0) {
        return -1;
    }

    token_count = build_gemma3_prompt(spec, text, tokens, max_tokens);
    if (token_count <= 0) {
        token_count = tokenize(tokenizer, text, tokens, max_tokens);
    }
    return token_count;
}

static int tape_capture_sample_cpu(inference_session_t *session,
                                   sapphire_tokenizer_t *tokenizer,
                                   const model_spec_t *spec,
                                   const char *text,
                                   activation_record_slot_t *slots,
                                   int slot_count)
{
    return sapphire_record_pass(session,
                                tokenizer,
                                spec,
                                text,
                                slots,
                                slot_count);
}

typedef struct {
    inference_session_t *session;
    sapphire_tokenizer_t *tokenizer;
    const model_spec_t *spec;
    const gemma3_270m_config_t *cfg;
    tape_staging_t *st;
    int n_unique;
    int *tokens;
    int max_tokens;
    float *capture_buffer;
    size_t capture_bytes;
} tape_vulkan_capture_ctx_t;

typedef struct {
    const activation_tape_record_config_t *config;
    const model_spec_t *ms;
    const calibration_corpus_t *corp;
    const gemma3_270m_config_t *cfg;
    tape_staging_t *st;
    activation_record_slot_t *slots;
    tape_write_plan_t write_plan;
    tape_vulkan_capture_ctx_t vk_capture;
    float *oracle_buffer;
    int use_vulkan_capture;
    int use_vulkan_oracle;
    int oracle_began;
    int token_capacity;
    int n_layers;
    int n_samples;
    int n_unique;
    int completed_samples;
    int rc;
} tape_record_runtime_t;

static int tape_capture_sample_vulkan(const tape_vulkan_capture_ctx_t *ctx,
                                      const char *text,
                                      int sample_idx)
{
    int token_count = 0;

    if (!ctx || !ctx->session || !ctx->tokenizer || !ctx->spec || !ctx->cfg || !text ||
        !ctx->st || ctx->n_unique <= 0 || sample_idx < 0 || !ctx->tokens ||
        ctx->max_tokens <= 0 || !ctx->capture_buffer || ctx->capture_bytes == 0u) {
        return -1;
    }

    token_count = tape_build_prompt_tokens(ctx->spec,
                                           ctx->tokenizer,
                                           text,
                                           ctx->tokens,
                                           ctx->max_tokens);
    if (token_count <= 0) {
        LOG_WARN("tape: tokenization failed for Vulkan capture");
        return -1;
    }
    if (backend_vulkan_capture_last_token_activations(ctx->session,
                                                      ctx->tokens,
                                                      token_count,
                                                      ctx->capture_buffer,
                                                      ctx->capture_bytes) != 0) {
        return -1;
    }

    for (int i = 0; i < ctx->n_unique; ++i) {
        size_t capture_offset = 0u;
        uint32_t vector_dim = 0u;
        float *dst = NULL;

        if (ternary_hessian_oracle_capture_view(ctx->cfg,
                                                ctx->st[i].layer_idx,
                                                ctx->st[i].target,
                                                &capture_offset,
                                                &vector_dim) != 0 ||
            vector_dim != ctx->st[i].vector_dim) {
            LOG_ERROR("tape: Vulkan capture layout mismatch at layer=%d target=%d",
                      ctx->st[i].layer_idx,
                      (int)ctx->st[i].target);
            return -1;
        }

        dst = ctx->st[i].buffer + (size_t)sample_idx * ctx->st[i].vector_dim;
        memcpy(dst,
               ctx->capture_buffer + (capture_offset / sizeof(float)),
               (size_t)vector_dim * sizeof(float));
    }

    return 0;
}

static int tape_record_runtime_init(tape_record_runtime_t *runtime,
                                    const activation_tape_record_config_t *config)
{
    if (!runtime || !config || !config->output_path || !config->session ||
        !config->tokenizer || !config->spec || !config->corpus) {
        return -1;
    }

    memset(runtime, 0, sizeof(*runtime));
    runtime->config = config;
    runtime->ms = (const model_spec_t *)config->spec;
    runtime->corp = (const calibration_corpus_t *)config->corpus;
    runtime->cfg = (const gemma3_270m_config_t *)runtime->ms->variant_config;
    runtime->rc = -1;
    runtime->token_capacity = (config->max_prompt_tokens > 0)
        ? config->max_prompt_tokens
        : 1024;
    if (!runtime->cfg || runtime->cfg->num_hidden_layers <= 0) {
        LOG_ERROR("tape: invalid variant_config");
        return -1;
    }

    runtime->n_layers = runtime->cfg->num_hidden_layers;
    runtime->n_samples = (config->sample_limit <= 0 ||
                          config->sample_limit > runtime->corp->sample_count)
        ? runtime->corp->sample_count
        : config->sample_limit;
    runtime->n_unique = runtime->n_layers * TAPE_N_UNIQUE;
    runtime->use_vulkan_capture = (config->session->backend &&
                                   config->session->backend->type == SAPPHIRE_BACKEND_TYPE_VULKAN);
    runtime->use_vulkan_oracle = runtime->use_vulkan_capture &&
                                 config->oracle_output_path &&
                                 config->oracle_output_path[0] != '\0';
    if (!runtime->use_vulkan_capture && runtime->use_vulkan_oracle) {
        LOG_ERROR("tape: oracle sidecar requires Vulkan tape recording");
        return -1;
    }

    LOG_INFO("tape: recording %d samples × %d layers → %s (%s backend)",
             runtime->n_samples,
             runtime->n_layers,
             config->output_path,
             runtime->use_vulkan_capture ? "vulkan" : "cpu");
    return 0;
}

static int tape_record_prepare_vulkan_capture(tape_record_runtime_t *runtime)
{
    if (!runtime || !runtime->use_vulkan_capture) {
        return 0;
    }

    runtime->vk_capture.capture_bytes = ternary_hessian_oracle_capture_bytes(runtime->cfg);
    runtime->vk_capture.capture_buffer = (float *)malloc(runtime->vk_capture.capture_bytes);
    runtime->vk_capture.tokens = (int *)malloc((size_t)runtime->token_capacity * sizeof(int));
    if (runtime->vk_capture.capture_bytes == 0u || !runtime->vk_capture.capture_buffer ||
        !runtime->vk_capture.tokens) {
        LOG_ERROR("tape: failed to allocate Vulkan capture buffers");
        return -1;
    }

    runtime->vk_capture.session = runtime->config->session;
    runtime->vk_capture.tokenizer = (sapphire_tokenizer_t *)runtime->config->tokenizer;
    runtime->vk_capture.spec = runtime->ms;
    runtime->vk_capture.cfg = runtime->cfg;
    runtime->vk_capture.st = runtime->st;
    runtime->vk_capture.n_unique = runtime->n_unique;
    runtime->vk_capture.max_tokens = runtime->token_capacity;
    return 0;
}

static int tape_record_alloc_resources(tape_record_runtime_t *runtime)
{
    if (!runtime) {
        return -1;
    }

    runtime->st = tape_alloc_staging(runtime->n_layers, runtime->n_samples, runtime->cfg);
    if (!runtime->st) {
        return -1;
    }

    runtime->slots = tape_build_slots(runtime->st, runtime->n_unique);
    if (!runtime->slots) {
        return -1;
    }

    runtime->vk_capture.st = runtime->st;
    runtime->vk_capture.n_unique = runtime->n_unique;
    if (tape_record_prepare_vulkan_capture(runtime) != 0) {
        return -1;
    }
    if (runtime->use_vulkan_oracle &&
        backend_vulkan_begin_hessian_oracle_capture(runtime->config->session) != 0) {
        LOG_ERROR("tape: failed to start Vulkan oracle capture");
        return -1;
    }
    runtime->oracle_began = runtime->use_vulkan_oracle;
    return 0;
}

static int tape_capture_one_sample(tape_record_runtime_t *runtime, int sample_idx)
{
    if (!runtime || sample_idx < 0 || sample_idx >= runtime->n_samples) {
        return -1;
    }

    if (runtime->use_vulkan_capture) {
        return tape_capture_sample_vulkan(&runtime->vk_capture,
                                          runtime->corp->samples[sample_idx],
                                          sample_idx);
    }

    tape_set_slot_vectors(runtime->slots, runtime->st, runtime->n_unique, sample_idx);
    return tape_capture_sample_cpu(runtime->config->session,
                                   (sapphire_tokenizer_t *)runtime->config->tokenizer,
                                   runtime->ms,
                                   runtime->corp->samples[sample_idx],
                                   runtime->slots,
                                   runtime->n_unique);
}

static void tape_record_samples(tape_record_runtime_t *runtime)
{
    if (!runtime) {
        return;
    }

    for (int sample_idx = 0; sample_idx < runtime->n_samples; ++sample_idx) {
        if (activation_tape_stop_requested()) {
            LOG_WARN("tape: stop requested before sample %d/%d",
                     sample_idx + 1,
                     runtime->n_samples);
            break;
        }

        if (tape_capture_one_sample(runtime, sample_idx) != 0) {
            LOG_WARN("tape: sample %d/%d failed", sample_idx + 1, runtime->n_samples);
        }
        runtime->completed_samples = sample_idx + 1;
        if ((sample_idx % 8 == 0) || (sample_idx + 1 == runtime->n_samples)) {
            LOG_INFO("tape: sample %d/%d", sample_idx + 1, runtime->n_samples);
        }
        if (activation_tape_stop_requested()) {
            LOG_WARN("tape: stop requested after sample %d/%d",
                     runtime->completed_samples,
                     runtime->n_samples);
            break;
        }
    }
}

static int tape_finalize_oracle_sidecar(tape_record_runtime_t *runtime)
{
    ternary_hessian_oracle_write_request_t write_request;
    backend_vulkan_oracle_finish_config_t finish_config;

    if (!runtime || !runtime->use_vulkan_oracle || runtime->completed_samples <= 0) {
        return 0;
    }

    runtime->oracle_buffer = (float *)malloc(runtime->vk_capture.capture_bytes);
    if (!runtime->oracle_buffer) {
        LOG_ERROR("tape: failed to allocate oracle readback buffer");
        return -1;
    }

    memset(&finish_config, 0, sizeof(finish_config));
    finish_config.sample_count = (uint32_t)runtime->completed_samples;
    finish_config.strength = runtime->config->hessian_proxy_strength;
    finish_config.floor = runtime->config->hessian_proxy_floor;
    finish_config.out_buffer = runtime->oracle_buffer;
    finish_config.out_size = runtime->vk_capture.capture_bytes;
    if (backend_vulkan_finish_hessian_oracle_capture(runtime->config->session, &finish_config) != 0) {
        LOG_ERROR("tape: failed to finalize Vulkan oracle capture");
        return -1;
    }
    runtime->oracle_began = 0;

    memset(&write_request, 0, sizeof(write_request));
    write_request.output_path = runtime->config->oracle_output_path;
    write_request.teacher_model_name = runtime->ms->model_id;
    write_request.config = runtime->cfg;
    write_request.manifest = runtime->write_plan.manifest;
    write_request.entry_count = (uint32_t)runtime->write_plan.entry_count;
    write_request.tape_crc32 = runtime->write_plan.file_crc32;
    write_request.sample_count = (uint32_t)runtime->completed_samples;
    write_request.diagonal_capture_buffer = runtime->oracle_buffer;
    return ternary_hessian_oracle_write_sidecar(&write_request);
}

static int tape_write_outputs(tape_record_runtime_t *runtime)
{
    tape_manifest_ctx_t ctx;

    if (!runtime) {
        return -1;
    }
    if (runtime->completed_samples < runtime->n_samples || activation_tape_stop_requested()) {
        LOG_WARN("tape: writing partial tape with %d/%d samples",
                 runtime->completed_samples,
                 runtime->n_samples);
    }

    memset(&ctx, 0, sizeof(ctx));
    ctx.n_unique = runtime->n_unique;
    ctx.n_layers = runtime->n_layers;
    ctx.n_samples = runtime->completed_samples;
    ctx.cfg = runtime->cfg;
    ctx.prefix = tape_tensor_prefix(runtime->ms);

    if (tape_prepare_write_plan(runtime->st, &ctx, runtime->ms, &runtime->write_plan) != 0) {
        return -1;
    }
    if (tape_write_file_from_plan(runtime->config->output_path,
                                  runtime->st,
                                  runtime->n_unique,
                                  &runtime->write_plan) != 0) {
        return -1;
    }
    return tape_finalize_oracle_sidecar(runtime);
}

static void tape_record_cleanup(tape_record_runtime_t *runtime)
{
    if (!runtime) {
        return;
    }
    if (runtime->oracle_began) {
        backend_vulkan_abort_hessian_oracle_capture(runtime->config->session);
    }

    free(runtime->oracle_buffer);
    free(runtime->vk_capture.tokens);
    free(runtime->vk_capture.capture_buffer);
    tape_write_plan_release(&runtime->write_plan);
    free(runtime->slots);
    tape_free_staging(runtime->st, runtime->n_unique);
}

/* -------------------------------------------------------------------------
 * Public: activation_tape_record
 * -------------------------------------------------------------------------*/

int activation_tape_record_ex(const activation_tape_record_config_t *config)
{
    tape_record_runtime_t runtime;

    if (tape_record_runtime_init(&runtime, config) != 0) {
        return -1;
    }
    if (tape_record_alloc_resources(&runtime) != 0) {
        tape_record_cleanup(&runtime);
        return -1;
    }

    tape_record_samples(&runtime);
    runtime.rc = tape_write_outputs(&runtime);
    tape_record_cleanup(&runtime);
    return runtime.rc;
}

int activation_tape_record(const char                        *output_path,
                           struct inference_session_t        *session,
                           struct sapphire_tokenizer_t       *tokenizer,
                           const struct model_spec           *spec,
                           const struct calibration_corpus_t *corpus,
                           int                                sample_limit)
{
    const activation_tape_record_config_t config = {
        .output_path = output_path,
        .oracle_output_path = NULL,
        .session = session,
        .tokenizer = tokenizer,
        .spec = spec,
        .corpus = corpus,
        .sample_limit = sample_limit,
        .max_prompt_tokens = 0,
        .hessian_proxy_strength = 1.0f,
        .hessian_proxy_floor = 0.05f
    };

    return activation_tape_record_ex(&config);
}

/* -------------------------------------------------------------------------
 * Public: playback
 * -------------------------------------------------------------------------*/

static const tape_manifest_entry_t *tape_find_entry(const activation_tape_t *tape,
                                                      const char *tensor_name)
{
    for (uint32_t i = 0; i < tape->entry_count; ++i) {
        if (strncmp(tape->manifest[i].tensor_name, tensor_name,
                    TAPE_TENSOR_NAME_MAX) == 0) {
            return &tape->manifest[i];
        }
    }
    return NULL;
}

int activation_tape_entry_index(const activation_tape_t *tape,
                                const char              *tensor_name)
{
    const tape_manifest_entry_t *entry = NULL;

    if (!tape || !tensor_name) {
        return -1;
    }

    entry = tape_find_entry(tape, tensor_name);
    if (!entry) {
        return -1;
    }
    if (entry->alias_of_entry != TAPE_NO_ALIAS) {
        if (entry->alias_of_entry >= tape->entry_count) {
            return -1;
        }
        return (int)entry->alias_of_entry;
    }
    return (int)(entry - tape->manifest);
}

uint32_t activation_tape_entry_count(const activation_tape_t *tape)
{
    return tape ? tape->entry_count : 0u;
}

const tape_manifest_entry_t *activation_tape_entry(const activation_tape_t *tape,
                                                   uint32_t                 entry_idx)
{
    if (!tape || entry_idx >= tape->entry_count) {
        return NULL;
    }

    return &tape->manifest[entry_idx];
}

activation_tape_t *activation_tape_open(const char *tape_path)
{
    if (!tape_path) return NULL;

    int fd = open(tape_path, O_RDONLY);
    if (fd < 0) { LOG_ERROR("tape open: %s: %s", tape_path, strerror(errno)); return NULL; }

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    size_t sz = (size_t)st.st_size;
    if (sz < sizeof(tape_file_header_t)) {
        LOG_ERROR("tape open: %s too small (%zu B)", tape_path, sz);
        close(fd);
        return NULL;
    }

    void *ptr = mmap(NULL, sz, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, fd, 0);
    if (ptr == MAP_FAILED) {
        LOG_ERROR("tape open: mmap failed: %s", strerror(errno));
        close(fd);
        return NULL;
    }

    activation_tape_t *tape =
        (activation_tape_t *)calloc(1, sizeof(*tape));
    if (!tape) { munmap(ptr, sz); close(fd); return NULL; }

    tape->fd         = fd;
    tape->mmap_ptr   = ptr;
    tape->mmap_size  = sz;
    memcpy(&tape->header, ptr, sizeof(tape_file_header_t));

    if (tape->header.magic != TAPE_MAGIC || tape->header.version != TAPE_VERSION) {
        LOG_ERROR("tape open: bad magic/version (0x%08x / %u)",
                  tape->header.magic, tape->header.version);
        activation_tape_close(tape);
        return NULL;
    }

    tape->entry_count = tape->header.entry_count;
    tape->manifest = (const tape_manifest_entry_t *)
                     ((const uint8_t *)ptr + sizeof(tape_file_header_t));

    LOG_INFO("tape open: %s — %u entries, %u samples",
             tape_path, tape->entry_count, tape->header.sample_count);
    return tape;
}

int activation_tape_get_vector(const activation_tape_t *tape,
                                const char              *tensor_name,
                                int                      sample_idx,
                                float                   *out_vector)
{
    if (!tape || !tensor_name || sample_idx < 0 || !out_vector) return -1;

    const tape_manifest_entry_t *e = tape_find_entry(tape, tensor_name);
    if (!e) { LOG_WARN("tape: tensor not found: %s", tensor_name); return -1; }
    if ((uint32_t)sample_idx >= e->sample_count) return -1;

    if (e->alias_of_entry != TAPE_NO_ALIAS) {
        if (e->alias_of_entry >= tape->entry_count) return -1;
        e = &tape->manifest[e->alias_of_entry];
    }

    const uint8_t *base =
        (const uint8_t *)tape->mmap_ptr
        + tape->header.data_section_offset
        + e->data_offset;
    memcpy(out_vector, base + (size_t)sample_idx * e->vector_dim * sizeof(float),
           e->vector_dim * sizeof(float));
    return 0;
}

int activation_tape_sample_count(const activation_tape_t *tape)
{
    return tape ? (int)tape->header.sample_count : -1;
}

uint32_t activation_tape_vector_dim(const activation_tape_t *tape,
                                     const char              *tensor_name)
{
    if (!tape || !tensor_name) return 0u;
    const tape_manifest_entry_t *e = tape_find_entry(tape, tensor_name);
    return e ? e->vector_dim : 0u;
}

const tape_file_header_t *activation_tape_header(const activation_tape_t *tape)
{
    return tape ? &tape->header : NULL;
}

uint32_t activation_tape_crc32(const activation_tape_t *tape)
{
    if (!tape || !tape->mmap_ptr || tape->mmap_size == 0u) {
        return 0u;
    }

    return io_crc32_update(0u, tape->mmap_ptr, tape->mmap_size);
}

void activation_tape_prefetch_entry(const activation_tape_t *tape,
                                     uint32_t                 entry_idx)
{
    if (!tape || entry_idx >= tape->entry_count) return;
    const tape_manifest_entry_t *e = &tape->manifest[entry_idx];
    const uint8_t *base = (const uint8_t *)tape->mmap_ptr
                          + tape->header.data_section_offset
                          + e->data_offset;
    madvise((void *)base, (size_t)e->data_bytes, MADV_WILLNEED);
}

void activation_tape_close(activation_tape_t *tape)
{
    if (!tape) return;
    if (tape->mmap_ptr && tape->mmap_size)
        munmap(tape->mmap_ptr, tape->mmap_size);
    if (tape->fd >= 0)
        close(tape->fd);
    free(tape);
}

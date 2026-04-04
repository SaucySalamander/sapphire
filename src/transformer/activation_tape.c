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
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "calibration_corpus.h"
#include "gemma3_270m_config.h"
#include "inference.h"
#include "log.h"
#include "model_spec.h"
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

/** Opaque tape handle. */
struct activation_tape_t {
    int                          fd;
    void                        *mmap_ptr;
    size_t                       mmap_size;
    tape_file_header_t           header;
    const tape_manifest_entry_t *manifest;  /**< Points into mmap. */
    uint32_t                     entry_count;
};

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
    ssize_t w = write(fd, hdr, sizeof(*hdr));
    if (w != (ssize_t)sizeof(*hdr)) {
        LOG_ERROR("tape: header write failed: %s", strerror(errno));
        return -1;
    }
    size_t msz = (size_t)n_entries * sizeof(*manifest);
    w = write(fd, manifest, msz);
    if (w != (ssize_t)msz) {
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
        ssize_t w = write(fd, st[i].buffer, block);
        if (w != (ssize_t)block) {
            LOG_ERROR("tape: data write failed at slot %d: %s", i, strerror(errno));
            return -1;
        }
    }
    return 0;
}

static int tape_write_file(const char *path, const tape_staging_t *st,
                            const tape_manifest_ctx_t *ctx, const model_spec_t *ms)
{
    int n_entries = ctx->n_layers * TAPE_N_WT;
    uint64_t data_off = (uint64_t)sizeof(tape_file_header_t)
                        + (uint64_t)n_entries * sizeof(tape_manifest_entry_t);
    uint64_t data_size = 0;
    int fd = -1;
    int rc = -1;

    tape_manifest_entry_t *manifest =
        (tape_manifest_entry_t *)calloc((size_t)n_entries, sizeof(*manifest));
    uint64_t *st_offsets = (uint64_t *)calloc((size_t)ctx->n_unique, sizeof(uint64_t));
    if (!manifest || !st_offsets) {
        LOG_ERROR("tape: write_file OOM");
        goto wf_cleanup;
    }

    tape_compute_staging_offsets(st, ctx->n_unique, ctx->n_samples, st_offsets);
    for (int i = 0; i < ctx->n_unique; ++i)
        data_size += (uint64_t)st[i].vector_dim * (uint32_t)ctx->n_samples * sizeof(float);

    tape_build_manifest(manifest, st, st_offsets, ctx);

    tape_file_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic               = TAPE_MAGIC;
    hdr.version             = TAPE_VERSION;
    hdr.entry_count         = (uint32_t)n_entries;
    hdr.sample_count        = (uint32_t)ctx->n_samples;
    hdr.hidden_size         = (uint32_t)ctx->cfg->hidden_size;
    hdr.data_section_offset = data_off;
    hdr.data_section_size   = data_size;
    hdr.crc32               = 0u;

    fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { LOG_ERROR("tape: open %s: %s", path, strerror(errno)); goto wf_cleanup; }
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    if (tape_write_header_and_manifest(fd, &hdr, manifest, n_entries) != 0) goto wf_cleanup;
    if (tape_write_data_section(fd, st, ctx->n_unique, ctx->n_samples) != 0) goto wf_cleanup;

    hdr.crc32 = io_crc32_update(0u, &hdr, offsetof(tape_file_header_t, crc32));
    if (lseek(fd, (off_t)offsetof(tape_file_header_t, crc32), SEEK_SET) < 0 ||
        write(fd, &hdr.crc32, sizeof(hdr.crc32)) != (ssize_t)sizeof(hdr.crc32)) {
        LOG_ERROR("tape: CRC patch failed");
        goto wf_cleanup;
    }
    rc = 0;
    LOG_INFO("tape: wrote %s (%d entries, %d samples, %.1f MB)",
             path, n_entries, ctx->n_samples, (double)data_size / (1024.0 * 1024.0));

wf_cleanup:
    if (fd >= 0 && close(fd) != 0) { LOG_ERROR("tape: close failed"); rc = -1; }
    free(st_offsets);
    free(manifest);
    return rc;
}

/* -------------------------------------------------------------------------
 * Public: activation_tape_record
 * -------------------------------------------------------------------------*/

int activation_tape_record(const char                        *output_path,
                            struct inference_session_t        *session,
                            struct sapphire_tokenizer_t       *tokenizer,
                            const struct model_spec           *spec,
                            const struct calibration_corpus_t *corpus,
                            int                                sample_limit)
{
    const model_spec_t         *ms   = (const model_spec_t *)spec;
    const calibration_corpus_t *corp = (const calibration_corpus_t *)corpus;
    tape_staging_t             *st   = NULL;
    activation_record_slot_t   *slots = NULL;
    int rc = -1;

    if (!output_path || !session || !tokenizer || !ms || !corp) return -1;

    const gemma3_270m_config_t *cfg =
        (const gemma3_270m_config_t *)ms->variant_config;
    if (!cfg || cfg->num_hidden_layers <= 0) {
        LOG_ERROR("tape: invalid variant_config");
        return -1;
    }

    int n_layers  = cfg->num_hidden_layers;
    int n_samples = (sample_limit <= 0 || sample_limit > corp->sample_count)
                    ? corp->sample_count : sample_limit;
    int n_unique  = n_layers * TAPE_N_UNIQUE;

    LOG_INFO("tape: recording %d samples × %d layers → %s",
             n_samples, n_layers, output_path);

    st = tape_alloc_staging(n_layers, n_samples, cfg);
    if (!st) return -1;

    slots = tape_build_slots(st, n_unique);
    if (!slots) { tape_free_staging(st, n_unique); return -1; }

    for (int s = 0; s < n_samples; ++s) {
        tape_set_slot_vectors(slots, st, n_unique, s);
        if (sapphire_record_pass(session,
                                  (struct sapphire_tokenizer_t *)tokenizer,
                                  (const struct model_spec *)ms,
                                  corp->samples[s], slots, n_unique) != 0) {
            LOG_WARN("tape: sample %d/%d failed", s + 1, n_samples);
        }
        if (s % 8 == 0)
            LOG_INFO("tape: sample %d/%d", s + 1, n_samples);
    }

    tape_manifest_ctx_t ctx = {
        .n_unique = n_unique,
        .n_layers = n_layers,
        .n_samples = n_samples,
        .cfg    = cfg,
        .prefix = tape_tensor_prefix(ms),
    };
    rc = tape_write_file(output_path, st, &ctx, ms);
    free(slots);
    tape_free_staging(st, n_unique);
    return rc;
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

/**
 * @file activation_tape.h
 * @brief Single-pass activation recorder and disk-backed STE replay.
 *
 * Records one forward pass per calibration sample, capturing all projection-input
 * activation vectors to a binary tape file. The tape is replayed via mmap during
 * the STE weight-update phase, eliminating the O(N_tensors × N_samples) re-run cost.
 *
 * File format (native-endian):
 *   [ tape_file_header_t   ]      — fixed 64 bytes
 *   [ tape_manifest_entry_t × N ] — TAPE_N_WT entries per layer
 *   [ float data blocks ...      ] — F32 activation vectors, packed by (layer, target)
 */

#ifndef ACTIVATION_TAPE_H
#define ACTIVATION_TAPE_H

#include <stddef.h>
#include <stdint.h>

/* Only the types used in the public API need to be visible here. */
struct inference_session_t;
struct sapphire_tokenizer_t;
struct model_spec;
struct calibration_corpus_t;

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * On-disk format constants
 * -------------------------------------------------------------------------*/

#define TAPE_MAGIC           0x54415045u  /* "TAPE" */
#define TAPE_VERSION         1u
#define TAPE_TENSOR_NAME_MAX 256u
#define TAPE_NO_ALIAS        0xFFFFFFFFu

/**
 * @brief Tape file header — exactly 64 bytes.
 *
 * All values are native-endian. The CRC32 field covers the first
 * (offsetof crc32) bytes of the header with the crc32 field set to 0.
 */
typedef struct {
    uint32_t magic;               /**< TAPE_MAGIC */
    uint32_t version;             /**< TAPE_VERSION */
    uint32_t entry_count;         /**< Total manifest entries (n_layers × 7) */
    uint32_t sample_count;        /**< Number of corpus samples recorded */
    uint32_t hidden_size;         /**< d_model of the recorded model */
    uint32_t _pad[3];
    uint64_t data_section_offset; /**< Byte offset from file start to float data */
    uint64_t data_section_size;   /**< Total bytes in the data section */
    uint32_t crc32;               /**< CRC32 of header (field = 0 during computation) */
    uint32_t _pad2[3];
} tape_file_header_t;

/**
 * @brief Manifest entry — one per (layer, weight_type) pair.
 *
 * Aliased entries (K→Q, V→Q, up_proj→gate_proj) share data_offset with
 * their primary entry; alias_of_entry holds the entry index of the primary.
 */
typedef struct {
    char     tensor_name[TAPE_TENSOR_NAME_MAX]; /**< Full HF tensor name */
    uint32_t vector_dim;       /**< Floats per sample */
    uint32_t sample_count;     /**< Equals header.sample_count */
    uint64_t data_offset;      /**< Offset from data_section_offset to first float */
    uint64_t data_bytes;       /**< vector_dim × sample_count × sizeof(float) */
    uint32_t alias_of_entry;   /**< TAPE_NO_ALIAS = primary; else = index of primary */
    uint32_t layer_type;       /**< 0 = local, 1 = global attention */
    uint32_t _pad[2];
} tape_manifest_entry_t;

/* -------------------------------------------------------------------------
 * Opaque tape handle (write and read)
 * -------------------------------------------------------------------------*/

typedef struct activation_tape_t activation_tape_t;

typedef struct {
    const char *output_path;
    const char *oracle_output_path;
    struct inference_session_t *session;
    struct sapphire_tokenizer_t *tokenizer;
    const struct model_spec *spec;
    const struct calibration_corpus_t *corpus;
    int sample_limit;
    int max_prompt_tokens;
    float hessian_proxy_strength;
    float hessian_proxy_floor;
} activation_tape_record_config_t;

/* -------------------------------------------------------------------------
 * Recording API
 * -------------------------------------------------------------------------*/

/**
 * @brief Run one forward pass per corpus sample; write all activation vectors
 *        to output_path.
 *
 * sample_limit ≤ 0 means use all corpus samples. Default recommended: 64.
 * Supports CPU and Vulkan backends. When oracle_output_path is set through
 * activation_tape_record_ex(), Vulkan recording also emits a Hessian sidecar
 * keyed to the tape CRC in the same pass.
 *
 * @return 0 on success, -1 on error.
 */
int activation_tape_record_ex(const activation_tape_record_config_t *config);

int activation_tape_record(const char                    *output_path,
                           struct inference_session_t    *session,
                           struct sapphire_tokenizer_t   *tokenizer,
                           const struct model_spec       *spec,
                           const struct calibration_corpus_t *corpus,
                           int                            sample_limit);

/**
 * @brief Request that any in-flight tape recording stop at the next safe point.
 *
 * Intended for SIGINT / SIGTERM handlers. The recorder polls this flag between
 * samples and will still flush any completed partial tape before returning.
 */
void activation_tape_request_stop(void);

/** @brief Clear any pending tape-recording stop request. */
void activation_tape_clear_stop_request(void);

/** @return non-zero if a tape-recording stop request has been raised. */
int activation_tape_stop_requested(void);

/* -------------------------------------------------------------------------
 * Playback API
 * -------------------------------------------------------------------------*/

/**
 * @brief Open an existing tape for random-access mmap reading.
 *
 * Uses MAP_NORESERVE so the full tape is mapped into virtual address space
 * without committing swap backing. With 64 GB RAM the kernel uses physical
 * pages as a transparent cache.
 *
 * @return Opaque handle on success, NULL on error.
 */
activation_tape_t *activation_tape_open(const char *tape_path);

/**
 * @brief Copy the activation vector for tensor_name / sample_idx into out_vector.
 *
 * tensor_name is resolved against the manifest, and layer tensors may fall
 * back to the modulo-equivalent teacher layer when the exact student entry is
 * missing.
 * out_vector must hold at least activation_tape_vector_dim() floats.
 *
 * @return 0 on success, -1 on error.
 */
int activation_tape_get_vector(const activation_tape_t *tape,
                               const char              *tensor_name,
                               int                      sample_idx,
                               float                   *out_vector);

/** @return primary manifest entry index for tensor_name, or -1 if not found. */
int activation_tape_entry_index(const activation_tape_t *tape,
                                const char              *tensor_name);

/** @return number of manifest entries in the tape, or 0 if tape is NULL. */
uint32_t activation_tape_entry_count(const activation_tape_t *tape);

/** @return manifest entry by index, or NULL if out of range. */
const tape_manifest_entry_t *activation_tape_entry(const activation_tape_t *tape,
                                                   uint32_t                 entry_idx);

/** @return number of samples in the tape, or -1 if tape is NULL. */
int activation_tape_sample_count(const activation_tape_t *tape);

/** @return vector_dim for tensor_name, or 0 if not found. */
uint32_t activation_tape_vector_dim(const activation_tape_t *tape,
                                    const char              *tensor_name);

/** @return pointer to the tape header, or NULL if tape is NULL. */
const tape_file_header_t *activation_tape_header(const activation_tape_t *tape);

/** @return CRC32 of the mapped tape contents, or 0 if tape is NULL. */
uint32_t activation_tape_crc32(const activation_tape_t *tape);

/**
 * @brief Issue MADV_WILLNEED on the data range of manifest entry entry_idx.
 *
 * Call while the GPU processes layer N to prefetch layer N+1 activation
 * vectors into the kernel page cache. Returns immediately; the kernel
 * fulfils the readahead asynchronously.
 */
void activation_tape_prefetch_entry(const activation_tape_t *tape,
                                    uint32_t                 entry_idx);

/**
 * @brief Issue MADV_DONTNEED on the data region for tensor_name.
 *
 * Call once after all sample vectors for a tensor have been read.
 * Allows the kernel to reclaim those tape pages, reducing RSS while
 * leaving other entries resident. No-op if tensor_name is not found.
 */
void activation_tape_discard_entry(const activation_tape_t *tape,
                                   const char              *tensor_name);

/** @brief Unmap and close all resources. */
void activation_tape_close(activation_tape_t *tape);

#ifdef __cplusplus
}
#endif

#endif /* ACTIVATION_TAPE_H */

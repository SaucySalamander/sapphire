/**
 * @file ternary_anchor.h
 * @brief Saliency-driven mixed-precision anchor representation for ternary tensors.
 *
 * This header defines the data structures and I/O interfaces for tensors that
 * combine a grouped ternary bulk (99.5%+ of weights) with a small BF16 anchor
 * patch (0.1-1.0% of weights) for outlier-sensitive coordinates.
 *
 * The hybrid representation is:
 *   W ≈ W_bulk-ternary(γ) + A_anchors
 *
 * where protected coordinates bypass ternary quantization entirely.
 */

#ifndef TERNARY_ANCHOR_H
#define TERNARY_ANCHOR_H

#include <stddef.h>
#include <stdint.h>

#include "ternary_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum supported anchor budget as a fraction of total weights. */
#define TERNARY_ANCHOR_MAX_BUDGET_FRACTION 0.01f

/* Default anchor budget for down_proj experiments. */
#define TERNARY_ANCHOR_DEFAULT_BUDGET_FRACTION 0.001f

/* Minimum supported anchor budget (10 ppm). */
#define TERNARY_ANCHOR_MIN_BUDGET_PPM 10u

/* Maximum supported anchor budget (10000 ppm = 1.0%). */
#define TERNARY_ANCHOR_MAX_BUDGET_PPM 10000u

/* Legacy binary sidecar magic number. */
#define TERNARY_ANCHOR_MAGIC 0x414E4331u /* "ANC1" */

/* Legacy binary sidecar version. */
#define TERNARY_ANCHOR_LEGACY_VERSION 1u

/* Current production safetensors anchor contract version. */
#define TERNARY_ANCHOR_VERSION 2u

/**
 * @brief Saliency score computation mode.
 */
typedef enum {
    TERNARY_ANCHOR_SALIENCY_NONE = 0,
    TERNARY_ANCHOR_SALIENCY_WEIGHT_TIMES_HESSIAN = 1,
    TERNARY_ANCHOR_SALIENCY_HESSIAN_ONLY = 2,
    TERNARY_ANCHOR_SALIENCY_WEIGHT_ONLY = 3
} ternary_anchor_saliency_mode_t;

/**
 * @brief Single anchor entry (8 bytes, cache-line friendly).
 *
 * Padded to 8 bytes to ensure aligned traversal in hot loops.
 * Row and col are 16-bit, sufficient for Gemma 3 7B tensors (max dim ~14336).
 */
typedef struct ternary_anchor_entry_t {
    uint16_t row;        /* Row index */
    uint16_t col;        /* Column index */
    uint16_t value_bf16; /* BF16-encoded value */
    uint16_t _pad;       /* Padding for 8-byte alignment */
} ternary_anchor_entry_t;

/**
 * @brief Anchor patch metadata stored in safetensors __metadata__.
 */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t rows;
    uint32_t cols;
    uint32_t anchor_count;
    uint32_t scale_group_size;
    uint32_t groups_per_row;
    uint32_t saliency_mode;
    uint32_t budget_ppm;          /* Parts-per-million budget */
    float saliency_cutoff;         /* Score threshold used for selection */
    float anchor_value_rms;        /* RMS of anchor values */
    float bulk_gamma_mean;         /* Mean gamma from ternary bulk (anchors excluded) */
    uint32_t max_row_nnz;          /* Maximum anchors in any single row */
    uint32_t crc32;                /* CRC32 of anchor payload */
} ternary_anchor_metadata_t;

/**
 * @brief Runtime view of the anchor patch.
 *
 * For efficient CPU inference, the loader builds row_offsets from sorted COO.
 * Row i's anchors span entries[row_offsets[i]..row_offsets[i+1]).
 */
typedef struct {
    const ternary_anchor_entry_t *entries; /* Sorted by (row, col) */
    const uint32_t *row_offsets;           /* CSR-like row pointer (rows+1 entries) */
    uint32_t anchor_count;
    uint32_t rows;
    uint32_t cols;
    ternary_anchor_metadata_t metadata;
    int owns_memory;                       /* 1 if this view owns entries/row_offsets */
} ternary_anchor_view_t;

/**
 * @brief Combined ternary bulk + anchor patch payload.
 *
 * This is the complete hybrid tensor representation for serialization.
 */
typedef struct {
    /* Ternary bulk (same as ternary_layer_t) */
    const uint8_t *packed_weights;
    size_t packed_weight_bytes;
    const void *scales;
    size_t scale_count;
    size_t scale_bytes;
    safetensors_dtype_t scale_dtype;
    uint32_t rows;
    uint32_t cols;
    uint32_t scale_group_size;
    uint32_t groups_per_row;
    ternary_io_integrity_t integrity;
    
    /* Anchor patch */
    const ternary_anchor_entry_t *anchor_entries;
    uint32_t anchor_count;
    ternary_anchor_metadata_t anchor_metadata;
} ternary_hybrid_layer_t;

/**
 * @brief Configuration for anchor selection during calibration.
 */
typedef struct {
    ternary_anchor_saliency_mode_t saliency_mode;
    uint32_t budget_ppm;              /* Target budget in parts-per-million */
    uint32_t min_anchors_per_tensor;  /* Minimum anchor count (floor) */
    uint32_t max_anchors_per_tensor;  /* Maximum anchor count (cap) */
    uint32_t max_anchors_per_row;     /* Per-row cap to ensure coverage */
    int deterministic_tiebreak;        /* Use (row, col) for tie resolution */
} ternary_anchor_config_t;

/**
 * @brief Result of anchor selection.
 */
typedef struct {
    ternary_anchor_entry_t *entries; /* Caller must free */
    uint32_t *row_offsets;           /* Caller must free */
    uint32_t anchor_count;
    uint32_t rows;
    uint32_t cols;
    float saliency_cutoff;
    float anchor_value_rms;
    uint32_t max_row_nnz;
} ternary_anchor_selection_t;

/* =========================================================================
 * Anchor Selection API
 * ========================================================================= */

/**
 * @brief Compute default anchor config for a tensor.
 */
ternary_anchor_config_t ternary_anchor_default_config(void);

/**
 * @brief Select anchors for a tensor using saliency ranking.
 *
 * @param bf16_weights  Original BF16 weights [rows x cols]
 * @param rows          Number of rows
 * @param cols          Number of columns
 * @param hessian_diag  Hessian diagonal [cols] (NULL if not available)
 * @param config        Selection configuration
 * @param out_selection Output selection (caller owns memory)
 * @return 0 on success, -1 on error
 */
int ternary_anchor_select(const uint16_t *bf16_weights,
                          uint32_t rows,
                          uint32_t cols,
                          const float *hessian_diag,
                          const ternary_anchor_config_t *config,
                          ternary_anchor_selection_t *out_selection);

/**
 * @brief Free memory owned by an anchor selection.
 */
void ternary_anchor_selection_free(ternary_anchor_selection_t *selection);

/* =========================================================================
 * Anchor I/O API
 * ========================================================================= */

/**
 * @brief Write a hybrid ternary+anchor tensor to a directory.
 *
 * Creates two payload files:
 * - <tensor_name>.packed : packed ternary symbols
 * - <tensor_name>.scales : grouped FP32 scales
 * - <tensor_name>.anchors.safetensors : explicit anchor COO payload tensors
 *
 * @param output_dir   Output directory path
 * @param tensor_name  Tensor name for file naming
 * @param layer        Hybrid layer payload
 * @param out_crc32    Receives CRC32 of written payload
 * @return 0 on success, -1 on error
 */
int ternary_anchor_write_layer(const char *output_dir,
                               const char *tensor_name,
                               const ternary_hybrid_layer_t *layer,
                               uint32_t *out_crc32);

/**
 * @brief Append a hybrid tensor entry to the manifest.
 *
 * Writes a manifest line with format:
 *   <tensor_name>\t<bulk_file_name>\t<rows>\t<cols>\t<bytes>\t<bulk_crc32>\tanchor\t<anchor_file_name>\t<anchor_count>
 */
int ternary_anchor_append_manifest(const char *output_dir,
                                   const char *tensor_name,
                                   const char *bulk_file_name,
                                   const char *anchor_file_name,
                                   const ternary_hybrid_layer_t *layer,
                                   uint32_t bulk_crc32);

/**
 * @brief Load an anchor patch from a production safetensors sidecar or legacy binary sidecar.
 *
 * @param anchor_path  Path to the anchor sidecar named in manifest.tsv
 * @param out_view     Output view (caller must release with ternary_anchor_view_release)
 * @return 0 on success, -1 on error
 */
int ternary_anchor_load(const char *anchor_path,
                        ternary_anchor_view_t *out_view);

/**
 * @brief Load an anchor patch and validate it belongs to a specific tensor.
 *
 * For safetensors sidecars, the loader validates the expected tensor name
 * against the sidecar metadata and the `<tensor_name>.anchor_entries` tensor.
 * Legacy binary sidecars do not carry tensor names and are still accepted
 * during the migration window.
 *
 * @param anchor_path            Path to the anchor sidecar named in manifest.tsv
 * @param expected_tensor_name   Tensor name from manifest.tsv (NULL disables the check)
 * @param out_view               Output view (caller must release with ternary_anchor_view_release)
 * @return 0 on success, -1 on error
 */
int ternary_anchor_load_for_tensor(const char *anchor_path,
                                   const char *expected_tensor_name,
                                   ternary_anchor_view_t *out_view);

/**
 * @brief Release memory owned by an anchor view.
 */
void ternary_anchor_view_release(ternary_anchor_view_t *view);

/**
 * @brief Build row_offsets from sorted COO entries.
 *
 * @param entries      Sorted anchor entries
 * @param anchor_count Number of entries
 * @param rows         Number of rows
 * @param out_offsets  Output array (rows+1 entries, caller allocates)
 * @return 0 on success, -1 on error
 */
int ternary_anchor_build_row_offsets(const ternary_anchor_entry_t *entries,
                                     uint32_t anchor_count,
                                     uint32_t rows,
                                     uint32_t *out_offsets);

/* =========================================================================
 * Validation and Debug API
 * ========================================================================= */

/**
 * @brief Validate anchor metadata consistency.
 *
 * Checks that row/col indices are in bounds, entries are sorted,
 * and metadata matches the payload.
 *
 * @return 0 if valid, -1 if invalid (errors logged)
 */
int ternary_anchor_validate(const ternary_anchor_view_t *view);

/**
 * @brief Compute anchor contribution for a single row.
 *
 * For debugging: computes sum of anchor[j].value * x[anchor[j].col] for row i.
 *
 * @param view   Anchor view
 * @param row    Row index
 * @param x      Input vector [cols]
 * @return Anchor contribution for this row
 */
float ternary_anchor_row_dot(const ternary_anchor_view_t *view,
                             uint32_t row,
                             const float *x);

/**
 * @brief Check if a coordinate is protected by an anchor.
 *
 * Binary search in the sorted COO entries.
 *
 * @param view  Anchor view
 * @param row   Row index
 * @param col   Column index
 * @return 1 if protected, 0 if not
 */
int ternary_anchor_is_protected(const ternary_anchor_view_t *view,
                                uint32_t row,
                                uint32_t col);

/**
 * @brief Get saliency mode name for logging.
 */
const char *ternary_anchor_saliency_mode_name(ternary_anchor_saliency_mode_t mode);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_ANCHOR_H */

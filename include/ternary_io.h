/**
 * @file ternary_io.h
 * @brief Layer-wise ternary weight mmap and safetensors write helpers.
 *
 * Prompt 1 implementation shell for Sapphire's 1.58-bit pipeline.
 * File ownership remains in src/io/ and all reads use safetensors mmap paths.
 */

#ifndef TERNARY_IO_H
#define TERNARY_IO_H

#include <stddef.h>
#include <stdint.h>

#include "safetensors_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TERNARY_PACKED_WEIGHTS_PER_BYTE 4u

typedef enum {
    TERNARY_IO_INTEGRITY_NONE = 0,
    TERNARY_IO_INTEGRITY_CRC32 = 1
} ternary_io_integrity_t;

/**
 * @brief Raw mmapped BF16 layer view backed by an open safetensors file.
 *
 * Ownership: caller must release with io_unmap_layer_bf16().
 */
typedef struct {
    safetensors_file_t *file;
    const uint16_t *bf16_weights;
    size_t weight_count;
    size_t weight_bytes;
    uint32_t rows;
    uint32_t cols;
    char tensor_name[256];
} ternary_bf16_layer_map_t;

/**
 * @brief Ternary layer payload prepared for safetensors serialization.
 *
 * `packed_weights` stores 2-bit packed symbols with 4 weights per byte.
 * `scales` stores one FP32/FP16 scale per row or output channel.
 */
typedef struct {
    const uint8_t *packed_weights;
    size_t packed_weight_bytes;
    const void *scales;
    size_t scale_count;
    size_t scale_bytes;
    safetensors_dtype_t scale_dtype;
    uint32_t rows;
    uint32_t cols;
    ternary_io_integrity_t integrity;
} ternary_layer_t;

typedef struct ternary_layer_payload_t {
    uint8_t *packed_weights;
    size_t packed_weight_bytes;
    float *scales;
    size_t scale_count;
    size_t scale_bytes;
    uint32_t rows;
    uint32_t cols;
} ternary_layer_payload_t;

/**
 * @brief mmap a BF16 layer tensor from a safetensors file.
 *
 * Opens the safetensors file, locates the named tensor, validates it is BF16,
 * and exposes a raw mmapped pointer for layer-by-layer streaming.
 *
 * @return 0 on success, -1 on error.
 */
int io_mmap_layer_bf16(const char *safetensors_path,
                       const char *tensor_name,
                       ternary_bf16_layer_map_t *out_map);

/**
 * @brief Release a BF16 mmap view created by io_mmap_layer_bf16().
 */
void io_unmap_layer_bf16(ternary_bf16_layer_map_t *map);

/**
 * @brief Resolve the shard safetensors path for a named tensor in a sharded model.
 *
 * Reads model.safetensors.index.json from model_dir and walks weight_map to find
 * which shard file contains tensor_name.
 *
 * @param model_dir  Directory containing the sharded safetensors files and index.
 * @param tensor_name Fully-qualified tensor name (e.g. "language_model.model.layers.0.self_attn.q_proj.weight").
 * @return malloc'd path string (caller must free()), or NULL on error.
 */
char *io_resolve_shard_path(const char *model_dir, const char *tensor_name);

/**
 * @brief mmap a BF16 layer tensor from a sharded safetensors model.
 *
 * Automatically locates the correct shard via model.safetensors.index.json,
 * then delegates to io_mmap_layer_bf16().
 *
 * @param model_dir  Directory containing shard files and model.safetensors.index.json.
 * @param tensor_name Fully-qualified tensor name.
 * @param out_map    Receives the mapped layer view.
 * @return 0 on success, -1 on error.
 */
int io_mmap_layer_bf16_sharded(const char *model_dir,
                                const char *tensor_name,
                                ternary_bf16_layer_map_t *out_map);

/**
 * @brief Incremental CRC32 helper for non-ECC staging validation.
 *
 * Pass `0` as the initial CRC for a fresh computation.
 */
uint32_t io_crc32_update(uint32_t crc, const void *data, size_t size);

/**
 * @brief Write a single ternary layer payload to a safetensors file.
 *
 * The file contains two tensors:
 * - `<tensor_name>.packed`  : packed U8 ternary symbols
 * - `<tensor_name>.scales`  : row/channel scale buffer (F32 or F16)
 *
 * A CRC32 over packed weights and scales is emitted in `__metadata__`.
 *
 * @param output_path Path to the output safetensors file.
 * @param tensor_name Base tensor name.
 * @param layer Ternary layer payload.
 * @param out_crc32 Optional output checksum.
 *
 * @return 0 on success, -1 on error.
 */
int io_write_layer_ternary(const char *output_path,
                           const char *tensor_name,
                           const ternary_layer_t *layer,
                           uint32_t *out_crc32);

/**
 * @brief Ensure a model-wide ternary output directory exists.
 */
int io_prepare_ternary_output_dir(const char *output_dir);

/**
 * @brief Write a layer payload into an output directory and append a manifest entry.
 *
 * The output directory receives one safetensors file per tensor plus
 * `manifest.tsv` describing the conversion results.
 */
int io_write_layer_ternary_into_dir(const char *output_dir,
                                    const char *tensor_name,
                                    const ternary_layer_t *layer,
                                    uint32_t *out_crc32);

/**
 * @brief Load a previously written ternary layer payload from a safetensors file.
 *
 * The payload is allocated by the loader and must be released with
 * io_free_layer_ternary_payload().
 */
int io_load_layer_ternary_payload(const char *layer_path,
                                  const char *tensor_name,
                                  uint32_t expected_rows,
                                  uint32_t expected_cols,
                                  ternary_layer_payload_t *out_payload);

/**
 * @brief Release a payload loaded by io_load_layer_ternary_payload().
 */
void io_free_layer_ternary_payload(ternary_layer_payload_t *payload);

typedef struct {
    int converted_count;
    const char *tensor_name;
    uint32_t crc32;
    float baseline_mean_nll;
    float current_mean_nll;
    float mean_kl;
    float max_kl;
    float top1_agreement;
    int sample_count;
} ternary_validation_checkpoint_t;

/**
 * @brief Append a validation checkpoint row for full-model ternary conversion.
 */
int io_append_validation_checkpoint(const char *output_dir,
                                    const ternary_validation_checkpoint_t *checkpoint);

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_IO_H */
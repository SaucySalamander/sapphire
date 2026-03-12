/**
 * @file ternary_io.c
 * @brief Prompt 1 ternary I/O shell for Sapphire.
 *
 * Responsibilities:
 * - mmap BF16 layer tensors for streaming calibration
 * - validate staging copies with CRC32 for non-ECC systems
 * - serialize packed 2-bit weights and row/channel scales to safetensors
 */

#include "ternary_io.h"

#include "file_reader.h"
#include "log.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const uint32_t g_crc32_table[256] = {
    0x00000000u, 0x77073096u, 0xEE0E612Cu, 0x990951BAu, 0x076DC419u, 0x706AF48Fu, 0xE963A535u, 0x9E6495A3u,
    0x0EDB8832u, 0x79DCB8A4u, 0xE0D5E91Eu, 0x97D2D988u, 0x09B64C2Bu, 0x7EB17CBDu, 0xE7B82D07u, 0x90BF1D91u,
    0x1DB71064u, 0x6AB020F2u, 0xF3B97148u, 0x84BE41DEu, 0x1ADAD47Du, 0x6DDDE4EBu, 0xF4D4B551u, 0x83D385C7u,
    0x136C9856u, 0x646BA8C0u, 0xFD62F97Au, 0x8A65C9ECu, 0x14015C4Fu, 0x63066CD9u, 0xFA0F3D63u, 0x8D080DF5u,
    0x3B6E20C8u, 0x4C69105Eu, 0xD56041E4u, 0xA2677172u, 0x3C03E4D1u, 0x4B04D447u, 0xD20D85FDu, 0xA50AB56Bu,
    0x35B5A8FAu, 0x42B2986Cu, 0xDBBBC9D6u, 0xACBCF940u, 0x32D86CE3u, 0x45DF5C75u, 0xDCD60DCFu, 0xABD13D59u,
    0x26D930ACu, 0x51DE003Au, 0xC8D75180u, 0xBFD06116u, 0x21B4F4B5u, 0x56B3C423u, 0xCFBA9599u, 0xB8BDA50Fu,
    0x2802B89Eu, 0x5F058808u, 0xC60CD9B2u, 0xB10BE924u, 0x2F6F7C87u, 0x58684C11u, 0xC1611DABu, 0xB6662D3Du,
    0x76DC4190u, 0x01DB7106u, 0x98D220BCu, 0xEFD5102Au, 0x71B18589u, 0x06B6B51Fu, 0x9FBFE4A5u, 0xE8B8D433u,
    0x7807C9A2u, 0x0F00F934u, 0x9609A88Eu, 0xE10E9818u, 0x7F6A0DBBu, 0x086D3D2Du, 0x91646C97u, 0xE6635C01u,
    0x6B6B51F4u, 0x1C6C6162u, 0x856530D8u, 0xF262004Eu, 0x6C0695EDu, 0x1B01A57Bu, 0x8208F4C1u, 0xF50FC457u,
    0x65B0D9C6u, 0x12B7E950u, 0x8BBEB8EAu, 0xFCB9887Cu, 0x62DD1DDFu, 0x15DA2D49u, 0x8CD37CF3u, 0xFBD44C65u,
    0x4DB26158u, 0x3AB551CEu, 0xA3BC0074u, 0xD4BB30E2u, 0x4ADFA541u, 0x3DD895D7u, 0xA4D1C46Du, 0xD3D6F4FBu,
    0x4369E96Au, 0x346ED9FCu, 0xAD678846u, 0xDA60B8D0u, 0x44042D73u, 0x33031DE5u, 0xAA0A4C5Fu, 0xDD0D7CC9u,
    0x5005713Cu, 0x270241AAu, 0xBE0B1010u, 0xC90C2086u, 0x5768B525u, 0x206F85B3u, 0xB966D409u, 0xCE61E49Fu,
    0x5EDEF90Eu, 0x29D9C998u, 0xB0D09822u, 0xC7D7A8B4u, 0x59B33D17u, 0x2EB40D81u, 0xB7BD5C3Bu, 0xC0BA6CADu,
    0xEDB88320u, 0x9ABFB3B6u, 0x03B6E20Cu, 0x74B1D29Au, 0xEAD54739u, 0x9DD277AFu, 0x04DB2615u, 0x73DC1683u,
    0xE3630B12u, 0x94643B84u, 0x0D6D6A3Eu, 0x7A6A5AA8u, 0xE40ECF0Bu, 0x9309FF9Du, 0x0A00AE27u, 0x7D079EB1u,
    0xF00F9344u, 0x8708A3D2u, 0x1E01F268u, 0x6906C2FEu, 0xF762575Du, 0x806567CBu, 0x196C3671u, 0x6E6B06E7u,
    0xFED41B76u, 0x89D32BE0u, 0x10DA7A5Au, 0x67DD4ACCu, 0xF9B9DF6Fu, 0x8EBEEFF9u, 0x17B7BE43u, 0x60B08ED5u,
    0xD6D6A3E8u, 0xA1D1937Eu, 0x38D8C2C4u, 0x4FDFF252u, 0xD1BB67F1u, 0xA6BC5767u, 0x3FB506DDu, 0x48B2364Bu,
    0xD80D2BDAu, 0xAF0A1B4Cu, 0x36034AF6u, 0x41047A60u, 0xDF60EFC3u, 0xA867DF55u, 0x316E8EEFu, 0x4669BE79u,
    0xCB61B38Cu, 0xBC66831Au, 0x256FD2A0u, 0x5268E236u, 0xCC0C7795u, 0xBB0B4703u, 0x220216B9u, 0x5505262Fu,
    0xC5BA3BBEu, 0xB2BD0B28u, 0x2BB45A92u, 0x5CB36A04u, 0xC2D7FFA7u, 0xB5D0CF31u, 0x2CD99E8Bu, 0x5BDEAE1Du,
    0x9B64C2B0u, 0xEC63F226u, 0x756AA39Cu, 0x026D930Au, 0x9C0906A9u, 0xEB0E363Fu, 0x72076785u, 0x05005713u,
    0x95BF4A82u, 0xE2B87A14u, 0x7BB12BAEu, 0x0CB61B38u, 0x92D28E9Bu, 0xE5D5BE0Du, 0x7CDCEFB7u, 0x0BDBDF21u,
    0x86D3D2D4u, 0xF1D4E242u, 0x68DDB3F8u, 0x1FDA836Eu, 0x81BE16CDu, 0xF6B9265Bu, 0x6FB077E1u, 0x18B74777u,
    0x88085AE6u, 0xFF0F6A70u, 0x66063BCAu, 0x11010B5Cu, 0x8F659EFFu, 0xF862AE69u, 0x616BFFD3u, 0x166CCF45u,
    0xA00AE278u, 0xD70DD2EEu, 0x4E048354u, 0x3903B3C2u, 0xA7672661u, 0xD06016F7u, 0x4969474Du, 0x3E6E77DBu,
    0xAED16A4Au, 0xD9D65ADCu, 0x40DF0B66u, 0x37D83BF0u, 0xA9BCAE53u, 0xDEBB9EC5u, 0x47B2CF7Fu, 0x30B5FFE9u,
    0xBDBDF21Cu, 0xCABAC28Au, 0x53B39330u, 0x24B4A3A6u, 0xBAD03605u, 0xCDD70693u, 0x54DE5729u, 0x23D967BFu,
    0xB3667A2Eu, 0xC4614AB8u, 0x5D681B02u, 0x2A6F2B94u, 0xB40BBE37u, 0xC30C8EA1u, 0x5A05DF1Bu, 0x2D02EF8Du
};

static int ternary_validate_tensor_name(const char *tensor_name) {
    if (!tensor_name || tensor_name[0] == '\0') {
        LOG_ERROR("ternary I/O: tensor name is NULL or empty");
        return -1;
    }
    if (strlen(tensor_name) >= sizeof(((ternary_bf16_layer_map_t *)0)->tensor_name)) {
        LOG_ERROR("ternary I/O: tensor name too long: %s", tensor_name);
        return -1;
    }
    return 0;
}

static void ternary_sanitize_tensor_name(const char *tensor_name,
                                         char *out_name,
                                         size_t out_name_size) {
    size_t i = 0;

    if (!out_name || out_name_size == 0) {
        return;
    }
    out_name[0] = '\0';
    if (!tensor_name) {
        return;
    }

    for (; tensor_name[i] != '\0' && i + 1 < out_name_size; ++i) {
        char ch = tensor_name[i];
        if ((ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '-' || ch == '_') {
            out_name[i] = ch;
        } else {
            out_name[i] = '_';
        }
    }
    out_name[i] = '\0';
}

static int ternary_append_manifest_entry(const char *output_dir,
                                         const char *tensor_name,
                                         const char *file_name,
                                         const ternary_layer_t *layer,
                                         uint32_t crc32) {
    char *manifest_path = NULL;
    char line[1024];
    int fd = -1;
    int rc = -1;
    int n = 0;
    ssize_t wrote = 0;

    manifest_path = construct_safe_path(output_dir, "manifest.tsv", NULL);
    if (!manifest_path) {
        return -1;
    }

    fd = open(manifest_path, O_CREAT | O_APPEND | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("ternary manifest: cannot open %s: %s", manifest_path, strerror(errno));
        free(manifest_path);
        return -1;
    }

    n = snprintf(line,
                 sizeof(line),
                 "%s\t%s\t%u\t%u\t%zu\t%08x\n",
                 tensor_name,
                 file_name,
                 layer->rows,
                 layer->cols,
                 layer->packed_weight_bytes,
                 crc32);
    if (n < 0 || (size_t)n >= sizeof(line)) {
        LOG_ERROR("ternary manifest: line construction failed for %s", tensor_name);
        goto cleanup;
    }

    wrote = write(fd, line, (size_t)n);
    if (wrote != (ssize_t)n) {
        LOG_ERROR("ternary manifest: write failed for %s: %s", manifest_path, strerror(errno));
        goto cleanup;
    }

    rc = 0;

cleanup:
    if (fd >= 0 && close(fd) != 0) {
        LOG_ERROR("ternary manifest: close failed for %s: %s", manifest_path, strerror(errno));
        rc = -1;
    }
    free(manifest_path);
    return rc;
}

int io_append_validation_checkpoint(const char *output_dir,
                                    const ternary_validation_checkpoint_t *checkpoint) {
    char *report_path = NULL;
    char line[1024];
    int fd = -1;
    int rc = -1;
    int n = 0;
    ssize_t wrote = 0;
    struct stat st;
    int write_header = 0;

    if (!output_dir || !checkpoint || !checkpoint->tensor_name ||
        checkpoint->converted_count < 0 || checkpoint->sample_count < 0) {
        LOG_ERROR("validation report: invalid arguments");
        return -1;
    }

    report_path = construct_safe_path(output_dir, "validation.tsv", NULL);
    if (!report_path) {
        return -1;
    }

    if (stat(report_path, &st) != 0 || st.st_size == 0) {
        write_header = 1;
    }

    fd = open(report_path, O_CREAT | O_APPEND | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("validation report: cannot open %s: %s", report_path, strerror(errno));
        free(report_path);
        return -1;
    }

    if (write_header) {
        static const char *header =
            "converted_count\ttensor_name\tcrc32\tbaseline_mean_nll\tcurrent_mean_nll\tbaseline_ppl\tcurrent_ppl\tmean_kl\tmax_kl\ttop1_agreement\tsample_count\n";
        wrote = write(fd, header, strlen(header));
        if (wrote != (ssize_t)strlen(header)) {
            LOG_ERROR("validation report: failed to write header for %s", report_path);
            goto cleanup;
        }
    }

    n = snprintf(line,
                 sizeof(line),
                 "%d\t%s\t%08x\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%d\n",
                 checkpoint->converted_count,
                 checkpoint->tensor_name,
                 checkpoint->crc32,
                 checkpoint->baseline_mean_nll,
                 checkpoint->current_mean_nll,
                 expf(checkpoint->baseline_mean_nll),
                 expf(checkpoint->current_mean_nll),
                 checkpoint->mean_kl,
                 checkpoint->max_kl,
                 checkpoint->top1_agreement,
                 checkpoint->sample_count);
    if (n < 0 || (size_t)n >= sizeof(line)) {
        LOG_ERROR("validation report: line construction failed for %s", checkpoint->tensor_name);
        goto cleanup;
    }

    wrote = write(fd, line, (size_t)n);
    if (wrote != (ssize_t)n) {
        LOG_ERROR("validation report: write failed for %s: %s", report_path, strerror(errno));
        goto cleanup;
    }

    rc = 0;

cleanup:
    if (fd >= 0 && close(fd) != 0) {
        LOG_ERROR("validation report: close failed for %s: %s", report_path, strerror(errno));
        rc = -1;
    }
    free(report_path);
    return rc;
}

static int ternary_get_matrix_shape(const safetensors_tensor_meta_t *meta,
                                    uint32_t *out_rows,
                                    uint32_t *out_cols,
                                    size_t *out_count) {
    size_t weight_count = 1;

    if (!meta || !out_rows || !out_cols || !out_count) {
        LOG_ERROR("ternary I/O: invalid matrix shape arguments");
        return -1;
    }

    if (meta->ndim <= 0 || meta->ndim > 2) {
        LOG_ERROR("ternary I/O: expected 1D/2D layer tensor, got ndim=%d for %s",
                  meta->ndim, meta->name);
        return -1;
    }

    for (int i = 0; i < meta->ndim; ++i) {
        if (meta->shape[i] == 0) {
            LOG_ERROR("ternary I/O: zero-sized dimension in tensor %s", meta->name);
            return -1;
        }
        weight_count *= meta->shape[i];
    }

    *out_rows = meta->shape[0];
    *out_cols = (meta->ndim == 1) ? 1u : meta->shape[1];
    *out_count = weight_count;
    return 0;
}

static const char* ternary_scale_dtype_name(safetensors_dtype_t dtype) {
    switch (dtype) {
        case SAFETENSORS_F32: return "F32";
        case SAFETENSORS_F16: return "F16";
        default: return NULL;
    }
}

static int ternary_validate_layer_payload(const char *tensor_name,
                                          const ternary_layer_t *layer,
                                          size_t *out_packed_cols,
                                          const char **out_scale_dtype_name) {
    const char *scale_name = NULL;
    size_t packed_cols = 0;

    if (ternary_validate_tensor_name(tensor_name) != 0 || !layer) {
        return -1;
    }
    if (!layer->packed_weights || !layer->scales) {
        LOG_ERROR("ternary I/O: layer payload contains NULL buffers");
        return -1;
    }
    if (layer->rows == 0 || layer->cols == 0 || layer->scale_count == 0) {
        LOG_ERROR("ternary I/O: invalid rows/cols/scale_count for %s", tensor_name);
        return -1;
    }
    if (layer->scale_count != layer->rows) {
        LOG_ERROR("ternary I/O: scale_count=%zu must match rows=%u for %s",
                  layer->scale_count, layer->rows, tensor_name);
        return -1;
    }

    packed_cols = ((size_t)layer->cols + (TERNARY_PACKED_WEIGHTS_PER_BYTE - 1u))
                  / TERNARY_PACKED_WEIGHTS_PER_BYTE;
    if (layer->packed_weight_bytes != (size_t)layer->rows * packed_cols) {
        LOG_ERROR("ternary I/O: packed byte count mismatch for %s (have=%zu expected=%zu)",
                  tensor_name, layer->packed_weight_bytes, (size_t)layer->rows * packed_cols);
        return -1;
    }

    scale_name = ternary_scale_dtype_name(layer->scale_dtype);
    if (!scale_name) {
        LOG_ERROR("ternary I/O: unsupported scale dtype=%d for %s", (int)layer->scale_dtype, tensor_name);
        return -1;
    }

    if (layer->scale_dtype == SAFETENSORS_F32 && layer->scale_bytes != layer->scale_count * sizeof(float)) {
        LOG_ERROR("ternary I/O: invalid F32 scale byte count for %s", tensor_name);
        return -1;
    }
    if (layer->scale_dtype == SAFETENSORS_F16 && layer->scale_bytes != layer->scale_count * sizeof(uint16_t)) {
        LOG_ERROR("ternary I/O: invalid F16 scale byte count for %s", tensor_name);
        return -1;
    }

    if (out_packed_cols) *out_packed_cols = packed_cols;
    if (out_scale_dtype_name) *out_scale_dtype_name = scale_name;
    return 0;
}

uint32_t io_crc32_update(uint32_t crc, const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t state = crc ^ 0xFFFFFFFFu;

    if (!bytes && size != 0) {
        LOG_ERROR("io_crc32_update: NULL data with non-zero size");
        return 0u;
    }

    for (size_t i = 0; i < size; ++i) {
        state = g_crc32_table[(state ^ bytes[i]) & 0xFFu] ^ (state >> 8);
    }

    return state ^ 0xFFFFFFFFu;
}

int io_mmap_layer_bf16(const char *safetensors_path,
                       const char *tensor_name,
                       ternary_bf16_layer_map_t *out_map) {
    safetensors_file_t *file = NULL;
    const safetensors_tensor_meta_t *meta = NULL;
    const void *raw_ptr = NULL;
    size_t weight_count = 0;

    if (!safetensors_path || !out_map || ternary_validate_tensor_name(tensor_name) != 0) {
        LOG_ERROR("io_mmap_layer_bf16: invalid arguments");
        return -1;
    }

    memset(out_map, 0, sizeof(*out_map));

    file = safetensors_open(safetensors_path);
    if (!file) {
        LOG_ERROR("io_mmap_layer_bf16: failed to open %s", safetensors_path);
        return -1;
    }

    meta = safetensors_get_tensor_by_name(file, tensor_name);
    if (!meta) {
        LOG_ERROR("io_mmap_layer_bf16: tensor not found: %s", tensor_name);
        safetensors_close(file);
        return -1;
    }
    if (meta->dtype != SAFETENSORS_BF16) {
        LOG_ERROR("io_mmap_layer_bf16: tensor %s has dtype=%d, expected BF16",
                  tensor_name, (int)meta->dtype);
        safetensors_close(file);
        return -1;
    }
    if (ternary_get_matrix_shape(meta, &out_map->rows, &out_map->cols, &weight_count) != 0) {
        safetensors_close(file);
        return -1;
    }

    raw_ptr = safetensors_data_ptr(file, meta);
    if (!raw_ptr) {
        safetensors_close(file);
        return -1;
    }

    out_map->file = file;
    out_map->bf16_weights = (const uint16_t *)raw_ptr;
    out_map->weight_count = weight_count;
    out_map->weight_bytes = meta->size_bytes;
    memcpy(out_map->tensor_name, tensor_name, strlen(tensor_name) + 1u);

    LOG_INFO("Mapped BF16 layer tensor %s: rows=%u cols=%u bytes=%zu",
             out_map->tensor_name, out_map->rows, out_map->cols, out_map->weight_bytes);
    return 0;
}

void io_unmap_layer_bf16(ternary_bf16_layer_map_t *map) {
    if (!map) {
        return;
    }
    if (map->file) {
        safetensors_close(map->file);
    }
    memset(map, 0, sizeof(*map));
}

int io_write_layer_ternary(const char *output_path,
                           const char *tensor_name,
                           const ternary_layer_t *layer,
                           uint32_t *out_crc32) {
    char header[2048];
    const char *scale_dtype_name = NULL;
    size_t packed_cols = 0;
    uint32_t crc32 = 0;
    uint64_t header_len = 0;
    uint64_t packed_begin = 0;
    uint64_t packed_end = 0;
    uint64_t scale_begin = 0;
    uint64_t scale_end = 0;
    int fd = -1;
    int header_rc = 0;
    ssize_t wrote = 0;

    if (!output_path || !layer) {
        LOG_ERROR("io_write_layer_ternary: invalid arguments");
        return -1;
    }
    if (ternary_validate_layer_payload(tensor_name, layer, &packed_cols, &scale_dtype_name) != 0) {
        return -1;
    }

    crc32 = io_crc32_update(0u, layer->packed_weights, layer->packed_weight_bytes);
    crc32 = io_crc32_update(crc32, layer->scales, layer->scale_bytes);
    if (out_crc32) {
        *out_crc32 = crc32;
    }

    packed_begin = 0;
    packed_end = (uint64_t)layer->packed_weight_bytes;
    scale_begin = packed_end;
    scale_end = scale_begin + (uint64_t)layer->scale_bytes;

    header_rc = snprintf(
        header,
        sizeof(header),
        "{\"__metadata__\":{\"sapphire_quant\":\"ternary-1.58\",\"integrity\":\"crc32\",\"crc32\":\"%08x\",\"rows\":\"%u\",\"cols\":\"%u\"},"
        "\"%s.packed\":{\"dtype\":\"U8\",\"shape\":[%u,%zu],\"data_offsets\":[%llu,%llu]},"
        "\"%s.scales\":{\"dtype\":\"%s\",\"shape\":[%zu],\"data_offsets\":[%llu,%llu]}}",
        crc32,
        layer->rows,
        layer->cols,
        tensor_name,
        layer->rows,
        packed_cols,
        (unsigned long long)packed_begin,
        (unsigned long long)packed_end,
        tensor_name,
        scale_dtype_name,
        layer->scale_count,
        (unsigned long long)scale_begin,
        (unsigned long long)scale_end);
    if (header_rc < 0 || (size_t)header_rc >= sizeof(header)) {
        LOG_ERROR("io_write_layer_ternary: header construction failed for %s", tensor_name);
        return -1;
    }
    header_len = (uint64_t)header_rc;

    fd = open(output_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        LOG_ERROR("io_write_layer_ternary: cannot open %s: %s", output_path, strerror(errno));
        return -1;
    }

    wrote = write(fd, &header_len, sizeof(header_len));
    if (wrote != (ssize_t)sizeof(header_len)) {
        LOG_ERROR("io_write_layer_ternary: failed to write header length: %s", strerror(errno));
        close(fd);
        return -1;
    }
    wrote = write(fd, header, (size_t)header_len);
    if (wrote != (ssize_t)header_len) {
        LOG_ERROR("io_write_layer_ternary: failed to write JSON header: %s", strerror(errno));
        close(fd);
        return -1;
    }
    wrote = write(fd, layer->packed_weights, layer->packed_weight_bytes);
    if (wrote != (ssize_t)layer->packed_weight_bytes) {
        LOG_ERROR("io_write_layer_ternary: failed to write packed weights: %s", strerror(errno));
        close(fd);
        return -1;
    }
    wrote = write(fd, layer->scales, layer->scale_bytes);
    if (wrote != (ssize_t)layer->scale_bytes) {
        LOG_ERROR("io_write_layer_ternary: failed to write scales: %s", strerror(errno));
        close(fd);
        return -1;
    }

    if (close(fd) != 0) {
        LOG_ERROR("io_write_layer_ternary: close failed for %s: %s", output_path, strerror(errno));
        return -1;
    }

    LOG_INFO("Wrote ternary safetensors layer %s -> %s (packed=%zuB scales=%zuB crc32=%08x)",
             tensor_name, output_path, layer->packed_weight_bytes, layer->scale_bytes, crc32);
    return 0;
}

int io_prepare_ternary_output_dir(const char *output_dir) {
    struct stat st;

    if (!output_dir || output_dir[0] == '\0') {
        LOG_ERROR("io_prepare_ternary_output_dir: invalid output directory");
        return -1;
    }

    if (stat(output_dir, &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            LOG_ERROR("io_prepare_ternary_output_dir: path exists and is not a directory: %s", output_dir);
            return -1;
        }
        return 0;
    }

    if (mkdir(output_dir, 0755) != 0) {
        LOG_ERROR("io_prepare_ternary_output_dir: mkdir failed for %s: %s", output_dir, strerror(errno));
        return -1;
    }

    return 0;
}

int io_write_layer_ternary_into_dir(const char *output_dir,
                                    const char *tensor_name,
                                    const ternary_layer_t *layer,
                                    uint32_t *out_crc32) {
    char safe_name[320];
    char file_name[352];
    char *output_path = NULL;
    uint32_t crc32 = 0;
    int rc = -1;

    if (!output_dir || !tensor_name || !layer) {
        LOG_ERROR("io_write_layer_ternary_into_dir: invalid arguments");
        return -1;
    }
    if (io_prepare_ternary_output_dir(output_dir) != 0) {
        return -1;
    }

    ternary_sanitize_tensor_name(tensor_name, safe_name, sizeof(safe_name));
    if (safe_name[0] == '\0') {
        LOG_ERROR("io_write_layer_ternary_into_dir: failed to sanitize tensor name %s", tensor_name);
        return -1;
    }
    if (snprintf(file_name, sizeof(file_name), "%s.safetensors", safe_name) < 0) {
        LOG_ERROR("io_write_layer_ternary_into_dir: failed to build filename for %s", tensor_name);
        return -1;
    }

    output_path = construct_safe_path(output_dir, file_name, NULL);
    if (!output_path) {
        return -1;
    }

    rc = io_write_layer_ternary(output_path, tensor_name, layer, &crc32);
    if (rc == 0) {
        if (ternary_append_manifest_entry(output_dir, tensor_name, file_name, layer, crc32) != 0) {
            rc = -1;
        }
    }

    if (rc == 0 && out_crc32) {
        *out_crc32 = crc32;
    }

    free(output_path);
    return rc;
}
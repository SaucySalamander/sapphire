/**
 * @file safetensors_reader.c
 * @brief Safetensors format reader implementation with mmap-based zero-copy loading.
 *
 * Provides efficient loading of Hugging Face Safetensors files (.safetensors)
 * using memory-mapped I/O. Tensor data is accessed directly from the mmap
 * without buffering the entire file in memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>
#include <stddef.h>
#include "../include/kernels.h"
#include "../include/log.h"
#include "../include/safetensors_reader.h"
#include "../include/tensor.h"
#include "../include/utils.h"
#include "../include/model_spec.h"
#include "../include/tensor_mapper.h"
#include "../include/simple_json.h"

/**
 * @brief Opaque structure managing an open Safetensors file.
 *
 * Holds mmapped memory, file metadata, and parsed tensor information.
 */
typedef struct safetensors_file_t {
    int fd;                            // File descriptor
    void *mmap_ptr;                    // Pointer to mmapped memory
    size_t mmap_size;                  // Size of mmapped region
    
    safetensors_tensor_meta_t *tensors; // Array of tensor metadata
    int tensor_count;                  // Number of tensors
    
    uint64_t header_size;              // Size of JSON header in bytes
    char *json_header;                 // Allocated copy of JSON (for parsing)
} safetensors_file_t;

/**
 * @brief Convert Safetensors dtype string to enum.
 */
static safetensors_dtype_t safetensors_dtype_from_string(const char *s) {
    if (!s) return SAFETENSORS_UNKNOWN;
    
    if (strcmp(s, "F32") == 0 || strcmp(s, "float32") == 0) return SAFETENSORS_F32;
    if (strcmp(s, "BF16") == 0 || strcmp(s, "bfloat16") == 0) return SAFETENSORS_BF16;
    if (strcmp(s, "F16") == 0 || strcmp(s, "float16") == 0) return SAFETENSORS_F16;
    if (strcmp(s, "I32") == 0 || strcmp(s, "int32") == 0) return SAFETENSORS_I32;
    if (strcmp(s, "I64") == 0 || strcmp(s, "int64") == 0) return SAFETENSORS_I64;
    
    return SAFETENSORS_UNKNOWN;
}

/**
 * @brief Get size in bytes for a Safetensors dtype.
 */
// Note: Currently unused but retained for future quantization support
// static size_t safetensors_dtype_size(safetensors_dtype_t dtype) {
//     switch (dtype) {
//         case SAFETENSORS_F32: return 4;
//         case SAFETENSORS_BF16: return 2;
//         case SAFETENSORS_F16: return 2;
//         case SAFETENSORS_I32: return 4;
//         case SAFETENSORS_I64: return 8;
//         default: return 0;
//     }
// }

/**
 * @brief Parse the shape array [d1, d2, ...] for a tensor.
 */
static int parse_tensor_shape(sjson_cursor_t *c, safetensors_tensor_meta_t *meta) {
    if (!sjson_cursor_consume(c, '[')) return -1;
    
    int shape_idx = 0;
    while (shape_idx < 8) {
        char ch = sjson_cursor_peek(c);
        
        if (ch == ']') {
            sjson_cursor_consume(c, ']');
            break;
        }
        
        if (ch == ',') {
            sjson_cursor_consume(c, ',');
            continue;
        }
        
        uint64_t dim = 0;
        if (sjson_cursor_parse_u64(c, &dim) != 0) return -1;
        meta->shape[shape_idx++] = (uint32_t)dim;
    }
    
    meta->ndim = shape_idx;
    return 0;
}

/**
 * @brief Parse the metadata object fields for a single tensor.
 */
static int parse_tensor_fields(sjson_cursor_t *c, safetensors_tensor_meta_t *meta) {
    if (!sjson_cursor_consume(c, '{')) return -1;
    
    // Initialize tensor metadata defaults
    meta->ndim = 0;
    meta->offset = 0;
    meta->size_bytes = 0;
    meta->dtype = SAFETENSORS_UNKNOWN;
    
    while (1) {
        char ch = sjson_cursor_peek(c);
        
        if (ch == '}') {
            sjson_cursor_consume(c, '}');
            return 0;
        }
        
        if (ch == ',') {
            sjson_cursor_consume(c, ',');
            continue;
        }
        
        if (ch != '"') return -1;
        
        char field[64] = {0};
        if (sjson_cursor_parse_string(c, field, sizeof(field)) != 0) return -1;
        
        if (!sjson_cursor_consume(c, ':')) return -1;
        
        // Parse field value based on key
        if (strcmp(field, "shape") == 0) {
            if (parse_tensor_shape(c, meta) != 0) return -1;
        } else if (strcmp(field, "dtype") == 0) {
            char dtype_str[64] = {0};
            if (sjson_cursor_parse_string(c, dtype_str, sizeof(dtype_str)) != 0) return -1;
            meta->dtype = safetensors_dtype_from_string(dtype_str);
        } else if (strcmp(field, "data_offsets") == 0) {
            if (!sjson_cursor_consume(c, '[')) return -1;
            uint64_t start = 0, end = 0;
            if (sjson_cursor_parse_u64(c, &start) != 0) return -1;
            if (!sjson_cursor_consume(c, ',')) return -1;
            if (sjson_cursor_parse_u64(c, &end) != 0) return -1;
            if (!sjson_cursor_consume(c, ']')) return -1;
            meta->offset = start;
            meta->size_bytes = end - start;
        } else if (strcmp(field, "offset") == 0) {
            if (sjson_cursor_parse_u64(c, &meta->offset) != 0) return -1;
        } else if (strcmp(field, "size") == 0) {
            if (sjson_cursor_parse_u64(c, &meta->size_bytes) != 0) return -1;
        } else {
            if (sjson_cursor_skip_value(c) != 0) return -1;
        }
    }
}

/**
 * @brief Simple JSON parser for Safetensors header using Cursor API.
 *
 * Uses the low-level Cursor API for streaming, zero-allocation parsing.
 * Extracts tensor names, shapes, dtypes, and offsets from the JSON header.
 *
 * @param json JSON string to parse.
 * @param json_len Length of JSON string.
 * @param out_tensors Pointer to output array (must be pre-allocated).
 * @param max_tensors Maximum number of tensors to parse.
 * @param out_count Pointer to store actual tensor count.
 *
 * @return 0 on success, -1 on parse error.
 */
static int parse_safetensors_json(const char *json, size_t json_len,
                                   safetensors_tensor_meta_t *out_tensors,
                                   int max_tensors, int *out_count) {
    if (!json || !out_tensors || !out_count) return -1;
    
    *out_count = 0;
    sjson_cursor_t c = sjson_cursor_init(json, json_len);
    
    // Expect opening '{'
    if (!sjson_cursor_consume(&c, '{')) return -1;
    
    while (*out_count < max_tensors) {
        char ch = sjson_cursor_peek(&c);
        
        // End of root object
        if (ch == '}') {
            sjson_cursor_consume(&c, '}');
            return 0;
        }
        
        // Skip comma between entries
        if (ch == ',') {
            sjson_cursor_consume(&c, ',');
            continue;
        }
        
        // Expect string key (tensor name)
        if (ch != '"') break;
        
        // Parse tensor name
        char name[256] = {0};
        if (sjson_cursor_parse_string(&c, name, sizeof(name)) != 0) break;

        // Expect ':'
        if (!sjson_cursor_consume(&c, ':')) break;
        
        // Skip metadata entries (start with __)
        if (name[0] == '_' && name[1] == '_') {
            if (sjson_cursor_skip_value(&c) != 0) break;
            continue;
        }
        
        // Copy tensor name safely
        strncpy(out_tensors[*out_count].name, name, sizeof(out_tensors[*out_count].name) - 1);
        out_tensors[*out_count].name[sizeof(out_tensors[*out_count].name) - 1] = '\0';
        
        // Delegate parsing of tensor fields
        if (parse_tensor_fields(&c, &out_tensors[*out_count]) != 0) return -1;
        (*out_count)++;
    }
    
    return 0;
}

/* Helper: Open and mmap the safetensors file */
static int map_safetensors_file(const char* path, int* fd_out, void** mmap_ptr_out, size_t* mmap_size_out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("safetensors_open: cannot open %s: %s", path, strerror(errno));
        return -1;
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        LOG_ERROR("safetensors_open: fstat failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    size_t file_size = (size_t)sb.st_size;
    if (file_size < 8) {
        LOG_ERROR("safetensors_open: file too small (< 8 bytes)");
        close(fd);
        return -1;
    }

    void *mmap_ptr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
        LOG_ERROR("safetensors_open: mmap failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    *fd_out = fd;
    *mmap_ptr_out = mmap_ptr;
    *mmap_size_out = file_size;
    return 0;
}

/* Helper: Extract JSON header and parse metadata */
static int parse_header_metadata(void* mmap_ptr, size_t mmap_size,
                                 uint64_t* header_len_out, char** json_header_out,
                                 safetensors_tensor_meta_t** tensors_out, int* tensor_count_out) {
    uint64_t header_len = *(uint64_t*)mmap_ptr;
    if (header_len + 8 > mmap_size) {
        LOG_ERROR("safetensors_open: header length exceeds file size");
        LOG_ERROR("       File size: %zu bytes", mmap_size);
        LOG_ERROR("       Declared header length: %lu bytes", (unsigned long)header_len);
        return -1;
    }

    char *json_header = (char*)malloc(header_len + 1);
    if (!json_header) {
        LOG_ERROR("safetensors_open: malloc failed for header");
        return -1;
    }
    memcpy(json_header, (char*)mmap_ptr + 8, header_len);
    json_header[header_len] = '\0';

    safetensors_tensor_meta_t *tensors = (safetensors_tensor_meta_t*)malloc(
        1000 * sizeof(safetensors_tensor_meta_t));
    if (!tensors) {
        LOG_ERROR("safetensors_open: malloc failed for tensors");
        free(json_header);
        return -1;
    }

    int tensor_count = 0;
    if (parse_safetensors_json(json_header, header_len, tensors, 1000, &tensor_count) < 0) {
        LOG_ERROR("safetensors_open: failed to parse JSON header");
        free(tensors);
        free(json_header);
        return -1;
    }

    *header_len_out = header_len;
    *json_header_out = json_header;
    *tensors_out = tensors;
    *tensor_count_out = tensor_count;
    return 0;
}

/**
 * @brief Open a Safetensors file and parse header.
 */
safetensors_file_t* safetensors_open(const char *path) {
    if (!path) {
        LOG_ERROR("safetensors_open: path is NULL");
        return NULL;
    }

    int fd = -1;
    void *mmap_ptr = NULL;
    size_t mmap_size = 0;
    if (map_safetensors_file(path, &fd, &mmap_ptr, &mmap_size) != 0) {
        return NULL;
    }

    uint64_t header_len = 0;
    char *json_header = NULL;
    safetensors_tensor_meta_t *tensors = NULL;
    int tensor_count = 0;
    safetensors_file_t *st = NULL;

    if (parse_header_metadata(mmap_ptr, mmap_size, &header_len, &json_header, &tensors, &tensor_count) != 0) {
        goto cleanup;
    }

    st = (safetensors_file_t*)malloc(sizeof(safetensors_file_t));
    if (!st) {
        LOG_ERROR("safetensors_open: malloc failed");
        goto cleanup;
    }

    st->fd = fd;
    st->mmap_ptr = mmap_ptr;
    st->mmap_size = mmap_size;
    st->tensors = tensors;
    st->tensor_count = tensor_count;
    st->header_size = header_len;
    st->json_header = json_header;

    LOG_INFO("✓ Safetensors file opened: %s", path);
    LOG_INFO("  - Header size: %lu bytes", (unsigned long)header_len);
    LOG_INFO("  - File size: %zu bytes", mmap_size);
    LOG_INFO("  - Tensor count: %d", tensor_count);

    return st;

cleanup:
    if (tensors) free(tensors);
    if (json_header) free(json_header);
    if (mmap_ptr && mmap_ptr != MAP_FAILED) munmap(mmap_ptr, mmap_size);
    if (fd >= 0) close(fd);
    return NULL;
}

/**
 * @brief Get tensor count.
 */
int safetensors_tensor_count(const safetensors_file_t *st) {
    return st ? st->tensor_count : 0;
}

/**
 * @brief Get tensor metadata by index.
 */
const safetensors_tensor_meta_t* safetensors_get_tensor_by_index(
    const safetensors_file_t *st, int index) {
    if (!st || index < 0 || index >= st->tensor_count) {
        return NULL;
    }
    return &st->tensors[index];
}

/**
 * @brief Get tensor metadata by name.
 */
const safetensors_tensor_meta_t* safetensors_get_tensor_by_name(
    const safetensors_file_t *st, const char *name) {
    if (!st || !name) return NULL;
    
    for (int i = 0; i < st->tensor_count; i++) {
        if (strcmp(st->tensors[i].name, name) == 0) {
            return &st->tensors[i];
        }
    }
    return NULL;
}

/**
 * @brief Create a tensor reference pointing to mmapped data.
 */
tensor_t* safetensors_create_tensor_ref(safetensors_file_t *st,
                                        const safetensors_tensor_meta_t *meta) {
    if (!st || !meta) return NULL;
    
    // Map Safetensors dtype to Sapphire dtype
    tensor_dtype_t dtype;
    switch (meta->dtype) {
        case SAFETENSORS_F32:  dtype = DTYPE_F32; break;
        case SAFETENSORS_BF16: dtype = DTYPE_BF16; break;
        case SAFETENSORS_F16:  dtype = DTYPE_F16; break;
        default:
            LOG_ERROR("Unsupported dtype in safetensors: %d", meta->dtype);
            return NULL;
    }
    
    uint64_t data_section_start = 8 + st->header_size;
    
    // Validate we're not reading beyond the mmap
    if (data_section_start + meta->offset + meta->size_bytes > st->mmap_size) {
        LOG_ERROR("Tensor '%s' data extends beyond file", meta->name);
        return NULL;
    }

    // Special-case: for BF16 norm/bias tensors, elevate to F32 (requires copy + conversion)
    if (meta->dtype == SAFETENSORS_BF16 && 
        (meta->ndim == 1 || strstr(meta->name, "norm") || strstr(meta->name, "bias"))) {
        
        tensor_t *t = tensor_create(meta->ndim, (int*)meta->shape, DTYPE_F32);
        if (!t) return NULL;
        
        size_t numel = 1;
        for (int i = 0; i < meta->ndim; i++) numel *= meta->shape[i];
        
        const uint16_t *src = (const uint16_t *)((char*)st->mmap_ptr + data_section_start + meta->offset);
        float *dst = (float *)tensor_data_mutable(t);
        
        // Use vectorized conversion assistant
        bf16_to_f32_vec(dst, src, (int)numel);
        
        return t;
    }

    // ZERO-COPY PATH: Create a view directly into the mmapped file data.
    // This keeps the massive projection weights in BF16/F32 in the mmapped file,
    // satisfying the 8GB RAM constraint.
    void *mmap_offset = (char*)st->mmap_ptr + data_section_start + meta->offset;
    tensor_t *t = tensor_create_view(dtype, meta->ndim, (int*)meta->shape, mmap_offset);
    
    return t;
}

/**
 * @brief Load tensor data with copy.
 */
tensor_t* safetensors_load_tensor_copy(const safetensors_file_t *st,
                                       const safetensors_tensor_meta_t *meta) {
    if (!st || !meta) return NULL;
    
    tensor_t *t = safetensors_create_tensor_ref((safetensors_file_t*)st, meta);
    return t;
}

/**
 * @brief Close Safetensors file.
 */
void safetensors_close(safetensors_file_t *st) {
    if (!st) return;
    
    if (st->mmap_ptr) {
        munmap(st->mmap_ptr, st->mmap_size);
    }
    
    if (st->fd >= 0) {
        close(st->fd);
    }
    
    if (st->tensors) {
        free(st->tensors);
    }
    
    if (st->json_header) {
        free(st->json_header);
    }
    
    free(st);
}

/**
 * @brief Print Safetensors file information.
 */
void safetensors_print_info(const safetensors_file_t *st) {
    if (!st) {
        LOG_INFO("NULL safetensors file");
        return;
    }
    
    LOG_INFO("================================================================================");
    LOG_INFO("                         Safetensors File Information");
    LOG_INFO("================================================================================");
    LOG_INFO("Header size: %lu bytes", (unsigned long)st->header_size);
    LOG_INFO("File size: %zu bytes", st->mmap_size);
    LOG_INFO("Tensor count: %d", st->tensor_count);
    
    for (int i = 0; i < st->tensor_count; i++) {
        const safetensors_tensor_meta_t *t = &st->tensors[i];
        
        char shape_buf[128];
        int pos = 0;
        pos += snprintf(shape_buf + pos, sizeof(shape_buf) - pos, "[");
        for (int j = 0; j < t->ndim; j++) {
            if (j > 0) pos += snprintf(shape_buf + pos, sizeof(shape_buf) - pos, ", ");
            pos += snprintf(shape_buf + pos, sizeof(shape_buf) - pos, "%u", t->shape[j]);
        }
        snprintf(shape_buf + pos, sizeof(shape_buf) - pos, "]");

        const char *dtype_str = "UNKNOWN";
        switch (t->dtype) {
            case SAFETENSORS_F32: dtype_str = "F32"; break;
            case SAFETENSORS_BF16: dtype_str = "BF16"; break;
            case SAFETENSORS_F16: dtype_str = "F16"; break;
            case SAFETENSORS_I32: dtype_str = "I32"; break;
            case SAFETENSORS_I64: dtype_str = "I64"; break;
            default: break;
        }

        LOG_INFO("Tensor %d: %s", i, t->name);
        LOG_INFO("  Shape: %s", shape_buf);
        LOG_INFO("  Dtype: %s", dtype_str);
        LOG_INFO("  Offset: %lu bytes", (unsigned long)t->offset);
        LOG_INFO("  Size: %lu bytes", (unsigned long)t->size_bytes);
    }
    
    LOG_INFO("================================================================================");
}

/* Helper for mapping individual layer fields */
typedef struct {
    const char *name;
    size_t offset;
} layer_field_map_t;

static const layer_field_map_t LAYER_FIELD_MAP[] = {
    {"norm_attn_weight", offsetof(model_layer_weights_t, norm_attn_weight)},
    {"norm_attn_post_weight", offsetof(model_layer_weights_t, norm_attn_post_weight)},
    {"q_proj_weight", offsetof(model_layer_weights_t, q_proj_weight)},
    {"k_proj_weight", offsetof(model_layer_weights_t, k_proj_weight)},
    {"v_proj_weight", offsetof(model_layer_weights_t, v_proj_weight)},
    {"q_norm_weight", offsetof(model_layer_weights_t, q_norm_weight)},
    {"k_norm_weight", offsetof(model_layer_weights_t, k_norm_weight)},
    {"out_proj_weight", offsetof(model_layer_weights_t, out_proj_weight)},
    {"norm_ffn_weight", offsetof(model_layer_weights_t, norm_ffn_weight)},
    {"norm_ffn_post_weight", offsetof(model_layer_weights_t, norm_ffn_post_weight)},
    {"up_proj_weight", offsetof(model_layer_weights_t, up_proj_weight)},
    {"gate_proj_weight", offsetof(model_layer_weights_t, gate_proj_weight)},
    {"down_proj_weight", offsetof(model_layer_weights_t, down_proj_weight)},
};

/* Handle case where a tensor is missing from the file */
static int handle_missing_tensor(safetensors_file_t* st,
                                 const tensor_map_entry_t* e,
                                 safetensors_dynamic_handler_t dyn_cb,
                                 llm_model_t* model) {
    // Special-case: allow lm_head to be tied to embeddings when missing
    if (e->field_name && strcmp(e->field_name, "lm_head_weight") == 0 && model->embedding_weight) {
        model->lm_head_weight = model->embedding_weight;
        tensor_ref_inc(model->lm_head_weight);
        return 0; // successfully handled
    }

    if (dyn_cb) {
        int dr = dyn_cb((const safetensors_file_t*)st, NULL, model);
        if (dr == 0) return 0; // handled
        if (dr == -1) return -1; // catastrophic error
        // dr == 1 means not handled, proceed to generic error
    }

    LOG_ERROR("Required tensor not found: %s", e->hf_name);
    return -1;
}

/* Map a tensor to block-level layer structures (blk.N) */
static int map_to_block_layer(llm_model_t* model, const tensor_map_entry_t* e, tensor_t* t) {
    int layer_idx = -1;
    if (sscanf(e->internal_key, "blk.%d", &layer_idx) != 1 || layer_idx < 0 || layer_idx >= SAPPHIRE_MAX_LAYERS) {
        LOG_ERROR("Invalid layer key: %s", e->internal_key);
        return -1;
    }

    model_layer_weights_t *lw = &model->layers[layer_idx];
    
    for (size_t i = 0; i < sizeof(LAYER_FIELD_MAP)/sizeof(LAYER_FIELD_MAP[0]); i++) {
        if (strcmp(e->field_name, LAYER_FIELD_MAP[i].name) == 0) {
            tensor_t **dest = (tensor_t **)((char *)lw + LAYER_FIELD_MAP[i].offset);
            *dest = t;
            
            if (i == 1) { // norm_attn_post_weight debug
                LOG_DEBUG("MAPPER: Assigned norm_attn_post_weight to layer %d", layer_idx);
            }
            return 0;
        }
    }

    LOG_ERROR("Unknown layer field: %s for key %s", e->field_name, e->internal_key);
    return -1;
}

/* Map a tensor to top-level model structures (embedding, final) */
static int map_to_top_level(llm_model_t* model, const tensor_map_entry_t* e, tensor_t* t) {
    if (strcmp(e->internal_key, "embedding") == 0) {
        model->embedding_weight = t;
        return 0;
    }

    if (strcmp(e->internal_key, "final") == 0) {
        if (strcmp(e->field_name, "norm_final_weight") == 0) {
            model->norm_final_weight = t;
            return 0;
        }
        if (strcmp(e->field_name, "lm_head_weight") == 0) {
            model->lm_head_weight = t;
            return 0;
        }
    }

    LOG_ERROR("Unknown top-level field/key: %s/%s", e->internal_key, e->field_name);
    return -1;
}

/**
 * Map all tensors in a Safetensors file using a static table and optional dynamic handler.
 * See include/tensor_mapper.h for the public declaration and semantics.
 */
int safetensors_map_all_tensors_with_table(safetensors_file_t* st,
                                           const tensor_map_entry_t* table,
                                           int table_size,
                                           safetensors_dynamic_handler_t dyn_cb,
                                           llm_model_t* model) {
    if (!st || !table || !model) {
        LOG_ERROR("Invalid argument to safetensors_map_all_tensors_with_table");
        return -1;
    }

    for (int i = 0; i < table_size; i++) {
        const tensor_map_entry_t *e = &table[i];
        if (!e->hf_name) continue;

        const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_name(st, e->hf_name);
        if (!meta) {
            if (handle_missing_tensor(st, e, dyn_cb, model) != 0) return -1;
            continue;
        }

        tensor_t *t = safetensors_create_tensor_ref(st, meta);
        if (!t) {
            LOG_ERROR("Failed to create tensor for %s (unsupported dtype or corrupt data)", e->hf_name);
            return -1;
        }

        int map_res = -1;
        if (strcmp(e->internal_key, "embedding") == 0 || strcmp(e->internal_key, "final") == 0) {
            map_res = map_to_top_level(model, e, t);
        } else if (strncmp(e->internal_key, "blk.", 4) == 0) {
            map_res = map_to_block_layer(model, e, t);
        } else {
            LOG_ERROR("Unknown internal key: %s", e->internal_key);
        }

        if (map_res != 0) return -1;
    }

    return 0;
}

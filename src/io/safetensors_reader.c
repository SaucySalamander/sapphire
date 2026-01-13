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
#include "../include/log.h"
#include "../include/safetensors_reader.h"
#include "../include/tensor.h"
#include "../include/utils.h"
#include "../include/model_spec.h"
#include "../include/tensor_mapper.h"

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
 * @brief Simple JSON parser for Safetensors header.
 *
 * Parses the JSON header to extract tensor names, shapes, dtypes, and offsets.
 * This is a minimal parser that assumes well-formed JSON (as per Safetensors spec).
 *
 * Expected format (simplified):
 * {
 *   "tensor_name": {"shape": [d1, d2, ...], "dtype": "F32", "offset": 0, "size": N},
 *   ...
 * }
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
    
    // Skip initial '{'
    const char *p = json;
    const char *end = json + json_len;
    
    while (p < end && *p != '{') p++;
    if (p >= end) return -1;
    p++;
    
    safetensors_tensor_meta_t *current = out_tensors;
    
    while (p < end && *out_count < max_tensors) {
        // Skip whitespace
        while (p < end && isspace(*p)) p++;
        
        // Check for end of object
        if (*p == '}') break;
        
        // Skip comma
        if (*p == ',') {
            p++;
            while (p < end && isspace(*p)) p++;
        }
        
        // Expect "tensor_name": ...
        if (*p != '"') break;
        p++;
        
        // Read tensor name
        char name[256] = {0};
        int name_len = 0;
        while (p < end && *p != '"' && name_len < 255) {
            name[name_len++] = *p++;
        }
        if (p >= end || *p != '"') break;
        p++; // skip closing "
        
        // Skip metadata entries (start with __)
        if (name[0] == '_' && name[1] == '_') {
            // Skip to next comma or close brace at this level
            int brace_depth = 0;
            while (p < end) {
                if (*p == '{') brace_depth++;
                else if (*p == '}') {
                    if (brace_depth == 0) break;
                    brace_depth--;
                }
                else if (*p == ',' && brace_depth == 0) {
                    p++;
                    break;
                }
                p++;
            }
            continue;
        }
        
        memcpy(current->name, name, name_len);
        current->name[name_len] = '\0';
        
        // Skip to ':' and then '{'
        while (p < end && *p != ':') p++;
        if (p >= end) break;
        p++;
        
        while (p < end && *p != '{') p++;
        if (p >= end) break;
        p++; // skip '{'
        
        // Parse shape, dtype, offset, size (or data_offsets)
        int shape_idx = 0;
        current->ndim = 0;
        current->offset = 0;
        current->size_bytes = 0;
        current->dtype = SAFETENSORS_UNKNOWN;
        
        while (p < end && *p != '}') {
            while (p < end && (isspace(*p) || *p == ',')) p++;
            if (*p != '"') break;
            p++;
            
            // Read key
            char key[64] = {0};
            int key_len = 0;
            while (p < end && *p != '"' && key_len < 63) {
                key[key_len++] = *p++;
            }
            if (p >= end) break;
            p++; // skip closing "
            
            while (p < end && *p != ':') p++;
            if (p >= end) break;
            p++;
            
            while (p < end && isspace(*p)) p++;
            
            // Parse value based on key
            if (strcmp(key, "shape") == 0) {
                // Expect '[' followed by numbers
                while (p < end && *p != '[') p++;
                if (p >= end) break;
                p++;
                
                shape_idx = 0;
                while (p < end && *p != ']' && shape_idx < 8) {
                    while (p < end && (isspace(*p) || *p == ',')) p++;
                    if (isdigit(*p)) {
                        current->shape[shape_idx++] = (uint32_t)strtoul(p, (char**)&p, 10);
                    } else {
                        break;
                    }
                }
                current->ndim = shape_idx;
                while (p < end && *p != ']') p++;
                if (p < end) p++;
                
            } else if (strcmp(key, "dtype") == 0) {
                // Expect string value
                while (p < end && *p != '"') p++;
                if (p >= end) break;
                p++;
                
                char dtype_str[64] = {0};
                int dtype_len = 0;
                while (p < end && *p != '"' && dtype_len < 63) {
                    dtype_str[dtype_len++] = *p++;
                }
                current->dtype = safetensors_dtype_from_string(dtype_str);
                
                if (p < end) p++; // skip closing "
                
            } else if (strcmp(key, "data_offsets") == 0) {
                // Modern format: [start_offset, end_offset]
                while (p < end && *p != '[') p++;
                if (p >= end) break;
                p++;
                
                // Skip whitespace
                while (p < end && isspace(*p)) p++;
                
                // Read start offset
                uint64_t start_offset = (uint64_t)strtoll(p, (char**)&p, 10);
                
                // Skip to comma
                while (p < end && isspace(*p)) p++;
                if (*p == ',') p++;
                while (p < end && isspace(*p)) p++;
                
                // Read end offset
                uint64_t end_offset = (uint64_t)strtoll(p, (char**)&p, 10);
                
                // Skip to closing bracket
                while (p < end && *p != ']') p++;
                if (p < end) p++;
                
                current->offset = start_offset;
                current->size_bytes = end_offset - start_offset;
                
            } else if (strcmp(key, "offset") == 0) {
                // Old format: offset field
                current->offset = (uint64_t)strtoll(p, (char**)&p, 10);
                
            } else if (strcmp(key, "size") == 0) {
                // Old format: size field
                current->size_bytes = (uint64_t)strtoll(p, (char**)&p, 10);
                
            } else {
                // Skip unknown key-value (including __metadata__)
                while (p < end && *p != ',' && *p != '}') p++;
            }
        }
        
        if (p < end && *p == '}') {
            p++; // skip '}'
            (*out_count)++;
            current++;
        }
    }
    
    return 0;
}

/**
 * @brief Open a Safetensors file and parse header.
 */
safetensors_file_t* safetensors_open(const char *path) {
    if (!path) {
        fprintf(stderr, "ERROR: safetensors_open: path is NULL\n");
        return NULL;
    }
    
    // Open file
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: safetensors_open: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    
    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        fprintf(stderr, "ERROR: safetensors_open: fstat failed: %s\n", strerror(errno));
        close(fd);
        return NULL;
    }
    
    size_t file_size = (size_t)sb.st_size;
    
    if (file_size < 8) {
        fprintf(stderr, "ERROR: safetensors_open: file too small (< 8 bytes)\n");
        close(fd);
        return NULL;
    }
    
    // mmap the entire file
    void *mmap_ptr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) {
        fprintf(stderr, "ERROR: safetensors_open: mmap failed: %s\n", strerror(errno));
        close(fd);
        return NULL;
    }
    
    // Read header length (first 8 bytes, little-endian)
    uint64_t header_len = *(uint64_t*)mmap_ptr;
    
    if (header_len + 8 > file_size) {
        fprintf(stderr, "ERROR: safetensors_open: header length exceeds file size\n");
        fprintf(stderr, "       File size: %zu bytes\n", file_size);
        fprintf(stderr, "       Declared header length: %lu bytes\n", (unsigned long)header_len);
        fprintf(stderr, "       This may be a Git LFS pointer file (usually 100-200 bytes)\n");
        fprintf(stderr, "       Download the actual model file and try again\n");
        munmap(mmap_ptr, file_size);
        close(fd);
        return NULL;
    }
    
    // Extract JSON header
    const char *json_start = (const char*)mmap_ptr + 8;
    char *json_header = (char*)malloc(header_len + 1);
    if (!json_header) {
        fprintf(stderr, "ERROR: safetensors_open: malloc failed\n");
        munmap(mmap_ptr, file_size);
        close(fd);
        return NULL;
    }
    
    memcpy(json_header, json_start, header_len);
    json_header[header_len] = '\0';
    
    // Parse JSON to extract tensor metadata
    // Allocate space for tensor metadata (up to 1000 tensors)
    safetensors_tensor_meta_t *tensors = (safetensors_tensor_meta_t*)malloc(
        1000 * sizeof(safetensors_tensor_meta_t));
    if (!tensors) {
        fprintf(stderr, "ERROR: safetensors_open: malloc failed for tensors\n");
        free(json_header);
        munmap(mmap_ptr, file_size);
        close(fd);
        return NULL;
    }
    
    int tensor_count = 0;
    if (parse_safetensors_json(json_header, header_len, tensors, 1000, &tensor_count) < 0) {
        fprintf(stderr, "ERROR: safetensors_open: failed to parse JSON header\n");
        free(tensors);
        free(json_header);
        munmap(mmap_ptr, file_size);
        close(fd);
        return NULL;
    }
    
    // Allocate safetensors_file_t structure
    safetensors_file_t *st = (safetensors_file_t*)malloc(sizeof(safetensors_file_t));
    if (!st) {
        fprintf(stderr, "ERROR: safetensors_open: malloc failed\n");
        free(tensors);
        free(json_header);
        munmap(mmap_ptr, file_size);
        close(fd);
        return NULL;
    }
    
    st->fd = fd;
    st->mmap_ptr = mmap_ptr;
    st->mmap_size = file_size;
    st->tensors = tensors;
    st->tensor_count = tensor_count;
    st->header_size = header_len;
    st->json_header = json_header;
    
    printf("✓ Safetensors file opened: %s\n", path);
    printf("  - Header size: %lu bytes\n", (unsigned long)header_len);
    printf("  - File size: %zu bytes\n", file_size);
    printf("  - Tensor count: %d\n", tensor_count);
    
    return st;
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
            fprintf(stderr, "ERROR: Unsupported dtype in safetensors: %d\n", meta->dtype);
            return NULL;
    }
    
    uint64_t data_section_start = 8 + st->header_size;
    
    // Validate we're not reading beyond the mmap
    if (data_section_start + meta->offset + meta->size_bytes > st->mmap_size) {
        fprintf(stderr, "ERROR: Tensor '%s' data extends beyond file\n", meta->name);
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
        printf("NULL safetensors file\n");
        return;
    }
    
    printf("================================================================================\n");
    printf("                         Safetensors File Information\n");
    printf("================================================================================\n");
    printf("Header size: %lu bytes\n", (unsigned long)st->header_size);
    printf("File size: %zu bytes\n", st->mmap_size);
    printf("Tensor count: %d\n\n", st->tensor_count);
    
    for (int i = 0; i < st->tensor_count; i++) {
        const safetensors_tensor_meta_t *t = &st->tensors[i];
        
        printf("Tensor %d: %s\n", i, t->name);
        printf("  Shape: [");
        for (int j = 0; j < t->ndim; j++) {
            printf("%u", t->shape[j]);
            if (j < t->ndim - 1) printf(", ");
        }
        printf("]\n");
        
        const char *dtype_str = "UNKNOWN";
        switch (t->dtype) {
            case SAFETENSORS_F32: dtype_str = "F32"; break;
            case SAFETENSORS_BF16: dtype_str = "BF16"; break;
            case SAFETENSORS_F16: dtype_str = "F16"; break;
            case SAFETENSORS_I32: dtype_str = "I32"; break;
            case SAFETENSORS_I64: dtype_str = "I64"; break;
            default: break;
        }
        printf("  Dtype: %s\n", dtype_str);
        printf("  Offset: %lu bytes\n", (unsigned long)t->offset);
        printf("  Size: %lu bytes\n\n", (unsigned long)t->size_bytes);
    }
    
    printf("================================================================================\n");
}

/**
 * Map all tensors in a Safetensors file using a static table and optional dynamic handler.
 * See include/tensor_mapper.h for the public declaration and semantics.
 */
int safetensors_map_all_tensors_with_table(safetensors_file_t* st,
                                           const tensor_map_entry_t* table,
                                           int table_size,
                                           safetensors_dynamic_handler_t dyn_cb,
                                           llm_model_t* model,
                                           char* error_msg, int max_error_len) {
    if (!st || !table || !model) {
        if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Invalid argument to safetensors_map_all_tensors_with_table");
        return -1;
    }

    for (int i = 0; i < table_size; i++) {
        const tensor_map_entry_t *e = &table[i];
        if (!e->hf_name) continue;

        const safetensors_tensor_meta_t *meta = safetensors_get_tensor_by_name(st, e->hf_name);
        if (!meta) {
            // Special-case: allow lm_head to be tied to embeddings when missing
            if (e->field_name && strcmp(e->field_name, "lm_head_weight") == 0 && model->embedding_weight) {
                model->lm_head_weight = model->embedding_weight;
                tensor_ref_inc(model->lm_head_weight);
                continue;
            }
            if (dyn_cb) {
                int dr = dyn_cb((const safetensors_file_t*)st, NULL, model, error_msg, max_error_len);
                if (dr == 0) {
                    continue; // dynamic handler handled this missing tensor
                } else if (dr == 1) {
                    if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Required tensor not found: %s", e->hf_name);
                    return -1;
                } else {
                    if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Dynamic handler error for tensor: %s", e->hf_name);
                    return -1;
                }
            }

            if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Required tensor not found: %s", e->hf_name);
            return -1;
        }

        tensor_t *t = safetensors_create_tensor_ref(st, meta);
        if (!t) {
            if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Failed to create tensor for %s (unsupported dtype or corrupt data)", e->hf_name);
            return -1;
        }

        // Top-level mappings
        if (strcmp(e->internal_key, "embedding") == 0) {
            model->embedding_weight = t;
            continue;
        }

        if (strcmp(e->internal_key, "final") == 0) {
            if (strcmp(e->field_name, "norm_final_weight") == 0) {
                model->norm_final_weight = t;
                continue;
            }
            if (strcmp(e->field_name, "lm_head_weight") == 0) {
                model->lm_head_weight = t;
                continue;
            }
            // Unknown field under final
            if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Unknown final field: %s", e->field_name);
            return -1;
        }

        // Per-layer mappings: expect keys like "blk.N"
        if (strncmp(e->internal_key, "blk.", 4) == 0) {
            int layer_idx = -1;
            if (sscanf(e->internal_key, "blk.%d", &layer_idx) != 1 || layer_idx < 0 || layer_idx >= SAPPHIRE_MAX_LAYERS) {
                if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Invalid layer key: %s", e->internal_key);
                return -1;
            }

            model_layer_weights_t *lw = &model->layers[layer_idx];

            // Map field names to struct members
            if (strcmp(e->field_name, "norm_attn_weight") == 0) lw->norm_attn_weight = t;
            else if (strcmp(e->field_name, "norm_attn_post_weight") == 0) {
                lw->norm_attn_post_weight = t;
                LOG_DEBUG("MAPPER: Assigned norm_attn_post_weight to layer %d", layer_idx);
            }
            else if (strcmp(e->field_name, "q_proj_weight") == 0) lw->q_proj_weight = t;
            else if (strcmp(e->field_name, "k_proj_weight") == 0) lw->k_proj_weight = t;
            else if (strcmp(e->field_name, "v_proj_weight") == 0) lw->v_proj_weight = t;
            else if (strcmp(e->field_name, "q_norm_weight") == 0) lw->q_norm_weight = t;
            else if (strcmp(e->field_name, "k_norm_weight") == 0) lw->k_norm_weight = t;
            else if (strcmp(e->field_name, "out_proj_weight") == 0) lw->out_proj_weight = t;
            else if (strcmp(e->field_name, "norm_ffn_weight") == 0) lw->norm_ffn_weight = t;
            else if (strcmp(e->field_name, "norm_ffn_post_weight") == 0) lw->norm_ffn_post_weight = t;
            else if (strcmp(e->field_name, "up_proj_weight") == 0) lw->up_proj_weight = t;
            else if (strcmp(e->field_name, "gate_proj_weight") == 0) lw->gate_proj_weight = t;
            else if (strcmp(e->field_name, "down_proj_weight") == 0) lw->down_proj_weight = t;
            else {
                if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Unknown layer field: %s for key %s", e->field_name, e->internal_key);
                return -1;
            }

            continue;
        }

        // Unknown internal_key
        if (error_msg && max_error_len > 0) snprintf(error_msg, max_error_len, "Unknown internal key: %s", e->internal_key);
        return -1;
    }

    return 0;
}

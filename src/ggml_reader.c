/**
 * @file ggml_reader.c
 * @brief GGML file format parser and tensor loader.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../include/ggml_model.h"
#include "../include/tensor.h"

#define GGML_MAGIC 0x67676d6c  // "ggml"
#define GGML_VERSION 1

/**
 * Read exactly n bytes from file.
 */
static int read_exact(FILE *fp, void *buf, size_t n) {
    size_t nread = fread(buf, 1, n, fp);
    return (nread == n) ? 0 : -1;
}

/**
 * Open and parse a GGML file header.
 */
FILE* ggml_file_open_and_parse_header(const char *filename, ggml_file_header_t *header) {
    if (!filename || !header) return NULL;
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open GGML file: %s\n", filename);
        return NULL;
    }
    
    // Read magic and version
    uint32_t magic, version;
    if (read_exact(fp, &magic, sizeof(magic)) < 0 ||
        read_exact(fp, &version, sizeof(version)) < 0) {
        fprintf(stderr, "ERROR: Failed to read GGML header magic/version\n");
        fclose(fp);
        return NULL;
    }
    
    if (magic != GGML_MAGIC) {
        fprintf(stderr, "ERROR: Invalid GGML magic number: 0x%08x\n", magic);
        fclose(fp);
        return NULL;
    }
    
    if (version != GGML_VERSION) {
        fprintf(stderr, "WARNING: GGML version mismatch: expected %u, got %u\n", 
                GGML_VERSION, version);
    }
    
    // Read tensor count
    uint32_t tensor_count;
    if (read_exact(fp, &tensor_count, sizeof(tensor_count)) < 0) {
        fprintf(stderr, "ERROR: Failed to read tensor count\n");
        fclose(fp);
        return NULL;
    }
    
    header->magic = magic;
    header->version = version;
    header->tensor_count = tensor_count;
    
    // Allocate tensor metadata array
    header->tensors = (ggml_tensor_meta_t *)malloc(tensor_count * sizeof(ggml_tensor_meta_t));
    if (!header->tensors) {
        fprintf(stderr, "ERROR: Failed to allocate tensor metadata\n");
        fclose(fp);
        return NULL;
    }
    
    // Parse each tensor's metadata
    for (uint32_t i = 0; i < tensor_count; i++) {
        ggml_tensor_meta_t *meta = &header->tensors[i];
        
        // Read name length and name
        uint32_t name_len;
        if (read_exact(fp, &name_len, sizeof(name_len)) < 0) {
            fprintf(stderr, "ERROR: Failed to read tensor %u name length\n", i);
            free(header->tensors);
            fclose(fp);
            return NULL;
        }
        
        if (name_len >= sizeof(meta->name)) {
            fprintf(stderr, "ERROR: Tensor %u name too long (%u bytes)\n", i, name_len);
            free(header->tensors);
            fclose(fp);
            return NULL;
        }
        
        if (read_exact(fp, meta->name, name_len) < 0) {
            fprintf(stderr, "ERROR: Failed to read tensor %u name\n", i);
            free(header->tensors);
            fclose(fp);
            return NULL;
        }
        meta->name[name_len] = '\0';
        
        // Read ndim and shape
        uint32_t ndim;
        if (read_exact(fp, &ndim, sizeof(ndim)) < 0) {
            fprintf(stderr, "ERROR: Failed to read tensor %u ndim\n", i);
            free(header->tensors);
            fclose(fp);
            return NULL;
        }
        
        if (ndim > 8) {
            fprintf(stderr, "ERROR: Tensor %u has too many dimensions (%u)\n", i, ndim);
            free(header->tensors);
            fclose(fp);
            return NULL;
        }
        
        meta->ndim = ndim;
        if (read_exact(fp, meta->shape, ndim * sizeof(uint32_t)) < 0) {
            fprintf(stderr, "ERROR: Failed to read tensor %u shape\n", i);
            free(header->tensors);
            fclose(fp);
            return NULL;
        }
        
        // Read dtype
        uint32_t dtype;
        if (read_exact(fp, &dtype, sizeof(dtype)) < 0) {
            fprintf(stderr, "ERROR: Failed to read tensor %u dtype\n", i);
            free(header->tensors);
            fclose(fp);
            return NULL;
        }
        meta->dtype = (tensor_dtype_t)dtype;
        
        // Calculate data size
        size_t nelems = 1;
        for (int j = 0; j < ndim; j++) {
            nelems *= meta->shape[j];
        }
        
        // Data size depends on dtype
        switch (meta->dtype) {
            case DTYPE_F32:
                meta->data_size = nelems * sizeof(float);
                break;
            case DTYPE_Q4_0:
                // Q4_0: 32 elements per block, 2 bytes overhead per block
                meta->data_size = (nelems / 32 + (nelems % 32 ? 1 : 0)) * (2 + 16);
                break;
            case DTYPE_Q8_0:
                // Q8_0: 32 elements per block, 1 byte scale + 32 bytes quantized
                meta->data_size = (nelems / 32 + (nelems % 32 ? 1 : 0)) * (1 + 32);
                break;
            default:
                fprintf(stderr, "ERROR: Unknown tensor dtype %d for tensor %u\n", meta->dtype, i);
                free(header->tensors);
                fclose(fp);
                return NULL;
        }
        
        // Record file offset
        meta->file_offset = ftell(fp);
        
        // Skip tensor data (we'll load on-demand)
        if (fseek(fp, (long)meta->data_size, SEEK_CUR) != 0) {
            fprintf(stderr, "ERROR: Failed to skip tensor %u data\n", i);
            free(header->tensors);
            fclose(fp);
            return NULL;
        }
    }
    
    return fp;
}

/**
 * Load a single tensor from file into memory.
 */
tensor_t* ggml_load_tensor(FILE *fp, const ggml_tensor_meta_t *meta) {
    if (!fp || !meta) return NULL;
    
    // Create tensor - convert shape to signed ints
    int shape_signed[8];
    for (int i = 0; i < meta->ndim; i++) {
        shape_signed[i] = (int)meta->shape[i];
    }
    
    tensor_t *t = tensor_create(meta->ndim, shape_signed, meta->dtype);
    if (!t) {
        fprintf(stderr, "ERROR: Failed to create tensor %s\n", meta->name);
        return NULL;
    }
    
    // Seek to data
    if (fseek(fp, (long)meta->file_offset, SEEK_SET) != 0) {
        fprintf(stderr, "ERROR: Failed to seek to tensor %s data\n", meta->name);
        tensor_release(t);
        return NULL;
    }
    
    // Read data
    const void *data_ptr = tensor_data(t);
    if (!data_ptr) {
        fprintf(stderr, "ERROR: Tensor %s has no data buffer\n", meta->name);
        tensor_release(t);
        return NULL;
    }
    
    // Cast away const for writing
    void *data = (void *)data_ptr;
    
    if (read_exact(fp, data, meta->data_size) < 0) {
        fprintf(stderr, "ERROR: Failed to read tensor %s data\n", meta->name);
        tensor_release(t);
        return NULL;
    }
    
    return t;
}

/**
 * Find a tensor by name in the file header.
 */
const ggml_tensor_meta_t* ggml_find_tensor_meta(const ggml_file_header_t *header, const char *name) {
    if (!header || !name) return NULL;
    
    for (uint32_t i = 0; i < header->tensor_count; i++) {
        if (strcmp(header->tensors[i].name, name) == 0) {
            return &header->tensors[i];
        }
    }
    
    return NULL;
}

/**
 * Free parsed header and metadata.
 */
void ggml_header_destroy(ggml_file_header_t *header) {
    if (header && header->tensors) {
        free(header->tensors);
        header->tensors = NULL;
    }
    header->tensor_count = 0;
}

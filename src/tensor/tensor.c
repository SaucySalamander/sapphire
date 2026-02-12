#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "../include/log.h"

// Concrete definition of the tensor structure (private to this .c file)
struct tensor_t {
    void *data;                  // Pointer to tensor data
    int ndim;                    // Number of dimensions (1-8)
    int shape[8];                // Shape array [ndim]
    tensor_dtype_t dtype;        // Data type (F32, Q4_0, Q8_0)
    memory_layout_t layout;      // Memory layout (row-major default)
    size_t nbytes;               // Total bytes allocated
    int ref_count;               // Reference count
    int is_external;             // Flag if data is managed elsewhere (e.g. mmap)
};

// ============================================================================
// Helper: Get element size for a dtype
// ============================================================================

/**
 * Returns element size in bytes.
 * For Q4_0: returns 1 (special handling: 0.5 bytes per element, or 2 elements per byte)
 * For Q8_0: returns 1 (1 byte per element)
 * For F16/BF16: returns 2 (2 bytes per element)
 * For F32: returns 4 (4 bytes per element)
 */
size_t dtype_element_size(tensor_dtype_t dtype) {
    switch (dtype) {
        case DTYPE_F32:   return 4;      // 32-bit float: 4 bytes per element
        case DTYPE_BF16:  return 2;      // 16-bit brain float: 2 bytes per element
        case DTYPE_F16:   return 2;      // 16-bit float: 2 bytes per element
        case DTYPE_Q4_0:  return 1;      // 4-bit quantized: 2 elements per 1 byte (special handling)
        case DTYPE_Q8_0:  return 1;      // 8-bit quantized: 1 byte per element
        default:          return 0;
    }
}

const char* dtype_name(tensor_dtype_t dtype) {
    switch (dtype) {
        case DTYPE_F32:   return "F32";
        case DTYPE_BF16:  return "BF16";
        case DTYPE_F16:   return "F16";
        case DTYPE_Q4_0:  return "Q4_0";
        case DTYPE_Q8_0:  return "Q8_0";
        default:          return "UNKNOWN";
    }
}

// ============================================================================
// Helper: Calculate tensor element count from shape
// ============================================================================

/**
 * Compute product of all shape dimensions.
 * Example: shape=[3, 4, 2] -> 3*4*2 = 24
 */
static size_t shape_product(int ndim, const int *shape) {
    size_t product = 1;
    for (int i = 0; i < ndim; i++) {
        product *= shape[i];
    }
    return product;
}

// ============================================================================
// Public API: tensor_create
// ============================================================================

/**
 * Allocate and initialize a tensor.
 * 
 * Memory allocation strategy:
 * - Calculate total elements: product of shape dimensions
 * - Calculate bytes needed: numel * dtype_element_size(dtype)
 * - Zero-initialize with calloc to ensure clean state
 * - Initialize reference count to 1 (caller owns reference)
 */
tensor_t* tensor_create(int ndim, const int *shape, tensor_dtype_t dtype) {
    // Validate inputs
    if (ndim < 1 || ndim > 8 || !shape) {
        LOG_ERROR("tensor_create invalid ndim=%d or null shape", ndim);
        return NULL;
    }

    // Allocate tensor struct
    tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
    if (!t) {
        LOG_ERROR("tensor_create malloc failed for tensor_t");
        return NULL;
    }

    // Copy shape and metadata
    t->ndim = ndim;
    t->dtype = dtype;
    t->layout = LAYOUT_ROW_MAJOR;  // Default layout
    memcpy(t->shape, shape, ndim * sizeof(int));

    // Calculate total elements and required bytes
    size_t numel = shape_product(ndim, shape);
    size_t element_size = dtype_element_size(dtype);
    
    // For Q4_0: 2 elements per byte, so nbytes = numel / 2
    // For Q8_0: 1 element per byte, so nbytes = numel * 1
    // For F32: 1 element per 4 bytes, so nbytes = numel * 4
    if (dtype == DTYPE_Q4_0) {
        t->nbytes = (numel + 1) / 2;   // Q4_0: 2 elements per byte (round up)
    } else {
        t->nbytes = numel * element_size;
    }

    // Allocate tensor data (zero-initialized)
    t->data = calloc(t->nbytes, 1);
    if (!t->data) {
        LOG_ERROR("tensor_create calloc failed for %zu bytes", t->nbytes);
        free(t);
        return NULL;
    }

    // Initialize reference count
    t->ref_count = 1;
    t->is_external = 0;

    return t;
}

// ============================================================================
// Public API: tensor_create_view
// ============================================================================

tensor_t* tensor_create_view(tensor_dtype_t dtype, int ndim, const int *shape, void *data) {
    tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
    if (!t) return NULL;
    
    t->data = data; // Point to existing memory (e.g. mmap offset)
    t->ndim = ndim;
    t->dtype = dtype;
    t->layout = LAYOUT_ROW_MAJOR;
    t->ref_count = 1;
    t->is_external = 1;

    size_t elements = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        elements *= shape[i];
    }
    t->nbytes = elements * dtype_element_size(dtype);
    
    return t;
}


// ============================================================================
// Public API: tensor_clone
// ============================================================================

/**
 * Deep copy of tensor.
 * Creates new tensor with identical metadata and data.
 */
tensor_t* tensor_clone(const tensor_t *src) {
    if (!src) return NULL;

    // Create new tensor with same shape and dtype
    tensor_t *clone = tensor_create(src->ndim, src->shape, src->dtype);
    if (!clone) return NULL;

    // Copy memory layout
    clone->layout = src->layout;

    // Copy data bytes
    memcpy(clone->data, src->data, src->nbytes);

    return clone;
}

int tensor_transpose(const tensor_t *src, tensor_t **dst) {
    if (!src || !dst) return -1;
    if (tensor_ndim(src) != 2) return -1;
    if (tensor_dtype(src) != DTYPE_F32) return -1;

    const int *shape = tensor_shape(src);
    int m = shape[0];
    int n = shape[1];

    const int dims[2] = { n, m };
    tensor_t *tnew = tensor_create(2, dims, DTYPE_F32);
    if (!tnew) return -1;

    const float *sdata = (const float *)tensor_data(src);
    float *ddata = (float *)tensor_data(tnew);

    // Transpose: d[i,j] = s[j,i]
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            ddata[j * m + i] = sdata[i * n + j];
        }
    }

    *dst = tnew;
    return 0;
}

// ============================================================================
// Public API: tensor_numel
// ============================================================================

size_t tensor_numel(const tensor_t *t) {
    if (!t) return 0;
    return shape_product(t->ndim, t->shape);
}

// ============================================================================
// Public API: tensor_get_f32
// ============================================================================

/**
 * Get element as F32 (with dequantization if needed).
 * 
 * For F32 tensors: direct read
 * For Q4_0/Q8_0: dequantize (slow! use for testing only)
 * 
 * Note: Quantized dequantization is placeholder.
 * Phase 4 will integrate actual dequantization from Phase 1.
 */
float tensor_get_f32(const tensor_t *t, size_t idx) {
    if (!t || idx >= tensor_numel(t)) {
        LOG_ERROR("tensor_get_f32 invalid index %zu", idx);
        return 0.0f;
    }

    if (t->dtype == DTYPE_F32) {
        // Direct float access
        float *fdata = (float *)t->data;
        return fdata[idx];
    } else if (t->dtype == DTYPE_Q4_0 || t->dtype == DTYPE_Q8_0) {
        // Placeholder: quantized dequantization
        // TODO: Integrate actual dequantization from Phase 1 (ggml_reader.c)
        // For now, return 0.0 to indicate unsupported
        LOG_WARN("tensor_get_f32 quantized dequantization not yet implemented");
        return 0.0f;
    }

    return 0.0f;
}

// ============================================================================
// Public API: tensor_set_f32
// ============================================================================

/**
 * Set element value (F32 tensors only).
 * Quantized tensors cannot be directly set element-wise.
 */
int tensor_set_f32(tensor_t *t, size_t idx, float val) {
    if (!t || idx >= tensor_numel(t)) {
        LOG_ERROR("tensor_set_f32 invalid index %zu", idx);
        return -1;
    }

    if (t->dtype != DTYPE_F32) {
        LOG_ERROR("tensor_set_f32 only works with DTYPE_F32");
        return -1;
    }

    float *fdata = (float *)t->data;
    fdata[idx] = val;
    return 0;
}

// ============================================================================
// Public API: Reference counting
// ============================================================================

void tensor_ref_inc(tensor_t *t) {
    if (t) {
        t->ref_count++;
    }
}

void tensor_release(tensor_t *t) {
    if (!t) return;

    t->ref_count--;
    if (t->ref_count <= 0) {
        if (t->data && !t->is_external) {
            free(t->data);
        }
        free(t);
    }
}

// ============================================================================
// Public API: tensor_print_info
// ============================================================================

/**
 * Pretty-print tensor metadata.
 * 
 * Example:
 *   Tensor shape=[3, 4] dtype=F32 layout=row-major 48 bytes ref_count=1
 */
void tensor_print_info(const tensor_t *t) {
    if (!t) {
        LOG_INFO("Tensor: NULL");
        return;
    }

    // Safely format shape into an allocated string
    // Max 12 chars per dimension (10 digits + comma + space) + 3 for brackets/null
    size_t shape_size = (size_t)t->ndim * 12 + 3;
    char *shape_buf = malloc(shape_size);
    if (!shape_buf) {
        LOG_ERROR("Failed to allocate shape buffer");
        return;
    }

    int pos = 0;
    int n = snprintf(shape_buf + pos, shape_size - pos, "[");
    if (n > 0 && (size_t)n < shape_size - pos) pos += n;

    for (int i = 0; i < t->ndim; i++) {
        if (i > 0) {
            n = snprintf(shape_buf + pos, shape_size - pos, ",");
            if (n > 0 && (size_t)n < shape_size - pos) pos += n;
        }
        n = snprintf(shape_buf + pos, shape_size - pos, "%d", t->shape[i]);
        if (n > 0 && (size_t)n < shape_size - pos) pos += n;
    }
    snprintf(shape_buf + pos, shape_size - pos, "]");

    LOG_INFO("Tensor shape=%s dtype=%s layout=%s %zu bytes ref_count=%d",
             shape_buf,
             dtype_name(t->dtype),
             t->layout == LAYOUT_ROW_MAJOR ? "row-major" : "col-major",
             t->nbytes,
             t->ref_count);
    
    free(shape_buf);
}

// ============================================================================
// Accessor helpers (non-breaking additions)
// ============================================================================

int tensor_ndim(const tensor_t *t) {
    if (!t) return 0;
    return t->ndim;
}

const int* tensor_shape(const tensor_t *t) {
    if (!t) return NULL;
    return t->shape;
}

const void* tensor_data(const tensor_t *t) {
    if (!t) return NULL;
    return t->data;
}

float* tensor_data_f32(const tensor_t *t) {
    if (!t) return NULL;
    if (t->dtype != DTYPE_F32) {
        LOG_WARN("tensor_data_f32 called on non-F32 tensor (dtype=%d)", t->dtype);
        return NULL;
    }
    return (float *)t->data;
}

void* tensor_data_mutable(const tensor_t *t) {
    if (!t) return NULL;
    return t->data;
}

tensor_dtype_t tensor_dtype(const tensor_t *t) {
    if (!t) return DTYPE_F32; // sensible default
    return t->dtype;
}

size_t tensor_nbytes(const tensor_t *t) {
    if (!t) return 0;
    return t->nbytes;
}

int tensor_ref_count(const tensor_t *t) {
    if (!t) return 0;
    return t->ref_count;
}

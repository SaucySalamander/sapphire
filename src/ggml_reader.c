// ggml_reader.c - Minimal GGML reader stubs for linking

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/sapphire.h"

int ggml_model_load(const char *path, ggml_tensor_t **out_tensors, size_t *out_count) {
    (void)path;
    if (out_tensors) *out_tensors = NULL;
    if (out_count) *out_count = 0;
    // Not implemented; caller should detect absence as non-fatal.
    return 0;
}

void ggml_model_free(ggml_tensor_t *tensors, size_t count) {
    (void)tensors; (void)count;
}

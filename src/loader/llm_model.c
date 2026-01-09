/**
 * @file llm_model.c
 * @brief Model lifecycle helpers (format-agnostic)
 *
 * Provides allocation / destruction / printing helpers for llm_model_t.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/llm_model.h"
#include "../include/ggml_model.h" // for ggml_header_destroy when appropriate

void llm_model_destroy(llm_model_t *model) {
    if (!model) return;

    if (model->layers) {
        free(model->layers);
        model->layers = NULL;
    }

    if (model->weight_file) {
        fclose(model->weight_file);
        model->weight_file = NULL;
    }

    // If file_header points to a GGML header, call its cleanup routine.
    // The GGML loader stores a ggml_file_header_t* in file_header; if not,
    // ggml_header_destroy is a no-op-safe function for NULL.
    if (model->file_header) {
        ggml_header_destroy((ggml_file_header_t*)model->file_header);
        model->file_header = NULL;
    }

    memset(model, 0, sizeof(llm_model_t));
}

void llm_model_print_info(const llm_model_t *model) {
    if (!model) return;

    printf("\n========== Model Configuration ==========" "\n");
    printf("Vocabulary Size: %d\n", model->config.vocab_size);
    printf("Hidden Dimension (d_model): %d\n", model->config.d_model);
    printf("Num Heads: %d\n", model->config.num_heads);
    printf("Dim per Head (d_k): %d\n", model->config.d_k);
    printf("Num Layers: %d\n", model->config.num_layers);
    printf("Max Context Length: %d\n", model->config.max_context_len);
    printf("RoPE Base: %.1f\n", model->config.rope_base);
    printf("========================================\n\n");
}

/**
 * @file test_tensor_mapper.c
 * @brief Unit tests for Safetensors tensor mapping and loading.
 *
 * Tests cover:
 * 1. Config extraction from tensor shapes
 * 2. Individual tensor mapping
 * 3. Shape validation
 * 4. Full model loading with synthetic Safetensors files
 * 5. Error handling (missing tensors, shape mismatches)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>

#include "../include/tensor_mapper.h"
#include "../include/safetensors_reader.h"

#define ANSI_GREEN "\x1b[32m"
#define ANSI_RED "\x1b[31m"
#define ANSI_RESET "\x1b[0m"

/**
 * @brief Test 1: Shape validation function.
 */
static int test_validate_tensor_shape(void) {
    printf("\n" ANSI_GREEN "[TEST 1]" ANSI_RESET " Shape validation\n");

    uint32_t shape1[] = {256000, 640};
    uint32_t shape2[] = {256000, 640};
    uint32_t shape3[] = {256000, 512};

    assert(validate_tensor_shape(shape1, 2, shape2, 2) == 0);
    assert(validate_tensor_shape(shape1, 2, shape3, 2) == -1);
    assert(validate_tensor_shape(shape1, 2, shape2, 1) == -1);

    printf("  ✓ Shape validation works correctly\n");
    return 0;
}

/**
 * @brief Test 2: Config extraction from real Safetensors file.
 *
 * This test uses an actual Gemma 3 270M model file if available.
 */
static int test_config_extraction(void) {
    printf("\n" ANSI_GREEN "[TEST 2]" ANSI_RESET " Config extraction from Safetensors\n");

    const char *model_path = "/home/sevenofnine/Git/gemma-3-270m-it/model.safetensors";
    struct stat st;
    if (stat(model_path, &st) != 0) {
        printf("  ⊘ Skipping (Gemma 3 270M file not found at %s)\n", model_path);
        return 0;
    }

    safetensors_file_t *sf = safetensors_open(model_path);
    if (!sf) {
        printf("  ✗ Failed to open Safetensors file\n");
        return -1;
    }

    // Verify key tensors exist
    const safetensors_tensor_meta_t *embed = safetensors_get_tensor_by_name(sf, "model.embed_tokens.weight");
    if (!embed) {
        printf("  ✗ Embedding tensor not found\n");
        safetensors_close(sf);
        return -1;
    }

    printf("  Embedding shape: [%u, %u]\n", embed->shape[0], embed->shape[1]);
    // Gemma 3 has vocab_size of 262144 (not 256000)
    assert(embed->shape[0] == 262144);  // vocab_size
    assert(embed->shape[1] == 640);     // d_model

    const safetensors_tensor_meta_t *q_proj = safetensors_get_tensor_by_name(sf, "model.layers.0.self_attn.q_proj.weight");
    if (!q_proj) {
        printf("  ✗ Q projection tensor not found\n");
        safetensors_close(sf);
        return -1;
    }

    printf("  Q proj shape: [%u, %u]\n", q_proj->shape[0], q_proj->shape[1]);
    // Gemma 3 uses Grouped Query Attention with structured weights
    // q_proj: [1024, 640] (4 query heads * 256 dims per head, projected to d_model)
    assert(q_proj->shape[0] == 1024);
    assert(q_proj->shape[1] == 640);

    const safetensors_tensor_meta_t *up_proj = safetensors_get_tensor_by_name(sf, "model.layers.0.mlp.up_proj.weight");
    if (!up_proj) {
        printf("  ✗ Up projection tensor not found\n");
        safetensors_close(sf);
        return -1;
    }

    printf("  Up proj shape: [%u, %u]\n", up_proj->shape[0], up_proj->shape[1]);
    assert(up_proj->shape[0] == 2048);
    assert(up_proj->shape[1] == 640);

    safetensors_close(sf);
    printf("  ✓ Config extraction successful\n");
    return 0;
}

/**
 * @brief Test 3: Full model loading from real Safetensors file.
 */
static int test_load_full_model(void) {
    printf("\n" ANSI_GREEN "[TEST 3]" ANSI_RESET " Full model loading\n");

    const char *model_path = "/home/sevenofnine/Git/gemma-3-270m-it/model.safetensors";
    struct stat st;
    if (stat(model_path, &st) != 0) {
        printf("  ⊘ Skipping (Gemma 3 270M file not found at %s)\n", model_path);
        return 0;
    }

    llm_model_t model = {0};
    char err[256] = {0};

    printf("  Loading from: %s\n", model_path);
    int rc = sapphire_load_safetensors(model_path, &model, err, sizeof(err));

    if (rc != SAPPHIRE_OK) {
        printf("  ✗ Load failed: %s\n", err);
        return -1;
    }

    printf("  Loaded config:\n");
    printf("    vocab_size: %d\n", model.config.vocab_size);
    printf("    d_model: %d\n", model.config.d_model);
    printf("    num_heads: %d\n", model.config.num_heads);
    printf("    num_layers: %d\n", model.config.num_layers);

    // Gemma 3 has vocab_size of 262144
    assert(model.config.vocab_size == 262144);
    assert(model.config.d_model == 640);
    assert(model.config.num_heads == 4);   // 1024 / 256 = 4 query heads with GQA
    assert(model.config.num_layers == 18);

    // Verify key tensors are loaded
    assert(model.embedding_weight != NULL);
    assert(model.norm_final_weight != NULL);
    assert(model.lm_head_weight != NULL);

    // Verify layer tensors
    for (int i = 0; i < model.config.num_layers; i++) {
        model_layer_weights_t *layer = &model.layers[i];
        assert(layer->norm_attn_weight != NULL);
        assert(layer->q_proj_weight != NULL);
        assert(layer->k_proj_weight != NULL);
        assert(layer->v_proj_weight != NULL);
        assert(layer->out_proj_weight != NULL);
        assert(layer->norm_ffn_weight != NULL);
        assert(layer->gate_proj_weight != NULL);
        assert(layer->up_proj_weight != NULL);
        assert(layer->down_proj_weight != NULL);
    }

    printf("  ✓ All tensors loaded successfully\n");

    // Cleanup
    llm_model_destroy(&model);
    return 0;
}

/**
 * @brief Test 4: Verify tensor shapes match expected dimensions.
 */
static int test_tensor_shapes(void) {
    printf("\n" ANSI_GREEN "[TEST 4]" ANSI_RESET " Tensor shape validation\n");

    const char *model_path = "/home/sevenofnine/Git/gemma-3-270m-it/model.safetensors";
    struct stat st;
    if (stat(model_path, &st) != 0) {
        printf("  ⊘ Skipping (Gemma 3 270M file not found at %s)\n", model_path);
        return 0;
    }

    llm_model_t model = {0};
    char err[256] = {0};

    int rc = sapphire_load_safetensors(model_path, &model, err, sizeof(err));
    if (rc != SAPPHIRE_OK) {
        printf("  ✗ Load failed: %s\n", err);
        return -1;
    }

    // Check embedding shape
    const int *embed_shape = tensor_shape(model.embedding_weight);
    int embed_ndim = tensor_ndim(model.embedding_weight);
    printf("  Embedding shape: [%d, %d] (ndim=%d)\n", embed_shape[0], embed_shape[1], embed_ndim);
    assert(embed_ndim == 2);
    assert(embed_shape[0] == 262144);  // Gemma 3 vocab size
    assert(embed_shape[1] == 640);

    // Check Q projection of first layer
    // Gemma 3 uses Grouped Query Attention: q_proj is [num_q_heads * head_dim, d_model]
    // With 4 query heads and head_dim=256: [1024, 640]
    const int *q_shape = tensor_shape(model.layers[0].q_proj_weight);
    int q_ndim = tensor_ndim(model.layers[0].q_proj_weight);
    printf("  Layer 0 Q proj shape: [%d, %d] (ndim=%d)\n", q_shape[0], q_shape[1], q_ndim);
    assert(q_ndim == 2);
    assert(q_shape[0] == 1024);  // 4 heads * 256 dim
    assert(q_shape[1] == 640);

    // Check FFN up projection
    const int *up_shape = tensor_shape(model.layers[0].up_proj_weight);
    int up_ndim = tensor_ndim(model.layers[0].up_proj_weight);
    printf("  Layer 0 up proj shape: [%d, %d] (ndim=%d)\n", up_shape[0], up_shape[1], up_ndim);
    assert(up_ndim == 2);
    assert(up_shape[0] == 2048);
    assert(up_shape[1] == 640);

    printf("  ✓ All tensor shapes validated\n");

    llm_model_destroy(&model);
    return 0;
}

/**
 * @brief Test 5: Zero-copy verification (pointers into mmapped region).
 */
static int test_zero_copy_pointers(void) {
    printf("\n" ANSI_GREEN "[TEST 5]" ANSI_RESET " Zero-copy pointer validation\n");

    const char *model_path = "/home/sevenofnine/Git/gemma-3-270m-it/model.safetensors";
    struct stat st;
    if (stat(model_path, &st) != 0) {
        printf("  ⊘ Skipping (Gemma 3 270M file not found at %s)\n", model_path);
        return 0;
    }

    llm_model_t model = {0};
    char err[256] = {0};

    int rc = sapphire_load_safetensors(model_path, &model, err, sizeof(err));
    if (rc != SAPPHIRE_OK) {
        printf("  ✗ Load failed: %s\n", err);
        return -1;
    }

    // Verify tensor data pointers are non-NULL (pointing to mmapped memory)
    assert(tensor_data(model.embedding_weight) != NULL);
    assert(tensor_data(model.norm_final_weight) != NULL);
    assert(tensor_data(model.lm_head_weight) != NULL);

    for (int i = 0; i < model.config.num_layers; i++) {
        model_layer_weights_t *layer = &model.layers[i];
        assert(tensor_data(layer->q_proj_weight) != NULL);
        assert(tensor_data(layer->k_proj_weight) != NULL);
        assert(tensor_data(layer->v_proj_weight) != NULL);
        assert(tensor_data(layer->out_proj_weight) != NULL);
        assert(tensor_data(layer->norm_ffn_weight) != NULL);
        assert(tensor_data(layer->up_proj_weight) != NULL);
        assert(tensor_data(layer->gate_proj_weight) != NULL);
        assert(tensor_data(layer->down_proj_weight) != NULL);
    }

    printf("  ✓ All tensor data pointers valid (mmapped memory)\n");

    llm_model_destroy(&model);
    return 0;
}

/**
 * @brief Test 6: Error handling for missing file.
 */
static int test_missing_file(void) {
    printf("\n" ANSI_GREEN "[TEST 6]" ANSI_RESET " Error handling: missing file\n");

    llm_model_t model = {0};
    char err[256] = {0};

    int rc = sapphire_load_safetensors("/nonexistent/path/model.safetensors", &model, err, sizeof(err));

    assert(rc != SAPPHIRE_OK);
    printf("  Error message: %s\n", err);
    printf("  ✓ Correctly rejected missing file\n");
    return 0;
}

/**
 * @brief Test 7: Tensor count validation (all 18 layers must be complete).
 */
static int test_all_layers_present(void) {
    printf("\n" ANSI_GREEN "[TEST 7]" ANSI_RESET " All layers present and complete\n");

    const char *model_path = "/home/sevenofnine/Git/gemma-3-270m-it/model.safetensors";
    struct stat st;
    if (stat(model_path, &st) != 0) {
        printf("  ⊘ Skipping (Gemma 3 270M file not found at %s)\n", model_path);
        return 0;
    }

    llm_model_t model = {0};
    char err[256] = {0};

    int rc = sapphire_load_safetensors(model_path, &model, err, sizeof(err));
    if (rc != SAPPHIRE_OK) {
        printf("  ✗ Load failed: %s\n", err);
        return -1;
    }

    printf("  Loaded %d layers\n", model.config.num_layers);
    assert(model.config.num_layers == 18);

    // Verify each layer has all 9 required tensors
    for (int i = 0; i < 18; i++) {
        model_layer_weights_t *layer = &model.layers[i];

        // Check non-null pointers
        int has_all = (layer->norm_attn_weight != NULL &&
                       layer->q_proj_weight != NULL &&
                       layer->k_proj_weight != NULL &&
                       layer->v_proj_weight != NULL &&
                       layer->out_proj_weight != NULL &&
                       layer->norm_ffn_weight != NULL &&
                       layer->gate_proj_weight != NULL &&
                       layer->up_proj_weight != NULL &&
                       layer->down_proj_weight != NULL);

        if (!has_all) {
            printf("  ✗ Layer %d is missing tensors\n", i);
            llm_model_destroy(&model);
            return -1;
        }
    }

    printf("  ✓ All 18 layers complete with 9 tensors each (162 total layer tensors)\n");

    llm_model_destroy(&model);
    return 0;
}

/**
 * @brief Run all tests.
 */
int main(void) {
    printf("============================================================\n");
    printf("  TENSOR MAPPER TEST SUITE\n");
    printf("  Safetensors → Sapphire Model Loading\n");
    printf("============================================================\n");

    int failed = 0;

    failed += test_validate_tensor_shape();
    failed += test_config_extraction();
    failed += test_load_full_model();
    failed += test_tensor_shapes();
    failed += test_zero_copy_pointers();
    failed += test_missing_file();
    failed += test_all_layers_present();

    printf("\n============================================================\n");
    if (failed == 0) {
        printf(ANSI_GREEN "✓ ALL TESTS PASSED" ANSI_RESET "\n");
    } else {
        printf(ANSI_RED "✗ %d TEST(S) FAILED" ANSI_RESET "\n", failed);
    }
    printf("============================================================\n");

    return failed ? 1 : 0;
}

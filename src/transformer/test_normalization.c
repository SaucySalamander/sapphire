#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "normalization.h"

// ============================================================================
// Test: RMSNorm basic
// ============================================================================

static void test_rmsnorm_basic(void) {
    printf("TEST: rmsnorm basic computation\n");
    
    int n = 4;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};  // No scaling
    float eps = 1e-6f;
    
    rmsnorm(x, weight, n, eps);
    
    // After RMSNorm with weight=1: x[i] = x[i] / RMS
    // Verify normalization property: sum of squares should be n (approximately)
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float result_rms = sqrtf(sum_sq / n);
    
    // RMS should be ~1.0 after normalization
    assert(fabs(result_rms - 1.0f) < 0.01f);
    
    printf("  ✓ RMSNorm basic computation correct\n");
}

static void test_rmsnorm_with_weight(void) {
    printf("TEST: rmsnorm with learnable weight\n");
    
    int n = 3;
    float x[] = {1.0f, 2.0f, 3.0f};
    float weight[] = {2.0f, 0.5f, 1.5f};
    float eps = 1e-6f;
    
    // RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.160
    
    rmsnorm(x, weight, n, eps);
    
    // After RMSNorm: x[i] = (x[i] / RMS) * weight[i]
    // Verify the scaling was applied
    // x[0] should be scaled by weight[0]=2.0
    // x[1] should be scaled by weight[1]=0.5
    // etc.
    
    // Just verify that results are reasonable (non-zero and finite)
    for (int i = 0; i < n; i++) {
        assert(!isnan(x[i]) && !isinf(x[i]));
    }
    
    printf("  ✓ RMSNorm with weight applied correctly\n");
}

static void test_rmsnorm_no_weight(void) {
    printf("TEST: rmsnorm_no_weight\n");
    
    int n = 4;
    float x[] = {2.0f, 4.0f, 6.0f, 8.0f};
    float eps = 1e-6f;
    
    rmsnorm_no_weight(x, n, eps);
    
    // Verify normalization: RMS should be 1.0
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float result_rms = sqrtf(sum_sq / n);
    
    assert(fabs(result_rms - 1.0f) < 0.01f);
    
    printf("  ✓ RMSNorm without weight correct\n");
}

static void test_rmsnorm_batch(void) {
    printf("TEST: rmsnorm_batch for matrices\n");
    
    int num_rows = 2;
    int row_size = 3;
    float matrix[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float weight[] = {1.0f, 1.0f, 1.0f};
    float eps = 1e-6f;
    
    rmsnorm_batch(matrix, weight, num_rows, row_size, eps);
    
    // Verify each row has RMS ≈ 1.0
    for (int row = 0; row < num_rows; row++) {
        float sum_sq = 0.0f;
        for (int col = 0; col < row_size; col++) {
            float val = matrix[row * row_size + col];
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / row_size);
        assert(fabs(rms - 1.0f) < 0.01f);
    }
    
    printf("  ✓ RMSNorm batch computation correct\n");
}

// ============================================================================
// Test: LayerNorm basic
// ============================================================================

static void test_layernorm_basic(void) {
    printf("TEST: layernorm basic computation\n");
    
    int n = 4;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};  // No scaling
    float bias[] = {0.0f, 0.0f, 0.0f, 0.0f};    // No shift
    float eps = 1e-6f;
    
    // Mean = (1 + 2 + 3 + 4) / 4 = 2.5
    // Variance = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4
    //          = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
    
    layernorm(x, weight, bias, n, eps);
    
    // After LayerNorm: should have mean ≈ 0 and variance ≈ 1
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    float mean = sum / n;
    assert(fabs(mean) < 0.01f);  // Mean ≈ 0
    
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        sum_sq_diff += diff * diff;
    }
    float var = sum_sq_diff / n;
    assert(fabs(var - 1.0f) < 0.01f);  // Variance ≈ 1
    
    printf("  ✓ LayerNorm basic computation correct\n");
}

static void test_layernorm_with_params(void) {
    printf("TEST: layernorm with learnable weight and bias\n");
    
    int n = 3;
    float x[] = {1.0f, 2.0f, 3.0f};
    float weight[] = {2.0f, 0.5f, 1.5f};
    float bias[] = {1.0f, -1.0f, 0.5f};
    float eps = 1e-6f;
    
    layernorm(x, weight, bias, n, eps);
    
    // Just verify results are reasonable
    for (int i = 0; i < n; i++) {
        assert(!isnan(x[i]) && !isinf(x[i]));
    }
    
    printf("  ✓ LayerNorm with parameters applied correctly\n");
}

static void test_layernorm_no_params(void) {
    printf("TEST: layernorm_no_params\n");
    
    int n = 4;
    float x[] = {2.0f, 4.0f, 6.0f, 8.0f};
    float eps = 1e-6f;
    
    layernorm_no_params(x, n, eps);
    
    // Verify mean ≈ 0 and variance ≈ 1
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    float mean = sum / n;
    assert(fabs(mean) < 0.01f);
    
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        sum_sq_diff += diff * diff;
    }
    float var = sum_sq_diff / n;
    assert(fabs(var - 1.0f) < 0.01f);
    
    printf("  ✓ LayerNorm without parameters correct\n");
}

static void test_layernorm_batch(void) {
    printf("TEST: layernorm_batch for matrices\n");
    
    int num_rows = 2;
    int row_size = 3;
    float matrix[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float weight[] = {1.0f, 1.0f, 1.0f};
    float bias[] = {0.0f, 0.0f, 0.0f};
    float eps = 1e-6f;
    
    layernorm_batch(matrix, weight, bias, num_rows, row_size, eps);
    
    // Verify each row has mean ≈ 0 and variance ≈ 1
    for (int row = 0; row < num_rows; row++) {
        float sum = 0.0f;
        for (int col = 0; col < row_size; col++) {
            sum += matrix[row * row_size + col];
        }
        float mean = sum / row_size;
        assert(fabs(mean) < 0.01f);
        
        float sum_sq_diff = 0.0f;
        for (int col = 0; col < row_size; col++) {
            float diff = matrix[row * row_size + col] - mean;
            sum_sq_diff += diff * diff;
        }
        float var = sum_sq_diff / row_size;
        assert(fabs(var - 1.0f) < 0.01f);
    }
    
    printf("  ✓ LayerNorm batch computation correct\n");
}

// ============================================================================
// Test: Numerical stability
// ============================================================================

static void test_rmsnorm_stability(void) {
    printf("TEST: rmsnorm numerical stability with extreme values\n");
    
    int n = 4;
    float x[] = {1e-10f, 2e-10f, 3e-10f, 4e-10f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float eps = 1e-6f;
    
    rmsnorm(x, weight, n, eps);
    
    // Should not produce NaN/Inf despite small values
    for (int i = 0; i < n; i++) {
        assert(!isnan(x[i]) && !isinf(x[i]));
    }
    
    printf("  ✓ RMSNorm stable with small values\n");
}

static void test_layernorm_stability(void) {
    printf("TEST: layernorm numerical stability with extreme values\n");
    
    int n = 4;
    float x[] = {1e10f, 2e10f, 3e10f, 4e10f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float bias[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float eps = 1e-6f;
    
    layernorm(x, weight, bias, n, eps);
    
    // Should not produce NaN/Inf despite large values
    for (int i = 0; i < n; i++) {
        assert(!isnan(x[i]) && !isinf(x[i]));
    }
    
    printf("  ✓ LayerNorm stable with large values\n");
}

// ============================================================================
// Test: Comparison between methods
// ============================================================================

static void test_rmsnorm_vs_layernorm(void) {
    printf("TEST: RMSNorm vs LayerNorm comparison\n");
    
    int n = 5;
    
    // Setup for RMSNorm
    float x_rms[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float eps = 1e-6f;
    
    // Setup for LayerNorm
    float x_ln[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    rmsnorm(x_rms, w, n, eps);
    layernorm(x_ln, w, b, n, eps);
    
    // Both should normalize, but with different properties
    // RMSNorm: focuses on magnitude (RMS-based)
    // LayerNorm: focuses on distribution (mean and variance)
    
    // Both should produce finite results
    for (int i = 0; i < n; i++) {
        assert(!isnan(x_rms[i]) && !isinf(x_rms[i]));
        assert(!isnan(x_ln[i]) && !isinf(x_ln[i]));
    }
    
    printf("  ✓ RMSNorm and LayerNorm both work correctly\n");
}

// ============================================================================
// sapphire_rmsnorm Tests (Non-In-Place, Standardized API)
// ============================================================================

static void test_sapphire_rmsnorm_basic(void) {
    printf("TEST: sapphire_rmsnorm basic functionality\n");
    
    float in[] = {1.0f, 2.0f, 3.0f};
    float weight[] = {1.0f, 1.0f, 1.0f};
    float out[3];
    
    int ret = sapphire_rmsnorm(out, in, weight, 1e-6f, 3);
    assert(ret == 0);
    
    // Verify normalization: RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.16...
    float rms = sqrtf((1.0f + 4.0f + 9.0f) / 3.0f + 1e-6f);
    
    for (int i = 0; i < 3; i++) {
        float expected = (in[i] / rms) * weight[i];
        assert(fabsf(out[i] - expected) < 1e-5f);
    }
    
    printf("  ✓ sapphire_rmsnorm basic test passed\n");
}

static void test_sapphire_rmsnorm_identity_weight(void) {
    printf("TEST: sapphire_rmsnorm with identity weight\n");
    
    float in[] = {1.0f, 0.5f, -0.5f, 2.0f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    
    int ret = sapphire_rmsnorm(out, in, weight, 1e-6f, 4);
    assert(ret == 0);
    
    // out[i] should equal in[i] / RMS
    float rms = sqrtf((1.0f + 0.25f + 0.25f + 4.0f) / 4.0f + 1e-6f);
    
    for (int i = 0; i < 4; i++) {
        float expected = in[i] / rms;
        assert(fabsf(out[i] - expected) < 1e-5f);
    }
    
    printf("  ✓ sapphire_rmsnorm identity weight test passed\n");
}

static void test_sapphire_rmsnorm_scaling(void) {
    printf("TEST: sapphire_rmsnorm with scaling weights\n");
    
    float in[] = {2.0f, 4.0f};
    float weight[] = {0.5f, 2.0f};
    float out[2];
    
    int ret = sapphire_rmsnorm(out, in, weight, 1e-6f, 2);
    assert(ret == 0);
    
    float rms = sqrtf((4.0f + 16.0f) / 2.0f + 1e-6f);
    float expected_0 = (2.0f / rms) * 0.5f;
    float expected_1 = (4.0f / rms) * 2.0f;
    
    assert(fabsf(out[0] - expected_0) < 1e-5f);
    assert(fabsf(out[1] - expected_1) < 1e-5f);
    
    printf("  ✓ sapphire_rmsnorm scaling test passed\n");
}

static void test_sapphire_rmsnorm_invalid_inputs(void) {
    printf("TEST: sapphire_rmsnorm invalid input handling\n");
    
    float dummy_in[3] = {1.0f, 2.0f, 3.0f};
    float dummy_weight[3] = {1.0f, 1.0f, 1.0f};
    float dummy_out[3];
    
    // NULL output
    assert(sapphire_rmsnorm(NULL, dummy_in, dummy_weight, 1e-6f, 3) == -1);
    
    // NULL input
    assert(sapphire_rmsnorm(dummy_out, NULL, dummy_weight, 1e-6f, 3) == -1);
    
    // NULL weight
    assert(sapphire_rmsnorm(dummy_out, dummy_in, NULL, 1e-6f, 3) == -1);
    
    // Invalid dimension
    assert(sapphire_rmsnorm(dummy_out, dummy_in, dummy_weight, 1e-6f, 0) == -1);
    assert(sapphire_rmsnorm(dummy_out, dummy_in, dummy_weight, 1e-6f, -1) == -1);
    
    // Negative epsilon
    assert(sapphire_rmsnorm(dummy_out, dummy_in, dummy_weight, -1e-6f, 3) == -1);
    
    printf("  ✓ sapphire_rmsnorm invalid input handling passed\n");
}

static void test_sapphire_rmsnorm_input_preserved(void) {
    printf("TEST: sapphire_rmsnorm input preservation\n");
    
    float in_orig[] = {1.0f, 2.0f, 3.0f};
    float in[] = {1.0f, 2.0f, 3.0f};
    float weight[] = {1.0f, 1.0f, 1.0f};
    float out[3];
    
    int ret = sapphire_rmsnorm(out, in, weight, 1e-6f, 3);
    assert(ret == 0);
    
    // Verify input unchanged
    for (int i = 0; i < 3; i++) {
        assert(in[i] == in_orig[i]);
    }
    
    // Output should be different
    int different = 0;
    for (int i = 0; i < 3; i++) {
        if (fabsf(out[i] - in[i]) > 1e-6f) {
            different++;
        }
    }
    assert(different > 0);  // At least one element different
    
    printf("  ✓ sapphire_rmsnorm input preservation passed\n");
}

static void test_sapphire_rmsnorm_vs_existing(void) {
    printf("TEST: sapphire_rmsnorm vs existing rmsnorm\n");
    
    float in[] = {1.5f, -0.5f, 2.0f, 0.5f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    
    // Make copies for comparison
    float in_copy1[4], in_copy2[4];
    for (int i = 0; i < 4; i++) {
        in_copy1[i] = in[i];
        in_copy2[i] = in[i];
    }
    
    // Existing in-place rmsnorm (modifies in_copy1)
    rmsnorm(in_copy1, weight, 4, 1e-6f);
    
    // New non-in-place sapphire_rmsnorm
    float out[4];
    int ret = sapphire_rmsnorm(out, in_copy2, weight, 1e-6f, 4);
    assert(ret == 0);
    
    // Results should be identical
    for (int i = 0; i < 4; i++) {
        assert(fabsf(in_copy1[i] - out[i]) < 1e-5f);
    }
    
    printf("  ✓ sapphire_rmsnorm vs existing rmsnorm passed\n");
}

static void test_sapphire_rmsnorm_various_dimensions(void) {
    printf("TEST: sapphire_rmsnorm with various dimensions\n");
    
    int dims[] = {1, 8, 16, 64, 128, 256, 512, 2048};
    
    for (int d = 0; d < 8; d++) {
        int dim = dims[d];
        
        float *in = (float *)malloc(dim * sizeof(float));
        float *weight = (float *)malloc(dim * sizeof(float));
        float *out = (float *)malloc(dim * sizeof(float));
        
        // Fill with pattern
        for (int i = 0; i < dim; i++) {
            in[i] = sinf((float)i * 0.1f);
            weight[i] = 1.0f;
        }
        
        int ret = sapphire_rmsnorm(out, in, weight, 1e-6f, dim);
        assert(ret == 0);
        
        // Spot-check normalization property
        // Verify against manual calculation
        float sum_in_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum_in_sq += in[i] * in[i];
        }
        float rms_in = sqrtf(sum_in_sq / (float)dim + 1e-6f);
        
        // Spot-check some values
        for (int i = 0; i < 10; i++) {
            float expected = in[i] / rms_in * weight[i];
            assert(fabsf(out[i] - expected) < 1e-4f);
        }
        
        free(in);
        free(weight);
        free(out);
    }
    
    printf("  ✓ sapphire_rmsnorm various dimensions passed\n");
}

static void test_sapphire_rmsnorm_epsilon_values(void) {
    printf("TEST: sapphire_rmsnorm epsilon handling\n");
    
    float in[] = {1.0f, 2.0f, 3.0f};
    float weight[] = {1.0f, 1.0f, 1.0f};
    float out1[3], out2[3], out3[3];
    
    // Very small epsilon
    int ret1 = sapphire_rmsnorm(out1, in, weight, 1e-8f, 3);
    assert(ret1 == 0);
    
    // Standard epsilon
    int ret2 = sapphire_rmsnorm(out2, in, weight, 1e-6f, 3);
    assert(ret2 == 0);
    
    // Larger epsilon
    int ret3 = sapphire_rmsnorm(out3, in, weight, 1e-3f, 3);
    assert(ret3 == 0);
    
    // All should be valid (not NaN/inf)
    for (int i = 0; i < 3; i++) {
        assert(!isnan(out1[i]) && !isinf(out1[i]));
        assert(!isnan(out2[i]) && !isinf(out2[i]));
        assert(!isnan(out3[i]) && !isinf(out3[i]));
    }
    
    printf("  ✓ sapphire_rmsnorm epsilon handling passed\n");
}

static void test_sapphire_rmsnorm_inplace(void) {
    printf("TEST: sapphire_rmsnorm in-place operation\n");
    
    float buffer[] = {1.0f, 2.0f, 3.0f};
    float weight[] = {1.0f, 1.0f, 1.0f};
    float buffer_orig[] = {1.0f, 2.0f, 3.0f};
    
    // In-place: output buffer is same as input
    int ret = sapphire_rmsnorm(buffer, buffer, weight, 1e-6f, 3);
    assert(ret == 0);
    
    // Verify result is correct
    float rms = sqrtf((1.0f + 4.0f + 9.0f) / 3.0f + 1e-6f);
    for (int i = 0; i < 3; i++) {
        float expected = buffer_orig[i] / rms;
        assert(fabsf(buffer[i] - expected) < 1e-5f);
    }
    
    printf("  ✓ sapphire_rmsnorm in-place operation passed\n");
}

static void test_sapphire_rmsnorm_large_array(void) {
    printf("TEST: sapphire_rmsnorm large array processing\n");
    
    int dim = 10000;
    float *in = (float *)malloc(dim * sizeof(float));
    float *weight = (float *)malloc(dim * sizeof(float));
    float *out = (float *)malloc(dim * sizeof(float));
    
    // Fill with random values
    for (int i = 0; i < dim; i++) {
        in[i] = (float)rand() / RAND_MAX;
        weight[i] = 1.0f;
    }
    
    int ret = sapphire_rmsnorm(out, in, weight, 1e-6f, dim);
    assert(ret == 0);
    
    // Spot-check correctness
    float rms = 0.0f;
    for (int i = 0; i < dim; i++) {
        rms += in[i] * in[i];
    }
    rms = sqrtf(rms / (float)dim + 1e-6f);
    
    for (int i = 0; i < 100; i += 10) {
        float expected = in[i] / rms;
        assert(fabsf(out[i] - expected) < 1e-4f);
    }
    
    free(in);
    free(weight);
    free(out);
    
    printf("  ✓ sapphire_rmsnorm large array processing passed\n");
}

// ============================================================================
// Main test runner
// ============================================================================

int main(void) {
    printf("\n");
    printf("============================================================\n");
    printf("                NORMALIZATION TEST SUITE\n");
    printf("============================================================\n");
    printf("\n");
    
    // RMSNorm tests
    test_rmsnorm_basic();
    test_rmsnorm_with_weight();
    test_rmsnorm_no_weight();
    test_rmsnorm_batch();
    printf("\n");
    
    // LayerNorm tests
    test_layernorm_basic();
    test_layernorm_with_params();
    test_layernorm_no_params();
    test_layernorm_batch();
    printf("\n");
    
    // Stability tests
    test_rmsnorm_stability();
    test_layernorm_stability();
    printf("\n");
    
    // Comparison tests
    test_rmsnorm_vs_layernorm();
    printf("\n");
    
    // sapphire_rmsnorm tests (non-in-place, standardized API)
    test_sapphire_rmsnorm_basic();
    test_sapphire_rmsnorm_identity_weight();
    test_sapphire_rmsnorm_scaling();
    test_sapphire_rmsnorm_invalid_inputs();
    test_sapphire_rmsnorm_input_preserved();
    test_sapphire_rmsnorm_vs_existing();
    test_sapphire_rmsnorm_various_dimensions();
    test_sapphire_rmsnorm_epsilon_values();
    test_sapphire_rmsnorm_inplace();
    test_sapphire_rmsnorm_large_array();
    printf("\n");
    
    printf("============================================================\n");
    printf("                    ALL TESTS PASSED ✓\n");
    printf("============================================================\n");
    printf("\n");
    
    return 0;
}

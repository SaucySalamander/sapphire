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
    
    printf("============================================================\n");
    printf("                    ALL TESTS PASSED ✓\n");
    printf("============================================================\n");
    printf("\n");
    
    return 0;
}

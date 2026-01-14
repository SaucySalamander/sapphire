#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "../include/tensor.h"
#include <stdlib.h>
#include <stdint.h>

int main(void) {
    printf("[TEST] tensor_transpose\n");
    int shape_src[2] = {2, 3};
    tensor_t *src = tensor_create(2, shape_src, DTYPE_F32);
    float *sdat = (float*)tensor_data(src);
    // fill src with row-major values: [ [1,2,3], [4,5,6] ]
    for (int i = 0; i < 6; ++i) sdat[i] = (float)(i+1);

    tensor_t *dst = NULL;
    int rc = tensor_transpose(src, &dst);
    assert(rc == 0);

    const int *shape_dst = tensor_shape(dst);
    assert(shape_dst[0] == 3 && shape_dst[1] == 2);
    float *dd = (float*)tensor_data(dst);
    // dst should be [ [1,4], [2,5], [3,6] ] row-major
    assert(dd[0] == 1.0f && dd[1] == 4.0f);
    assert(dd[2] == 2.0f && dd[3] == 5.0f);
    assert(dd[4] == 3.0f && dd[5] == 6.0f);

    tensor_release(src);
    tensor_release(dst);
    printf("  ✓ tensor_transpose passed\n");

    // BF16 -> F32 conversion + transpose path (simulate loader behavior)
    printf("[TEST] bf16 convert + transpose\n");
    tensor_t *bsrc = tensor_create(2, shape_src, DTYPE_BF16);
    uint16_t *bdata = (uint16_t*)tensor_data(bsrc);
    // write bf16 representation as high 16 bits of float
    for (int i = 0; i < 6; ++i) {
        float v = (float)(i+1);
        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        uint16_t bf = (uint16_t)(bits >> 16);
        bdata[i] = bf;
    }
    float *scratch = (float*)malloc(sizeof(float) * 6);
    // Manual BF16->F32 conversion (reinterpret bf16 as high 16 bits of f32)
    uint16_t *bptr = (uint16_t*)tensor_data(bsrc);
    for (int i = 0; i < 6; ++i) {
        uint32_t bits = ((uint32_t)bptr[i]) << 16;
        float v; memcpy(&v, &bits, sizeof(v));
        scratch[i] = v;
    }
    int dims_src[2] = {2,3};
    tensor_t *tsrc = tensor_create(2, dims_src, DTYPE_F32);
    memcpy((float*)tensor_data(tsrc), scratch, sizeof(float)*6);
    free(scratch);
    tensor_t *tdst = NULL;
    rc = tensor_transpose(tsrc, &tdst);
    assert(rc == 0);
    float *tdd = (float*)tensor_data(tdst);
    assert(tdd[0] == 1.0f && tdd[1] == 4.0f);
    assert(tdd[2] == 2.0f && tdd[3] == 5.0f);
    assert(tdd[4] == 3.0f && tdd[5] == 6.0f);
    tensor_release(bsrc);
    tensor_release(tsrc);
    tensor_release(tdst);
    printf("  ✓ bf16 convert + transpose passed\n");
    return 0;
}

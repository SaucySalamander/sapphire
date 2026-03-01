/**
 * @file bf16_helpers.glsl
 * @brief BF16 <-> FP32 conversion helpers for Vulkan compute shaders.
 *
 * BF16 = 1 sign + 8 exponent + 7 mantissa (16 bits total)
 * FP32 = 1 sign + 8 exponent + 23 mantissa (32 bits total)
 *
 * BF16 is stored as uint (packed two per uint in a uint[] SSBO).
 * These helpers extract/pack individual BF16 values within uint arrays.
 *
 * Buffer layout: uint[] where each uint holds two BF16 values:
 *   bits [15:0]  = BF16 value at even index
 *   bits [31:16] = BF16 value at odd index
 */

// ============================================================================
// Scalar BF16 <-> FP32 Conversions
// ============================================================================

/**
 * Unpack a BF16 value (given as uint16 in lower bits) to FP32.
 * BF16 is just the upper 16 bits of an IEEE 754 float32.
 */
float bf16_unpack(uint bf16_val) {
    uint fp32_bits = bf16_val << 16;
    return uintBitsToFloat(fp32_bits);
}

/**
 * Pack an FP32 value to BF16 (round-to-nearest-even).
 * Returns uint with BF16 in lower 16 bits.
 */
uint bf16_pack(float fp32_val) {
    uint bits = floatBitsToUint(fp32_val);
    // Round to nearest: add 0x7FFF + bit 16 (round-to-even)
    return (bits + 0x7FFFu) >> 16;
}

// ============================================================================
// Buffer Access Helpers (packed uint[] layout)
// ============================================================================

/**
 * Load a single BF16 value from a uint[] buffer at a logical BF16 index.
 * Each uint holds two BF16 values (low half = even index, high half = odd index).
 */
float bf16_load(readonly restrict uint[] buf, uint bf16_index) {
    uint word_idx = bf16_index >> 1;
    uint word = buf[word_idx];
    uint half_val = ((bf16_index & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16);
    return bf16_unpack(half_val);
}

/**
 * Store a single BF16 value into a uint[] buffer at a logical BF16 index.
 * WARNING: This does a read-modify-write on the containing uint.
 * For write-only buffers, prefer bf16_store_pair() or full-word writes.
 */
void bf16_store(restrict uint[] buf, uint bf16_index, float val) {
    uint word_idx = bf16_index >> 1;
    uint packed = bf16_pack(val);
    uint word = buf[word_idx];
    if ((bf16_index & 1u) == 0u) {
        word = (word & 0xFFFF0000u) | (packed & 0xFFFFu);
    } else {
        word = (word & 0x0000FFFFu) | (packed << 16);
    }
    buf[word_idx] = word;
}

/**
 * Store a pair of BF16 values as a single uint (avoids read-modify-write).
 * even_val goes to bits [15:0], odd_val goes to bits [31:16].
 */
uint bf16_pack_pair(float even_val, float odd_val) {
    uint lo = bf16_pack(even_val) & 0xFFFFu;
    uint hi = bf16_pack(odd_val) << 16;
    return lo | hi;
}

/**
 * Load two BF16 values from one uint word.
 * Returns them as a vec2 (x = even index, y = odd index).
 */
vec2 bf16_unpack_pair(uint word) {
    float lo = bf16_unpack(word & 0xFFFFu);
    float hi = bf16_unpack(word >> 16);
    return vec2(lo, hi);
}

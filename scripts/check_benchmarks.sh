#!/usr/bin/env bash
set -euo pipefail

# Default minimum throughput gates (wops/s). Override via env in CI if needed.
MIN_Q4_WOPS=${MIN_Q4_WOPS:-100000000}
MIN_Q8_WOPS=${MIN_Q8_WOPS:-100000000}
MIN_F32_WOPS=${MIN_F32_WOPS:-1000000000}
MIN_BF16_WOPS=${MIN_BF16_WOPS:-1000000000}

extract_wops() {
    local output="$1"
    local label="$2"
    echo "$output" | awk -v lbl="$label" '
        index($0, lbl) > 0 {
            split($0, parts, "wops/s=");
            if (length(parts) > 1) {
                split(parts[2], tail, " ");
                print tail[1];
                exit 0;
            }
        }
    '
}

assert_min_wops() {
    local name="$1"
    local output="$2"
    local label="$3"
    local min="$4"

    local measured
    measured="$(extract_wops "$output" "$label")"

    if [[ -z "$measured" ]]; then
        echo "❌ ${name}: could not parse wops/s from output"
        echo "$output"
        exit 1
    fi

    if ! awk -v m="$measured" -v min="$min" 'BEGIN { exit !(m >= min) }'; then
        echo "❌ ${name}: throughput gate failed (measured=${measured}, min=${min})"
        exit 1
    fi

    echo "✓ ${name}: throughput gate passed (measured=${measured}, min=${min})"
}

echo "Running benchmark performance gates..."

echo "- Q4"
q4_out="$(./out/bench_q4)"
echo "$q4_out"
assert_min_wops "Q4 aligned" "$q4_out" "Q4 aligned:" "$MIN_Q4_WOPS"

echo "- Q8"
q8_out="$(./out/bench_q8)"
echo "$q8_out"
assert_min_wops "Q8 aligned" "$q8_out" "Q8 aligned:" "$MIN_Q8_WOPS"

echo "- F32"
f32_out="$(./out/bench_f32)"
echo "$f32_out"
assert_min_wops "F32 AVX2" "$f32_out" "F32 AVX2:" "$MIN_F32_WOPS"

echo "- BF16"
bf16_out="$(./out/bench_bf16)"
echo "$bf16_out"
assert_min_wops "BF16 AVX2" "$bf16_out" "BF16 AVX2:" "$MIN_BF16_WOPS"

echo "✓ All benchmark throughput gates passed"

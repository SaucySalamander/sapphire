#!/bin/bash
# Run sanitizer tests (ASan + UBSan) and capture violations

set -e

BINARY="${1:-./out/asan/sapphire}"
REPORT_FILE="asan-report.txt"

if [ ! -f "$BINARY" ]; then
  echo "Error: Binary not found at $BINARY"
  echo "Run 'make bin-asan' first"
  exit 1
fi

echo "=========================================="
echo "AddressSanitizer + UndefinedBehaviorSanitizer"
echo "=========================================="
echo "Binary: $BINARY"
echo "Report: $REPORT_FILE"
echo ""

# Set sanitizer options for detailed output
export ASAN_OPTIONS="verbosity=1:halt_on_error=0:log_path=$REPORT_FILE"
export UBSAN_OPTIONS="verbosity=1:halt_on_error=0"

# Run the binary with a simple one-shot inference test
# This exercises model loading, tokenizer, inference, and memory management
echo "Running binary with one-shot inference test..."
"$BINARY" -m gemma3-270m-it -p "write a poem about the sun" 2>&1 || true

# Check if sanitizer detected any violations
if [ -f "$REPORT_FILE" ] && [ -s "$REPORT_FILE" ]; then
  echo ""
  echo "=========================================="
  echo "SANITIZER VIOLATIONS DETECTED"
  echo "=========================================="
  cat "$REPORT_FILE"
  echo ""
  echo "Full report saved to: $REPORT_FILE"
  exit 1
else
  echo ""
  echo "=========================================="
  echo "✓ No sanitizer violations detected"
  echo "=========================================="
  rm -f "$REPORT_FILE"
  exit 0
fi

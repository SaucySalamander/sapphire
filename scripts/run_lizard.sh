#!/bin/bash
# Run Lizard complexity analysis with reporting

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Configuration
LIZARD="${LIZARD:-.venv/bin/lizard}"
THRESHOLD_CC="${LIZARD_THRESHOLD_CC:-15}"
THRESHOLD_LENGTH="${LIZARD_THRESHOLD_LENGTH:-1000}"
REPORTS_DIR="reports"
CSV_REPORT="$REPORTS_DIR/lizard-report.csv"
HTML_REPORT="$REPORTS_DIR/lizard-report.html"

mkdir -p "$REPORTS_DIR"

if [ ! -f "$LIZARD" ]; then
  echo "Error: lizard not found at $LIZARD"
  echo "Install lizard: pip install lizard"
  exit 1
fi

echo "=========================================="
echo "Lizard Complexity Analysis"
echo "=========================================="
echo "Threshold (Cyclomatic Complexity): $THRESHOLD_CC"
echo "Threshold (Function Length): $THRESHOLD_LENGTH"
echo ""

# Run lizard with thresholds
echo "Running analysis on src/..."
$LIZARD -l c \
  -m \
  -C $THRESHOLD_CC \
  -L $THRESHOLD_LENGTH \
  --csv > "$CSV_REPORT" || true

$LIZARD -l c \
  -m \
  -C $THRESHOLD_CC \
  -L $THRESHOLD_LENGTH \
  -H > "$HTML_REPORT" || true

echo ""
echo "Reports generated:"
echo "  CSV:  $CSV_REPORT"
echo "  HTML: $HTML_REPORT"
echo ""

# Display summary from CSV
if [ -f "$CSV_REPORT" ]; then
  echo "=========================================="
  echo "Summary (functions exceeding thresholds)"
  echo "=========================================="
  tail -n +2 "$CSV_REPORT" | head -10
  echo ""
  LINE_COUNT=$(wc -l < "$CSV_REPORT")
  echo "Total functions analyzed: $((LINE_COUNT - 1))"
fi

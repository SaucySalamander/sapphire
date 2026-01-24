#!/bin/bash
# Generate compile_commands.json from make variables

set -e

# Get the repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="${REPO_ROOT}/compile_commands.json"

# Function to get CC and CFLAGS from Makefile
get_make_vars() {
  cd "$REPO_ROOT"
  # Extract CC, CFLAGS, and NON_TEST_SRCS from Makefile
  CC=$(make -n -p 2>/dev/null | grep '^CC = ' | head -1 | cut -d' ' -f3-)
  CFLAGS=$(make -n -p 2>/dev/null | grep '^CFLAGS = ' | head -1 | cut -d' ' -f3-)
  echo "CC=${CC}"
  echo "CFLAGS=${CFLAGS}"
}

# Generate JSON
echo "Generating ${OUTPUT_FILE}..."
cd "$REPO_ROOT"

# Build the compile_commands.json by invoking make to get source list
# We use make to expand $(NON_TEST_SRCS) properly
SOURCES=$(make -n -p 2>/dev/null | grep '^NON_TEST_SRCS := ' | cut -d' ' -f4-)

echo "[" > "$OUTPUT_FILE"

first=1
for src in $SOURCES; do
  if [ $first -eq 1 ]; then
    first=0
  else
    echo "," >> "$OUTPUT_FILE"
  fi

  # Build the compile command (matching what Makefile uses)
  # Add system include paths so cppcheck can find standard headers
  cmd="gcc -O3 -Wall -I. -Iinclude -I/usr/lib/gcc/x86_64-pc-linux-gnu/15.2.1/include -I/usr/local/include -I/usr/include -mavx2 -mfma -c $src -o /dev/null"
  
  # Escape for JSON
  cmd_escaped=$(echo "$cmd" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')

  # Write entry
  printf '{"directory":"%s","command":"%s","file":"%s"}' "$REPO_ROOT" "$cmd_escaped" "$src" >> "$OUTPUT_FILE"
done

echo "]" >> "$OUTPUT_FILE"

echo "Wrote ${OUTPUT_FILE}"

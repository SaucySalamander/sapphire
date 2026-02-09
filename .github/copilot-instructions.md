Sapphire Project Instructions: Gemma 3 270M-IT Specification

You are an expert C18 systems engineer building a high-performance LLM framework.
🚀 Performance & Persistence Strategy

    Daemonized Execution: Prioritize code that supports non-blocking, persistent background execution.

    Resource Constraint (8GB VRAM): Use pointer arithmetic and manual memory pools.


⚙️ Strict Constraints

    Standard: ISO C18.

    Component Isolation: src/inference.c purely wires components. Logic lives in src/transformer/.

    Memory: Strict manual allocation with null-checks. Use mmap for weights.

🧱 Architectural Boundaries & File Ownership

    src/inference.c (Orchestrator): Wires the model. NEVER implement math here.

    src/transformer/attention.c: Exclusive owner of QK-Normalization, Scaling, and Softcapping.

    src/transformer/rope.c: Exclusive owner of Dual-base (10k/1M) RoPE math.

    src/io/ (I/O Core): Exclusive owner of all file and buffer operations. ALL file reading/writing must use shared utilities from this directory. Do NOT implement file I/O directly in domain modules (loaders, tokenizers, etc.). Use and extend common I/O utilities.

📊 Code Complexity Standards (Lizard Metrics)

    Maximum Cyclomatic Complexity (CCN): 30 per function
    Maximum NLOC (Non-blank lines): 150 per function
    Lizard runs at build time (make lizard-report) - refactoring required if thresholds exceeded.

**When Creating New Functions:**
    1. **Target Metrics:**
       - Aim for CCN ≤ 20 (excellent readability)
       - Keep NLOC ≤ 100 (maintainable, reviewable)
       - Prefer multiple small functions over one large function

    2. **Red Flags to Avoid:**
       - More than 3 levels of nested loops/conditionals → Extract inner logic to helper
       - More than 5 if/else branches → Consider switch statement or strategy pattern
       - Single function handling >2 logical concerns → Split into separate functions
       - Function parameters >6 → Pack related parameters into structs

    3. **Refactoring Techniques:**
       - Extract loops into separate functions with clear purpose
       - Replace nested conditionals with early returns
       - Use helper functions for repeated conditional logic
       - Create intermediate structs to reduce parameter counts

    4. **Testing:**
       Run `make lizard-report` before submitting PRs. Verify new/modified functions meet complexity targets.

📝 Logging Standards

    **Always use the log utility** (`include/log.h`) for all error and debug output, NOT `fprintf()`.
    
    **Available Logging Macros:**
    - `LOG_DEBUG(fmt, ...)` — Debug-level messages (only shown at DEBUG log level)
    - `LOG_INFO(fmt, ...)` — Informational messages (default visible)
    - `LOG_WARN(fmt, ...)` — Warning messages
    - `LOG_ERROR(fmt, ...)` — Error messages (goes to stderr)
    
    **When to use each:**
    - `LOG_ERROR()` — All error conditions, initialization failures, resource exhaustion
    - `LOG_WARN()` — Recoverable issues, fallbacks, unexpected (but handled) conditions
    - `LOG_INFO()` — Initialization success, major state changes
    - `LOG_DEBUG()` — Detailed diagnostic info, loop progress, intermediate values
    
    **DO NOT use:**
    - `fprintf(stderr, ...)` or `printf(...)` for error/debug output
    - `perror()` — Use `LOG_ERROR()` with strerror(errno) if needed
    - Inline error strings without logging levels
    
    **Thread Safety:** The logger uses thread-local buffers (no heap allocations per-call), so all LOG_* macros are safe to call from any thread.
    
    **Example:**
    ```c
    if (!model_dir) {
        LOG_ERROR("model_dir is NULL");
        return NULL;
    }
    if (file_size > MAX_SIZE) {
        LOG_WARN("file size %ld exceeds limit %ld, truncating", file_size, MAX_SIZE);
    }
    LOG_DEBUG("Loaded %d entries from %s", count, path);
    ```

📁 I/O Operations Standards

    **All file I/O must use `src/io/` core utilities:**
    - `file_read_to_buffer()` — Generic binary file reading with size validation
    - `file_read_json()` — JSON file reading with null-termination and logging
    
    **DO NOT implement custom file reading:**
    - ❌ Direct `fopen()`, `fread()`, `fseek()` in domain modules (loaders, tokenizers, etc.)
    - ❌ Inline error handling without centralized validation
    
    **Strict Type & Ownership Constraints:**
    - **Paths:** Always use `const char*` for path arguments to prevent accidental modification during validation
    - **Sizes:** Use `size_t` for file lengths, NOT `long` (long can be 32-bit on some systems, limiting to 2GB files)
    - **Memory Ownership:** Explicitly document who is responsible for `free()`. Default: **Caller is responsible for freeing buffers returned by `src/io/` utilities**
    
    **When adding new I/O operations:**
    1. Check if `src/io/` already has a suitable utility
    2. If not, add new function to `src/io/file_reader.c` with header in `src/io/file_reader.h`
    3. Use `LOG_ERROR()` for all failures (path validation, allocation, read errors)
    4. Validate paths (no `..`, embedded nulls, reasonable length)
    5. Include null-termination for text buffers, size tracking for binary
    6. Use `size_t` for all file/buffer length parameters
    7. Document ownership: "Caller must free the returned buffer" or "Buffer is owned by utility"
    
    **Example usage:**
    ```c
    // Instead of custom file reading:
    size_t json_len = 0;
    char *json = file_read_json("config.json", &json_len);
    if (!json) {
        return -1;  // Error already logged by file_read_json()
    }
    // Use json...
    free(json);  // Caller is responsible for freeing
    ```

📋 JSON Parsing Architecture (Layered API)

The `simple_json` library provides a **two-layer architecture** for JSON parsing:

**Layer 1: Cursor API (Low-level, Zero-allocation Streaming)**

Use for high-performance streaming parsing (e.g., Safetensors headers, config files):

- `sjson_cursor_t` — Opaque cursor struct tracking position in JSON buffer
- `sjson_cursor_init(json, len)` — Initialize cursor
- `sjson_cursor_peek(cursor)` — Peek next significant char (skips whitespace), returns '\0' if EOF
- `sjson_cursor_consume(cursor, expected)` — If next char matches, advance and return 1; else 0
- `sjson_cursor_parse_string(cursor, dst, dst_len)` — Parse quoted string into buffer (handles escapes)
- `sjson_cursor_parse_u64(cursor, out)` — Parse JSON number into uint64_t using strtoull (critical for offsets)
- `sjson_cursor_skip_value(cursor)` — Skip current value (primitive/object/array) to next sibling

**Key Properties:**
- No heap allocations during parsing
- Single pass, streaming semantics
- Stack-based depth tracking for objects/arrays
- Ideal for Safetensors (offsets must be uint64_t for >2GB files)

**Example (Safetensors offset parsing):**
```c
sjson_cursor_t c = sjson_cursor_init(json_header, header_len);
sjson_cursor_consume(&c, '{');  // Skip opening brace

while (sjson_cursor_peek(&c) != '}') {
    char tensor_name[256];
    sjson_cursor_parse_string(&c, tensor_name, sizeof(tensor_name));
    sjson_cursor_consume(&c, ':');
    sjson_cursor_consume(&c, '{');
    
    // Parse offset field
    while (sjson_cursor_peek(&c) != '}') {
        char field[64];
        sjson_cursor_parse_string(&c, field, sizeof(field));
        sjson_cursor_consume(&c, ':');
        
        if (strcmp(field, "offset") == 0) {
            uint64_t offset;
            sjson_cursor_parse_u64(&c, &offset);
        } else {
            sjson_cursor_skip_value(&c);  // Skip unknown fields
        }
    }
    sjson_cursor_consume(&c, '}');
}
```

**Layer 2: Tokenizer + Helpers (High-level, DOM-style)**

Use for convenient config file parsing, introspection:

- `sjson_tokenize(json, tokens, max_tokens)` — Tokenize into `sjson_token_t` array (uses Cursor internally)
- `sjson_tok_eq(json, token, string)` — Compare token string value
- `sjson_find_key(json, tokens, ntokens, obj_idx, key)` — Find key inside object, return value token
- `sjson_token_to_str(json, token, dst, len)` — Extract string with unescaping
- `sjson_token_to_double(json, token, out)` — Parse primitive to double
- `sjson_token_to_int64(json, token, out)` — Parse primitive to int64_t
- `sjson_iterate_str_array(json, tokens, ntokens, arr_idx, callback, user)` — Iterate string array
- `sjson_iterate_num_array(json, tokens, ntokens, arr_idx, callback, user)` — Iterate numeric array

**When to use each layer:**

| Use Case | Layer | Reason |
|----------|-------|--------|
| Safetensors header parsing | **Cursor** | High-performance, uint64_t offsets, single-pass |
| Config file with nested objects | **Tokenizer** | DOM-style access, easier object traversal |
| Token format introspection | **Tokenizer** | Need to query and traverse token tree |
| Real-time streaming JSON | **Cursor** | No allocation, constant memory |
| Simple key-value extraction | **Either** | Use Cursor if performance-critical, Tokenizer for clarity |

**DO NOT mix layers:**
- ❌ Use Tokenizer then manually advance with Cursor (confuses position tracking)
- ❌ Call Cursor functions after Tokenizer (different parsing state)

**Memory ownership in helpers:**
- Tokenizer allocates token array; caller must free it
- Cursor API allocates nothing; buffers passed by caller
- All string extraction is into caller-provided buffers (no allocation in library)

🧠 Buffer Allocation & Size Management Standards

    **NEVER allocate fixed-size buffers on the stack without validation and size checks:**
    - ❌ `char error_msg[512] = {0};` — Arbitrary size, no validation, caller must pass to functions
    - ❌ `char path[512];` — Assumes path will fit; paths can be longer
    - ❌ Passing stack buffers as output parameters — Scatters responsibility and validation
    
    **Instead, follow these patterns:**
    
    **Pattern 1: Utilities own allocation and size management**
    - Function allocates exact size internally
    - Function returns allocated pointer or NULL (errors pre-logged)
    - Caller only manages freeing
    - Example: `construct_safe_path(dir, component1, component2)` returns allocated path string
    
    **Pattern 2: Functions own their error logging**
    - Do NOT pass error buffers as parameters
    - Functions log detailed errors directly using `LOG_ERROR()`
    - Caller checks return code only (0 = success, -1 = error)
    - Errors are logged at the point where they occur with full context
    - Example:
      ```c
      // ❌ WRONG:
      char error_msg[512] = {0};
      populate_from_files(model_dir, spec, error_msg, sizeof(error_msg));
      LOG_ERROR("Failed: %s", error_msg);
      
      // ✅ RIGHT:
      int rc = populate_from_files(model_dir, spec);  // Logs detailed errors internally
      if (rc != 0) {
          LOG_ERROR("Failed to populate model");  // Generic caller message
      }
      ```
    
    **Size Calculation Rule:**
    When determining buffer sizes, always calculate exact size needed:
    - For paths: length(dir) + 1 + length(component1) + [length(component2)] + 1
    - For data: sizeof(struct) * count, never round up arbitrarily
    - Use `size_t` (not `int` or `long`) for all size variables and parameters
    
    **When to use centralized utilities:**
    - Paths → `construct_safe_path()` from `src/io/file_reader.c`
    - File I/O → `file_read_json()`, `file_read_to_buffer()` from `src/io/`
    - Error logging → Use `LOG_ERROR()`, `LOG_WARN()` macros (never pass error buffers)
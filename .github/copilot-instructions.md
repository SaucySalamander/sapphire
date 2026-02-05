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
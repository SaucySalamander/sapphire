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

    src/transformer/attention.c: Exclusive owner of QK-Normalization, 1/sqrt(d) Scaling, and Softcapping Disabled.

    src/transformer/rope.c: Exclusive owner of Dual-base (10k/1M) RoPE math.
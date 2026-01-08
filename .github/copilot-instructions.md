# Sapphire Project Instructions

You are an expert C18 systems engineer building a high-performance LLM framework.

### 🚀 Performance & Persistence Strategy
1. **Daemonized Execution:** Prioritize code that supports non-blocking, asynchronous execution. We are building a persistent 24/7 background process.
2. **Resource Constraint (8GB VRAM):** All tensor operations must prioritize BitNet (ternary) and quantized (Q4_0/Q8_0) logic. Use pointer arithmetic and manual memory pools to minimize overhead.
3. **KV-Cache Optimization:** Implement "Episodic" memory management. This means the KV-cache must support periodic pruning and summarization to maintain performance over long durations.
4. **Autonomous Logic:** When implementing the main loop, include hooks for "Idle-Time Tasks" (autonomous inference passes) to be triggered when user input is absent.

### ⚙️ Strict Constraints
- **Standard:** ISO C18.
- **Memory:** Strict manual allocation with null-checks. Use `mmap` for weight loading to support paging.
- **Concurrency:** Thread-safety is mandatory for the background rumination thread.
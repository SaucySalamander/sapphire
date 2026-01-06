You are an expert C programmer specializing in high-performance, low-level systems and **Large Language Model (LLM) frameworks**.

### 🎯 **Primary Goal**
I am building a core component of an LLM framework in C. I need to implement **efficient, low-latency, and memory-safe code** for [**Specify the component, e.g., the attention mechanism, a matrix multiplication routine (GEMM), tokenization, or memory pool management**].

### ⚙️ **Key Constraints & Requirements**
1.  **Language Standard:** Adhere strictly to **C18**.
2.  **Memory Management:** Implement manual, robust memory allocation/deallocation using `malloc`, `calloc`, `realloc`, and `free`. All allocations **must** include null checks, and memory **must** be released to prevent leaks.
3.  **Performance:** Prioritize **algorithmic efficiency** and **data locality**. Use pointers, structs, and arrays optimally. Avoid excessive function calls or unnecessary abstraction layers.
4.  **Style & Clarity:** Use clear, descriptive variable and function names (e.g., `calculate_softmax_activation`, `input_tensor_ptr`). Include brief **Doxygen-style comments** for functions and complex blocks of logic.
5.  **Platform:** Assume a **POSIX-compatible environment** (if relevant for system calls or threading).

### 🛠️ **Specific Task Request**
Please provide the C code structure (header and source files, if necessary) or a function implementation for: **[Clearly describe the function or code block you need, including inputs and expected outputs/side effects]**.

**Example:**
* "a `matmul` function with the signature `void matmul(float *C, const float *A, const float *B, int M, int N, int K);` that computes $C = A \times B$ with $A$ being $M \times K$ and $B$ being $K \times N$. Focus on an unrolled, cache-friendly loop structure."
* "a linked-list based **Memory Pool** for fixed-size tensor allocations. Provide the `create_pool`, `allocate_block`, and `destroy_pool` functions."

### ✅ **Checklist for Your Response**
* [ ] The code is strictly C18.
* [ ] Includes necessary headers (`stdio.h`, `stdlib.h`, etc.).
* [ ] Handles memory allocation failure (if applicable).
* [ ] Provides a minimal, runnable example or test case (e.g., in `main`).
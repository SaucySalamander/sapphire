#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <stddef.h>

/**
 * @file activations.h
 * @brief Activation functions for neural network layers.
 * 
 * This module provides common activation functions used in transformers:
 * - SiLU (Swish): x * sigmoid(x), modern alternative to ReLU
 * - ReLU: Rectified Linear Unit, max(0, x)
 * - GELU: Gaussian Error Linear Unit, smooth approximation
 */

/**
 * @brief SiLU (Sigmoid Linear Unit) activation, also known as Swish.
 * 
 * Formula: output = x * sigmoid(x) = x / (1 + exp(-x))
 * 
 * Used in modern LLMs (GPT-style models). Smooth, non-monotonic, and empirically
 * performs well. More expensive than ReLU but often worth the cost.
 * 
 * @param x Input value.
 * @return x * sigmoid(x)
 */
float silu(float x);

/**
 * @brief ReLU (Rectified Linear Unit) activation.
 * 
 * Formula: output = max(0, x)
 * 
 * Classic activation. Simple and fast, but suffers from "dying ReLU" problem
 * (neurons can get stuck outputting 0).
 * 
 * @param x Input value.
 * @return max(0, x)
 */
float relu(float x);

/**
 * @brief GELU (Gaussian Error Linear Unit) activation.
 * 
 * Formula (approximate): output ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 * 
 * Smooth alternative to ReLU used in BERT and similar models.
 * Approximation is faster than the exact error function implementation.
 * 
 * @param x Input value.
 * @return Approximate GELU output.
 */
float gelu(float x);

/**
 * @brief In-place SiLU activation for an array.
 * 
 * Applies SiLU to each element: output[i] = silu(input[i])
 * 
 * @param x Array to activate (modified in-place).
 * @param n Number of elements.
 */
void silu_inplace(float *x, int n);

/**
 * @brief In-place ReLU activation for an array.
 * 
 * Applies ReLU to each element: output[i] = relu(input[i])
 * 
 * @param x Array to activate (modified in-place).
 * @param n Number of elements.
 */
void relu_inplace(float *x, int n);

/**
 * @brief In-place GELU activation for an array.
 * 
 * Applies GELU to each element: output[i] = gelu(input[i])
 * 
 * @param x Array to activate (modified in-place).
 * @param n Number of elements.
 */
void gelu_inplace(float *x, int n);

/**
 * @brief GeGLU (Gated GELU) activation for a single pair of values.
 *
 * Formula: output = x * GELU(y)
 *
 * GeGLU is a gated variant of GELU used in modern transformers.
 * One input acts as the main signal (x), the other as a gate (y).
 *
 * @param x The main signal value.
 * @param y The gating value.
 * @return x * GELU(y)
 */
float geglu(float x, float y);

/**
 * @brief Vectorized GeGLU (Gated GELU) activation for an array.
 *
 * Formula: output[i] = input[i] * GELU(input[i + n/2])
 *
 * Input layout: [x_1, x_2, ..., x_n, y_1, y_2, ..., y_n]
 * where size = 2n (total elements), n is the number of (x,y) pairs.
 *
 * Memory layout (row-major):
 * - First half: x values (signal)
 * - Second half: y values (gate)
 *
 * Algorithm:
 *   for i in 0..n-1:
 *     output[i] = input[i] * GELU(input[i + n])
 *
 * Optimization: Inner loop is unrolled (factor 4-8) for cache efficiency.
 *
 * @param output Output array where results are written (size >= size).
 *               Must be pre-allocated.
 * @param input Input array with layout [x_1..x_n, y_1..y_n].
 *              Must contain size elements (size must be even).
 * @param size Total number of elements in input (2n where n is pair count).
 *             Must be even and > 0.
 *
 * @return 0 on success, -1 on error (NULL pointers, invalid size).
 *
 * @note Input and output may be the same array (in-place operation).
 * @note size must be even; the function will assert this.
 * @note Loop unrolling assumes sequential memory layout.
 *
 * Example:
 *   float input[] = {1.0, 2.0, 0.0, 0.5};  // [x_1, x_2, y_1, y_2]
 *   float output[2];
 *   sapphire_geglu(output, input, 4);
 *   // output[0] = 1.0 * GELU(0.0) ≈ 0.0
 *   // output[1] = 2.0 * GELU(0.5) ≈ 2.0 * 0.38...
 */
int sapphire_geglu(float *output, const float *input, size_t size);

#endif // ACTIVATIONS_H

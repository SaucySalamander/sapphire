#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

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

#endif // ACTIVATIONS_H

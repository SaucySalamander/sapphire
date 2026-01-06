#ifndef UTILS_H
#define UTILS_H

/**
 * @brief Computes softmax with numerical stability.
 * 
 * Implements the numerically stable softmax using the log-space trick:
 * softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)) for all j)
 * 
 * This prevents overflow/underflow by subtracting the maximum value before
 * computing exponentials.
 * 
 * @param scores Input array of scores. Modified in-place to contain softmax output.
 * @param n The number of elements in the scores array.
 */
void softmax(float *scores, int n);

#endif // UTILS_H

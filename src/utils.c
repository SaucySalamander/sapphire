#include "utils.h"
#include <math.h>
#include <stddef.h>

/**
 * @brief Computes softmax with numerical stability.
 * 
 * Implements the numerically stable softmax using the log-space trick:
 * softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)) for all j)
 * 
 * @param scores Input array of scores. Modified in-place to contain softmax output.
 * @param n The number of elements in the scores array.
 */
void softmax(float *scores, int n) {
    if (n <= 0 || scores == NULL) {
        return;
    }

    // Step 1: Find the maximum score to prevent overflow.
    float max_score = scores[0];
    for (int i = 1; i < n; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    // Step 2: Compute exp(scores[i] - max_score) and accumulate sum.
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }

    // Step 3: Normalize by the sum to produce final softmax weights.
    for (int i = 0; i < n; i++) {
        scores[i] /= sum;
    }
}

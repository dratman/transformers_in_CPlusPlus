#include "layer_normalization.h"
#include <cmath>
#include <numeric>

// Constructor: Initialize gamma and beta
LayerNormalization::LayerNormalization(size_t feature_dim)
    : gamma(1, feature_dim), beta(1, feature_dim) {
    for (size_t i = 0; i < feature_dim; ++i) {
        gamma.data[0][i] = 1.0f;
        beta.data[0][i] = 0.0f;
    }
}

// Forward pass
Matrix LayerNormalization::forward(const Matrix &input) {
    Matrix output(input.rows, input.cols);

    for (size_t i = 0; i < input.rows; ++i) {
        float mean = std::accumulate(input.data[i].begin(), input.data[i].end(), 0.0f) / input.cols;
        float variance = 0.0f;
        for (float x : input.data[i]) variance += (x - mean) * (x - mean);
        variance /= input.cols;

        for (size_t j = 0; j < input.cols; ++j) {
            float norm = (input.data[i][j] - mean) / std::sqrt(variance + 1e-6f);
            output.data[i][j] = gamma.data[0][j] * norm + beta.data[0][j];
        }
    }
    return output;
}

// Backward pass (simplified, assumes precomputed gradients for gamma/beta)
Matrix LayerNormalization::backward(const Matrix &input, const Matrix &grad_output) {
    return grad_output; // Placeholder for full gradient computation
}

#include "attention.h"
#include "atomic_operations.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

// Constructor: Initialize scale factor
ScaledDotProductAttention::ScaledDotProductAttention(size_t d_k)
    : scale_factor(1.0f / std::sqrt(static_cast<float>(d_k))) {}

// Forward pass
Matrix ScaledDotProductAttention::forward(const Matrix &query, const Matrix &key, const Matrix &value) {
    // Step 1: Compute scaled dot-product attention scores
    Matrix scores = matmul(query, transpose(key));
    for (size_t i = 0; i < scores.rows; ++i) {
        for (size_t j = 0; j < scores.cols; ++j) {
            scores.data[i][j] *= scale_factor;
        }
    }
    // Step 2: Apply softmax to get attention weights
    Matrix attention_weights = softmax(scores);
    // Step 3: Compute the output by multiplying attention weights with the value matrix
    Matrix output = matmul(attention_weights, value);
    return output;
}

// Softmax backward helper function
Matrix softmax_backward(const Matrix &softmax_output, const Matrix &grad_output) {
    Matrix grad_input(softmax_output.rows, softmax_output.cols);
    for (size_t i = 0; i < softmax_output.rows; ++i) {
        for (size_t j = 0; j < softmax_output.cols; ++j) {
            float s_j = softmax_output.data[i][j];
            float grad = 0.0f;
            for (size_t k = 0; k < softmax_output.cols; ++k) {
                float s_k = softmax_output.data[i][k];
                if (j == k) {
                    grad += grad_output.data[i][k] * s_j * (1 - s_j);
                } else {
                    grad -= grad_output.data[i][k] * s_j * s_k;
                }
            }
            grad_input.data[i][j] = grad;
        }
    }
    return grad_input;
}

// Backward pass
void ScaledDotProductAttention::backward(const Matrix &query, const Matrix &key, const Matrix &value,
                                         const Matrix &grad_output, Matrix &grad_query,
                                         Matrix &grad_key, Matrix &grad_value) {
    // Step 1: Compute scaled attention scores
    Matrix scores = matmul(query, transpose(key));
    for (size_t i = 0; i < scores.rows; ++i) {
        for (size_t j = 0; j < scores.cols; ++j) {
            scores.data[i][j] *= scale_factor;
        }
    }

    // Step 2: Apply softmax to get attention weights
    Matrix attention_weights = softmax(scores);

    // Step 3: Compute grad_value
    grad_value = matmul(transpose(attention_weights), grad_output);

    // Step 4: Compute grad_attention_weights
    Matrix grad_attention_weights = matmul(grad_output, transpose(value));

    // Step 5: Compute grad_scores using the softmax backward function
    Matrix grad_scores = softmax_backward(attention_weights, grad_attention_weights);

    // Step 6: Scale grad_scores by the scaling factor
    for (size_t i = 0; i < grad_scores.rows; ++i) {
        for (size_t j = 0; j < grad_scores.cols; ++j) {
            grad_scores.data[i][j] *= scale_factor;
        }
    }

    // Step 7: Compute grad_query
    grad_query = matmul(grad_scores, key);

    // Step 8: Compute grad_key
    grad_key = matmul(transpose(grad_scores), query);
}

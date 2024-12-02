#include "linear_layer.h"
#include <random>
#include <iostream>
#include "atomic_operations.h"

// Helper function to initialize a matrix with random values
void random_initialize(Matrix &m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            m.data[i][j] = dist(gen);
        }
    }
}

// Constructor: Initialize weights and biases
LinearLayer::LinearLayer(size_t input_dim, size_t output_dim)
    : weights(input_dim, output_dim), biases(1, output_dim),
      grad_weights(input_dim, output_dim), grad_biases(1, output_dim) {
    random_initialize(weights);
    random_initialize(biases);
}

// Forward pass
Matrix LinearLayer::forward(const Matrix &input) {
    Matrix output = matmul(input, weights);
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            output.data[i][j] += biases.data[0][j]; // Add bias
        }
    }
    return output;
}

// Backward pass
Matrix LinearLayer::backward(const Matrix &input, const Matrix &grad_output) {
    // Gradients w.r.t. weights: input^T × grad_output
    grad_weights = matmul(transpose(input), grad_output);

    // Gradients w.r.t. biases: sum over batch (rows of grad_output)
    for (size_t j = 0; j < grad_output.cols; ++j) {
        grad_biases.data[0][j] = 0;
        for (size_t i = 0; i < grad_output.rows; ++i) {
            grad_biases.data[0][j] += grad_output.data[i][j];
        }
    }

    // Gradients w.r.t. input: grad_output × weights^T
    return matmul(grad_output, transpose(weights));
}

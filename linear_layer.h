#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include "atomic_operations.h"

struct LinearLayer {
    Matrix weights;
    Matrix biases;
    Matrix grad_weights;
    Matrix grad_biases;

    // Constructor: Initialize weights and biases with random values
    LinearLayer(size_t input_dim, size_t output_dim);

    // Forward pass: input × weights + biases
    Matrix forward(const Matrix &input);

    // Backward pass:
    // 1. Gradients w.r.t. input, weights, and biases
    Matrix backward(const Matrix &input, const Matrix &grad_output);
};

#endif

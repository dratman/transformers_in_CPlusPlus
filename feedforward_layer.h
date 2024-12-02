#ifndef FEEDFORWARD_LAYER_H
#define FEEDFORWARD_LAYER_H

#include "linear_layer.h"

struct FeedforwardLayer {
    LinearLayer linear1;
    LinearLayer linear2;

    // Constructor
    FeedforwardLayer(size_t input_dim, size_t hidden_dim);

    // Forward pass
    Matrix forward(const Matrix &input);

    // Backward pass
    Matrix backward(const Matrix &input, const Matrix &grad_output);
};

#endif

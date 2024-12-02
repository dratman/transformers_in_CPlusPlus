#ifndef LAYER_NORMALIZATION_H
#define LAYER_NORMALIZATION_H

#include "atomic_operations.h"

struct LayerNormalization {
    Matrix gamma;
    Matrix beta;

    LayerNormalization(size_t feature_dim);

    // Forward pass
    Matrix forward(const Matrix &input);

    // Backward pass
    Matrix backward(const Matrix &input, const Matrix &grad_output);
};

#endif

#ifndef ATTENTION_H
#define ATTENTION_H

#include "atomic_operations.h"

struct ScaledDotProductAttention {
    float scale_factor;

    ScaledDotProductAttention(size_t d_k);

    // Forward pass
    Matrix forward(const Matrix &query, const Matrix &key, const Matrix &value);

    // Backward pass
    void backward(const Matrix &query, const Matrix &key, const Matrix &value,
                  const Matrix &grad_output, Matrix &grad_query,
                  Matrix &grad_key, Matrix &grad_value);
};

#endif

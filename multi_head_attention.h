#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include "attention.h"

struct MultiHeadAttention {
    size_t d_model;
    size_t num_heads;
    ScaledDotProductAttention attention;

    // Constructor
    MultiHeadAttention(size_t d_model, size_t num_heads);

    // Forward pass
    Matrix forward(const Matrix &query, const Matrix &key, const Matrix &value);

    // Backward pass
    Matrix backward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &grad_output);
};

#endif
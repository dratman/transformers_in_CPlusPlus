#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "multi_head_attention.h"  // Include MultiHeadAttention
#include "layer_normalization.h"
#include "feedforward_layer.h"

struct TransformerBlock {
    MultiHeadAttention attention;
    LayerNormalization norm1;
    FeedforwardLayer feedforward;
    LayerNormalization norm2;

    // Constructor
    TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff);

    // Forward pass
    Matrix forward(const Matrix &input);

    // Backward pass
    Matrix backward(const Matrix &input, const Matrix &grad_output);
};

#endif

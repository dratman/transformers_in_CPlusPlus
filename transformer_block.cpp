#include "transformer_block.h"

// Constructor
TransformerBlock::TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff)
    : attention(d_model, num_heads),
      norm1(d_model),
      feedforward(d_model, d_ff),
      norm2(d_model) {}

// Forward pass
Matrix TransformerBlock::forward(const Matrix &input) {
    // Multi-head attention
    Matrix attention_output = attention.forward(input, input, input);

    // Add residual and normalize
    Matrix residual1 = add(input, attention_output);
    Matrix normalized1 = norm1.forward(residual1);

    // Feedforward
    Matrix feedforward_output = feedforward.forward(normalized1);

    // Add residual and normalize again
    Matrix residual2 = add(normalized1, feedforward_output);
    return norm2.forward(residual2);
}

// Backward pass
Matrix TransformerBlock::backward(const Matrix &input, const Matrix &grad_output) {
    // Backprop through second normalization
    Matrix grad_residual2 = norm2.backward(input, grad_output);

    // Backprop through feedforward layer
    Matrix grad_feedforward = feedforward.backward(input, grad_residual2);

    // Backprop through first normalization
    Matrix grad_residual1 = norm1.backward(input, grad_feedforward);

    // Backprop through multi-head attention
    return attention.backward(input, input, input, grad_residual1);
}

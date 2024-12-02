#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

#include "atomic_operations.h"
#include "positional_encoding.h"
#include "multi_head_attention.h"
#include "feedforward_layer.h"
#include "layer_normalization.h"
#include <vector>

// transformer_model.h

class TransformerBlock {
public:
    // Constructor
    //TransformerBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim);

    // Forward pass
    Matrix forward(const Matrix &input);

    // Backward pass
    Matrix backward(const Matrix &input, const Matrix &grad_output);

private:
    // Internal layers
    MultiHeadAttention attention_layer;
    FeedForward feedforward_layer;
    LayerNorm norm1;
    LayerNorm norm2;

    // Dimensions
    size_t embed_dim;
    size_t num_heads;
    size_t hidden_dim;
};

class Transformer {
public:
    TransformerBlock(size_t vocab_size, size_t embed_dim, size_t num_heads, size_t hidden_dim, size_t num_layers);

    Matrix forward(const std::vector<size_t> &input_tokens);
    void backward(const std::vector<size_t> &input_tokens, const Matrix &grad_output);

    // Additional methods for training, inference, etc., can be added here.

private:
    size_t vocab_size;
    size_t embed_dim;
    size_t num_heads;
    size_t hidden_dim;
    size_t num_layers;

    // Embedding layer (you may need to implement this)
    Matrix embeddings;

    // Positional encoding
//    PositionalEncoding positional_encoding;

    // Transformer blocks
//    std::vector<TransformerBlock> transformer_blocks;
    TransformerBlock transformer_blocks

    // Output layer (e.g., a linear layer projecting to the vocab size)
    LinearLayer output_layer;

    // Intermediate variables for storing forward pass outputs
    Matrix input_embedding;
    Matrix model_output;
};

#endif // TRANSFORMER_MODEL_H

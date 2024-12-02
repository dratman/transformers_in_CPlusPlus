#include "transformer_model.h"
#include "embedding_layer.h" // Assume you have an embedding layer implementation
#include "positional_encoding.h"
#include "atomic_operations.h"
#include <cmath>

// Constructor
TransformerModel::TransformerModel(size_t vocab_size, size_t embed_dim, size_t num_heads, size_t hidden_dim, size_t num_layers)
    : vocab_size(vocab_size),
      embed_dim(embed_dim),
      num_heads(num_heads),
      hidden_dim(hidden_dim),
      num_layers(num_layers),
//      positional_encoding(embed_dim),
      output_layer(embed_dim, vocab_size)
{
    // Initialize embeddings matrix (vocab_size x embed_dim)
    embeddings = Matrix(vocab_size, embed_dim);

    // Initialize embeddings with random values or use a specific initialization
    for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < embed_dim; ++j) {
            embeddings.data[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Initialize Transformer blocks
//    for (size_t i = 0; i < num_layers; ++i) {
//        transformer_blocks.emplace_back(embed_dim, num_heads, hidden_dim);
//    }
}

// Forward pass
Matrix TransformerModel::forward(const std::vector<size_t> &input_tokens) {
    // Step 1: Embed the input tokens
    size_t seq_length = input_tokens.size();
    input_embedding = Matrix(seq_length, embed_dim);

    for (size_t i = 0; i < seq_length; ++i) {
        size_t token = input_tokens[i];
        if (token >= vocab_size) {
            throw std::out_of_range("Input token index exceeds vocabulary size.");
        }
        input_embedding.data[i] = embeddings.data[token]; // Copy embedding vector
    }

    // Step 2: Add positional encoding
    Matrix position_encoded = positional_encoding.forward(input_embedding);

    // Step 3: Pass through Transformer blocks
    Matrix transformer_output = position_encoded;

    for (size_t i = 0; i < num_layers; ++i) {
        transformer_output = transformer_blocks[i].forward(transformer_output);
    }

    // Step 4: Pass through output layer (e.g., linear layer to project to vocab size)
    model_output = output_layer.forward(transformer_output);

    // Optionally apply softmax if needed
    // model_output = softmax(model_output);

    return model_output;
}

// Backward pass
void TransformerModel::backward(const std::vector<size_t> &input_tokens, const Matrix &grad_output) {
    // Step 1: Backprop through output layer
    Matrix grad_transformer_output = output_layer.backward(transformer_output, grad_output);

    // Step 2: Backprop through Transformer blocks
    for (int i = num_layers - 1; i >= 0; --i) {
        grad_transformer_output = transformer_blocks[i].backward(transformer_blocks[i].get_output(), grad_transformer_output);
    }

    // Step 3: Backprop through positional encoding
    Matrix grad_input_embedding = positional_encoding.backward(input_embedding, grad_transformer_output);

    // Step 4: Backprop through embeddings
    // Update embeddings based on grad_input_embedding
    // Since embeddings are a lookup, we need to accumulate gradients for each token

    // Initialize gradients for embeddings
    Matrix grad_embeddings(vocab_size, embed_dim);

    for (size_t i = 0; i < input_tokens.size(); ++i) {
        size_t token = input_tokens[i];
        for (size_t j = 0; j < embed_dim; ++j) {
            grad_embeddings.data[token][j] += grad_input_embedding.data[i][j];
        }
    }

    // Update embeddings using gradients (you might want to apply learning rate, etc.)
    // For example:
    // float learning_rate = 0.001f;
    // for (size_t i = 0; i < vocab_size; ++i) {
    //     for (size_t j = 0; j < embed_dim; ++j) {
    //         embeddings.data[i][j] -= learning_rate * grad_embeddings.data[i][j];
    //     }
    // }
}


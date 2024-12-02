#include "embedding_layer.h"
#include <cstdlib>

EmbeddingLayer::EmbeddingLayer(size_t vocab_size, size_t embed_dim)
    : embeddings(vocab_size, embed_dim) {
    // Initialize embeddings with random values
    for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < embed_dim; ++j) {
            embeddings.data[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

Matrix EmbeddingLayer::forward(const std::vector<size_t> &input_tokens) {
    size_t seq_length = input_tokens.size();
    Matrix output(seq_length, embeddings.cols);

    for (size_t i = 0; i < seq_length; ++i) {
        size_t token = input_tokens[i];
        output.data[i] = embeddings.data[token];
    }

    return output;
}

void EmbeddingLayer::backward(const std::vector<size_t> &input_tokens, const Matrix &grad_output) {
    // Accumulate gradients for embeddings
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        size_t token = input_tokens[i];
        for (size_t j = 0; j < embeddings.cols; ++j) {
            // Update embeddings directly or store gradients for an optimizer
            embeddings.data[token][j] -= grad_output.data[i][j]; // Example update
        }
    }
}

const Matrix& EmbeddingLayer::get_embeddings() const {
    return embeddings;
}

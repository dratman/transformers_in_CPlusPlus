#include "embedding_layer.h"
#include <cassert>
#include <iostream>

void test_embedding_layer_forward() {
    size_t vocab_size = 10;
    size_t embed_dim = 4;
    EmbeddingLayer embedding(vocab_size, embed_dim);

    // Initialize embeddings for deterministic behavior
    for (size_t i = 0; i < vocab_size; ++i) {
        for (size_t j = 0; j < embed_dim; ++j) {
            embedding.embeddings.data[i][j] = static_cast<float>(i + j);
        }
    }

    std::vector<size_t> input_tokens = {1, 3, 5};

    Matrix output = embedding.forward(input_tokens);

    assert(output.rows == input_tokens.size());
    assert(output.cols == embed_dim);

    // Check that output embeddings match expected values
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        size_t token = input_tokens[i];
        for (size_t j = 0; j < embed_dim; ++j) {
            assert(output.data[i][j] == embedding.embeddings.data[token][j]);
        }
    }

    std::cout << "test_embedding_layer_forward passed!" << std::endl;
}

int main() {
    test_embedding_layer_forward();
    std::cout << "All embedding layer tests passed!" << std::endl;
    return 0;
}

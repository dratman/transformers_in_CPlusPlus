#ifndef EMBEDDING_LAYER_H
#define EMBEDDING_LAYER_H

#include "atomic_operations.h"
#include <vector>

class EmbeddingLayer {
public:
    EmbeddingLayer(size_t vocab_size, size_t embed_dim);

    Matrix forward(const std::vector<size_t> &input_tokens);
    void backward(const std::vector<size_t> &input_tokens, const Matrix &grad_output);

    // Accessor for embeddings (optional)
    const Matrix& get_embeddings() const;

private:
    Matrix embeddings;
};

#endif // EMBEDDING_LAYER_H

#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <vector>
#include <cmath>

class positional_encoding {
public:
    // Constructor to initialize with sequence length and embedding dimension
    positional_encoding(size_t sequence_length, size_t embedding_dim);

    // Returns the generated positional encoding matrix
    const std::vector<std::vector<float>>& getMatrix() const;

private:
    size_t sequence_length_;
    size_t embedding_dim_;
    std::vector<std::vector<float>> matrix_;
    int cols;

    // Helper function to calculate the positional encoding
    void generateMatrix();
};

#endif // POSITIONAL_ENCODING_H

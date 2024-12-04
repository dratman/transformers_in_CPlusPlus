#include "positional_encoding.h"

positional_encoding::positional_encoding(size_t sequence_length, size_t embedding_dim)
    : sequence_length_(sequence_length), embedding_dim_(embedding_dim), matrix_(sequence_length, std::vector<float>(embedding_dim, 0.0f)) {
    generateMatrix();
}

void positional_encoding::generateMatrix() {
    for (size_t pos = 0; pos < sequence_length_; ++pos) {
        for (size_t i = 0; i < embedding_dim_; ++i) {
            if (i % 2 == 0) {
                matrix_[pos][i] = std::sin(pos / std::pow(10000.0, static_cast<float>(i) / embedding_dim_));
            } else {
                matrix_[pos][i] = std::cos(pos / std::pow(10000.0, static_cast<float>(i) / embedding_dim_));
            }
        }
    }
}

const std::vector<std::vector<float>>& positional_encoding::getMatrix() const {
    return matrix_;
}

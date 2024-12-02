#include "layer_normalization.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_layer_normalization_forward() {
    size_t embed_dim = 4;
    LayerNormalization layer_norm(embed_dim);

    Matrix input(2, embed_dim);
    input.data = {{1.0f, 2.0f, 3.0f, 4.0f},
                  {5.0f, 6.0f, 7.0f, 8.0f}};

    Matrix output = layer_norm.forward(input);

    // Check dimensions
    assert(output.rows == input.rows);
    assert(output.cols == input.cols);

    // Since layer normalization normalizes each sample independently,
    // we can compute expected outputs manually if desired.

    std::cout << "test_layer_normalization_forward passed!" << std::endl;
}

int main() {
    test_layer_normalization_forward();
    std::cout << "All layer normalization tests passed!" << std::endl;
    return 0;
}

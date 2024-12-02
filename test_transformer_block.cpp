#include "transformer_block.h"
#include <cassert>
#include <iostream>

void test_transformer_block_forward() {
    size_t embed_dim = 4;
    size_t num_heads = 2;
    size_t hidden_dim = 8;

    TransformerBlock transformer_block(embed_dim, num_heads, hidden_dim);

    Matrix input(1, embed_dim);
    input.data = {{1.0f, 0.0f, 1.0f, 0.0f}};

    Matrix output = transformer_block.forward(input);

    // Check output dimensions
    assert(output.rows == input.rows);
    assert(output.cols == embed_dim);

    std::cout << "test_transformer_block_forward passed!" << std::endl;
}

int main() {
    test_transformer_block_forward();
    std::cout << "All transformer block tests passed!" << std::endl;
    return 0;
}

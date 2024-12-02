#include "multi_head_attention.h"
#include <cassert>
#include <iostream>

void test_multi_head_attention_forward() {
    size_t embed_dim = 4;
    size_t num_heads = 2;

    MultiHeadAttention mha(embed_dim, num_heads);

    Matrix query(1, embed_dim);
    query.data = {{1.0f, 0.0f, 1.0f, 0.0f}};
    Matrix key = query;
    Matrix value = query;

    Matrix output = mha.forward(query, key, value);

    // Check output dimensions
    assert(output.rows == query.rows);
    assert(output.cols == embed_dim);

    std::cout << "test_multi_head_attention_forward passed!" << std::endl;
}

int main() {
    test_multi_head_attention_forward();
    std::cout << "All multi-head attention tests passed!" << std::endl;
    return 0;
}

#include "transformer_model.h"
#include <cassert>
#include <iostream>

void test_transformer_model_forward() {
    size_t vocab_size = 100;
    size_t embed_dim = 8;
    size_t num_heads = 2;
    size_t hidden_dim = 16;
    size_t num_layers = 2;

    TransformerModel model(vocab_size, embed_dim, num_heads, hidden_dim, num_layers);

    // Create a simple input sequence
    std::vector<size_t> input_tokens = {10, 20, 30, 40};

    Matrix output = model.forward(input_tokens);

    // Check output dimensions
    assert(output.rows == input_tokens.size());
    assert(output.cols == vocab_size); // Output projects to vocab size

    std::cout << "test_transformer_model_forward passed!" << std::endl;
}

int main() {
    test_transformer_model_forward();
    std::cout << "All transformer model tests passed!" << std::endl;
    return 0;
}

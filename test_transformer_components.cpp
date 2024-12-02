#include "feedforward_layer.h"
#include "transformer_block.h"
#include <iostream>
#include <cassert>

// Utility function to print a matrix
void print_matrix(const Matrix &m) {
    for (const auto &row : m.data) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Test for Feedforward Layer
void test_feedforward_layer() {
    size_t input_dim = 4, hidden_dim = 8;
    FeedforwardLayer feedforward(input_dim, hidden_dim);

    // Create dummy input
    Matrix input(2, input_dim); // Batch size = 2, Input dim = 4
    input.data = {{1, 2, 3, 4}, {5, 6, 7, 8}};

    // Forward pass
    Matrix output = feedforward.forward(input);
    assert(output.rows == 2 && output.cols == input_dim);
    std::cout << "Forward pass output (Feedforward Layer):\n";
    print_matrix(output);

    // Backward pass
    Matrix grad_output(2, input_dim); // Same shape as output
    grad_output.data = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    Matrix grad_input = feedforward.backward(input, grad_output);

    assert(grad_input.rows == 2 && grad_input.cols == input_dim);
    std::cout << "Backward pass gradients (Feedforward Layer):\n";
    print_matrix(grad_input);

    std::cout << "test_feedforward_layer passed.\n";
}

// Test for Transformer Block
void test_transformer_block() {
    size_t d_model = 4, num_heads = 2, d_ff = 8;
    TransformerBlock transformer(d_model, num_heads, d_ff);

    // Create dummy input
    Matrix input(2, d_model); // Batch size = 2, Model dim = 4
    input.data = {{1, 2, 3, 4}, {5, 6, 7, 8}};

    // Forward pass
    Matrix output = transformer.forward(input);
    assert(output.rows == 2 && output.cols == d_model);
    std::cout << "Forward pass output (Transformer Block):\n";
    print_matrix(output);

    // Backward pass
    Matrix grad_output(2, d_model); // Same shape as output
    grad_output.data = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    Matrix grad_input = transformer.backward(input, grad_output);

    assert(grad_input.rows == 2 && grad_input.cols == d_model);
    std::cout << "Backward pass gradients (Transformer Block):\n";
    print_matrix(grad_input);

    std::cout << "test_transformer_block passed.\n";
}

// Main test runner
int main() {
    test_feedforward_layer();
    test_transformer_block();

    std::cout << "All transformer component tests passed!\n";
    return 0;
}

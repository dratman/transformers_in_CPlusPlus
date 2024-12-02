#include "linear_layer.h"
#include "attention.h"
#include "positional_encoding.h"
#include "layer_normalization.h"
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

// Test for Linear Layer
void test_linear_layer() {
    size_t input_dim = 3, output_dim = 2;
    LinearLayer linear(input_dim, output_dim);

    Matrix input(2, input_dim); // 2x3 matrix
    input.data = {{1, 2, 3}, {4, 5, 6}};

    // Forward pass
    Matrix output = linear.forward(input);
    assert(output.rows == 2 && output.cols == output_dim);
    std::cout << "Forward pass output (Linear Layer):\n";
    print_matrix(output);

    // Backward pass
    Matrix grad_output(2, output_dim);
    grad_output.data = {{1, 1}, {1, 1}}; // Dummy gradients
    Matrix grad_input = linear.backward(input, grad_output);

    assert(grad_input.rows == 2 && grad_input.cols == input_dim);
    std::cout << "Backward pass gradients (Linear Layer):\n";
    print_matrix(grad_input);

    std::cout << "test_linear_layer passed.\n";
}

// Test for Scaled Dot-Product Attention
void test_attention() {
    size_t d_k = 3;
    ScaledDotProductAttention attention(d_k);

    Matrix query(2, d_k), key(2, d_k), value(2, d_k);
    query.data = {{1, 0, 1}, {0, 1, 0}};
    key.data = {{1, 0, 1}, {0, 1, 0}};
    value.data = {{1, 2, 3}, {4, 5, 6}};

    // Forward pass
    Matrix output = attention.forward(query, key, value);
    assert(output.rows == 2 && output.cols == d_k);
    std::cout << "Forward pass output (Attention):\n";
    print_matrix(output);

    // Backward pass
    Matrix grad_output(2, d_k);
    grad_output.data = {{1, 1, 1}, {1, 1, 1}}; // Dummy gradients
    Matrix grad_query(query.rows, query.cols);
    Matrix grad_key(key.rows, key.cols);
    Matrix grad_value(value.rows, value.cols);
    attention.backward(query, key, value, grad_output, grad_query, grad_key, grad_value);

    assert(grad_value.rows == 2 && grad_value.cols == d_k);
    std::cout << "Backward pass gradients (Attention, Value):\n";
    print_matrix(grad_value);

    std::cout << "test_attention passed.\n";
}

// Test for Positional Encoding
void test_positional_encoding() {
    size_t seq_len = 4, d_model = 8;
    Matrix encodings = positional_encoding(seq_len, d_model);

    assert(encodings.rows == seq_len && encodings.cols == d_model);
    std::cout << "Positional Encodings:\n";
    print_matrix(encodings);

    std::cout << "test_positional_encoding passed.\n";
}

// Test for Layer Normalization
void test_layer_normalization() {
    size_t feature_dim = 3;
    LayerNormalization layer_norm(feature_dim);

    Matrix input(2, feature_dim); // 2x3 matrix
    input.data = {{1, 2, 3}, {4, 5, 6}};

    // Forward pass
    Matrix output = layer_norm.forward(input);
    assert(output.rows == 2 && output.cols == feature_dim);
    std::cout << "Forward pass output (Layer Normalization):\n";
    print_matrix(output);

    // Backward pass
    Matrix grad_output(2, feature_dim);
    grad_output.data = {{1, 1, 1}, {1, 1, 1}}; // Dummy gradients
    Matrix grad_input = layer_norm.backward(input, grad_output);

    assert(grad_input.rows == 2 && grad_input.cols == feature_dim);
    std::cout << "Backward pass gradients (Layer Normalization):\n";
    print_matrix(grad_input);

    std::cout << "test_layer_normalization passed.\n";
}

// Main test runner
int main() {
    test_linear_layer();
    test_attention();
    test_positional_encoding();
    test_layer_normalization();

    std::cout << "All layer component tests passed!\n";
    return 0;
}

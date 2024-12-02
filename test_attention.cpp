#include "attention.h"
#include "atomic_operations.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_scaled_dot_product_attention_forward() {
    Matrix query(1, 2);
    query.data = {{1.0f, 0.0f}};
    Matrix key(1, 2);
    key.data = {{1.0f, 0.0f}};
    Matrix value(1, 2);
    value.data = {{0.0f, 2.0f}};

    ScaledDotProductAttention attention(2);

    Matrix output = attention.forward(query, key, value);

    // Expected output is value since the query perfectly matches the key
    Matrix expected_output(1, 2);
    expected_output.data = {{0.0f, 2.0f}};

    // Compare output to expected output
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            assert(fabs(output.data[i][j] - expected_output.data[i][j]) < 1e-5);
        }
    }
    std::cout << "test_scaled_dot_product_attention_forward passed!" << std::endl;
}

void test_scaled_dot_product_attention_backward() {
    // This test will check if backward pass runs without errors.
    // For more thorough testing, perform gradient checking.

    Matrix query(1, 2);
    query.data = {{1.0f, 0.0f}};
    Matrix key(1, 2);
    key.data = {{1.0f, 0.0f}};
    Matrix value(1, 2);
    value.data = {{0.0f, 2.0f}};

    ScaledDotProductAttention attention(2);

    // Forward pass
    Matrix output = attention.forward(query, key, value);

    // Assume some gradient coming from next layer
    Matrix grad_output(1, 2);
    grad_output.data = {{0.1f, 0.2f}};

    // Initialize gradients
    Matrix grad_query(query.rows, query.cols);
    Matrix grad_key(key.rows, key.cols);
    Matrix grad_value(value.rows, value.cols);

    // Backward pass
    attention.backward(query, key, value, grad_output, grad_query, grad_key, grad_value);

    // Check that gradients have correct dimensions
    assert(grad_query.rows == query.rows && grad_query.cols == query.cols);
    assert(grad_key.rows == key.rows && grad_key.cols == key.cols);
    assert(grad_value.rows == value.rows && grad_value.cols == value.cols);

    std::cout << "test_scaled_dot_product_attention_backward passed!" << std::endl;
}

int main() {
    test_scaled_dot_product_attention_forward();
    test_scaled_dot_product_attention_backward();
    std::cout << "All attention tests passed!" << std::endl;
    return 0;
}

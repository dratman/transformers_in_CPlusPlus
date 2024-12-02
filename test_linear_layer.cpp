#include "linear_layer.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_linear_layer_forward() {
    LinearLayer linear(2, 2);
    linear.weights.data = {{1, 2}, {3, 4}};
    linear.biases.data = {{0, 0}};

    Matrix input(1, 2);
    input.data = {{5, 6}};

    Matrix expected_output(1, 2);
    expected_output.data = {{5 * 1 + 6 * 3, 5 * 2 + 6 * 4}}; // {23, 34}

    Matrix output = linear.forward(input);

    assert(output.rows == expected_output.rows);
    assert(output.cols == expected_output.cols);
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            assert(fabs(output.data[i][j] - expected_output.data[i][j]) < 1e-5);
        }
    }
    std::cout << "test_linear_layer_forward passed!" << std::endl;
}

void test_linear_layer_backward() {
    LinearLayer linear(2, 2);
    linear.weights.data = {{1, 2}, {3, 4}};
    linear.biases.data = {{0, 0}};

    Matrix input(1, 2);
    input.data = {{5, 6}};

    Matrix grad_output(1, 2);
    grad_output.data = {{1, 1}}; // Simple gradient

    // Perform forward pass
    Matrix output = linear.forward(input);

    // Perform backward pass
    Matrix grad_input = linear.backward(input, grad_output);

    // Expected gradients
    Matrix expected_grad_input(1, 2);
    expected_grad_input.data = {{1 * 1 + 1 * 3, 1 * 2 + 1 * 4}}; // {4, 6}

    Matrix expected_grad_weights(2, 2);
    expected_grad_weights.data = {{5, 5}, {6, 6}};

    Matrix expected_grad_biases(1, 2);
    expected_grad_biases.data = {{1, 1}};

    // Check gradients w.r.t input
    for (size_t i = 0; i < grad_input.rows; ++i) {
        for (size_t j = 0; j < grad_input.cols; ++j) {
            assert(fabs(grad_input.data[i][j] - expected_grad_input.data[i][j]) < 1e-5);
        }
    }

    // Check gradients w.r.t weights
    for (size_t i = 0; i < linear.grad_weights.rows; ++i) {
        for (size_t j = 0; j < linear.grad_weights.cols; ++j) {
            assert(fabs(linear.grad_weights.data[i][j] - expected_grad_weights.data[i][j]) < 1e-5);
        }
    }

    // Check gradients w.r.t biases
    for (size_t j = 0; j < linear.grad_biases.cols; ++j) {
        assert(fabs(linear.grad_biases.data[0][j] - expected_grad_biases.data[0][j]) < 1e-5);
    }

    std::cout << "test_linear_layer_backward passed!" << std::endl;
}

int main() {
    test_linear_layer_forward();
    test_linear_layer_backward();
    std::cout << "All linear layer tests passed!" << std::endl;
    return 0;
}

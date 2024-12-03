#include "atomic_operations.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_relu() {
    Matrix input(2, 2);
    input.data = {{-1, 0}, {1, 2}};
    Matrix expected(2, 2);
    expected.data = {{0, 0}, {1, 2}};

    Matrix output = relu(input);

    assert(output.rows == expected.rows);
    assert(output.cols == expected.cols);
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            assert(output.data[i][j] == expected.data[i][j]);
        }
    }
    std::cout << "test_relu passed!" << std::endl;
}

void test_softmax() {
    Matrix input(1, 3);
    input.data = {{1, 2, 3}};
    Matrix expected(1, 3);
    float sum_exp = std::exp(1) + std::exp(2) + std::exp(3);
    expected.data = {
        {static_cast<float>(std::exp(1)) / sum_exp, static_cast<float>(std::exp(2) / sum_exp), static_cast<float>(std::exp(3) / sum_exp)}};

    Matrix output = softmax(input);

    assert(output.rows == expected.rows);
    assert(output.cols == expected.cols);
    for (size_t j = 0; j < output.cols; ++j) {
        assert(fabs(output.data[0][j] - expected.data[0][j]) < 1e-5);
    }
    std::cout << "test_softmax passed!" << std::endl;
}

int main() {
    test_relu();
    test_softmax();
    std::cout << "All activation function tests passed!" << std::endl;
    return 0;
}

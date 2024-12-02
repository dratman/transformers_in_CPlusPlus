#include "atomic_operations.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_matmul() {
    Matrix A(2, 2);
    A.data = {{1, 2}, {3, 4}};
    Matrix B(2, 2);
    B.data = {{5, 6}, {7, 8}};
    Matrix expected(2, 2);
    expected.data = {{19, 22}, {43, 50}};

    Matrix result = matmul(A, B);

    assert(result.rows == expected.rows);
    assert(result.cols == expected.cols);
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            assert(fabs(result.data[i][j] - expected.data[i][j]) < 1e-5);
        }
    }
    std::cout << "test_matmul passed!" << std::endl;
}

void test_transpose() {
    Matrix A(2, 3);
    A.data = {{1, 2, 3}, {4, 5, 6}};
    Matrix expected(3, 2);
    expected.data = {{1, 4}, {2, 5}, {3, 6}};

    Matrix result = transpose(A);

    assert(result.rows == expected.rows);
    assert(result.cols == expected.cols);
    for (size_t i = 0; i < expected.rows; ++i) {
        for (size_t j = 0; j < expected.cols; ++j) {
            assert(result.data[i][j] == expected.data[i][j]);
        }
    }
    std::cout << "test_transpose passed!" << std::endl;
}

void test_add() {
    Matrix A(2, 2);
    A.data = {{1, 2}, {3, 4}};
    Matrix B(2, 2);
    B.data = {{5, 6}, {7, 8}};
    Matrix expected(2, 2);
    expected.data = {{6, 8}, {10, 12}};

    Matrix result = add(A, B);

    assert(result.rows == expected.rows);
    assert(result.cols == expected.cols);
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            assert(result.data[i][j] == expected.data[i][j]);
        }
    }
    std::cout << "test_add passed!" << std::endl;
}

void test_elementwise_multiply() {
    Matrix A(2, 2);
    A.data = {{1, 2}, {3, 4}};
    Matrix B(2, 2);
    B.data = {{5, 6}, {7, 8}};
    Matrix expected(2, 2);
    expected.data = {{5, 12}, {21, 32}};

    Matrix result = elementwise_multiply(A, B);

    assert(result.rows == expected.rows);
    assert(result.cols == expected.cols);
    for (size_t i = 0; i < result.rows; ++i) {
        for (size_t j = 0; j < result.cols; ++j) {
            assert(result.data[i][j] == expected.data[i][j]);
        }
    }
    std::cout << "test_elementwise_multiply passed!" << std::endl;
}

int main() {
    test_matmul();
    test_transpose();
    test_add();
    test_elementwise_multiply();
    std::cout << "All atomic operations tests passed!" << std::endl;
    return 0;
}

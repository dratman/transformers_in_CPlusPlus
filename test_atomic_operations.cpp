#include "atomic_operations.h"
#include <iostream>
#include <cassert>

// Utility function to print a matrix (for debugging)
void print_matrix(const Matrix &m) {
    for (const auto &row : m.data) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Test for matrix multiplication
void test_matmul() {
    Matrix a(2, 3);
    Matrix b(3, 2);
    a.data = {{1, 2, 3}, {4, 5, 6}};
    b.data = {{7, 8}, {9, 10}, {11, 12}};

    Matrix result = matmul(a, b);
    assert(result.rows == 2 && result.cols == 2);
    assert(result.data[0][0] == 58 && result.data[0][1] == 64);
    assert(result.data[1][0] == 139 && result.data[1][1] == 154);
    std::cout << "test_matmul passed.\n";
}

// Test for matrix addition
void test_add() {
    Matrix a(2, 2);
    Matrix b(2, 2);
    a.data = {{1, 2}, {3, 4}};
    b.data = {{5, 6}, {7, 8}};

    Matrix result = add(a, b);
    assert(result.rows == 2 && result.cols == 2);
    assert(result.data[0][0] == 6 && result.data[0][1] == 8);
    assert(result.data[1][0] == 10 && result.data[1][1] == 12);
    std::cout << "test_add passed.\n";
}

// Test for ReLU activation
void test_relu() {
    Matrix a(2, 2);
    a.data = {{-1, 2}, {-3, 4}};

    Matrix result = relu(a);
    assert(result.rows == 2 && result.cols == 2);
    assert(result.data[0][0] == 0 && result.data[0][1] == 2);
    assert(result.data[1][0] == 0 && result.data[1][1] == 4);
    std::cout << "test_relu passed.\n";
}

// Test for softmax activation
void test_softmax() {
    Matrix a(1, 3);
    a.data = {{1, 2, 3}};

    Matrix result = softmax(a);
    float sum = result.data[0][0] + result.data[0][1] + result.data[0][2];
    assert(result.rows == 1 && result.cols == 3);
    assert(sum > 0.99 && sum < 1.01); // Ensure softmax normalizes to 1
    assert(result.data[0][0] < result.data[0][1] && result.data[0][1] < result.data[0][2]); // Higher inputs -> higher outputs
    std::cout << "test_softmax passed.\n";
}

// Test for sum reduction
void test_sum() {
    Matrix a(2, 2);
    a.data = {{1, 2}, {3, 4}};

    float result = sum(a);
    assert(result == 10);
    std::cout << "test_sum passed.\n";
}

// Test for mean reduction
void test_mean() {
    Matrix a(2, 2);
    a.data = {{1, 2}, {3, 4}};

    float result = mean(a);
    assert(result == 2.5);
    std::cout << "test_mean passed.\n";
}

// Main test runner
int main() {
    test_matmul();
    test_add();
    test_relu();
    test_softmax();
    test_sum();
    test_mean();

    std::cout << "All tests passed!\n";
    return 0;
}

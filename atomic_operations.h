#ifndef ATOMIC_OPERATIONS_H
#define ATOMIC_OPERATIONS_H

#include <vector>

// Matrix representation
struct Matrix {
    std::vector<std::vector<float>> data;
    size_t rows;
    size_t cols;

    // Default constructor
    Matrix() : data(), rows(0), cols(0) {}

    // Constructor with dimensions
    Matrix(size_t r, size_t c) : data(r, std::vector<float>(c, 0.0f)), rows(r), cols(c) {}
};

// Matrix transpose
Matrix transpose(const Matrix &m);

// Matrix multiplication
Matrix matmul(const Matrix &a, const Matrix &b);

// Addition
Matrix add(const Matrix &a, const Matrix &b);

// Elementwise multiplication
Matrix multiply(const Matrix &a, const Matrix &b);

// Activation functions
Matrix relu(const Matrix &input);
Matrix softmax(const Matrix &input);

// Reduction operations
float mean(const Matrix &input);
float sum(const Matrix &input);

// Activation function derivatives
Matrix relu_backward(const Matrix &input, const Matrix &grad_output);

#endif

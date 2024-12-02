#include "atomic_operations.h"
#include <cmath>
#include <stdexcept>
#include <numeric>


// Matrix multiplication
Matrix matmul(const Matrix &a, const Matrix &b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    Matrix result(a.rows, b.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.cols; ++j) {
            for (size_t k = 0; k < a.cols; ++k) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return result;
}

// Elementwise addition
Matrix add(const Matrix &a, const Matrix &b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return result;
}

// Element-wise multiplication (Hadamard product)
Matrix elementwise_multiply(const Matrix &a, const Matrix &b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for element-wise multiplication.");
    }
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }
    return result;
}

// ReLU activation
Matrix relu(const Matrix &input) {
    Matrix result(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            result.data[i][j] = std::max(0.0f, input.data[i][j]);
        }
    }
    return result;
}

// Softmax activation
Matrix softmax(const Matrix &input) {
    Matrix result(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        float max_val = *std::max_element(input.data[i].begin(), input.data[i].end());
        float sum_exp = 0.0f;
        // Compute exponentials and sum
        for (size_t j = 0; j < input.cols; ++j) {
            float exp_val = std::exp(input.data[i][j] - max_val);
            result.data[i][j] = exp_val;
            sum_exp += exp_val;
        }
        // Normalize
        for (size_t j = 0; j < input.cols; ++j) {
            result.data[i][j] /= sum_exp;
        }
    }
    return result;
}

// Mean reduction
float mean(const Matrix &input) {
    float total_sum = 0.0f;
    size_t count = input.rows * input.cols;
    for (size_t i = 0; i < input.rows; ++i) {
        total_sum += std::accumulate(input.data[i].begin(), input.data[i].end(), 0.0f);
    }
    return total_sum / count;
}

// Sum reduction
float sum(const Matrix &input) {
    float total_sum = 0.0f;
    for (size_t i = 0; i < input.rows; ++i) {
        total_sum += std::accumulate(input.data[i].begin(), input.data[i].end(), 0.0f);
    }
    return total_sum;
}

Matrix transpose(const Matrix &m) {
    Matrix result(m.cols, m.rows);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            result.data[j][i] = m.data[i][j];
        }
    }
    return result;
}

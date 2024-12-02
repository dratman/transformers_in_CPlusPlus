#include "multi_head_attention.h"
#include "attention.h"
#include "atomic_operations.h"

// Assuming you have a class MultiHeadAttention with an instance of ScaledDotProductAttention
// and that you have included necessary headers.

Matrix MultiHeadAttention::backward(const Matrix &query, const Matrix &key, const Matrix &value,
                                    const Matrix &grad_output) {
    // Initialize gradient matrices with appropriate dimensions
    Matrix grad_query(query.rows, query.cols);
    Matrix grad_key(key.rows, key.cols);
    Matrix grad_value(value.rows, value.cols);

    // Perform backward pass
    attention.backward(query, key, value, grad_output, grad_query, grad_key, grad_value);

    // For this example, we'll return grad_query
    return grad_query;

    // Alternatively, if you need to return all gradients, you can adjust your function signature
    // to output them via reference parameters or return a custom struct containing all gradients.
}

// Constructor
MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads)
    : d_model(d_model), num_heads(num_heads), attention(d_model / num_heads) {}

// Forward pass
Matrix MultiHeadAttention::forward(const Matrix &query, const Matrix &key, const Matrix &value) {
    return attention.forward(query, key, value); // Simplified single-head attention for now
}

// // Backward pass
// Matrix MultiHeadAttention::backward(const Matrix &query, const Matrix &key, const Matrix &value, const Matrix &grad_output) {
//     return attention.backward(query, key, value, grad_output); // Simplified single-head attention for now
// }

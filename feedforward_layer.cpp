#include "feedforward_layer.h"

// Constructor
FeedforwardLayer::FeedforwardLayer(size_t input_dim, size_t hidden_dim)
    : linear1(input_dim, hidden_dim), linear2(hidden_dim, input_dim) {}

// Member variables to store intermediate activations
Matrix hidden;
Matrix activated;

// Forward pass
Matrix FeedforwardLayer::forward(const Matrix &input) {
    // Apply first linear layer
    hidden = linear1.forward(input);

    // Apply ReLU activation
    activated = relu(hidden);

    // Apply second linear layer
    return linear2.forward(activated);
}

// Backward pass
Matrix FeedforwardLayer::backward(const Matrix &input, const Matrix &grad_output) {
    // Backprop through second linear layer
    Matrix grad_activated = linear2.backward(activated, grad_output);

    // Backprop through ReLU activation
    Matrix grad_hidden = relu_backward(hidden, grad_activated);

    // Backprop through first linear layer
    Matrix grad_input = linear1.backward(input, grad_hidden);

    return grad_input;
}

#include "atomic_operations.h"

// ReLU backward function
Matrix relu_backward(const Matrix &input, const Matrix &grad_output) {
    Matrix grad_input(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            grad_input.data[i][j] = (input.data[i][j] > 0.0f) ? grad_output.data[i][j] : 0.0f;
        }
    }
    return grad_input;
}

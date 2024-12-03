// linear.cpp
#include "linear.h"

Linear::Linear(int input_dim, int output_dim) {
    // Initialize weights and biases
    weights_.resize(output_dim, std::vector<float>(input_dim));
    biases_.resize(output_dim);

    // TODO: Implement weight initialization
}

std::vector<std::vector<float>> Linear::forward(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output;

    for (const auto& input_vector : input) {
        std::vector<float> output_vector(biases_);

        // Matrix multiplication
        for (int i = 0; i < weights_.size(); ++i) {
            for (int j = 0; j < weights_[0].size(); ++j) {
                output_vector[i] += weights_[i][j] * input_vector[j];
            }
        }
        output.push_back(output_vector);
    }

    return output;
}

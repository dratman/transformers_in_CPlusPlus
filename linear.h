// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include <vector>

class Linear {
public:
    Linear(int input_dim, int output_dim);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);

private:
    std::vector<std::vector<float>> weights_;
    std::vector<float> biases_;
};

#endif // LINEAR_H

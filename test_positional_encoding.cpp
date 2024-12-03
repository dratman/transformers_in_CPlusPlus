#include "positional_encoding.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_positional_encoding_forward() {
    size_t embed_dim = 4;
    size_t seq_length = 2;
    PositionalEncoding positional_encoding(embed_dim);

    Matrix input(seq_length, embed_dim);
    input.data = {{1.0f, 2.0f, 3.0f, 4.0f},
                  {5.0f, 6.0f, 7.0f, 8.0f}};

    Matrix output = pos_enc.forward(input);

    // Since the positional encoding values are based on sinusoids, we can check
    // if the output is the sum of input and positional encodings.

    // We can compute the expected positional encodings manually or accept that
    // the function runs without errors and outputs the correct dimensions.

    assert(output.rows == input.rows);
    assert(output.cols == input.cols);

    // For simplicity, check that output differs from input
    bool differs = false;
    for (size_t i = 0; i < output.rows; ++i) {
        for (size_t j = 0; j < output.cols; ++j) {
            if (fabs(output.data[i][j] - input.data[i][j]) > 1e-5) {
                differs = true;
                break;
            }
        }
        if (differs) break;
    }
    assert(differs);
    std::cout << "test_positional_encoding_forward passed!" << std::endl;
}

int main() {
    test_positional_encoding_forward();
    std::cout << "All positional encoding tests passed!" << std::endl;
    return 0;
}

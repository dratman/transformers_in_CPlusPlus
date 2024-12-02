#include "positional_encoding.h"
#include <cmath>

// Compute positional encodings
Matrix positional_encoding(size_t seq_len, size_t d_model) {
    Matrix enc(seq_len, d_model);
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
            float angle = pos / std::pow(10000.0f, 2 * (i / 2) / static_cast<float>(d_model));
            enc.data[pos][i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
    return enc;
}

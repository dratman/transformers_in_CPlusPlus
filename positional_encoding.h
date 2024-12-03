#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "atomic_operations.h"

typedef Matrix PositionalEncoding;

PositionalEncoding positional_encoding(size_t seq_len, size_t d_model);

#endif
//
//
//
// #ifndef POSITIONAL_ENCODING_H
// #define POSITIONAL_ENCODING_H
//
// #include "atomic_operations.h"
//
// Matrix positional_encoding(size_t seq_len, size_t d_model);
//
// #endif

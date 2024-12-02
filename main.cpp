#include "atomic_operations.h"
#include <iostream>

int main() {
    Matrix a(2, 3);
    Matrix b(3, 2);
    Matrix c(2, 2);

    // Fill matrices
    a.data = {{1, 2, 3}, {4, 5, 6}};
    b.data = {{7, 8}, {9, 10}, {11, 12}};

    // Test matmul
    try {
        c = matmul(a, b);
        std::cout << "Matrix multiplication result:\n";
        for (const auto &row : c.data) {
            for (float val : row) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
    }

    return 0;
}

#include "invert.h"
#include <cassert>
#include <iostream>

int main() {
    Matrix A(2, 2, {1, 2, 3, 4});
    Matrix B(2, 2, {5, 6, 7, 8});

    Matrix C = A.multiply(B);

    assert(C.at(0,0) == 19);
    assert(C.at(0,1) == 22);
    assert(C.at(1,0) == 43);
    assert(C.at(1,1) == 50);

    std::cout << "C++ test passed ✅\n";
}

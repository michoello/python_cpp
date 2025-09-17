#include "invert.h"
#include <cassert>
#include <iostream>

void test_multiply() {
    Matrix A(2, 2);
    Matrix B(2, 2);
    A.set_data({1, 2, 3, 4});
    B.set_data({5, 6, 7, 8});

    Matrix C = A.multiply(B);

    assert(C.at(0,0) == 19);
    assert(C.at(0,1) == 22);
    assert(C.at(1,0) == 43);
    assert(C.at(1,1) == 50);

    std::cout << "Multiply test passed ✅\n";
}


void test_random() {
    Matrix A(10, 15);
    A.fill_uniform();
    for(size_t r = 0; r < 10; ++r) {
        for(size_t c = 0; c < 15; ++c) {
           assert(A.at(r, c) <= 1);
           assert(A.at(r, c) >= -1);
        }
    }

    std::cout << "Random test passed ✅\n";
}


template <typename T>
void assertEqualVectors(const std::vector<std::vector<T>>& got,
                        const std::vector<std::vector<T>>& expected,
                        const char* msg = "")
{
    if (got == expected) {
        return; // success
    }

    std::cerr << "Assertion failed";
    if (msg && *msg) std::cerr << ": " << msg;
    std::cerr << "\n";

    std::cerr << "Expected:\n";
    for (const auto& row : expected) {
        std::cerr << "  { ";
        for (const auto& val : row) std::cerr << val << " ";
        std::cerr << "}\n";
    }

    std::cerr << "Got:\n";
    for (const auto& row : got) {
        std::cerr << "  { ";
        for (const auto& val : row) std::cerr << val << " ";
        std::cerr << "}\n";
    }

    std::exit(1);
}



void test_matmul() {

    Matrix ma(2, 3);
    ma.set_data({1, 2, 3, 4, 5, 6});
    Matrix mb(3, 4);
    mb.set_data({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    DataBlock da(2, 3);
    DataBlock db(3, 4);


    MulBlock dc(&da, &db);

    da.SetVal(ma);
    db.SetVal(mb);


    assertEqualVectors(da.GetVal().value(), {
      {1, 2, 3},
      {4, 5, 6},
    });

    dc.CalcVal();

    assertEqualVectors(dc.GetVal().value(), {
      { 38, 44, 50, 56 },
      { 83, 98, 113, 128 },
    });

    std::cout << "MatMul test passed ✅\n";
}




int main() {
    test_multiply();
    test_random();
    test_matmul();
}

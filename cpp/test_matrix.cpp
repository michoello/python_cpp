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


    MatMulBlock dc(&da, &db);

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


void test_sqrt_matrix() {
    DataBlock da(2, 3);

    SqrtBlock dc(&da);
    SqrtBlock dc2(&dc);

    Matrix ma(2, 3);
    ma.set_data({1, 2, 3, 4, 5, 6});
    da.SetVal(ma);

    dc2.CalcVal();
    assertEqualVectors(dc2.GetVal().value(), {
      {1, 16, 81},
      {256, 625, 1296},
    });

    // dc is also calculated
    assertEqualVectors(dc.GetVal().value(), {
      {1, 4, 9},
      {16, 25, 36},
    });

    std::cout << "Sqrt matrix test passed ✅\n";
}

void test_add_matrix() {
    DataBlock da(2, 3);
    DataBlock db(2, 3);
    DataBlock dc(2, 3);


    Matrix ma(2, 3);
    ma.set_data({1, 2, 3, 4, 5, 6});
    da.SetVal(ma);

    Matrix mb(2, 3);
    mb.set_data({4, 5, 6, 1, 2, 3 });
    db.SetVal(mb);

    Matrix mc(2, 3);
    mc.set_data({1, 1, 1, 2, 2, 2});
    dc.SetVal(mc);

    AddBlock ds1(&da, &db);
    AddBlock ds2(&ds1, &dc);


    ds2.CalcVal();
    assertEqualVectors(ds2.GetVal().value(), {
      {6, 8, 10},
      {7, 9, 11},
    });

    // ds1 is also calculated
    assertEqualVectors(ds1.GetVal().value(), {
      {5, 7, 9},
      {5, 7, 9},
    });

    std::cout << "Sum matrix test passed ✅\n";
}


void test_dif_matrix() {
    DataBlock da(2, 3);
    DataBlock db(2, 3);

    Matrix ma(2, 3);
    ma.set_data({1, 2, 3, 4, 5, 6});
    da.SetVal(ma);

    Matrix mb(2, 3);
    mb.set_data({2, 3, 5, 8, 13, 21 });
    db.SetVal(mb);

    DifBlock dd(&db, &da); // db - da

    dd.CalcVal();
    assertEqualVectors(dd.GetVal().value(), {
      {1, 1, 2},
      {4, 8, 15},
    });

    std::cout << "Differences between 2 matrices test passed ✅\n";
}

void test_mul_el() {
    DataBlock da(2, 3);

    Matrix ma(2, 3);
    ma.set_data({1, 2, 3, 4, 5, 6});
    da.SetVal(ma);

    MulElBlock db(&da, 2);
    MulElBlock dc(&db, -1);

    dc.CalcVal();

    assertEqualVectors(db.GetVal().value(), {
      {2, 4, 6},
      {8, 10, 12},
    });

    assertEqualVectors(dc.GetVal().value(), {
      {-2, -4, -6},
      {-8, -10, -12},
    });

    std::cout << "Mul el matrix test passed ✅\n";
}

void test_sum_mat() {
    DataBlock da(2, 3);

    Matrix ma(2, 3);
    ma.set_data({1, 2, 3, 4, 5, 6});
    da.SetVal(ma);

    SumBlock ds(&da);

    ds.CalcVal();

    assertEqualVectors(ds.GetVal().value(), {
      {21},
    });

    std::cout << "Sum matrix test passed ✅\n";
}



int main() {
    test_multiply();
    test_random();
    test_matmul();
    test_sqrt_matrix();
    test_add_matrix();
    test_mul_el();
    test_dif_matrix();
    test_sum_mat();
}

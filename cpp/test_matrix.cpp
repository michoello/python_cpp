#include "invert.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <type_traits>


void test_multiply() {
    Matrix A(2, 2);
    Matrix B(2, 2);
    A.set_data({{1, 2}, {3, 4}});
    B.set_data({{5, 6}, {7, 8}});

    Matrix C(2, 2); 
		::multiply_matrix(A, B, &C);

    assert(C.at(0,0) == 19);
    assert(C.at(0,1) == 22);
    assert(C.at(1,0) == 43);
    assert(C.at(1,1) == 50);

    std::cout << "Multiply test passed ✅\n";
}

void test_shared_data() {
    Matrix A(2, 2);
		A.set_data({{ 1, 2}, {3, 4}});

    Matrix B(A);

    A.at(1, 1) = 5;

    assert(B.rows == 2);
    assert(B.cols == 2);
    assert(B.at(0,0) == 1);
    assert(B.at(1,1) == 5);

    std::cout << "Shared data passed ✅\n";
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
bool approxEqual(T a, T b, double tol = 1e-3) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::fabs(a - b) <= tol;
    } else {
        return a == b; // exact for integers
    }
}

template <typename T>
void assertEqualVectors(const std::vector<std::vector<T>>& got,
                        const std::vector<std::vector<T>>& expected,
                        double tol = 1e-3)
{
    if (got.size() != expected.size()) {
        std::cerr << "Assertion failed (different number of rows):" << got.size() << " vs " << expected.size();
        std::cerr << "\n";
        std::exit(1);
    }

    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i].size() != expected[i].size()) {
            std::cerr << "Assertion failed (different number of columns in row " << i << ")";
            std::cerr << "\n";
            std::exit(1);
        }
        for (size_t j = 0; j < got[i].size(); ++j) {
            if (!approxEqual(got[i][j], expected[i][j], tol)) {
                std::cerr << "Assertion failed";
                std::cerr << "\n";

                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << "expected " << expected[i][j]
                          << " but got " << got[i][j]
                          << " (tolerance " << tol << ")\n";

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
        }
    }
}


void test_matmul() {

    Matrix ma(2, 3);
    ma.set_data({{1, 2, 3}, {4, 5, 6}});
    Matrix mb(3, 4);
    mb.set_data({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

    DataBlock da(ma);
    DataBlock db(mb);

    MatMulBlock dc(&da, &db);

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
    Matrix ma(2, 3);
    DataBlock da(ma);

    SqrtBlock dc(&da);
    SqrtBlock dc2(&dc);

    ma.set_data({{1, 2, 3}, {4, 5, 6}});

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
    Matrix ma(2, 3);
    Matrix mb(2, 3);
    Matrix mc(2, 3);
    DataBlock da(ma);
    DataBlock db(mb);
    DataBlock dc(mc);

    ma.set_data({{1, 2, 3}, {4, 5, 6}});
    mb.set_data({{4, 5, 6}, {1, 2, 3 }});
    mc.set_data({{1, 1, 1}, {2, 2, 2}});

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
    Matrix ma(2, 3);
    Matrix mb(2, 3);
    DataBlock da(ma);
    DataBlock db(mb);

    ma.set_data({{1, 2, 3}, {4, 5, 6}});
    mb.set_data({{2, 3, 5}, {8, 13, 21} });

    DifBlock dd(&db, &da); // db - da

    dd.CalcVal();
    assertEqualVectors(dd.GetVal().value(), {
      {1, 1, 2},
      {4, 8, 15},
    });

    std::cout << "Differences between 2 matrices test passed ✅\n";
}

void test_mul_el() {
    Matrix ma(2, 3);
    DataBlock da(ma);

    ma.set_data({{1, 2, 3}, {4, 5, 6}});

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
    Matrix ma(2, 3);
    DataBlock da(ma);

    ma.set_data({{1, 2, 3}, {4, 5, 6}});

    SumBlock ds(&da);

    ds.CalcVal();

    assertEqualVectors(ds.GetVal().value(), {
      {21},
    });

    ds.CalcDval();
    assertEqualVectors(da.GetDval().value(), {
      {1, 1, 1},
      {1, 1, 1},
    });

    std::cout << "Sum matrix test passed ✅\n";
}



void test_sse() {
    Matrix ma(2, 3);
    DataBlock da(ma);
    ma.set_data({{1, 2, 3}, {4, 5, 6}});

    Matrix mb(2, 3);
    DataBlock db(mb);
    mb.set_data({{1, 2, 4}, {4, 5, 4}});


    SSEBlock ds(&da, &db);

    ds.CalcVal();

    assertEqualVectors(ds.GetVal().value(), {
      {5},
    });

    std::cout << "SSE matrix test passed ✅\n";
}


void test_sse_with_grads() {
    // "output"
    Matrix my(1, 2);
    DataBlock dy(my);
    my.set_data({{1, 2}});  // true labels

    // "labels"
    Matrix ml(1, 2);
    DataBlock dl(ml);
    ml.set_data({{0, 4}});


    SSEBlock ds(&dy, &dl);

    ds.CalcVal();

    assertEqualVectors(ds.GetVal().value(), {
      {5},
    });


    // Calc derivatives
    ds.CalcDval();

    // Derivative of loss function is its value
    assertEqualVectors(ds.GetDval().value(), {
      {5},
    });

    // Derivative of its args
    assertEqualVectors(dy.GetDval().value(), {
      {-2, 4},
    });

    dy.ApplyGrad(0.1);
    assertEqualVectors(dy.GetVal().value(), {
      {0.8, 2.4},
    });

    // Calc loss again
    ds.CalcVal();

    // Derivative of loss function is its value
    assertEqualVectors(ds.GetDval().value(), {
      {3.2},
    });


    std::cout << "SSE matrix test with_gradients passed ✅\n";
}




int main() {
    test_multiply();
    test_shared_data();
    test_random();
    test_matmul();
    test_sqrt_matrix();
    test_add_matrix();
    test_mul_el();
    test_dif_matrix();
    test_sum_mat();
    test_sse();
    test_sse_with_grads();
}

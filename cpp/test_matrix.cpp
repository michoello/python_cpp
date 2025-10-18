#include "invert.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

//-------------------------------------------------------------
// Tiny unittests libraryset
//
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// Shared state per running test
inline int __checks_failed = 0;
inline int __checks_total = 0;

// Store all registered tests
struct TestCase {
  std::string name;
  std::function<void()> func;
};

inline std::vector<TestCase> &get_tests() {
  static std::vector<TestCase> tests;
  return tests;
}

// Register a test case
#define TEST_CASE(name)                                                        \
  void name();                                                                 \
  struct name##_registrar {                                                    \
    name##_registrar() { get_tests().push_back({#name, name}); }               \
  };                                                                           \
  static name##_registrar name##_instance;                                     \
  void name()

// Assertion macro
#define CHECK(agent)                                                           \
  do {                                                                         \
    ++__checks_total;                                                          \
    if (!(agent)) {                                                            \
      ++__checks_failed;                                                       \
      std::cerr << "    Failed: " #agent << " at " << __FILE__ << ":"          \
                << __LINE__ << "\n";                                           \
    }                                                                          \
  } while (0)

// Run a single test
inline bool run_test(const TestCase &t) {
  __checks_failed = 0;
  __checks_total = 0;
  std::cout << "[     ... ] " << t.name << "\n";
  t.func();
  if (__checks_failed == 0) {
    std::cout << "[ ✅ PASS ] " << t.name << " (" << __checks_total
              << " checks)\n";
    return true;
  } else {
    std::cout << "[ ❌ FAIL ] " << t.name << " (" << __checks_failed << "/"
              << __checks_total << " failed)\n";
    return false;
  }
}

// Run all or one test based on CLI
inline int run_tests(int argc, char **argv) {
  if (argc > 1) {
    std::string filter = argv[1];
    for (auto &t : get_tests()) {
      if (t.name == filter) {
        return run_test(t) ? 0 : 1;
      }
    }
    std::cerr << "No test case named '" << filter << "'\n";
    return 1;
  } else {
    int total_failed = 0;
    for (auto &t : get_tests()) {
      if (!run_test(t)) {
        ++total_failed;
      }
    }
    std::cout << "\n=== Summary: " << (get_tests().size() - total_failed)
              << " passed, " << total_failed << " failed ===\n";
    return total_failed == 0 ? 0 : 1;
  }
}

//-------------------------------------------------------------

TEST_CASE(multiply) {
  Matrix A(2, 2);
  Matrix B(2, 2);
  A.set_data({{1, 2}, {3, 4}});
  B.set_data({{5, 6}, {7, 8}});

  Matrix C(2, 2);
  ::multiply_matrix(A, B, &C);

  assert(C.at(0, 0) == 19);
  assert(C.at(0, 1) == 22);
  assert(C.at(1, 0) == 43);
  assert(C.at(1, 1) == 50);
}

TEST_CASE(shared_data) {
  Matrix A(2, 2);
  A.set_data({{1, 2}, {3, 4}});

  Matrix B(A);

  A.at(1, 1) = 5;

  assert(B.rows == 2);
  assert(B.cols == 2);
  assert(B.at(0, 0) == 1);
  assert(B.at(1, 1) == 5);
}

TEST_CASE(random_matrix) {
  Matrix A(10, 15);
  A.fill_uniform();
  for (size_t r = 0; r < 10; ++r) {
    for (size_t c = 0; c < 15; ++c) {
      assert(A.at(r, c) <= 1);
      assert(A.at(r, c) >= -1);
    }
  }
}

template <typename T> bool approxEqual(T a, T b, double tol = 1e-3) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fabs(a - b) <= tol;
  } else {
    return a == b; // exact for integers
  }
}

template <typename T>
bool assertEqualVectors(const std::vector<std::vector<T>> &got,
                        const std::vector<std::vector<T>> &expected,
                        int round = 3) {
  float tol = std::pow(10.0f, -round);

  if (got.size() != expected.size()) {
    std::cerr << "Assertion failed (different number of rows):" << got.size()
              << " vs " << expected.size();
    std::cerr << "\n";
    return false;
  }

  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i].size() != expected[i].size()) {
      std::cerr << "Assertion failed (different number of columns in row " << i
                << ")";
      std::cerr << "\n";
      return false;
    }
    for (size_t j = 0; j < got[i].size(); ++j) {
      if (!approxEqual(got[i][j], expected[i][j], tol)) {
        std::cerr << "Assertion failed";
        std::cerr << "\n";

        std::cerr << "Mismatch at (" << i << "," << j << "): "
                  << "expected " << expected[i][j] << " but got " << std::fixed
                  << std::setprecision(round) << got[i][j] << std::defaultfloat
                  << " (tolerance " << tol << ")\n";

        std::cerr << "Expected:\n";
        for (const auto &row : expected) {
          std::cerr << "  { ";
          for (int i = 0; i < row.size(); ++i) {
            std::cerr << row[i] << (i < row.size() - 1 ? ", " : " ");
          }
          std::cerr << "}\n";
        }

        std::cerr << "Got:\n";
        for (const auto &row : got) {
          std::cerr << "  { ";
          for (int i = 0; i < row.size(); ++i) {
            std::cerr << row[i] << (i < row.size() - 1 ? ", " : " ");
          }
          std::cerr << "}\n";
        }
        return false;
      }
    }
  }
  return true;
}

TEST_CASE(matmul) {
  Block da = Data(2, 3);
  da.val().set_data({{1, 2, 3}, {4, 5, 6}});

  Block db = Data(3, 4);
  db.val().set_data({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

  // Block dc = MatMul(&da, &db);
  // Block dc = Blocks::MatMul(&da, &db);
  Block dc = MatMul(&da, &db);

  CHECK(assertEqualVectors(da.val().value(), {
                                               {1, 2, 3},
                                               {4, 5, 6},
                                           }));

  dc.CalcVal();

  CHECK(assertEqualVectors(dc.val().value(), {
                                               {38, 44, 50, 56},
                                               {83, 98, 113, 128},
                                           }));
}

TEST_CASE(matmul_with_grads) {

  Block da = Data(1, 2);
  da.val().set_data({{1, 2}});

  Block db = Data(2, 3);
  db.val().set_data({{3, 4, 5}, {6, 7, 8}});

  Block dc = MatMul(&da, &db);

  dc.CalcVal();
  CHECK(assertEqualVectors(dc.val().value(), {
                                               {15, 18, 21},
                                           }));

  dc.CalcGrad();
  CHECK(assertEqualVectors(db.back->val().value(), {{1, 1, 1}, {2, 2, 2}}));

  CHECK(assertEqualVectors(da.back->val().value(), {{12, 21}}));

  // TODO: see test_mse_loss in test_hello.py and extend this test with loss
}

TEST_CASE(sqrt_matrix) {
  Block da = Data(2, 3);

  Block *dc = Sqrt(&da);
  Block *dc2 = Sqrt(dc);

  da.val().set_data({{1, 2, 3}, {4, 5, 6}});

  dc2->CalcVal();
  CHECK(assertEqualVectors(dc2->val().value(), {
                                                 {1, 16, 81},
                                                 {256, 625, 1296},
                                             }));

  // dc is also calculated
  CHECK(assertEqualVectors(dc->val().value(), {
                                                {1, 4, 9},
                                                {16, 25, 36},
                                            }));
}

TEST_CASE(add_matrix) {
  Block da = Data(2, 3);
  Block db = Data(2, 3);
  Block dc = Data(2, 3);

  da.val().set_data({{1, 2, 3}, {4, 5, 6}});
  db.val().set_data({{4, 5, 6}, {1, 2, 3}});
  dc.val().set_data({{1, 1, 1}, {2, 2, 2}});

  Block ds1 = Add(&da, &db);
  Block ds2 = Add(&ds1, &dc);

  ds2.CalcVal();
  CHECK(assertEqualVectors(ds2.val().value(), {
                                                {6, 8, 10},
                                                {7, 9, 11},
                                            }));

  // ds1 is also calculated
  CHECK(assertEqualVectors(ds1.val().value(), {
                                                {5, 7, 9},
                                                {5, 7, 9},
                                            }));
}

TEST_CASE(dif_matrix) {
  Block da = Data(2, 3);
  Block db = Data(2, 3);

  da.val().set_data({{1, 2, 3}, {4, 5, 6}});
  db.val().set_data({{2, 3, 5}, {8, 13, 21}});

  // DifBlock dd(&db, &da); // db - da
  Block dd = Dif(&db, &da); // db - da

  dd.CalcVal();
  CHECK(assertEqualVectors(dd.val().value(), {
                                               {1, 1, 2},
                                               {4, 8, 15},
                                           }));
}

TEST_CASE(mul_el) {
  Block da = Data(2, 3);

  da.val().set_data({{1, 2, 3}, {4, 5, 6}});

  Block *db = MulEl(&da, 2);
  Block *dc = MulEl(db, -1);

  dc->CalcVal();

  CHECK(assertEqualVectors(db->val().value(), {
                                                {2, 4, 6},
                                                {8, 10, 12},
                                            }));

  CHECK(assertEqualVectors(dc->val().value(), {
                                                {-2, -4, -6},
                                                {-8, -10, -12},
                                            }));
}

TEST_CASE(sum_mat) {
  Block da = Data(2, 3);
  da.val().set_data({{1, 2, 3}, {4, 5, 6}});

  Block ds = Sum(&da);

  ds.CalcVal();
  CHECK(assertEqualVectors(ds.val().value(), {
                                               {21},
                                           }));

  da.CalcGrada();
  CHECK(assertEqualVectors(da.back->val().value(), {
                                                    {1, 1, 1},
                                                    {1, 1, 1},
                                                }));
}

TEST_CASE(sse) {
  Block da = Data(2, 3);
  da.val().set_data({{1, 2, 3}, {4, 5, 6}});

  Block db = Data(2, 3);
  db.val().set_data({{1, 2, 4}, {4, 5, 4}});

  // SSEBlock ds(&da, &db);
  Block ds = SSE(&da, &db);

  ds.CalcVal();

  CHECK(assertEqualVectors(ds.val().value(), {
                                               {5},
                                           }));
}

TEST_CASE(sse_with_grads) {
  // "output"
  Block dy = Data(1, 2);
  dy.val().set_data({{1, 2}}); // true labels

  // "labels"
  Block dl = Data(1, 2);
  dl.val().set_data({{0, 4}});

  Block ds = SSE(&dy, &dl);

  ds.CalcVal();

  CHECK(assertEqualVectors(ds.val().value(), {{5}}));

  // Calc derivatives
  dy.CalcGrada();

  // Derivative of loss function is its value is 1.0 (aka df/df)
  CHECK(assertEqualVectors(ds.back->val().value(), {
                                                    {1},
                                                }));
  // Derivative of its args
  CHECK(assertEqualVectors(dy.back->val().value(), {
                                                    {2, -4},
                                                }));

  dy.ApplyGrad(0.1);
  CHECK(assertEqualVectors(dy.val().value(), {
                                               {0.8, 2.4},
                                           }));

  // Calc loss again
  ds.CalcVal();
  CHECK(assertEqualVectors(ds.val().value(), {
                                               {3.2},
                                           }));
}

TEST_CASE(sigmoid_with_grads) {

  Block x = Data(1, 2);
  x.val().set_data({{0.1, -0.2}});

  Block w = Data(2, 3);
  w.val().set_data({{-0.1, 0.5, 0.3}, {-0.6, 0.7, 0.8}});

  Block mm = MatMul(&x, &w);

  Block sb = Sigmoid(&mm);

  sb.CalcVal();

  CHECK(assertEqualVectors(sb.val().value(), {{0.527, 0.478, 0.468}}));

  mm.CalcGrada();
  // TODO: add bce loss and check
  // see test_bce_loss in python tests
  CHECK(assertEqualVectors(mm.back->val().value(), {{0.2492, 0.2495, 0.2489}}));
}

TEST_CASE(sigmoid_with_gradas) {

  Block x = Data(1, 2);
  x.val().set_data({{0.1, -0.2}});

  Block w = Data(2, 3);
  w.val().set_data({{-0.1, 0.5, 0.3}, {-0.6, 0.7, 0.8}});

  Block mm = MatMul(&x, &w);

  Block sb = Sigmoid(&mm);

  sb.CalcVal();
  CHECK(assertEqualVectors(sb.val().value(), {{0.527, 0.478, 0.468}}));

  mm.CalcGrada();
  // TODO: add bce loss and check
  // see test_bce_loss in python tests
  CHECK(assertEqualVectors(mm.back->val().value(), {{0.2492, 0.2495, 0.2489}}));
}



// see test_bce_loss in python tests
TEST_CASE(bce_with_grads) {
  Block ypred = Data(1, 3);
  ypred.val().set_data({{0.527, 0.478, 0.468}});
  Block ytrue = Data(1, 3);
  ytrue.val().set_data({{0, 1, 0.468}});

  Block bce = BCE(&ypred, &ytrue);

  bce.CalcVal();
  CHECK(assertEqualVectors(bce.val().value(), {{0.749, 0.738, 0.691}}));

  ypred.CalcGrada();
  CHECK(assertEqualVectors(ypred.back->val().value(), {{2.11416, -2.09205, 0}}));
}


// see test_bce_loss in python tests
TEST_CASE(bce_with_gradas) {
  Block ypred = Data(1, 3);
  ypred.val().set_data({{0.527, 0.478, 0.468}});
  Block ytrue = Data(1, 3);
  ytrue.val().set_data({{0, 1, 0.468}});

  Block bce = BCE(&ypred, &ytrue);

  bce.CalcVal();
  CHECK(assertEqualVectors(bce.val().value(), {{0.749, 0.738, 0.691}}));

  ypred.CalcGrada();
  CHECK(assertEqualVectors(ypred.back->val().value(), {{2.11416, -2.09205, 0}}));
}


TEST_CASE(full_layer_with_loss_with_grads) {
  Block x = Data(1, 2);
  x.val().set_data({{0.1, -0.2}});

  Block w = Data(2, 3);
  w.val().set_data({{-0.1, 0.5, 0.3}, {-0.6, 0.7, 0.8}});

  Block mm = MatMul(&x, &w);
  Block sb = Sigmoid(&mm);

  Block y = Data(1, 3);
  y.val().set_data({{0, 1, 0.468}});

  // loss
  Block bce = BCE(&sb, &y);

  // Forward
  bce.CalcVal();
  CHECK(assertEqualVectors(sb.val().value(), {{0.527, 0.478, 0.468}}));
  CHECK(assertEqualVectors(bce.val().value(), {{0.75, 0.739, 0.691}}));

  // Calc diff and check the loss values
  bce.CalcGrad();

  // TODO: make this work:
  //w.CalcGrada();
  // Derivative of loss against itself is ones
  CHECK(assertEqualVectors(bce.back->val().value(), {{1, 1, 1}}));

  // Make sure the gradient flows backwards
  // Check sigmoid diff
  CHECK(assertEqualVectors(sb.back->val().value(), {{2.116, -2.094, -0.002}}));

  // Check the matrix diff
  CHECK(assertEqualVectors(
      w.back->val().value(),
      {{0.0527, -0.052, -4.543 / 100000}, {-0.105, 0.104, 9.086 / 100000}}));

  // TODO: apply grads to w, calc loss value and check that it is reduced
  // see test_bce_loss inpython
  //
  // Check that w values are still the same
  CHECK(
      assertEqualVectors(w.val().value(), {{-0.1, 0.5, 0.3}, {-0.6, 0.7, 0.8}}));

  w.ApplyGrad(1.0);

  // Check that w values have changed
  CHECK(assertEqualVectors(w.val().value(),
                           {{-0.153, 0.552, 0.3}, {-0.495, 0.596, 0.8}}));

  // Recalculate the loss
  bce.CalcVal();
  // Assure it got smaller!
  CHECK(assertEqualVectors(sb.val().value(), {{0.521, 0.484, 0.468}}));
  CHECK(assertEqualVectors(bce.val().value(), {{0.736, 0.726, 0.691}}));

  // Update the inputs, and check that it also reduces the loss
  x.ApplyGrad(0.01);
  CHECK(assertEqualVectors(x.val().value(), {{0.103, -0.193}}));

  bce.CalcVal();
  CHECK(assertEqualVectors(bce.val().value(), {{0.734, 0.723, 0.691}}));
}

int main(int argc, char **argv) { return run_tests(argc, argv); }

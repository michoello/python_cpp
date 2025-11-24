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

//template <typename T>
bool assertEqualVectors(const std::vector<std::vector<double>> &got,
                        const std::vector<std::vector<double>> &expected,
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
          for (size_t i = 0; i < row.size(); ++i) {
            std::cerr << std::fixed << std::setprecision(round) << row[i] << (i < row.size() - 1 ? ", " : " ");
          }
          std::cerr << "}\n";
        }

        std::cerr << "Got:\n";
        for (const auto &row : got) {
          std::cerr << "  { ";
          for (size_t i = 0; i < row.size(); ++i) {
            std::cerr << std::fixed << std::setprecision(round) << row[i] << (i < row.size() - 1 ? ", " : " ");
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
  Mod3l m;

  Block *da = Data(&m, 2, 3);
  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});

  Block *db = Data(&m, 3, 4);
  m.set_data(db, {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

  Block *dc = MatMul(da, db);

  CHECK(assertEqualVectors(da->fval(), {
                                           {1, 2, 3},
                                           {4, 5, 6},
                                       }));

  dc->calc_fval();

  CHECK(assertEqualVectors(dc->fval(), {
                                           {38, 44, 50, 56},
                                           {83, 98, 113, 128},
                                       }));
}

TEST_CASE(reshape) {
  Mod3l m;

  Block *db = Data(&m, 3, 4);
  m.set_data(db, {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

  Block *dc = Reshape(db, 6, 2);

  CHECK(assertEqualVectors(db->fval(), {
		{1, 2, 3, 4}, 
    {5, 6, 7, 8}, 
    {9, 10, 11, 12}
  }));

  dc->calc_fval();
 
  CHECK(assertEqualVectors(dc->fval(), {
  	{ 1, 2 },
	  { 3, 4 },
	  { 5, 6 },
	  { 7, 8 },
	  { 9, 10 },
	  {   11, 12 }
	}));
}



TEST_CASE(matmul_with_grads) {
  Mod3l m;

  Block *da = Data(&m, 1, 2);
  m.set_data(da, {{1, 2}});

  Block *db = Data(&m, 2, 3);
  m.set_data(db, {{3, 4, 5}, {6, 7, 8}});

  Block *dc = MatMul(da, db);

  dc->calc_fval();
  CHECK(assertEqualVectors(dc->fval(), {
                                           {15, 18, 21},
                                       }));

  da->calc_bval();
  db->calc_bval();
  CHECK(assertEqualVectors(db->bval(), {{1, 1, 1}, {2, 2, 2}}));

  CHECK(assertEqualVectors(da->bval(), {{12, 21}}));

  // TODO: see test_mse_loss in test_hello.py and extend this test with loss
}

TEST_CASE(sqrt_matrix) {
  Mod3l m;
  Block *da = Data(&m, 2, 3);

  Block *dc = Sqrt(da);
  Block *dc2 = Sqrt(dc);

  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});

  dc2->calc_fval();
  CHECK(assertEqualVectors(dc2->fval(), {
                                            {1, 16, 81},
                                            {256, 625, 1296},
                                        }));

  // dc is also calculated
  CHECK(assertEqualVectors(dc->fval(), {
                                           {1, 4, 9},
                                           {16, 25, 36},
                                       }));
}

TEST_CASE(add_matrix) {
  Mod3l m;
  Block *da = Data(&m, 2, 3);
  Block *db = Data(&m, 2, 3);
  Block *dc = Data(&m, 2, 3);
  Block *dy = Data(&m, 2, 3);

  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});
  m.set_data(db, {{4, 5, 6}, {1, 2, 3}});
  m.set_data(dc, {{1, 1, 1}, {2, 2, 2}});
  m.set_data(dy, {{0.1, 0.3, 0.7}, {0.99, 0.5, 0.001}});

  Block *ds2 = Add(Add(da, db), dc);

  ds2->calc_fval();
  CHECK(assertEqualVectors(ds2->fval(), {
                                            {6, 8, 10},
                                            {7, 9, 11},
                                        }));

  Block* dsig = Sigmoid(ds2);
  Block *dl = BCE(dsig, dy);

  dl->calc_fval();
  CHECK(assertEqualVectors(dl->fval(), {
		{ 5.402, 5.600, 3 },
		{ 0.071, 4.500, 10.989 }
  }));

  // Calc derivatives
  da->calc_bval();

  CHECK(assertEqualVectors(dsig->bval(), {
		{ 363.886, 2087.070, 6607.540 },
		{ 9.985, 4051.542, 59815.266 },
                                       }));

  // From Sum and backwards it all goes the same:
  CHECK(assertEqualVectors(ds2->bval(), {
		{ 0.898, 0.700, 0.300 },
		{ 0.009, 0.500, 0.999 }
                                       }));


  CHECK(assertEqualVectors(da->bval(), ds2->bval()));

  // Db is not yet calculated
  CHECK(assertEqualVectors(db->bval(), {
		{ 1, 1, 1},
		{ 1, 1, 1},
                                       }));
  db->calc_bval();
  dc->calc_bval();
  // Now it is calculated
  CHECK(assertEqualVectors(db->bval(), ds2->bval()));
  CHECK(assertEqualVectors(dc->bval(), ds2->bval()));
}

TEST_CASE(dif_matrix) {

  Mod3l m;
  Block *da = Data(&m, 2, 3);
  Block *db = Data(&m, 2, 3);

  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});
  m.set_data(db, {{2, 3, 5}, {8, 13, 21}});

  Block *dd = Dif(db, da); // db - da

  dd->calc_fval();
  CHECK(assertEqualVectors(dd->fval(), {
                                           {1, 1, 2},
                                           {4, 8, 15},
                                       }));
}

TEST_CASE(mul_el) {
  Mod3l m;
  Block *da = Data(&m, 2, 3);

  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});

  Block *db = MulEl(da, 2);
  Block *dc = MulEl(db, -1);

  dc->calc_fval();

  CHECK(assertEqualVectors(db->fval(), {
                                           {2, 4, 6},
                                           {8, 10, 12},
                                       }));

  CHECK(assertEqualVectors(dc->fval(), {
                                           {-2, -4, -6},
                                           {-8, -10, -12},
                                       }));
}

TEST_CASE(sum_mat) {
  Mod3l m;
  Block *da = Data(&m, 2, 3);
  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});

  Block *ds = Sum(da);

  ds->calc_fval();
  CHECK(assertEqualVectors(ds->fval(), {
                                           {21},
                                       }));

  da->calc_bval();
  CHECK(assertEqualVectors(da->bval(), {
                                           {1, 1, 1},
                                           {1, 1, 1},
                                       }));
}

TEST_CASE(sse) {
  Mod3l m;
  Block *da = Data(&m, 2, 3);
  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});

  Block *db = Data(&m, 2, 3);
  m.set_data(db, {{1, 2, 4}, {4, 5, 4}});

  Block *ds = SSE(da, db);

  ds->calc_fval();

  CHECK(assertEqualVectors(ds->fval(), {
                                           {5},
                                       }));
}

TEST_CASE(sse_with_grads) {
  Mod3l m;
  // "output"
  Block *dy = Data(&m, 1, 2);
  m.set_data(dy, {{1, 2}}); // true labels

  // "labels"
  Block *dl = Data(&m, 1, 2);
  m.set_data(dl, {{0, 4}});

  Block *ds = SSE(dy, dl);

  ds->calc_fval();

  CHECK(assertEqualVectors(ds->fval(), {{5}}));

  // Calc derivatives
  dy->calc_bval();

  // Derivative of loss function is its value is 1.0 (aka df/df)
  CHECK(assertEqualVectors(ds->bval(), {
                                           {1},
                                       }));
  // Derivative of its args
  CHECK(assertEqualVectors(dy->bval(), {
                                           {2, -4},
                                       }));

  dy->apply_bval(0.1);
  CHECK(assertEqualVectors(dy->fval(), {
                                           {0.8, 2.4},
                                       }));

  // Calc loss again
  ds->calc_fval();
  CHECK(assertEqualVectors(ds->fval(), {
                                           {3.2},
                                       }));
}

TEST_CASE(sigmoid_with_grads) {
  Mod3l m;

  Block *x = Data(&m, 1, 2);
  m.set_data(x, {{0.1, -0.2}});

  Block *w = Data(&m, 2, 3);
  m.set_data(w, {{-0.1, 0.5, 0.3}, {-0.6, 0.7, 0.8}});

  Block *mm = MatMul(x, w);

  Block *sb = Sigmoid(mm);

  sb->calc_fval();

  CHECK(assertEqualVectors(sb->fval(), {{0.527, 0.478, 0.468}}));

  mm->calc_bval();
  // TODO: add bce loss and check
  // see test_bce_loss in python tests
  CHECK(assertEqualVectors(mm->bval(), {{0.2492, 0.2495, 0.2489}}));
}

TEST_CASE(sigmoid_with_gradas) {
  Mod3l m;

  Block *x = Data(&m, 1, 2);
  m.set_data(x, {{0.1, -0.2}});

  Block *w = Data(&m, 2, 3);
  m.set_data(w, {{-0.1, 0.5, 0.3}, {-0.6, 0.7, 0.8}});

  Block *mm = MatMul(x, w);
  Block *sb = Sigmoid(mm);

  sb->calc_fval();
  CHECK(assertEqualVectors(sb->fval(), {{0.527, 0.478, 0.468}}));

  mm->calc_bval();
  // TODO: add bce loss and check
  // see test_bce_loss in python tests
  CHECK(assertEqualVectors(mm->bval(), {{0.2492, 0.2495, 0.2489}}));
}

// see test_bce_loss in python tests
TEST_CASE(bce_with_grads) {
  Mod3l m;
  Block *ypred = Data(&m, 1, 3);
  m.set_data(ypred, {{0.527, 0.478, 0.468}});
  Block *ytrue = Data(&m, 1, 3);
  m.set_data(ytrue, {{0, 1, 0.468}});

  Block *bce = BCE(ypred, ytrue);

  bce->calc_fval();
  CHECK(assertEqualVectors(bce->fval(), {{0.749, 0.738, 0.691}}));

  ypred->calc_bval();
  CHECK(assertEqualVectors(ypred->bval(), {{2.11416, -2.09205, 0}}));
}

// see test_bce_loss in python tests
TEST_CASE(bce_with_gradas) {
  Mod3l m;
  Block *ypred = Data(&m, 1, 3);
  m.set_data(ypred, {{0.527, 0.478, 0.468}});
  Block *ytrue = Data(&m, 1, 3);
  m.set_data(ytrue, {{0, 1, 0.468}});

  Block *bce = BCE(ypred, ytrue);

  bce->calc_fval();
  CHECK(assertEqualVectors(bce->fval(), {{0.749, 0.738, 0.691}}));

  ypred->calc_bval();
  CHECK(assertEqualVectors(ypred->bval(), {{2.11416, -2.09205, 0}}));
}

TEST_CASE(full_layer_with_loss_with_grads) {
  Mod3l m;
  Block *x = Data(&m, 1, 2);
  m.set_data(x, {{0.1, -0.2}});

  Block *w = Data(&m, 2, 3);
  m.set_data(w, {{-0.1, 0.5, 0.3}, {-0.6, 0.7, 0.8}});

  Block *mm = MatMul(x, w);
  Block *sb = Sigmoid(mm);

  Block *y = Data(&m, 1, 3);
  m.set_data(y, {{0, 1, 0.468}});

  // loss
  Block *bce = BCE(sb, y);

  // Forward
  bce->calc_fval();
  CHECK(assertEqualVectors(sb->fval(), {{0.527, 0.478, 0.468}}));
  CHECK(assertEqualVectors(bce->fval(), {{0.75, 0.739, 0.691}}));

  // Calc diff and check the loss values
  w->calc_bval();
  x->calc_bval();

  // Derivative of loss against itself is ones
  CHECK(assertEqualVectors(bce->bval(), {{1, 1, 1}}));

  // Make sure the gradient flows backwards
  // Check sigmoid diff
  CHECK(assertEqualVectors(sb->bval(), {{2.116, -2.094, -0.002}}));

  // Check the matrix diff
  CHECK(assertEqualVectors(w->bval(), {{0.0527, -0.052, -4.543 / 100000},
                                       {-0.105, 0.104, 9.086 / 100000}}));

  // TODO: apply grads to w, calc loss value and check that it is reduced
  // see test_bce_loss inpython
  //
  // Check that w values are still the same
  CHECK(assertEqualVectors(w->fval(), {{-0.1, 0.5, 0.3}, {-0.6, 0.7, 0.8}}));

  w->apply_bval(1.0);

  // Check that w values have changed
  CHECK(assertEqualVectors(w->fval(),
                           {{-0.153, 0.552, 0.3}, {-0.495, 0.596, 0.8}}));

  // Recalculate the loss
  bce->calc_fval();
  // Assure it got smaller!
  CHECK(assertEqualVectors(sb->fval(), {{0.521, 0.484, 0.468}}));
  CHECK(assertEqualVectors(bce->fval(), {{0.736, 0.726, 0.691}}));

  // Update the inputs, and check that it also reduces the loss
  x->apply_bval(0.01);
  CHECK(assertEqualVectors(x->fval(), {{0.103, -0.193}}));

  bce->calc_fval();
  CHECK(assertEqualVectors(bce->fval(), {{0.734, 0.723, 0.691}}));
}

TEST_CASE(matrix_views) {
  Matrix A(2, 3);
  A.set_data({{1, 2, 3}, {3, 4, 5}});

  // Transposed view
  TransposedView t(A);

  CHECK(assertEqualVectors(value(t),{
    {1, 3},
    {2, 4},
    {3, 5}
  }));

  TransposedView tb(t);
  CHECK(assertEqualVectors(value(tb), value(A)));

  // Reshaped view
  ReshapedView rv(A, 1, 6);
  CHECK(assertEqualVectors(value(rv),{ {1,2,3,3,4,5} }));

  ReshapedView rv2(rv, 6, 1);
  CHECK(assertEqualVectors(value(rv2),{ {1},{2},{3},{3},{4},{5} }));

  ReshapedView rv3(t, 2, 3);
  CHECK(assertEqualVectors(value(rv3),{ {1, 3, 2},{4, 3, 5} }));


  // Sliding window view
  Matrix b(3, 4);
  b.set_data({
     {1,  2,  3,  4},
     {5,  6,  7,  8}, 
     {9, 10, 11, 12}
  });

  SlidingWindowView swv(b, 3, 3);
  // Each row represents the content of
  // sliding window 3*3 rolling over b
  // circular
  CHECK(swv.rows = b.rows * b.cols);
  CHECK(swv.cols = 3 * 3);
  CHECK(assertEqualVectors(value(swv),{ 
    { 1, 2, 3,   5, 6, 7,  9, 10, 11 },
    { 2, 3, 4,   6, 7, 8,  10, 11,12 },
    { 3, 4, 1,   7, 8, 5,  11, 12, 9 },
    { 4, 1, 2,   8, 5, 6,  12, 9, 10 },
    { 5, 6, 7,   9, 10,11,  1, 2, 3 },
    { 6, 7, 8,   10,11,12,  2, 3, 4 },
    { 7, 8, 5,   11, 12,9,  3, 4, 1 },
    { 8, 5, 6,   12, 9,10,  4, 1, 2 },
    { 9, 10, 11,  1, 2, 3,  5, 6, 7 },
    { 10, 11,12,  2, 3, 4,  6, 7, 8 },
    { 11, 12, 9,  3, 4, 1,  7, 8, 5 },
    { 12, 9, 10,  4, 1, 2,  8, 5, 6 },
  }));
}

TEST_CASE(convolutions) {
  // Convolution op using ReshapeView and SlidingWindowView
  Matrix input(3, 4);
  input.set_data({
     {1, 2,  3,  4},
     {1, 0, -1,  0}, 
     {0, 2,  0, -2}
  });

  Matrix kernel(2,2);
  kernel.set_data({
     { 1, 0 },
     { 0, 1 },
  });


  ReshapedView kernel_flat(kernel, 4, 1);
  SlidingWindowView input_view(input, 2, 2);

  Matrix result(3, 4);
  ReshapedView result_flat(result, 12, 1);
  
  ::multiply_matrix(input_view, kernel_flat, &result_flat);

  // Result of convolution: each element is a sum of diagonal elements of input
  CHECK(assertEqualVectors(value(result), { 
    { 1, 1, 3, 5  },
    { 3, 0, -3, 0 },
    { 2, 5, 4, -1 }
  }));

  CHECK(result.at(0, 0) == input.at(0, 0) + input.at(1, 1)); // 1  = 1 + 0
  CHECK(result.at(1, 2) == input.at(1, 2) + input.at(2, 3)); // -3 = -1 + -2
  CHECK(result.at(2, 1) == input.at(2, 1) + input.at(0, 2)); // 5 = 2 + 3
  //
  // Now in model
  Mod3l m;

  Block *dinput = Data(&m, 3, 4);
  m.set_data(dinput, value(input));
  

  Block *dkernel = Data(&m, 2, 2);
  m.set_data(dkernel, value(kernel));

  Block *dc = Convolution(dinput, dkernel);
  dc->calc_fval();

  CHECK(assertEqualVectors(dc->fval(), value(result)));
}




int main(int argc, char **argv) { return run_tests(argc, argv); }

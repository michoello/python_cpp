#include "invert.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>
#include <regex>

//-------------------------------------------------------------
// Tiny unittests libraryset
//
#include "test_fw.h"

//-------------------------------------------------------------
TEST_CASE(matrix_wrong_set_data) {
  Matrix A(2, 2);
  ASSERT_THROWS(A.set_data({{1, 2}, {3, 4}, {5, 6}}), "set_data arg must have 2 rows. Provided 3 rows");
  ASSERT_THROWS(A.set_data({{1, 2}, {3}}), "all rows must have the 2 cols, provided 1 in row 1");
}


TEST_CASE(multiply) {
  Matrix A(2, 2);
  Matrix B(2, 2);
  A.set_data({{1, 2}, {3, 4}});
  B.set_data({{5, 6}, {7, 8}});

  Matrix C(2, 2);
  ::multiply_matrix(A, B, &C);

  assert(C.get(0, 0) == 19);
  assert(C.get(0, 1) == 22);
  assert(C.get(1, 0) == 43);
  assert(C.get(1, 1) == 50);
}

TEST_CASE(value_semantics) {
  Matrix A(2, 2);
  A.set_data({{1, 2}, {3, 4}});

  Matrix B(A);

  A.set(1, 1, 5);

  assert(B.rows == 2);
  assert(B.cols == 2);
  assert(B.get(0, 0) == 1);
  assert(B.get(1, 1) == 4);
}

#include "test_helpers.h"


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

  CHECK(assertEqualVectors(dc->fval(), {
                                           {38, 44, 50, 56},
                                           {83, 98, 113, 128},
                                       }));

}



TEST_CASE(matmul_grads) {
  Mod3l m;

  Block *da = Data(&m, 3, 3);
  m.set_data(da, {
     {1, 2, 3}, 
     {4, 5, 6},
     {7, 8, 9}
  });

  // Identity matrix
  Block *did = Data(&m, 3, 3);
  m.set_data(did, {
     {1, 0, 0}, 
     {0, 1, 0},
     {0, 0, 1}
  });

  Block *dc = MatMul(da, did);
  Abs(dc);  // trivial loss for grads check

  // Multiply by identity and get copy of input!
  CHECK(assertEqualVectors(da->fval(), dc->fval()));

  // Grads should all be ones for outputs
  CHECK(assertEqualVectors(dc->bval(), {
     {1, 1, 1}, 
     {1, 1, 1}, 
     {1, 1, 1}, 
  }));

  // and for inputs:
  CHECK(assertEqualVectors(da->bval(), {
     {1, 1, 1}, 
     {1, 1, 1}, 
     {1, 1, 1}, 
  }));
  //
  // Except for identity matrix, they are different
  // TODO: check if this is correct
  CHECK(assertEqualVectors(did->bval(), {
     {12, 12, 12}, 
     {15, 15, 15}, 
     {18, 18, 18}, 
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

  CHECK(assertEqualVectors(dc->fval(), {
  	{ 1, 2 },
	  { 3, 4 },
	  { 5, 6 },
	  { 7, 8 },
	  { 9, 10 },
	  {   11, 12 }
	}));
}

TEST_CASE(explode) {
  Mod3l m;

  Block *db = Data(&m, 3, 4);
  Block *de = Explode(db, 3, 3);
  Abs(de);
  m.set_data(db, {
     {1,  2,  3,  4},
     {5,  6,  7,  8}, 
     {9, 10, 11, 12}
  });

  // Each row represents the content of
  // sliding window 3*3 rolling over db
  // circularly
  CHECK(assertEqualVectors(de->fval(), {
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

  // Gradient is imploded back (1 in -> 9 out in this case)
  CHECK(assertEqualVectors(db->bval(), {
    { 9, 9, 9, 9 },
    { 9, 9, 9, 9 },
    { 9, 9, 9, 9 }
  }));
}


TEST_CASE(matmul_with_grads) {
  Mod3l m;

  Block *da = Data(&m, 1, 2);
  m.set_data(da, {{1, 2}});

  Block *db = Data(&m, 2, 3);
  m.set_data(db, {{3, 4, 5}, {6, 7, 8}});

  Block *dc = MatMul(da, db);
  Abs(dc);

  CHECK(assertEqualVectors(dc->fval(), {
                                           {15, 18, 21},
                                       }));

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

  CHECK(assertEqualVectors(ds2->fval(), {
                                            {6, 8, 10},
                                            {7, 9, 11},
                                        }));

  Block* dsig = Sigmoid(ds2);
  Block *dl = BCE(dsig, dy);

  CHECK(assertEqualVectors(dl->fval(), {
		{ 5.402, 5.600, 3 },
		{ 0.071, 4.500, 10.989 }
  }));

  // Backend gradient is always zeroes
  CHECK(assertEqualVectors(dl->bval(), {
		{ 0, 0, 0},
		{ 0, 0, 0},
  }));

  // Calc derivatives
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

  // Now check that grads are the same
  CHECK(assertEqualVectors(db->bval(), ds2->bval()));
  CHECK(assertEqualVectors(dc->bval(), ds2->bval()));
}


TEST_CASE(func_laziness) {
  Mod3l m;
  Block *da = Data(&m, 2, 3);
  Block *db = Data(&m, 2, 3);

  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});
  m.set_data(db, {{4, 5, 6}, {1, 2, 3}});

  Block *ds = Add(da, db);

  // wrap sum function into another one with call counter
  auto old_fu = ds->fowd_fun.fun;
  int counter = 0;
  ds->set_fowd_fun([&](Matrix *out) {
    counter++;
    old_fu(out);
  });

  CHECK(assertEqualVectors(ds->fval(), {
                                            {5, 7, 9},
                                            {5, 7, 9},
                                        }));
  CHECK(counter == 1);

  // get the value again
  CHECK(assertEqualVectors(ds->fval(), {
                                            {5, 7, 9},
                                            {5, 7, 9},
                                        }));
  // make sure counter has not changed
  CHECK(counter == 1);

  // set the same data, but the model is not smart enough, it resets everythign anyway
  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});

  CHECK(assertEqualVectors(ds->fval(), {
                                            {5, 7, 9},
                                            {5, 7, 9},
                                        }));
  // now the counter has been incremented
  CHECK(counter == 2);
}

TEST_CASE(dif_matrix) {

  Mod3l m;
  Block *da = Data(&m, 2, 3);
  Block *db = Data(&m, 2, 3);

  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});
  m.set_data(db, {{2, 3, 5}, {8, 13, 21}});

  Block *dd = Dif(db, da); // db - da

  CHECK(assertEqualVectors(dd->fval(), {
                                           {1, 1, 2},
                                           {4, 8, 15},
                                       }));
}

TEST_CASE(mul_el) {
  Mod3l m;
  Block *da = Data(&m, 2, 3);
  Block *dn = Data(&m, 1, 1);
  Block *db = MulEl2(da, dn);

  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});

  m.set_data(dn, {{2}});
  CHECK(assertEqualVectors(db->fval(), {
                                           {2, 4, 6},
                                           {8, 10, 12},
                                       }));

  m.set_data(dn, {{-2}});
  CHECK(assertEqualVectors(db->fval(), {
                                           {-2, -4, -6},
                                           {-8, -10, -12},
                                       }));
}

TEST_CASE(sum_mat) {
  Mod3l m;
  Block *da = Data(&m, 2, 3);
  m.set_data(da, {{1, 2, 3}, {4, 5, 6}});

  Block *ds = Sum(da);

  CHECK(assertEqualVectors(ds->fval(), {
                                           {21},
                                       }));
  // Grads of const are zeroes                                     
  CHECK(assertEqualVectors(da->bval(), {
                                           {0, 0, 0}, 
                                           {0, 0, 0}, 
                                       }));

  // Adding abs loss function to get non-empty grads
  Block *dabs = Abs(ds); 
  CHECK(assertEqualVectors(dabs->fval(), {
                                           {21},
                                       }));


  // Now grads are propagated back from Abs to const input values
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

  CHECK(assertEqualVectors(ds->fval(), {
                                           {5},
                                       }));

  // ---
  Block *da1 = Data(&m, 1, 1);
  m.set_data(da1, {{0}});

  Block *db1 = Data(&m, 1, 1);
  m.set_data(db1, {{3}});

  Block *ds1 = SSE(da1, db1);
  Abs(ds1);

  CHECK(assertEqualVectors(ds1->fval(), { {9}, }));

  CHECK(assertEqualVectors(ds1->bval(), { {1}, }));
  CHECK(assertEqualVectors(da1->bval(), { {-6}, }));
  CHECK(assertEqualVectors(db1->bval(), { {6}, }));

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

  Abs(ds);

  CHECK(assertEqualVectors(ds->fval(), {{5}}));

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
  Abs(sb);

  CHECK(assertEqualVectors(sb->fval(), {{0.527, 0.478, 0.468}}));

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

  Abs(sb);

  CHECK(assertEqualVectors(sb->fval(), {{0.527, 0.478, 0.468}}));

  // TODO: add bce loss and check
  // see test_bce_loss in python tests
  CHECK(assertEqualVectors(mm->bval(), {{0.2492, 0.2495, 0.2489}}));
}

TEST_CASE(bce_values) {
  Mod3l m;
  Block *ypred = Data(&m, 1, 1);
  Block *ytrue = Data(&m, 1, 1);
  Block *bce = BCE(ypred, ytrue);
  

  //auto check_values = [&m, &ypred, &ytrue, &bce](double yp, double yt, double expected) {
  auto check_values = [&](double yp, double yt, double expected) {
    m.set_data(ypred, {{yp}});
    m.set_data(ytrue, {{yt}});
    CHECK(assertEqualVectors(bce->fval(), {{expected}}));
  };

  check_values(1.0, 1.0, 0);       // complete certainty
  check_values(0.9, 1.0, 0.105);
  check_values(0.5, 1.0, 0.693);   // full uncertainty
  check_values(0.0, 1.0, 27.631);  // epsiloned kinda infinity

  check_values(0.0, 0.0, 0);       // complete certainty
  check_values(0.5, 0.0, 0.693);   // full uncertainty
  check_values(0.9, 0.0, 2.303);
  check_values(1.0, 0.0, 27.631);  // epsiloned kinda infinity

  check_values(0.0, 0.5, 13.816);  // half infinity
  check_values(0.5, 0.5, 0.693);   // full uncertainty
  check_values(0.6, 0.5, 0.714);
  check_values(0.9, 0.5, 1.204);
  check_values(1.0, 0.5, 13.816);  // another half
}



// see test_bce_loss in python tests
TEST_CASE(bce_with_grads) {
  Mod3l m;
  Block *ypred = Data(&m, 1, 3);
  m.set_data(ypred, {{0.527, 0.478, 0.468}});
  Block *ytrue = Data(&m, 1, 3);
  m.set_data(ytrue, {{0, 1, 0.468}});

  Block *bce = BCE(ypred, ytrue);

  CHECK(assertEqualVectors(bce->fval(), {{0.749, 0.738, 0.691}}));
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

  CHECK(assertEqualVectors(bce->fval(), {{0.749, 0.738, 0.691}}));
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
  CHECK(assertEqualVectors(sb->fval(), {{0.527, 0.478, 0.468}}));
  CHECK(assertEqualVectors(bce->fval(), {{0.75, 0.739, 0.691}}));

  Abs(bce);  // for grads

  // Calc diff and check the loss values
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
  // Assure it got smaller!
  CHECK(assertEqualVectors(sb->fval(), {{0.521, 0.484, 0.468}}));
  CHECK(assertEqualVectors(bce->fval(), {{0.736, 0.726, 0.691}}));

  // Update the inputs, and check that it also reduces the loss
  x->apply_bval(0.01);
  CHECK(assertEqualVectors(x->fval(), {{0.104, -0.194}}));

  CHECK(assertEqualVectors(bce->fval(), {{0.734, 0.723, 0.691}}));
}

TEST_CASE(grad_fork) {
  Mod3l m;
  Block *da = Data(&m, 1, 3);
  Block *db = Data(&m, 1, 3);
  Block *dc = Data(&m, 1, 3);

  m.set_data(da, {{1, 1, 1}});
  m.set_data(db, {{2, 2, 2}});
  m.set_data(dc, {{3, 3, 3}});

  Block *ds1 = Add(da, db); // 1 + 2 = 3
  Block *ds2 = Add(dc, db); // 3 + 2 = 5

  Block *ds = Add(ds1, ds2);  // 3 + 5 = 8 
  
  Abs(ds);  // to enable grads

  CHECK(assertEqualVectors(ds->fval(), { {8, 8, 8} }));

  CHECK(assertEqualVectors(ds->bval(), { {1, 1, 1} }));
  CHECK(assertEqualVectors(ds1->bval(), { {1, 1, 1} }));
  CHECK(assertEqualVectors(da->bval(), { {1, 1, 1} }));
  CHECK(assertEqualVectors(db->bval(), { {2, 2, 2} }));  // two gradients meet each other here
  CHECK(assertEqualVectors(dc->bval(), { {1, 1, 1} }));

  da->apply_bval(0.1);
  db->apply_bval(0.1);
  dc->apply_bval(0.1);
  CHECK(assertEqualVectors(da->fval(), { {0.9, 0.9, 0.9} })); // 1 - 1 * 0.1 (one grad)
  CHECK(assertEqualVectors(db->fval(), { {1.8, 1.8, 1.8} })); // 2 - (1 + 1) * 0.1 (two grads)
  CHECK(assertEqualVectors(dc->fval(), { {2.9, 2.9, 2.9} })); // 3 - 1 * 0.1
  CHECK(assertEqualVectors(ds->fval(), { {7.4, 7.4, 7.4} })); // sum of all of them: 3.6 + 0.9 + 2.9 = 7.4
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

 for(int i = 0; i < 10000; i++) {
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
  CHECK(swv.rows == b.rows * b.cols);
  CHECK(swv.cols == 3 * 3);
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
}


TEST_CASE(sliding_window_set) {
  // Sliding window view
  Matrix b(2, 2);
  b.set_data({
     {0, 0},
     {0, 0},
  });

  SlidingWindowView swv(b, 2, 2);
  // Each row represents the content of
  // sliding window 3*3 rolling over b
  // circular
  CHECK(swv.rows == b.rows * b.cols);
  CHECK(swv.cols == 2 * 2);
  CHECK(assertEqualVectors(value(swv),{ 
    { 0, 0, 0, 0},
    { 0, 0, 0, 0},
    { 0, 0, 0, 0},
    { 0, 0, 0, 0},
  }));

  swv.set(0, 0, 1);
  CHECK(assertEqualVectors(value(swv),{ 
    { 1, 0, 0, 0 },
    { 0, 1, 0, 0 },
    { 0, 0, 1, 0 },
    { 0, 0, 0, 1 },
  }));

  swv.set(0, 0, 2);  // adding 2
  CHECK(assertEqualVectors(value(swv),{ 
    { 3, 0, 0, 0 },
    { 0, 3, 0, 0 },
    { 0, 0, 3, 0 },
    { 0, 0, 0, 3 },
  }));

  CHECK(assertEqualVectors(value(b),{ 
    { 3, 0 },
    { 0, 0 },
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

  CHECK(result.get(0, 0) == input.get(0, 0) + input.get(1, 1)); // 1  = 1 + 0
  CHECK(result.get(1, 2) == input.get(1, 2) + input.get(2, 3)); // -3 = -1 + -2
  CHECK(result.get(2, 1) == input.get(2, 1) + input.get(0, 2)); // 5 = 2 + 3
  //
  // Now in model
  Mod3l m;

  Block *dinput = Data(&m, 3, 4);
  m.set_data(dinput, value(input));
  

  Block *dkernel = Data(&m, 2, 2);
  m.set_data(dkernel, value(kernel));

  Block *dc = Convo(dinput, dkernel);

  CHECK(assertEqualVectors(dc->fval(), value(result)));
}


TEST_CASE(convolutions_grads) {
  Mod3l m;

  Block *dinput = Data(&m, 3, 3);
  m.set_data(dinput, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  Block *dkernel = Data(&m, 2, 2);
  m.set_data(dkernel, {
     { 1, 0 },
     { 0, 0 },
  });

  Block *dc = Convo(dinput, dkernel);

  // This was identity convolution
  CHECK(assertEqualVectors(dc->fval(), dinput->fval()));

  // Now "rotate to left" convolution kernel
  m.set_data(dkernel, {
     { 0, 1 },
     { 0, 0 },
  });

  CHECK(assertEqualVectors(dc->fval(), {
     {2, 3, 1},
     {5, 6, 4}, 
     {8, 9, 7}
  }));

  Abs(dc); // for non-empty grads

  // Check grads
  // The output grads are ones:
  CHECK(assertEqualVectors(dc->bval(), {
     {1, 1, 1},
     {1, 1, 1},
     {1, 1, 1},
  }));


  // Is this right? 
  // Seems reasonable (45 is sum of all input elements), and each contributes to each
  // kernel cell but TODO: double check
  CHECK(assertEqualVectors(dkernel->bval(), {
     {45, 45},
     {45, 45},
  }));

  // The grads are passed to the input as is, since it is identity
  // TODO: check if that is correct
  CHECK(assertEqualVectors(dinput->bval(), {
     {1, 1, 1}, 
     {1, 1, 1},
     {1, 1, 1},
     // Should not they be equal to input instead?
     //{1, 2, 3},
     //{4, 5, 6}, 
     //{7, 8, 9}
  }));
}

TEST_CASE(convolutions_grads_apply) {
  Mod3l m;

  Block *dinput = Data(&m, 3, 3);
  m.set_data(dinput, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  Block *dkernel = Data(&m, 2, 2);
  m.set_data(dkernel, {
     { -0.2, 1.1 },
     { 2.3, 0.4 },
  });

  Block *dc = Convo(dinput, dkernel);

  CHECK(assertEqualVectors(dc->fval(), {
   { 13.200, 16.800, 15.900 },
   { 24.000, 27.600, 26.700 },
   { 10.500, 14.100, 13.200 }
  }));

  // Labels are the same as input, we want to check if kernel will get closer
  Block *dlabels = Data(&m, 3, 3);
  m.set_data(dlabels, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  // Let's have this crazy hand-crafted loss function:
  Block* dif = Dif(dc, dlabels);
  CHECK(assertEqualVectors(dif->fval(), {
    { 12.200, 14.800, 12.900 },
    { 20.000, 22.600, 20.700 },
    { 3.500, 6.100, 4.200 }
  }));

  Block* sq = Sqrt(dif);

  Block* sum = Sum(sq);


  Block* loss = Abs(sum);
  CHECK(assertEqualVectors(loss->fval(), {
    { 1940.64 },
  }));

  CHECK(assertEqualVectors(sum->bval(), {
    { 1.00 },
  }));

  CHECK(assertEqualVectors(dif->bval(), {
    { 24.400, 29.600, 25.800 },
    { 40.000, 45.200, 41.400 },
    { 7.000, 12.200, 8.400 }
  }));

  CHECK(assertEqualVectors(dkernel->bval(), {
		{ 1017.600, 1024.800 },
		{ 1471.200, 1478.400 }
  }));
  dkernel->apply_bval(0.0001);

  // See if effect is a least abit towards 
  CHECK(assertEqualVectors(loss->fval(), { {1357.199} }));
  CHECK(assertEqualVectors(dc->fval(), {
    { 11.566, 14.666, 14.018 },
    { 20.868, 23.969, 23.321 },
    { 8.525, 11.626, 10.978 }
    // Was:
    // { 13.200, 16.800, 15.900 },
    // { 24.000, 27.600, 26.700 },
    // { 10.500, 14.100, 13.200 }
  }));

  for(int i = 0; i < 200; ++i) {
     dkernel->apply_bval(0.001);
	}

  CHECK(assertEqualVectors(loss->fval(), { {0.0} }));

  // The gradient descent discovered another "identity kernel" for 
  // this particular input  convolution:
  CHECK(assertEqualVectors(dkernel->fval(), {
    { -0.050, 1.050 },
    { 1.050, -1.050 }
  }));
  // Now result of convo is exactly the input:
  CHECK(assertEqualVectors(dc->fval(), {
		{ 1.000, 2.000, 3.000 },
		{ 4.000, 5.000, 6.000 },
		{ 7.000, 8.000, 9.000 }
  }));
}

// New version of convolutions! 
// Will work with multi-convolutions as well
TEST_CASE(convolutions2_grads_apply) {
  Mod3l m;

  Block *dinput = Data(&m, 3, 3);
  m.set_data(dinput, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  Block *dkernel = Data(&m, 2, 2);
  m.set_data(dkernel, {
     { -0.2, 1.1 },
     { 2.3, 0.4 },
  });

  Block *dc = Convo2(dinput, dkernel);

  CHECK(assertEqualVectors(dc->fval(), {
   { 13.200, 16.800, 15.900 },
   { 24.000, 27.600, 26.700 },
   { 10.500, 14.100, 13.200 }
  }));

  // Labels are the same as input, we want to check if kernel will get closer
  Block *dlabels = Data(&m, 3, 3);
  m.set_data(dlabels, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  // Let's have this crazy hand-crafted loss function:
  Block* dif = Dif(dc, dlabels);
  CHECK(assertEqualVectors(dif->fval(), {
    { 12.200, 14.800, 12.900 },
    { 20.000, 22.600, 20.700 },
    { 3.500, 6.100, 4.200 }
  }));

  Block* sq = Sqrt(dif);
  Block* sum = Sum(sq);
  Block* loss = Abs(sum);
  CHECK(assertEqualVectors(loss->fval(), {
    { 1940.64 },
  }));

  CHECK(assertEqualVectors(sum->bval(), {
    { 1.00 },
  }));

  CHECK(assertEqualVectors(dif->bval(), {
    { 24.400, 29.600, 25.800 },
    { 40.000, 45.200, 41.400 },
    { 7.000, 12.200, 8.400 }
  }));

  CHECK(assertEqualVectors(dkernel->bval(), {
		{ 1017.600, 1024.800 },
		{ 1471.200, 1478.400 }
  }));
  dkernel->apply_bval(0.0001);

  // See if effect is a least abit towards 
  CHECK(assertEqualVectors(loss->fval(), { {1357.199} }));
  CHECK(assertEqualVectors(dc->fval(), {
    { 11.566, 14.666, 14.018 },
    { 20.868, 23.969, 23.321 },
    { 8.525, 11.626, 10.978 }
    // Was:
    // { 13.200, 16.800, 15.900 },
    // { 24.000, 27.600, 26.700 },
    // { 10.500, 14.100, 13.200 }
  }));

  for(int i = 0; i < 200; ++i) {
     dkernel->apply_bval(0.001);
	}

  CHECK(assertEqualVectors(loss->fval(), { {0.0} }));

  // The gradient descent discovered another "identity kernel" for 
  // this particular input  convolution:
  CHECK(assertEqualVectors(dkernel->fval(), {
    { -0.050, 1.050 },
    { 1.050, -1.050 }
  }));
  // Now result of convo is exactly the input:
  CHECK(assertEqualVectors(dc->fval(), {
		{ 1.000, 2.000, 3.000 },
		{ 4.000, 5.000, 6.000 },
		{ 7.000, 8.000, 9.000 }
  }));
}


TEST_CASE(convolutions_grads_propagate) {
  Mod3l m;

  Block *dinput = Data(&m, 3, 3);
  m.set_data(dinput, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  Block *dkernel = Data(&m, 2, 2);
  m.set_data(dkernel, {
     { -0.2, 1.1 },
     { 2.3, 0.4 },
  });

  Block *dc = Convo(dinput, dkernel);

  CHECK(assertEqualVectors(dc->fval(), {
   { 13.200, 16.800, 15.900 },
   { 24.000, 27.600, 26.700 },
   { 10.500, 14.100, 13.200 }
  }));

  // Labels are the same as input, we want to check if kernel will get closer
  Block *dlabels = Data(&m, 3, 3);
  m.set_data(dlabels, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  // Let's have this crazy hand-crafted loss function:
  Block* loss = Abs(Sum(Sqrt(Dif(dc, dlabels))));
  CHECK(assertEqualVectors(loss->fval(), {
    { 1940.64 },
  }));

  CHECK(assertEqualVectors(dc->fval(), {
    { 13.200, 16.800, 15.900 },
    { 24.000, 27.600, 26.700 },
    { 10.500, 14.100, 13.200 }
  }));

  // This time the kernel is fixed, but we apply gradients to the input
  dinput->apply_bval(0.001);

  // See if effect is a least abit towards 
  CHECK(assertEqualVectors(loss->fval(), { {1853.443} }));
  CHECK(assertEqualVectors(dc->fval(), {
    { 12.867, 16.449, 15.562 },
    { 23.579, 27.161, 26.274 },
    { 10.266, 13.848, 12.961 }
    // Was:
    // { 13.200, 16.800, 15.900 },
    // { 24.000, 27.600, 26.700 },
    // { 10.500, 14.100, 13.200 }
  }));

  // It takes quite a bit to squeeze the input while fixing the kernel
  for(int i = 0; i < 1100; ++i) {
     dinput->apply_bval(0.001);
	}
  CHECK(assertEqualVectors(loss->fval(), { {0.0} }));

  // Kernel is not changed:
  CHECK(assertEqualVectors(dkernel->fval(), {
     { -0.2, 1.1 },
     { 2.3, 0.4 },
  }));

  // But result of convo is exactly the labels:
  CHECK(assertEqualVectors(dc->fval(), {
		{ 1.000, 2.000, 3.000 },
		{ 4.000, 5.000, 6.000 },
		{ 7.000, 8.000, 9.000 }
  }));

  // But the input is barely recognizeable
  CHECK(assertEqualVectors(dinput->fval(), {
    { 2.170, 1.914, 2.940 },
    { -0.211, -0.467, 0.559 },
    { 1.694, 1.438, 2.464 }
  }));


}

TEST_CASE(convolutions2_grads_propagate) {
  Mod3l m;

  Block *dinput = Data(&m, 3, 3);
  m.set_data(dinput, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  Block *dkernel = Data(&m, 2, 2);
  m.set_data(dkernel, {
     { -0.2, 1.1 },
     { 2.3, 0.4 },
  });

  Block *dc = Convo2(dinput, dkernel);

  CHECK(assertEqualVectors(dc->fval(), {
   { 13.200, 16.800, 15.900 },
   { 24.000, 27.600, 26.700 },
   { 10.500, 14.100, 13.200 }
  }));

  // Labels are the same as input, we want to check if kernel will get closer
  Block *dlabels = Data(&m, 3, 3);
  m.set_data(dlabels, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  // Let's have this crazy hand-crafted loss function:
  Block* loss = Abs(Sum(Sqrt(Dif(dc, dlabels))));
  CHECK(assertEqualVectors(loss->fval(), {
    { 1940.64 },
  }));

  CHECK(assertEqualVectors(dc->fval(), {
    { 13.200, 16.800, 15.900 },
    { 24.000, 27.600, 26.700 },
    { 10.500, 14.100, 13.200 }
  }));

  // Grads are high!
  CHECK(assertEqualVectors(dc->bval(), {
    { 24.400, 29.600, 25.800 },
    { 40.000, 45.200, 41.400 },
    { 7.000, 12.200, 8.400 }
  }));


  CHECK(assertEqualVectors(dkernel->bval(), {
    { 1017.600, 1024.800 },
    { 1471.200, 1478.400 }
  }));

  CHECK(assertEqualVectors(dinput->bval(), {
    { 42.960, 51.780, 51.600 },
    { 103.980, 112.800, 112.620 },
    { 116.400, 125.220, 125.040 }
  }));



  // This time the kernel is fixed, but we apply gradients to the input
  dinput->apply_bval(0.001);

  // See if effect is a least abit towards 
  CHECK(assertEqualVectors(loss->fval(), { {1853.443} }));
  CHECK(assertEqualVectors(dc->fval(), {
    { 12.867, 16.449, 15.562 },
    { 23.579, 27.161, 26.274 },
    { 10.266, 13.848, 12.961 }
    // Was:
    // { 13.200, 16.800, 15.900 },
    // { 24.000, 27.600, 26.700 },
    // { 10.500, 14.100, 13.200 }
  }));

  // It takes quite a bit to squeeze the input while fixing the kernel
  for(int i = 0; i < 1100; ++i) {
     dinput->apply_bval(0.001);
	}
  CHECK(assertEqualVectors(loss->fval(), { {0.0} }));

  // Kernel is not changed:
  CHECK(assertEqualVectors(dkernel->fval(), {
     { -0.2, 1.1 },
     { 2.3, 0.4 },
  }));

  // But result of convo is exactly the labels:
  CHECK(assertEqualVectors(dc->fval(), {
		{ 1.000, 2.000, 3.000 },
		{ 4.000, 5.000, 6.000 },
		{ 7.000, 8.000, 9.000 }
  }));

  // But the input is barely recognizeable
  CHECK(assertEqualVectors(dinput->fval(), {
    { 2.170, 1.914, 2.940 },
    { -0.211, -0.467, 0.559 },
    { 1.694, 1.438, 2.464 }
  }));
}


TEST_CASE(convolutions2_grads_clipping) {
  Mod3l m;

  Block *dinput = Data(&m, 3, 3);
  m.set_data(dinput, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  Block *dkernel = Data(&m, 2, 2);
  m.set_data(dkernel, {
     { -0.2, 1.1 },
     { 2.3, 0.4 },
  });

  Block *dc = Convo2(GradClipper(dinput, 1.0), GradClipper(dkernel, 1.0));
  Block *dclipper = GradClipper(dc, 1.0);

  CHECK(assertEqualVectors(dc->fval(), {
   { 13.200, 16.800, 15.900 },
   { 24.000, 27.600, 26.700 },
   { 10.500, 14.100, 13.200 }
  }));

  CHECK(assertEqualVectors(dclipper->fval(), {
   { 13.200, 16.800, 15.900 },
   { 24.000, 27.600, 26.700 },
   { 10.500, 14.100, 13.200 }
  }));

  // Labels are the same as input, we want to check if kernel will get closer
  Block *dlabels = Data(&m, 3, 3);
  m.set_data(dlabels, {
     {1, 2, 3},
     {4, 5, 6}, 
     {7, 8, 9}
  });

  // Let's have this crazy hand-crafted loss function:
  Block* loss = Abs(Sum(Sqrt(Dif(dclipper, dlabels))));
  CHECK(assertEqualVectors(loss->fval(), { { 1940.64 } }));

  // Unclipped grads are high as in the testcase convolutions2_grads_propagate.
  CHECK(assertEqualVectors(dclipper->bval(), {
    { 24.400, 29.600, 25.800 },
    { 40.000, 45.200, 41.400 },
    { 7.000, 12.200, 8.400 }
  }));

  CHECK(approxEqual(m.global_grad_norm({dclipper}), 88.105));

  // Grads are much much lower than in the testcase convolutions2_grads_propagate.
  CHECK(assertEqualVectors(dc->bval(), {
    { 0.277, 0.336, 0.293 },
    { 0.454, 0.513, 0.470 },
    { 0.079, 0.138, 0.095 }
  }));
  CHECK(approxEqual(m.global_grad_norm({dc}), 1.0));

  CHECK(assertEqualVectors(dkernel->bval(), {
    { 0.401, 0.404 },
    { 0.580, 0.583 }
    // With no clipping:
    //{ 1017.600, 1024.800 },
    //{ 1471.200, 1478.400 }
  }));

  CHECK(assertEqualVectors(dinput->bval(), {
    { 0.145, 0.174, 0.174 },
    { 0.350, 0.380, 0.379 },
    { 0.392, 0.421, 0.421 }
  }));

  // This time the kernel is fixed, but we apply gradients to the input
  dinput->apply_bval(0.01);

  // See if effect is a least abit towards, but much more smoothan without clipping
  CHECK(assertEqualVectors(loss->fval(), { {1937.67} }));
  CHECK(assertEqualVectors(dc->fval(), {
    { 13.189, 16.788, 15.889 },
    { 23.986, 27.585, 26.686 },
    { 10.492, 14.092, 13.192 }
    // Was:
    // { 13.200, 16.800, 15.900 },
    // { 24.000, 27.600, 26.700 },
    // { 10.500, 14.100, 13.200 }
  }));

  // It takes a little longer to squeeze the input while fixing the kernel
  // but we can safely operate at higher learning rate than without clipping
  for(int i = 0; i < 1600; ++i) {
     dinput->apply_bval(0.01);
	}
  CHECK(assertEqualVectors(loss->fval(), { {0.0} }));

  // Kernel is not changed:
  CHECK(assertEqualVectors(dkernel->fval(), {
     { -0.2, 1.1 },
     { 2.3, 0.4 },
  }));

  // But result of convo is exactly the labels:
  CHECK(assertEqualVectors(dc->fval(), {
		{ 1.000, 2.000, 3.000 },
		{ 4.000, 5.000, 6.000 },
		{ 7.000, 8.000, 9.000 }
  }));

  // But the input is barely recognizeable
  CHECK(assertEqualVectors(dinput->fval(), {
    { 2.170, 1.914, 2.940 },
    { -0.211, -0.467, 0.559 },
    { 1.694, 1.438, 2.464 }
  }));
}

TEST_CASE(per_element_block_gradients_smoketest) {
  Mod3l m;

  Block *dother = Data(&m, 3, 3);
	m.set_data(dother, {
       {-0.5, 0.25, 0.125},
       {1.1, 1.2, 1.3},
       {0, 17, -3.1415},
	});

  std::vector<std::function<Block* (Block*)>> block_makers = {
     &ReLU,
     &Tanh,
     &Sigmoid,
     &Sqrt,
     [](Block* in) { return MulEl(in, 1.618); },

     // Testing Matrix addition block
     [dother](Block* in) { return Add(in, dother); },
     [dother](Block* in) { return Add(dother, in); },

     // Testing Matrix substraction block
     [dother](Block* in) { return Dif(in, dother); },
     [dother](Block* in) { return Dif(dother, in); },

     // Testing Matrix multiplication block
     [dother](Block* in) { return MatMul(in, dother); },
     [dother](Block* in) { return MatMul(dother, in); },
     
     // Reshaper
     [](Block* in) { return Reshape(in, 9, 1); },
     //
     // This one will fail, as it disconnects from the input block
     // TODO: add it to the test to check the sanity?
     // [dother](Block* in) { return dother; },
  };

  for(const auto& block_maker: block_makers) { 
		Block *dinput = Data(&m, 3, 3);
		m.set_data(dinput, {
			 {1, 2, 3},
			 {4, -5, 6}, 
			 {7, 8, 9}
		});
		Block *dc = block_maker(dinput);

		Block* loss = Abs(Sum(dc));

		double before = loss->fval().get(0, 0);
		dinput->apply_bval(0.01);
		double after = loss->fval().get(0, 0);

		CHECK(before > after);
	}
}

// This is a prototype of smoke test for a more realistic complex model
// to be used for game training. As of now, it is simpler, toy version,
// though still including all major blocks.
TEST_CASE(larger_model) {
  Mod3l m;

  Block *dinput = Data(&m, 3, 3);
	m.set_data(dinput, {
       { 1, 0, -1},
       { 1, 0, -1},
       { 0, 1, -1},
	});

  Block *dkernel1 = Data(&m, 2, 2);
  m.set_data(dkernel1, {
     { 0.3, 0.1 },
     { 0.2, 0.0 },
  });
  Block *dc1 = Convo(dinput, dkernel1);
  Block *rl1 = ReLU(dc1);


  Block *dkernel2 = Data(&m, 2, 2);
  m.set_data(dkernel2, {
     { -0.3, 0.1 },
     { -0.2, 0.4 },
  });
  Block *dc2 = Convo(dinput, dkernel2);
  Block *rl2 = ReLU(dc2);

  Block *rl = Add(rl1, rl2);
  
  Block *dw = Data(&m, 3, 3);
  m.set_data(dw, {{1, 2, 3}, {5, 6, 7}, {9, 10, 11}});

  Block *dlogits = MatMul(rl, dw);
  Block *dsoftmax = SoftMax(dlogits);
  // !!!

  // This is our toy policy network head
  Block *dlabels = Data(&m, 3, 3);
  m.set_data(dlabels, {
    {0, 1, 0},
    {0, 0, 0},
    {0, 0, 0},
  });
  Block *policy_loss = SoftMaxCrossEntropy(dlogits, dsoftmax, dlabels);

  //
  Block *dw2 = Data(&m, 3, 1);
  m.set_data(dw2, {{1.5}, {2.5}, {3.5}});

  Block *dvalue = Tanh(MatMul(rl, dw2));
  
  Block *dlabel = Data(&m, 1, 1);
  m.set_data(dlabel, {{-1}});

  Block *dvalue_loss = SSE(dvalue, dlabel);

  CHECK(assertEqualVectors(dvalue_loss->fval(), { {3.998} }));
  CHECK(assertEqualVectors(policy_loss->fval(), { {2.302} }));

  for(size_t i = 0; i < 10; ++i) {
  	double value_before = dvalue_loss->fval().get(0, 0);
  	double policy_before = policy_loss->fval().get(0, 0);
  	dkernel1->apply_bval(0.01);
  	dkernel2->apply_bval(0.01);
  	dw->apply_bval(0.01);
  	dw2->apply_bval(0.01);
  	double value_after = dvalue_loss->fval().get(0, 0);
  	double policy_after = policy_loss->fval().get(0, 0);
  
    CHECK(value_before > value_after);
    CHECK(policy_before > policy_after);
  }

  CHECK(assertEqualVectors(dvalue_loss->fval(), { {3.995} }));
  CHECK(assertEqualVectors(policy_loss->fval(), { {1.593} }));

}


TEST_CASE(softmax) {
  Mod3l m;
  Block *da = Data(&m, 2, 2);
  Block *ds = SoftMax(da);

  // all equal
  m.set_data(da, {{1, 1}, {1, 1}});
  CHECK(assertEqualVectors(ds->fval(), {
                                           {0.25, 0.25},
                                           {0.25, 0.25},
                                       }));
  // one stands out
  m.set_data(da, {{0, 0}, {0, 8}});
  CHECK(assertEqualVectors(ds->fval(), {
                                           {0.0, 0.0},
                                           {0.0, 0.999},
                                       }));
  // same ordering
  m.set_data(da, {{1, 2}, {3, 4}});
  CHECK(assertEqualVectors(ds->fval(), {
                                           {0.032, 0.087},
                                           {0.237, 0.644},
                                       }));
}

TEST_CASE(softmax_cross_entropy) {
  Mod3l m;
  Block *dlogits = Data(&m, 2, 2);
  Block *dsoftmax = SoftMax(dlogits);

  // all equal
  m.set_data(dlogits, {{3, 3}, {3, 3}});
  CHECK(assertEqualVectors(dsoftmax->fval(), {
                                           {0.25, 0.25},
                                           {0.25, 0.25},
                                       }));

  Block *dlabels = Data(&m, 2, 2);
  m.set_data(dlabels, {{0, 1}, {0, 0}});

  Block *cre_logits = SoftMaxCrossEntropy(dlogits, dsoftmax, dlabels);
  CHECK(assertEqualVectors(cre_logits->fval(), { {1.386} }));


  
  // gradient of logits is difference between their softmaxed values 
  // and corresponding labels
  CHECK(assertEqualVectors(dlogits->bval(), {
                                           {0.25, -0.75},
                                           {0.25, 0.25},
                                       }));

  dlogits->apply_bval(0.1);
  // Apply grads and check that loss dropped a bit
  CHECK(assertEqualVectors(cre_logits->fval(), { {1.312} }));

  // And softmax values are getting a bit closer to labels
  CHECK(assertEqualVectors(dsoftmax->fval(), {
																					 { 0.244, 0.269 },
																					 { 0.244, 0.244 }
                                       }));

  // Let's run 99 more iterations, to ensure it still makes it better
  for(size_t iter = 1; iter < 100; iter++) {
    dlogits->apply_bval(0.1);
  }
  CHECK(assertEqualVectors(cre_logits->fval(), { {0.092} }));

  CHECK(assertEqualVectors(dsoftmax->fval(), {
																					 { 0.029, 0.912 },
																					 { 0.029, 0.029 },
                                       }));
}

int main(int argc, char **argv) { return run_tests(argc, argv); }

/*
 *  LLM stands for Little Lazy Matrix
 *
 */

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

std::vector<int> invert(const std::vector<int> &v);

class Matrix;
// class Funcs;

struct Matrix {
  int rows;
  int cols;
  std::shared_ptr<std::vector<double>> data; // flat row-major storage

  Matrix(int r, int c, double val = 0.0)
      : rows(r), cols(c),
        data(std::make_shared<std::vector<double>>(r * c, val)) {}

  Matrix(const Matrix &other) = default;

  void set_data(const std::vector<std::vector<double>> &vals) {
    for (size_t r = 0; r < vals.size(); ++r) {
      if ((int)vals[r].size() != cols)
        throw std::invalid_argument(
            "All rows must have the same number of columns");
      for (int c = 0; c < cols; ++c) {
        at(r, c) = vals[r][c];
      }
    }
  }

  inline double &at(int r, int c) { return (*data)[r * cols + c]; }
  inline const double &at(int r, int c) const { return (*data)[r * cols + c]; }

  // Convert bawd_fun to nested vector (Python list-of-lists)
};

// Extracts the full value of Matrix-like object(i.e. Matrix or matrix view)
template <class M> std::vector<std::vector<double>> value(const M &m) {
  std::vector<std::vector<double>> out(m.rows, std::vector<double>(m.cols));
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      out[i][j] = m.at(i, j);
    }
  }
  return out;
}

template <class T, class U, class V>
void multiply_matrix(const T &a, const U &b, V *c) {
  // TODO: remove it out of here?
  if (a.cols != b.rows || a.rows != c->rows || b.cols != c->cols) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for multiplication");
  }

  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < b.cols; j++) {
      double sum = 0.0;
      for (int k = 0; k < a.cols; k++) {
        sum += a.at(i, k) * b.at(k, j);
      }
      c->at(i, j) = sum;
    }
  }
}

using Matrices = std::vector<Matrix>;

struct LazyFunc {
  std::vector<LazyFunc *> args;
  Matrices arg_mats;
  Matrix mtx;
  std::function<void(const Matrices&, Matrix *)> fun = [](const Matrices&, Matrix *) {};

  Matrix &val() { return mtx; }
  const Matrix &val() const { return mtx; }

  template <typename F> void set_fun(F &&f) { fun = std::forward<F>(f); }

  LazyFunc(const std::vector<LazyFunc *> &argz, int r, int c)
      : args(argz), mtx(r, c, 1.0) {
  }

  void calc() {
    for (auto *arg : args) {
      arg->calc();
    }
    fun(arg_mats, &mtx);
  }
};

struct Mod3l;

struct Block {
  Mod3l *model = nullptr;

  // TODO: why are they pointers?
  LazyFunc *fowd_fun = nullptr;
  // Backward for gradient propagation
  LazyFunc *bawd_fun = nullptr;

  const Matrix& fval() const {
    //fowd_fun->calc();
    return fowd_fun->val();
  }
  Matrix& fval() {
    //fowd_fun->calc();
    return fowd_fun->val();
  }

  const Matrix& bval() const {
    //bawd_fun->calc();
    return bawd_fun->val();
  }
  Matrix& bval() {
    //bawd_fun->calc();
    return bawd_fun->val();
  }

  template <typename F> void set_fowd_fun(F &&f) { fowd_fun->set_fun(f); }
  template <typename F> void set_bawd_fun(F &&f) { bawd_fun->set_fun(f); }


  // -------
  Block(const std::vector<Block *> &argz, int r, int c);
  ~Block() {
    if (fowd_fun != nullptr) {
      delete fowd_fun;
    }
    if (bawd_fun != nullptr) {
      delete bawd_fun;
    }
  }

  void calc_fval() { fowd_fun->calc(); }
  void calc_bval() { bawd_fun->calc(); }
  void apply_bval(float learning_rate) {
    Matrix &val = fval();
    Matrix &grads = bval();
    for (int i = 0; i < val.rows; i++) {
      for (int j = 0; j < val.cols; j++) {
        val.at(i, j) -= grads.at(i, j) * learning_rate;
      }
    }
  }
};

struct Mod3l {
private:
  std::unordered_map<Block *, bool> blocks;

public:
  Mod3l() {}

  Block* add(Block *block) {
    blocks.insert({block, false});
    block->model = this;
    return block;
  }

  ~Mod3l() {
    for (auto &[block, _] : blocks) {
      delete block;
    }
  }


  bool& is_calculated(Block* block) {
     return blocks[block];
	}

  void set_data(Block *block, const std::vector<std::vector<double>> &vals) {
    block->fval().set_data(vals);
    for(auto& [block, calculated]: blocks) {
       calculated = false;
    } 
  }
};

static Block *Data(Mod3l *model, int rows, int cols) {
  return model->add(new Block({}, rows, cols));
}

// TransposedView view of the matrix with no overhead. For MatMul bawd_fun
// gradient propagation
template <class M> struct TransposedView {
  const M &src;
  int rows;
  int cols;
  TransposedView(const M &src) : src(src), rows(src.cols), cols(src.rows) {}
  inline const double &at(int r, int c) const { return src.at(c, r); }
};

// This is requried to build view of a view
template <class M>
TransposedView(const TransposedView<M> &) -> TransposedView<TransposedView<M>>;

static Block *MatMul(Block *inputs, Block *weights) {
  const Matrix &in = inputs->fval();
  const Matrix &w = weights->fval();
  Block *res = new Block({inputs, weights}, in.rows, w.cols);

  const Matrix &dout = res->bval();

  res->set_fowd_fun([](const Matrices& ins, Matrix *out) {
    auto [in, w] = std::pair(ins[0], ins[1]);
    multiply_matrix(in,   // m, n
                    w,    // n, k
                    out); // m, k
  });

  inputs->set_bawd_fun([w, dout](const Matrices& ins, Matrix *dinputs) {
    multiply_matrix(dout,              // m, k
                    TransposedView(w), // k, n
                    dinputs);          // m, n
  });

  weights->set_bawd_fun([in, dout](const Matrices& ins, Matrix *dweights) {
    multiply_matrix(TransposedView(in), // n, m
                    dout,               // m, k
                    dweights);          // n, k
  });

  return res;
}

// TransposedView view of the matrix with no overhead. For MatMul bawd_fun
// gradient propagation
template <class M> struct ReshapedView {
  M *src;
  int rows;
  int cols;
  ReshapedView(M &src, size_t rows, size_t cols)
      : src(&src), rows(rows),
        cols(cols) { /* TODO: check rows*cols=rows*cols */
  }
  inline const double &at(int r, int c) const {
    size_t idx = r * cols + c;
    size_t src_r = idx / src->cols;
    size_t src_c = idx % src->cols;
    return src->at(src_r, src_c);
  }

  inline double &at(int r, int c) {
    size_t idx = r * cols + c;
    size_t src_r = idx / src->cols;
    size_t src_c = idx % src->cols;
    return src->at(src_r, src_c);
  }
};

// This is requried to build view of a view
template <class M>
ReshapedView(const ReshapedView<M> &) -> ReshapedView<ReshapedView<M>>;

template <class M> struct SlidingWindowView {
  M *src;
  int rows;
  int cols;
  size_t window_rows;
  size_t window_cols;
  SlidingWindowView(M &src, size_t window_rows, size_t window_cols)
      : src(&src), window_rows(window_rows), window_cols(window_cols) {
    rows = src.rows * src.cols;
    cols = window_rows * window_cols;
  }
  inline const double &at(int r, int c) const {
    auto [base_row, base_col, delta_row, delta_col] = std::tuple(
        r / src->cols, r % src->cols, c / window_cols, c % window_cols);

    size_t src_r = (base_row + delta_row) % src->rows;
    size_t src_c = (base_col + delta_col) % src->cols;
    return src->at(src_r, src_c);
  }

  inline double &at(int r, int c) {
    // TODO:
    // this is gona be a bit crazy. instead of assigning, it should calc the
    // difference between prev value and new value, and update source with that
    // difference.
    size_t base_row = r / src->cols;
    size_t base_col = r % src->cols;
    size_t delta_row = c / window_cols;
    size_t delta_col = c % window_cols;
    size_t row = base_row + delta_row;
    size_t col = base_col + delta_col;
    row = row % src->rows;
    col = col % src->cols;
    return src->at(row, col);
  }
};

// Circular convolution, to keep it simple
// Output size is same as input
static Block *Convolution(Block *input, Block *kernel) {
  Block *res = new Block({input, kernel}, input->fval().rows, input->fval().cols);
  // input -> m, n
  // kernel -> k, l
  // output -> m, n
  res->set_fowd_fun([input, kernel](const Matrices& ins, Matrix *out) {
    auto [k, l] = std::pair(kernel->fval().rows, kernel->fval().cols);
    SlidingWindowView input_slide(input->fval(), k, l);
    ReshapedView kernel_flat(kernel->fval(), k * l, 1);
    ReshapedView out_flat(*out, out->rows * out->cols, 1);
    //
    multiply_matrix(input_slide, // m * n, k * l
                    kernel_flat, // k * l, 1
                    &out_flat);  // m * n, 1
  });

  // TODO: test everything below
  const Matrix &dout = res->bval();

  input->set_bawd_fun([kernel, dout](const Matrices& ins, Matrix *dinputs) {
    auto [k, l] = std::pair(kernel->fval().rows, kernel->fval().cols);
    ReshapedView dout_flat(dout, dout.rows * dout.cols, 1);
    ReshapedView kernel_flat(kernel->fval(), k * l, 1);
    SlidingWindowView dinput_slide(*dinputs, k, l);
    //
    multiply_matrix(dout_flat,                   // m * n, 1
                    TransposedView(kernel_flat), // 1, k * l
                    &dinput_slide                // m * n, k * l
    );
  });

  kernel->set_bawd_fun([input, dout](const Matrices& ins, Matrix *dkernel) {
    auto [k, l] = std::pair(dkernel->rows, dkernel->cols);
    SlidingWindowView input_slide(input->fval(), k, l);
    ReshapedView dout_flat(dout, dout.rows * dout.cols, 1);
    ReshapedView dkernel_flat(*dkernel, k * l, 1);
    //
    multiply_matrix(TransposedView(input_slide), // k * l, m * n
                    dout_flat,                   // m * n, 1
                    &dkernel_flat                // k * l, 1
    );
  });

  return res;
}

static double square(double d) { return d * d; }

static double sigmoid(double x) {
  if (x >= 0) {
    double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  } else {
    double z = std::exp(x);
    return z / (1.0 + z);
  }
}

static double sigmoid_derivative(double x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

static double tbd(double) { return 0; }

template <typename F> 
static void for_each_el(const Matrix &in, F fu) {
  for (size_t i = 0; i < in.data->size(); ++i) {
    fu((*in.data)[i]);
  }
}

template <typename F>
static void for_each_el(const Matrix &in, Matrix *out, F fu) {
  for (size_t i = 0; i < in.data->size(); ++i) {
    (*out->data)[i] = fu((*in.data)[i]);
  }
}


template <typename F>
static void for_each_el(const Matrix &in1, const Matrix &in2, F fu) {
  for (size_t i = 0; i < in1.data->size(); ++i) {
    fu((*in1.data)[i], (*in2.data)[i]);
  }
}

template <typename F>
static void for_each_el(const Matrix &in1, const Matrix &in2, Matrix *out,
                        F fu) {
  for (size_t i = 0; i < in1.data->size(); ++i) {
    (*out->data)[i] = fu((*in1.data)[i], (*in2.data)[i]);
  }
}


static Block *Reshape(Block *a, int rows, int cols) {
  Block *res = new Block({a}, rows, cols);
  res->set_fowd_fun([](const Matrices& ins, Matrix *out) {
    for_each_el(ins[0], out, [](double a) { return a; });
  });

  a->set_bawd_fun([res](const Matrices& ins, Matrix *) {
    // TODO
  });
  return res;
}

template <typename F1, typename F2>
// static Block *ElFun(Block *a, DifFu1 fwd, DifFu1 bwd) {
static Block *ElFun(Block *a, F1 fwd, F2 bwd) {
  Block *res = new Block({a}, a->fval().rows, a->fval().cols);

  res->set_fowd_fun([fwd](const Matrices& ins, Matrix *out) { for_each_el(ins[0], out, fwd); });
    
  const Matrix& grads = res->bval();

  a->set_bawd_fun([a, grads, bwd](const Matrices& ins, Matrix *out) {
    for_each_el(a->fval(), grads, out,
                [bwd](double in, double grad) { return bwd(in) * grad; });
  });

  return res;
}

static Block *Sqrt(Block *a) { return ElFun(a, &square, &tbd); }

static Block *Sigmoid(Block *a) {
  return ElFun(a, &sigmoid, &sigmoid_derivative);
}

static Block *MulEl(Block *a, double n) {
  return ElFun(
      a, [n](double d) { return n * d; }, &tbd);
}

static Block *Add(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, a1->fval().rows, a1->fval().cols);

  res->set_fowd_fun([](const Matrices& ins, Matrix *out) {
    for_each_el(ins[0], ins[1], out,
                [](double a, double b) { return a + b; });
  });

  a1->set_bawd_fun([res](const Matrices& ins, Matrix *out) {
    for_each_el(res->bval(), out, [](double g) { return g; });
  });

  a2->set_bawd_fun([res](const Matrices& ins, Matrix *out) {
    for_each_el(res->bval(), out, [](double g) { return g; });
  });

  return res;
};

// Difference
static Block *Dif(Block *a1, Block *a2) { return Add(a1, MulEl(a2, -1)); };

static Block *Sum(Block *a) {
  auto *res = new Block({a}, 1, 1);

  res->set_fowd_fun([](const Matrices& ins, Matrix *out) {
    double s = 0;
    for_each_el(ins[0], [&s](double a) { s += a; });
    out->at(0, 0) = s;
  });

  a->set_bawd_fun([a, res](const Matrices& ins, Matrix *out) {
    double grad = res->bval().at(0, 0);
    for_each_el(a->fval(), out, [grad](double) { return grad; });
  });

  return res;
}

static Block *SSE(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, 1, 1);

  res->set_fowd_fun([](const Matrices& ins, Matrix *out) {
    double s = 0;
    for_each_el(ins[0], ins[1],
                [&s](double a, double b) { s += square(b - a); });
    out->at(0, 0) = s;
  });

  a1->set_bawd_fun([a1, a2](const Matrices& ins, Matrix *da1) {
    for_each_el(a1->fval(), a2->fval(), da1,
                [](double a, double b) { return 2 * (a - b); });
  });

  a2->set_bawd_fun([a1, a2](const Matrices& ins, Matrix *da2) {
    for_each_el(a1->fval(), a2->fval(), da2,
                [](double a, double b) { return 2 * (b - a); });
  });

  return res;
}


static double clip(double p) {
	double epsilon = 1e-12; // small value to avoid log(0)
	return std::min(std::max(p, epsilon), 1.0 - epsilon);
}

// Binary Cross Enthropy
// TODO: calc average as a single value. Currently it is consistent with
// python impl having same flaw
static Block *BCE(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, a1->fval().rows, a1->fval().cols);

  res->set_fowd_fun([](const Matrices& ins, Matrix *out) {
    for_each_el(ins[0], ins[1], out, [](double y_p, double y_t) {
      double p = clip(y_p); 
      return -(y_t * std::log(p) + (1.0 - y_t) * std::log(1.0 - p));
    });
  });

  a1->set_bawd_fun([a1, a2](const Matrices& ins, Matrix *out) {
    for_each_el(a1->fval(), a2->fval(), out, [](double y_p, double y_t) {
      double p = clip(y_p); 
      return -(y_t / p) + ((1.0 - y_t) / (1.0 - p));
    });
  });

  return res;
}

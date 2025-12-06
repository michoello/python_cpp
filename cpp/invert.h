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
#include <unordered_set>

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
    size_t i = 0;
    for (size_t r = 0; r < vals.size(); ++r) {
      if ((int)vals[r].size() != cols)
        throw std::invalid_argument(
            "All rows must have the same number of columns");
      for (int c = 0; c < cols; ++c) {
        (*data)[i++] = vals[r][c];
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

struct LazyFunc {
  Matrix mtx;
  std::function<void(Matrix *)> fun = [](Matrix *) {};
  bool is_calculated;

  Matrix &val() { return mtx; }
  const Matrix &val() const { return mtx; }

  template <typename F> void set_fun(F &&f) { 
     fun = std::forward<F>(f); 
     is_calculated = false;
  }

  LazyFunc(int r, int c) : mtx(r, c, 1.0) { }

  void calc() {
    if(is_calculated) return;
    fun(&mtx);
    is_calculated = true;
  }
};

struct Mod3l;

struct Block {
  Mod3l *model = nullptr;

  mutable LazyFunc fowd_fun;
  mutable std::vector<LazyFunc> bawd_funs;

  const Matrix& fval() const {
    fowd_fun.calc();
    return fowd_fun.val();
  }
  Matrix& fval() {
    fowd_fun.calc();
    return fowd_fun.val();
  }

  const Matrix& bval(size_t idx = 0) const {
    // ugly.. think about better
    if (bawd_funs.empty()) {
       return default_grads;
    }
    bawd_funs[idx].calc();
    return bawd_funs[idx].val();
  }
  Matrix& bval(size_t idx = 0) {
    if (bawd_funs.empty()) {
       return default_grads;
    }
    bawd_funs[idx].calc();
    return bawd_funs[idx].val();
  }

  template <typename F> void set_fowd_fun(F &&f) { fowd_fun.set_fun(f); }
  template <typename F> void add_bawd_fun(F &&f) { 
      LazyFunc bawd_fun(fowd_fun.mtx.rows, fowd_fun.mtx.cols);
      bawd_fun.set_fun(f); 
      bawd_funs.push_back(bawd_fun);
  }

  // -------
  Matrix default_grads;
  Block(const std::vector<Block *> &argz, int r, int c);

  void reset_both_lazy_funcs() {
     fowd_fun.is_calculated = false;
     for(auto& bawd_fun: bawd_funs) {
        bawd_fun.is_calculated = false;
     }
	}

  void apply_bval(float learning_rate);
};

struct Mod3l {
private:
  std::unordered_set<Block *> blocks;

public:
  Mod3l() {}

  Block* add(Block *block) {
    blocks.insert(block);
    block->model = this;
    return block;
  }

  ~Mod3l() {
    for (auto &block: blocks) {
      delete block;
    }
  }

  void set_data(Block *block, const std::vector<std::vector<double>> &vals) {
    block->fval().set_data(vals);
    reset_all_lazy_funcs();
  }

  void reset_all_lazy_funcs() {
    for(auto& block: blocks) {
       block->reset_both_lazy_funcs();
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

  res->set_fowd_fun([=](Matrix *out) {
    auto [in, w] = std::tuple(inputs->fval(), weights->fval());
    multiply_matrix(in,   // m, n
                    w,    // n, k
                    out); // m, k
  });

  inputs->add_bawd_fun([=](Matrix *dinputs) {
    auto [in, w] = std::pair(inputs->fval(), weights->fval());
    const Matrix &dout = res->bval();
    multiply_matrix(dout,              // m, k
                    TransposedView(w), // k, n
                    dinputs);          // m, n
  });

  weights->add_bawd_fun([=](Matrix *dweights) {
    auto [in, w] = std::pair(inputs->fval(), weights->fval());
    const Matrix &dout = res->bval();
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
  res->set_fowd_fun([=](Matrix *out) {
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

  input->add_bawd_fun([kernel, dout](Matrix *dinputs) {
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

  kernel->add_bawd_fun([input, dout](Matrix *dkernel) {
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
  res->set_fowd_fun([=](Matrix *out) {
    for_each_el(a->fval(), out, [](double a) { return a; });
  });

  a->add_bawd_fun([res](Matrix *) {
    // TODO
  });
  return res;
}


template <typename F1, typename F2>
static Block *ElFun(Block *arg, F1 fwd, F2 bwd) {
  Block *block = new Block({arg}, arg->fval().rows, arg->fval().cols);

  block->set_fowd_fun([=](Matrix *out) { 
     for_each_el(arg->fval(), out, fwd); 
  });
    
  arg->add_bawd_fun([=](Matrix *out) {
    for_each_el(arg->fval(), block->bval(), out,
                [bwd](double in, double grad) { return bwd(in) * grad; });
  });

  return block;
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

  res->set_fowd_fun([=](Matrix *out) {
    for_each_el(a1->fval(), a2->fval(), out,
                [](double a, double b) { return a + b; });
  });

  a1->add_bawd_fun([res](Matrix *out) {
    for_each_el(res->bval(), out, [](double g) { return g; });
  });

  a2->add_bawd_fun([res](Matrix *out) {
    for_each_el(res->bval(), out, [](double g) { return g; });
  });

  return res;
};

// Difference
static Block *Dif(Block *a1, Block *a2) { return Add(a1, MulEl(a2, -1)); };

static Block *Sum(Block *a) {
  auto *res = new Block({a}, 1, 1);

  res->set_fowd_fun([=](Matrix *out) {
    double s = 0;
    for_each_el(a->fval(), [&s](double a) { s += a; });
    out->at(0, 0) = s;
  });

  a->add_bawd_fun([a, res](Matrix *out) {
    double grad = res->bval().at(0, 0);
    for_each_el(a->fval(), out, [grad](double) { return grad; });
  });

  return res;
}

static Block *SSE(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, 1, 1);

  res->set_fowd_fun([=](Matrix *out) {
    double s = 0;
    for_each_el(a1->fval(), a2->fval(),
                [&s](double a, double b) { s += square(b - a); });
    out->at(0, 0) = s;
  });

  a1->add_bawd_fun([=](Matrix *da1) {
    for_each_el(a1->fval(), a2->fval(), da1,
                [](double a, double b) { return 2 * (a - b); });
  });

  a2->add_bawd_fun([=](Matrix *da2) {
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

  res->set_fowd_fun([=](Matrix *out) {
    for_each_el(a1->fval(), a2->fval(), out, [](double y_p, double y_t) {
      double p = clip(y_p); 
      return -(y_t * std::log(p) + (1.0 - y_t) * std::log(1.0 - p));
    });
  });

  a1->add_bawd_fun([a1, a2](Matrix *out) {
    for_each_el(a1->fval(), a2->fval(), out, [](double y_p, double y_t) {
      double p = clip(y_p); 
      return -(y_t / p) + ((1.0 - y_t) / (1.0 - p));
    });
  });

  return res;
}

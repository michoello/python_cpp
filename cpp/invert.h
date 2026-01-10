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
#include <unordered_set>
#include <vector>

#include "matrix.h"

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

  LazyFunc(int r, int c) : mtx(r, c, 1.0) {}

  void calc() {
    if (!is_calculated) {
      fun(&mtx);
      is_calculated = true;
    }
  }
};

struct Mod3l;

struct Block {
  Mod3l *model = nullptr;

  mutable LazyFunc fowd_fun;
  mutable LazyFunc bawd_fun;
  mutable std::vector<LazyFunc> bawd_funs;

  const Matrix &fval() const {
    fowd_fun.calc();
    return fowd_fun.val();
  }

  const Matrix &bval() const {
    bawd_fun.calc();
    return bawd_fun.val();
  }

  template <typename F> void set_fowd_fun(F &&f) { fowd_fun.set_fun(f); }
  template <typename F> void add_bawd_fun(F &&f) {
    // !!! 
    LazyFunc prev_one = bawd_funs[0];
    auto ff = [prev_one, f](Matrix *out) mutable {
        prev_one.is_calculated = false;
        prev_one.calc();
        Matrix prev_grad =  prev_one.val();

        f(out);

        for_each_ella([](double p, double &out) { out += p; }, prev_grad, *out);
      };

    bawd_fun.set_fun(ff);
    
    bawd_funs[0] = bawd_fun;

    // The graph updated, invalidate all values
    reset_model();
  }

  void reset_model();

  // -------
  Block() : Block({}, 1, 1) {} // temporay
  Block(const std::vector<Block *> &argz, int r, int c);

  void reset_both_lazy_funcs() {
    fowd_fun.is_calculated = false;
    bawd_fun.is_calculated = false;
  }

  void apply_bval(float learning_rate);
};

struct Mod3l {
private:
  std::unordered_set<Block *> blocks;

public:
  Mod3l() {}

  Block *add(Block *block) {
    blocks.insert(block);
    block->model = this;
    return block;
  }

  ~Mod3l() {
    for (auto &block : blocks) {
      delete block;
    }
  }

  void set_data(Block *block, const std::vector<std::vector<double>> &vals) {
    // block->fval().set_data(vals);
    block->fowd_fun.val().set_data(vals);
    reset_all_lazy_funcs();
  }

  void reset_all_lazy_funcs() {
    for (auto &block : blocks) {
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
  inline double get(int r, int c) const { return src.get(c, r); }
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

  std::pair<size_t, size_t> convert(size_t r, size_t c) const {
    size_t idx = r * cols + c;
    return {idx / src->cols, idx % src->cols};
  }

  inline double get(int r, int c) const {
    auto [src_r, src_c] = convert(r, c);
    return src->get(src_r, src_c);
  }

  inline void set(int r, int c, double value) {
    auto [src_r, src_c] = convert(r, c);
    src->set(src_r, src_c, value);
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

  std::pair<size_t, size_t> convert(size_t r, size_t c) const {
    size_t base_row = r / src->cols;
    size_t base_col = r % src->cols;
    size_t delta_row = c / window_cols;
    size_t delta_col = c % window_cols;
    size_t row = base_row + delta_row;
    size_t col = base_col + delta_col;
    return {row % src->rows, col % src->cols};
  }

  inline double get(int r, int c) const {
    auto [src_r, src_c] = convert(r, c);
    return src->get(src_r, src_c);
  }

  inline void set(int r, int c, double value) {
    // This is a bit crazy. instead of assigning the result, we increase it
    // each time. Since convolution is essentially faning out the source matrix into
    // list of shingles, each cell is multiplied many times, thus grads sum up.
    auto [row, col] = convert(r, c);
    src->set(row, col, src->get(row, col) + value);
  }
};

// Circular convolution, to keep it simple
// Output size is same as input
static Block *Convo(Block *input, Block *kernel) {
  Block *res =
      new Block({input, kernel}, input->fval().rows, input->fval().cols);
  // input -> m, n
  // kernel -> k, l
  // output -> m, n
  res->set_fowd_fun([=](Matrix *out) {
    auto [k, l] = std::pair(kernel->fval().rows, kernel->fval().cols);
    SlidingWindowView input_slide(input->fval(), k, l);
    ReshapedView kernel_flat(kernel->fval(), k * l, 1);
    ReshapedView out_flat(*out, out->rows * out->cols, 1);
    multiply_matrix(input_slide, // m * n, k * l
                    kernel_flat, // k * l, 1
                    &out_flat);  // m * n, 1
  });

  // TODO: test everything below

  input->add_bawd_fun([kernel, res](Matrix *dinputs) {
    const Matrix &dout = res->bval();

    auto [k, l] = std::pair(kernel->fval().rows, kernel->fval().cols);
    ReshapedView dout_flat(dout, dout.rows * dout.cols, 1);
    ReshapedView kernel_flat(kernel->fval(), k * l, 1);
    SlidingWindowView dinput_slide(*dinputs, k, l);
    //
    // Set dinputs to all zeros
    // TODO: this will not work if we have several grads coming
    // NEeds foxing.
    for_each_ella([](double &a) { a = 0; }, *dinputs);

    multiply_matrix(dout_flat,                   // m * n, 1
                    TransposedView(kernel_flat), // 1, k * l
                    &dinput_slide                // m * n, k * l
    );
  });

  kernel->add_bawd_fun([input, res](Matrix *dkernel) {
    const Matrix &dout = res->bval();
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


static double tanh_custom(double x) {
  if (x >= 0.0) {
      double e = std::exp(-2.0 * x);
      return (1.0 - e) / (1.0 + e);
  } else {
      double e = std::exp(2.0 * x);
      return (e - 1.0) / (e + 1.0);
  }
}

static double tanh_derivative(double x) {
  double t = tanh(x);
	return 1 - t * t;
}



static double tbd(double) { return 0; }

static Block *Reshape(Block *a, int rows, int cols) {
  Block *res = new Block({a}, rows, cols);
  res->set_fowd_fun([=](Matrix *out) {
    for_each_ella([](double in, double &out) { out = in; }, a->fval(), *out);
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
    for_each_ella([fwd](double in, double &out) { out = fwd(in); }, arg->fval(),
                  *out);
  });

  arg->add_bawd_fun([=](Matrix *out) {
    for_each_ella([bwd](double in, double grad_in,
                        double &grad_back) { grad_back = bwd(in) * grad_in; },
                  arg->fval(), block->bval(), *out);
  });

  return block;
}

static Block *Sqrt(Block *a) { return ElFun(a, &square, &tbd); }

static Block *Sigmoid(Block *a) {
  return ElFun(a, &sigmoid, &sigmoid_derivative);
}

static Block *Tanh(Block *a) {
  return ElFun(a, &tanh_custom, &tanh_derivative);
}

static Block *MulEl(Block *a, double n) {
  return ElFun(
      a, [n](double d) { return n * d; }, &tbd);
}

static Block *Add(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, a1->fval().rows, a1->fval().cols);

  res->set_fowd_fun([=](Matrix *out) {
    for_each_ella([](double a, double b, double &c) { c = a + b; }, a1->fval(),
                  a2->fval(), *out);
  });

  a1->add_bawd_fun([res](Matrix *out) {
    for_each_ella([](double grad_in, double &grad_out) { grad_out = grad_in; },
                  res->bval(), *out);
  });

  a2->add_bawd_fun([res](Matrix *out) {
    for_each_ella([](double grad_in, double &grad_out) { grad_out = grad_in; },
                  res->bval(), *out);
  });

  return res;
};

// Difference
static Block *Dif(Block *a1, Block *a2) { return Add(a1, MulEl(a2, -1)); };


// This implementation does not respect rows of matrix,
//and calculates the softmax over entire matrix
static Block *SoftMax(Block *a) {
  auto *res = new Block({a}, a->fval().rows, a->fval().cols);

  res->set_fowd_fun([=](Matrix *out) {
    const Matrix& in = a->fval();
    double max_val = in.get(0, 0);
    for_each_ella([&max_val](double i) { max_val = std::max(max_val, i); }, in);
    double sum = 0.0;
    for_each_ella([&sum, max_val](double i/*n*/, double& o/*ut*/) { 
       o = std::exp(i - max_val);
       sum += o;
    }, in, *out);
    for_each_ella([sum](double& o) { o /= sum; }, *out);
  });
 
  // TODO: bawd fun.

  return res; 
}

static Block *Abs(Block *a) {
  auto *res = new Block({a}, a->fval().rows, a->fval().cols);

  res->set_fowd_fun([=](Matrix *out) {
    for_each_ella([](double i/*n*/, double& o/*ut*/) { 
       o = i >= 0 ? i : -i;
    }, a->fval(), *out);
  });

  a->add_bawd_fun([a, res](Matrix *out) {
    for_each_ella([](double in, double &grad_out) { 
      grad_out = in >= 0 ? 1 : -1; 
    }, a->fval(), *out);
  });

  return res;
}

static Block *Sum(Block *a) {
  auto *res = new Block({a}, 1, 1);

  res->set_fowd_fun([=](Matrix *out) {
    double s = 0;
    for_each_ella([&s](double a) { s += a; }, a->fval());
    out->set(0, 0, s);
  });

  a->add_bawd_fun([a, res](Matrix *out) {
    double grad_in = res->bval().get(0, 0);
    for_each_ella([grad_in](double &grad_out) { grad_out = grad_in; }, *out);
  });

  return res;
}

static Block *SSE(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, 1, 1);

  res->set_fowd_fun([=](Matrix *out) {
    double s = 0;
    for_each_ella([&s](double a, double b) { s += square(b - a); }, a1->fval(),
                  a2->fval());
    out->set(0, 0, s);
  });

  a1->add_bawd_fun([=](Matrix *da1) {
    for_each_ella(
        [](double a, double b, double &grad_out) { grad_out = 2 * (a - b); },
        a1->fval(), a2->fval(), *da1);
  });

  a2->add_bawd_fun([=](Matrix *da2) {
    for_each_ella(
        [](double a, double b, double &grad_out) { grad_out = 2 * (b - a); },
        a1->fval(), a2->fval(), *da2);
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
    for_each_ella(
        [](double y_p, double y_t, double &res) {
          double p = clip(y_p);
          res = -(y_t * std::log(p) + (1.0 - y_t) * std::log(1.0 - p));
        },
        a1->fval(), a2->fval(), *out);
  });

  a1->add_bawd_fun([a1, a2](Matrix *out) {
    for_each_ella(
        [](double y_p, double y_t, double &grads_back) {
          double p = clip(y_p);
          grads_back = -(y_t / p) + ((1.0 - y_t) / (1.0 - p));
        },
        a1->fval(), a2->fval(), *out);
  });

  return res;
}

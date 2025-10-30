/*
 *  LLM stands for Little Lazy Matrix
 *
 */

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
class Funcs;

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

  void fill_uniform() {
    // Random engine
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Uniform distribution in [-1, 1]
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (auto &x : *data) {
      x = dist(gen);
    }
  }

  inline double &at(int r, int c) { return (*data)[r * cols + c]; }

  inline const double &at(int r, int c) const { return (*data)[r * cols + c]; }

  // Convert bawd_fun to nested vector (Python list-of-lists)
  std::vector<std::vector<double>> value() const {
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i][j] = at(i, j);
      }
    }
    return out;
  }
};

template <class T, class U>
void multiply_matrix(const T &a, const U &b, Matrix *c) {
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
  std::vector<LazyFunc *> args;
  Matrix _val;
  std::function<void(Matrix *)> fun = [](Matrix *) {};

  Matrix &val() { return _val; }
  const Matrix &val() const { return _val; }

  template <typename F> void set_fun(F &&f) { fun = std::forward<F>(f); }

  LazyFunc(const std::vector<LazyFunc *> &argz, int r, int c)
      : args(argz), _val(r, c, 1.0) {}

  void calc_fval() {
    for (auto *arg : args) {
      arg->calc_fval();
    }
    fun(&val());
  }
};

struct Mod3l;

struct Block {

  // TODO: delete these two guys?
  Matrix &val() { return fowd_fun->_val; }
  const Matrix &val() const { return fowd_fun->_val; }

  // forward value
  std::vector<std::vector<double>> fval() const {
    return fowd_fun->_val.value();
  }

  std::vector<std::vector<double>> bval() const {
    return bawd_fun->val().value();
  }

  template <typename F> void set_fun(F &&f) { fowd_fun->set_fun(f); }

  Mod3l *model = nullptr;

  // TODO: why are they pointers?
  LazyFunc *fowd_fun = nullptr;
  // Backward for gradient propagation
  LazyFunc *bawd_fun = nullptr;

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

  void calc_fval() { fowd_fun->calc_fval(); }

  void calc_bval() { bawd_fun->calc_fval(); }

  void apply_bval(float learning_rate) {
    for (int i = 0; i < val().rows; i++) {
      for (int j = 0; j < val().cols; j++) {
        val().at(i, j) -= bawd_fun->val().at(i, j) * learning_rate;
      }
    }
  }
};

struct Mod3l {
private:
  std::unordered_map<Block *, bool> blocks;

public:
  Mod3l() {}

  void add(Block *block) {
    blocks.insert({block, false});
    block->model = this;
  }

  ~Mod3l() {
    for (auto &[block, _] : blocks) {
      delete block;
    }
  }

  void set_data(Block *block, const std::vector<std::vector<double>> &vals) {
    block->val().set_data(vals);
  }
};

static Block *Data(Mod3l *model, int rows, int cols) {
  Block *res = new Block({}, rows, cols);
  model->add(res);
  return res;
}

static Block *MatMul(Block *a1, Block *a2) {

  // Transposed view of the matrix with no overhead. For MatMul bawd_fun
  // gradient propagation
  struct Transposed {
    const Matrix &matrix;
    int rows;
    int cols;
    Transposed(const Matrix &src)
        : matrix(src), rows(src.cols), cols(src.rows) {}
    inline const double &at(int r, int c) const { return matrix.at(c, r); }
  };

  Block *res = new Block({a1, a2}, a1->val().rows, a2->val().cols);
  res->set_fun(
      [a1, a2](Matrix *out) { multiply_matrix(a1->val(), a2->val(), out); });

  a1->bawd_fun->set_fun([a2, res](Matrix *out) {
    multiply_matrix(res->bawd_fun->val(), Transposed(a2->val()), out);
  });

  a2->bawd_fun->set_fun([a1, res](Matrix *out) {
    multiply_matrix(Transposed(a1->val()), res->bawd_fun->val(), out);
  });

  return res;
}

using DifFu0 = std::function<void(double)>;
using DifFu1 = std::function<double(double)>;
using DifFu2 = std::function<double(double, double)>;
using DifFu20 = std::function<void(double, double)>;

class Funcs {
public:
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

  static DifFu1 get_mul_el(double n) {
    return [n](double d) { return n * d; };
  }

  static void for_each_el(const Matrix &in, DifFu0 fu) {
    for (size_t i = 0; i < in.data->size(); ++i) {
      fu((*in.data)[i]);
    }
  }

  static void for_each_el(const Matrix &in, Matrix *out, DifFu1 fu) {
    for (size_t i = 0; i < in.data->size(); ++i) {
      (*out->data)[i] = fu((*in.data)[i]);
    }
  }

  static void for_each_el(const Matrix &in1, const Matrix &in2, Matrix *out,
                          DifFu2 fu) {
    for (size_t i = 0; i < in1.data->size(); ++i) {
      (*out->data)[i] = fu((*in1.data)[i], (*in2.data)[i]);
    }
  }

  static void for_each_el(const Matrix &in1, const Matrix &in2, 
                          DifFu20 fu) {
    for (size_t i = 0; i < in1.data->size(); ++i) {
      fu((*in1.data)[i], (*in2.data)[i]);
    }
  }


};

static Block *ElFun(Block *a, DifFu1 fwd, DifFu1 bwd) {
  Block *res = new Block({a}, a->val().rows, a->val().cols);

  res->set_fun(
      [a, fwd, res](Matrix *out) { Funcs::for_each_el(a->val(), out, fwd); });

  a->bawd_fun->set_fun([a, res, bwd](Matrix *out) {
    // Calc local derivative
    Funcs::for_each_el(a->val(), out, bwd);

    // Multiply by incoming gradient
    Funcs::for_each_el(
        *out, res->bawd_fun->val(), out,
        [](double local, double grads) { return local * grads; });
  });

  return res;
}

static Block *Sqrt(Block *a) { return ElFun(a, &Funcs::square, &Funcs::tbd); }

static Block *Sigmoid(Block *a) {
  return ElFun(a, &Funcs::sigmoid, &Funcs::sigmoid_derivative);
}

static Block *MulEl(Block *a, double n) {
  return ElFun(a, Funcs::get_mul_el(n), &Funcs::tbd);
}

static Block *Add(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, a1->val().rows, a1->val().cols);

  res->set_fun([a1, a2](Matrix *out) {
    Funcs::for_each_el(a1->val(), a2->val(), out,
                       [](double a, double b) { return a + b; });
  });
  // TODO: backward

  return res;
};

// Difference
static Block *Dif(Block *a1, Block *a2) { return Add(a1, MulEl(a2, -1)); };

static Block *Sum(Block *a) {
  auto *res = new Block({a}, 1, 1);

  res->set_fun([a](Matrix *out) {
    double s = 0;
    Funcs::for_each_el(a->val(), [&s](double a){ s += a; });
    out->at(0, 0) = s;
  });

  a->bawd_fun->set_fun([a, res](Matrix *out) {
    double grad = res->bawd_fun->val().at(0, 0);
    Funcs::for_each_el(a->val(), out, [grad](double){ return grad; });
  });

  return res;
}

static Block *SSE(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, 1, 1);

  res->set_fun([a1, a2](Matrix *out) {
    double s = 0;
    Funcs::for_each_el(a1->val(), a2->val(), [&s](double a, double b){ 
        s += Funcs::square(b - a);
    });
    out->at(0, 0) = s;
  });

  a1->bawd_fun->set_fun([a1, a2](Matrix *out) {
    Funcs::for_each_el(a1->val(), a2->val(), out, [](double a, double b){ 
        return 2 * (a - b);
    });


  });

  return res;
}

// Binary Cross Enthropy
// TODO: calc average as a single value. Currently it is consistent with
// python impl having same flaw
static Block *BCE(Block *a1, Block *a2) {
  auto *res = new Block({a1, a2}, a1->val().rows, a1->val().cols);

  res->set_fun([a1, a2](Matrix *out) {
    Funcs::for_each_el(a1->val(), a2->val(), out, [](double y_p, double y_t) {
      double epsilon = 1e-12; // small value to avoid log(0)
      double p = std::min(std::max(y_p, epsilon), 1.0 - epsilon);
      return -(y_t * std::log(p) + (1.0 - y_t) * std::log(1.0 - p));
    });
  });

  a1->bawd_fun->set_fun([a1, a2](Matrix *out) {
    Funcs::for_each_el(a1->val(), a2->val(), out, [](double y_p, double y_t) {
      double epsilon = 1e-12;
      double p = std::min(std::max(y_p, epsilon), 1.0 - epsilon);
      return -(y_t / p) + ((1.0 - y_t) / (1.0 - p));
    });
  });

  return res;
}

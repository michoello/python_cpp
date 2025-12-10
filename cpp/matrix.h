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

  inline double get(int r, int c) const { return (*data)[r * cols + c]; }
  inline void set(int r, int c, double value) { (*data)[r * cols + c] = value; }

  // Convert bawd_fun to nested vector (Python list-of-lists)
};

// Extracts the full value of Matrix-like object(i.e. Matrix or matrix view)
template <class M> std::vector<std::vector<double>> value(const M &m) {
  std::vector<std::vector<double>> out(m.rows, std::vector<double>(m.cols));
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      out[i][j] = m.get(i, j);
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
        sum += a.get(i, k) * b.get(k, j);
      }
      c->set(i, j, sum);
    }
  }
}


template <typename F>
static void for_each_el(const Matrix &in, F fu, Matrix *out = nullptr) {
  using Ret = std::invoke_result_t<F, double&>;
  if constexpr (std::is_void_v<Ret>) {
    for (size_t i = 0; i < in.data->size(); ++i) {
      fu((*in.data)[i]);
    }
  } else {
    for (size_t i = 0; i < in.data->size(); ++i) {
      (*out->data)[i] = fu((*in.data)[i]);
    }
  }
}

template <typename F>
static void for_each_el(const Matrix &in1, const Matrix &in2, F fu, Matrix *out = nullptr) {
  using Ret = std::invoke_result_t<F, double&, double&>;
  if constexpr (std::is_void_v<Ret>) {
    for (size_t i = 0; i < in1.data->size(); ++i) {
      fu((*in1.data)[i], (*in2.data)[i]);
    }
  } else {
    for (size_t i = 0; i < in1.data->size(); ++i) {
      (*out->data)[i] = fu((*in1.data)[i], (*in2.data)[i]);
    }
  }
}



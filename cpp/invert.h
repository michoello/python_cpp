/*
*  LLM stands for Little Lazy Matrix
*
*/

#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include <functional>
#include <memory>
#include <cmath>

std::vector<int> invert(const std::vector<int> &v);

class Matrix;
class Funcs;

void sum_matrix(const Matrix& a, const Matrix& y_true, Matrix* c);
void mul_el_matrix(const Matrix& a, const Matrix& b, Matrix* c);

class Matrix {
public:
    int rows;
    int cols;
    std::shared_ptr<std::vector<double>> data;  // flat row-major storage

    Matrix(int r, int c, double val=0.0) : rows(r), cols(c), data(std::make_shared<std::vector<double>>(r * c, val)) {}

    Matrix(const Matrix& other) = default;

    void set_data(const std::vector<std::vector<double>>& vals) {
        for (int r=0; r < vals.size(); ++r) {
            if ((int)vals[r].size() != cols)
                throw std::invalid_argument("All rows must have the same number of columns");
            for(int c=0; c < cols; ++c) {
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

      for (auto& x : *data) {
          x = dist(gen);
      }
    }

    inline double& at(int r, int c) {
        return (*data)[r * cols + c];
    }

    inline const double& at(int r, int c) const {
        return (*data)[r * cols + c];
    }

    // Convert back to nested vector (Python list-of-lists)
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
void multiply_matrix(const T& a, const U& b, Matrix* c) {
  // TODO: remove it out of here?
    if (a.cols != b.rows || a.rows != c->rows || b.cols != c->cols) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
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

// Transposed view of the matrix with no overhead. For MatMul back gradient propagation
class Transposed {
    const Matrix& matrix;
  public:
    int rows;
    int cols;
    Transposed(const Matrix& src) : matrix(src), rows(src.cols), cols(src.rows) {}
    inline const double& at(int r, int c) const {
        return matrix.at(c, r);
    }
};

struct FuncPair {
  // Forward uses vector of input args, and pointer to result
  std::function<void()> forward;
  // Backward for gradient propagation
  std::function<void()> backward;
};

struct Block {
  std::vector<Block*> args;
  FuncPair funcs;
  Matrix val;
  Matrix grads_in;

  Block(const std::vector<Block*>& argz, int r, int c) : args(argz), val(r, c), grads_in(r, c, 1.0) {
  }

  void CalcVal() {
      for(auto* arg: args) {
         arg->CalcVal();
      }
		  funcs.forward();
  }

  virtual void CalcGrad() {
    funcs.backward();
    for(auto* arg: args) {
        arg->CalcGrad();
    } 
  }

  void ApplyGrad(float learning_rate) {
    for (int i = 0; i < val.rows; i++) {
      for (int j = 0; j < val.cols; j++) {
         val.at(i, j) -= grads_in.at(i, j) * learning_rate;
      }
    }
  }
};


class DataBlock: public Block {
public: 
   DataBlock(int rows, int cols) : Block({}, rows, cols) {
     funcs = FuncPair{
       []() {},
       []() {}
     };
   }
};

// Matrix multiplication
class MatMulBlock: public Block {
public:
  MatMulBlock(Block* a1, Block* a2) : Block({a1, a2}, a1->val.rows, a2->val.cols) {
    funcs = FuncPair{
      // forward
      [a1, a2, this]() {
        multiply_matrix(a1->val, a2->val, &this->val);
      },
      // backward
      [a1, a2, this]() {
        multiply_matrix(Transposed(a1->val), this->grads_in, &a2->grads_in);
        multiply_matrix(this->grads_in, Transposed(a2->val), &a1->grads_in);
      }
    };
  }
};

class AddBlock: public Block {
public:
  AddBlock(Block* a1, Block* a2) : Block({a1, a2}, a1->val.rows, a1->val.cols) {
     // TODO: check dimensions
     funcs = FuncPair{
       [a1, a2, this]() {
          sum_matrix(a1->val, a2->val, &this->val);
       },
       []() {
          // TODO
       }
     };
  }
};

using DifFu = std::function<double(double)>;

class Funcs {
public:
  static double square(double d) {
    return d * d;
  } 

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

  static double tbd(double x) {
    return 0;
  }

  static DifFu get_mul_el(double n) {
    return [n](double d) {
      return n * d;
    };
  }

  static void for_each_el(const Matrix& in, Matrix* out, DifFu fu) {
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            out->at(i, j) = fu(in.at(i, j));
        }
    }
	}

};

class ElFunBlock: public Block {
public:
  ElFunBlock(Block *a, DifFu fwd, DifFu bwd) : Block({a}, a->val.rows, a->val.cols) {
    funcs = FuncPair{
      // forward
      [a, fwd, this]() {
        Funcs::for_each_el(a->val, &this->val, fwd);
      },
      // backward
      [a, bwd, this]() {
        Funcs::for_each_el(a->val, &a->grads_in, bwd);
        mul_el_matrix(a->grads_in, this->grads_in, &a->grads_in);
      }
    };
  }
};


class SqrtBlock: public ElFunBlock {
public:
  SqrtBlock(Block *a) : ElFunBlock(a, &Funcs::square, &Funcs::tbd) {
  }
};


class SigmoidBlock: public ElFunBlock {
public:
  SigmoidBlock(Block *a) : ElFunBlock(a, &Funcs::sigmoid, &Funcs::sigmoid_derivative) {
  }
};

class MulElBlock: public ElFunBlock {
public:
  MulElBlock(Block *a, double n) : ElFunBlock(a, Funcs::get_mul_el(n), &Funcs::tbd) {
  }
};

// Difference
class DifBlock: public AddBlock {
public:
  // TODO: fix the memory leak here:
  DifBlock(Block* a1, Block* a2) : AddBlock(a1, new MulElBlock(a2, -1)) {
  }
};


class SumBlock: public Block {
public:
  SumBlock(Block *a) : Block({a}, 1, 1) {
    funcs = FuncPair{
      // forward
      [a, this]() {
        float s = 0.0;
        for (int i = 0; i < a->val.rows; i++) {
           for (int j = 0; j < a->val.cols; j++) {
              s += a->val.at(i, j);
           }
        }
        this->val.at(0, 0) = s;
      },
      // backward
      [a, this]() {
        double grad = this->grads_in.at(0, 0);
        for(int r = 0; r < a->val.rows; ++r) {
           for(int c = 0; c < a->val.cols; ++c) {
               a->grads_in.at(r, c) = grad;
           }
        }
      }
    };

  }
};


class SSEBlock: public Block {
public:
  SSEBlock(Block* a1, Block* a2) : Block({a1, a2}, 1, 1) {
    funcs = FuncPair{
      // forward
      [a1, a2, this]() {
        float s = 0.0;
        for (int i = 0; i < a1->val.rows; i++) {
           for (int j = 0; j < a1->val.cols; j++) {
              s += Funcs::square(a2->val.at(i, j) - a1->val.at(i, j));
           }
        }
        this->val.at(0, 0) = s;
      },
      // backward
      [a1, a2, this]() {
        for (int i = 0; i < a1->val.rows; i++) {
           for (int j = 0; j < a1->val.cols; j++) {
              a1->grads_in.at(i, j) = 2* (a1->val.at(i, j) - a2->val.at(i, j));
           }
        }
      }
    };
  }
};

// Binary Cross Enthropy
// TODO: calc average as a single value. Currently it is consistent with 
// python impl having same flaw
class BCEBlock: public Block {
protected:
public:
  BCEBlock(Block* a1, Block* a2) : Block({a1, a2}, a1->val.rows, a1->val.cols) {
    assert(a1->val.rows == a2->val.rows && "Rows not equal");
    assert(a1->val.cols == a2->val.cols && "Cols not equal");

	  const auto& y_pred = a1->val;
		const auto& y_true = a2->val;
    funcs = FuncPair{
      // forward
      [&y_pred, &y_true, this]() {
        double epsilon = 1e-12; // small value to avoid log(0)
        for (int i = 0; i < y_pred.rows; i++) {
        	for (int j = 0; j < y_true.cols; j++) {
            double p = std::min(std::max(y_pred.at(i, j), epsilon), 1.0 - epsilon);
            double t = y_true.at(i, j);
            this->val.at(i, j) = -(t * std::log(p) + (1.0 - t) * std::log(1.0 - p));
          }
        }
        // TODO: result matrix should be [1, 1] dimensional and be an average across all elements.
      },
      // backward
      [&y_pred, &y_true, a1]() {
        double epsilon = 1e-12;
        for (int i = 0; i < y_pred.rows; i++) {
          for (int j = 0; j < y_true.cols; j++) {
             double p = std::min(std::max(y_pred.at(i, j), epsilon), 1.0 - epsilon);
             double t = y_true.at(i, j);
             a1->grads_in.at(i, j) = -(t / p) + ((1.0 - t) / (1.0 - p));
          }
        }
      }
    };
  }
};


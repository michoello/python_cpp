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

void multiply_matrix(const Matrix& a, const Matrix& y_true, Matrix* c);
void sum_matrix(const Matrix& a, const Matrix& y_true, Matrix* c);
void mul_el_matrix(const Matrix& a, const Matrix& b, Matrix* c);

class Matrix {
public:
    int rows;
    int cols;
    std::shared_ptr<std::vector<double>> data;  // flat row-major storage

    // Constructor with rows, cols (zero initialized)
    Matrix(int r, int c) : rows(r), cols(c), data(std::make_shared<std::vector<double>>(r * c, 0.0)) {}

    Matrix(const Matrix& other) {
       rows = other.rows;
       cols = other.cols;
       data = other.data;
		}

    // TODO: remove it
    Matrix(const std::vector<std::vector<double>>& vals) {
        set_data(vals);
    }

    // Constructor with values (nested vector)
    void set_data(const std::vector<std::vector<double>>& vals) {
        for (int r=0; r < vals.size(); ++r) {
            if ((int)vals[r].size() != cols)
                throw std::invalid_argument("All rows must have the same number of columns");
            for(int c=0; c < cols; ++c) {
                at(r, c) = vals[r][c];
            }
        }
    }

    void print() const {
       for (int r=0; r < rows; ++r) {
          for(int c=0; c < cols; ++c) {
              std::cout << at(r, c) << " ";
          }
          std::cout << "\n";
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

  // TODO: CRTP-based view block, avoid mem allocation on each round
  Matrix transpose() const {
     Matrix r(cols, rows);
     for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                r.at(j, i) = at(i, j);
            }
     }
     return r;
  }

};


class Block {
protected:

public:
  std::vector<Block*> args;
  Matrix val;
  Matrix dval;

  Block(const std::vector<Block*>& argz, int r, int c) : args(argz), val(r, c), dval(r, c) {
  }

  const Matrix& GetVal() const {
    return val;
  }

  Matrix& GetVal() {
    return val;
  }

  virtual void CalcValImpl(const std::vector<Matrix>& ins, Matrix* out) = 0;

  void CalcVal() {
      std::vector<Matrix> ins;
      for(auto* arg: args) {
         arg->CalcVal();
         ins.push_back(arg->GetVal());
      }

      CalcValImpl(ins, &val);
  }

  virtual void CalcDval() {}; // TODO = 0
  virtual void CalcDval(const Matrix& dvalue) {}; // TODO = 0

  Matrix& GetDval() {
     return dval;
  }
  const Matrix& GetDval() const {
     return dval;
  }

  void ApplyGrad(float learning_rate) {
    for (int i = 0; i < val.rows; i++) {
        for (int j = 0; j < val.cols; j++) {
            val.at(i, j) -= dval.at(i, j) * learning_rate;
        }
    }

  }
};


class DataBlock: public Block {
public: 
   DataBlock(const Matrix m) : Block({}, m.rows, m.cols) {
       val = m;
   }
  
   void CalcValImpl(const std::vector<Matrix>& ins, Matrix* out) override {
      // nothing
   }
};

struct FuncPair {
  // Forward takes vector of input args, and pointer to result
  using ForwardFn  = std::function<void(const std::vector<Matrix>&, Matrix*)>;
  // Backward takes vector of input args, grads matrix, and vector of matrixes to write results to
  using BackwardFn = std::function<void(const std::vector<Matrix>&, const Matrix&, const std::vector<Matrix*>&)>;

  ForwardFn forward;
  BackwardFn backward;
};


// Matrix multiplication
class MatMulBlock: public Block {
  FuncPair funcs;
public:
  MatMulBlock(Block* a1, Block* a2) : Block({a1, a2}, a1->GetVal().rows, a2->GetVal().cols) {
    funcs = FuncPair{
      // forward
      [](const std::vector<Matrix>& ins, Matrix* out) {
        multiply_matrix(ins[0], ins[1], out);
      },
      // backward
      [](const std::vector<Matrix>& ins, const Matrix& grads, const std::vector<Matrix*>& out) {
        multiply_matrix(ins[0].transpose(), grads, out[1]);
        multiply_matrix(grads, ins[1].transpose(), out[0]);
      }
    };
  }

  void CalcValImpl(const std::vector<Matrix>& ins, Matrix* out) override {
    funcs.forward(ins, out);
  }

  void CalcDval(const Matrix& grads) {
    // TODO: check dimensions
    funcs.backward({args[0]->GetVal(), args[1]->GetVal()}, grads, {&args[0]->dval, &args[1]->dval});
  }
};


class AddBlock: public Block {
public:
  AddBlock(Block* a1, Block* a2) : Block({a1, a2}, a1->GetVal().rows, a1->GetVal().cols) {
    // TODO: check dimensions
  }

  void CalcValImpl(const std::vector<Matrix>& ins, Matrix* out) override {
    sum_matrix(ins[0], ins[1], out);
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
protected:
  DifFu forward;
  DifFu backward;
public:
  ElFunBlock(Block *a, DifFu f) : Block({a}, a->GetVal().rows, a->GetVal().cols) {
     forward = f; 
  }

  void CalcValImpl(const std::vector<Matrix>& ins, Matrix* out) override {
    Funcs::for_each_el(ins[0], out, forward);
  }
  void CalcDval() override {
    Funcs::for_each_el(args[0]->GetVal(), &this->dval, backward);
    // TODO: args[0]->CalcDval();
  }
};


class SqrtBlock: public ElFunBlock {
public:
  SqrtBlock(Block *a) : ElFunBlock(a, &Funcs::square) {
  }
};


class SigmoidBlock: public ElFunBlock {
public:
  SigmoidBlock(Block *a) : ElFunBlock(a, &Funcs::sigmoid) {
      backward = &Funcs::sigmoid_derivative;
  }


  using ElFunBlock::CalcDval;
  void CalcDval(const Matrix& grads) override {
     assert(grads.rows == dval.rows && "Rows not equal");
     assert(grads.cols == dval.cols && "Cols not equal");
     this->CalcDval();
     mul_el_matrix(dval, grads, &dval);

     args[0]->CalcDval(dval);
  }
};

class MulElBlock: public ElFunBlock {
public:
  MulElBlock(Block *a, double n) : ElFunBlock(a, Funcs::get_mul_el(n)) {
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
  }

  void CalcValImpl(const std::vector<Matrix>& ins, Matrix* out) override {
    float s = 0.0;
    for (int i = 0; i < ins[0].rows; i++) {
        for (int j = 0; j < ins[0].cols; j++) {
            s += ins[0].at(i, j);
        }
    }
    val.at(0, 0) = s;
  }

  void CalcDval() {
       int rows = args[0]->GetVal().rows;
       int cols = args[0]->GetVal().cols;
       for(int r = 0; r < rows; ++r) {
         for(int c = 0; c < cols; ++c) {
           args[0]->dval.at(r, c) = 1;
         }
       }
  }

};

// Sum Square Error
// TODO: this inheritance is no good. Use composition instead.
class SSEBlock: public SumBlock {
  Block *arg1; // model output
  Block *arg2; // labels 

public:

  SSEBlock(Block *a, Block *y_true) : SumBlock(new SqrtBlock(new DifBlock(a, y_true))) {
     //args.push_back(a);  // model output
     //args.push_back(y_true);  // labels
     arg1 = a;
     arg2 = y_true;
  }
  
  void CalcDval() override {
     dval = val;
     auto* dblock =  new MulElBlock(new DifBlock(arg1, arg2), 2);
     dblock->CalcVal();
     arg1->dval = dblock->GetVal();
  } 
};


// Binary Cross Enthropy
// TODO: calc average as a single value. Currently it is consistent with 
// python impl having same flaw
class BCEBlock: public Block {
protected:
public:
  BCEBlock(Block* a1, Block* a2) : Block({a1, a2}, a1->GetVal().rows, a1->GetVal().cols) {
  }

  void CalcValImpl(const std::vector<Matrix>& ins, Matrix* out) override {
    const auto& y_pred = ins[0];
    const auto& y_true = ins[1];
    auto* c = out;

    assert(y_pred.rows == y_true.rows && "Rows not equal");
    assert(y_pred.cols == y_true.cols && "Cols not equal");

    double epsilon = 1e-15; // small value to avoid log(0)
    for (int i = 0; i < y_pred.rows; i++) {
        for (int j = 0; j < y_true.cols; j++) {

            // Clip predictions to avoid log(0)
            double p = std::min(std::max(y_pred.at(i, j), epsilon), 1.0 - epsilon);
            c->at(i, j) = -(y_true.at(i, j) * std::log(p) + (1.0 - y_true.at(i, j)) * std::log(1.0 - p));
        }
    }
    // TODO: result matrix should be [1, 1] dimensional and be an average across all elements.
    // 
  }

  //void CalcDval(const Matrix& dif) {
  void CalcDval() override {
    const auto& y_pred = args[0]->GetVal();
    const auto& y_true = args[1]->GetVal();
    auto* c = &dval;

    double epsilon = 1e-12;

    for (int i = 0; i < y_pred.rows; i++) {
        for (int j = 0; j < y_true.cols; j++) {
            double p = std::min(std::max(y_pred.at(i, j), epsilon), 1.0 - epsilon);
            c->at(i, j) = -(y_true.at(i, j) / p) + ((1.0 - y_true.at(i, j)) / (1.0 - p));
        }
    }

    args[0]->CalcDval(dval);

  }
};


#include <vector>
#include <iostream>
#include <random>
#include <functional>

std::vector<int> invert(const std::vector<int> &v);

class Matrix;

void multiply_matrix(const Matrix& a, const Matrix& b, Matrix* c);
void sum_matrix(const Matrix& a, const Matrix& b, Matrix* c);



class Matrix {
public:
    int rows;
    int cols;
    std::vector<double> data;  // flat row-major storage

    // Constructor with rows, cols (zero initialized)
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    // Constructor with values (nested vector)
    void set_data(const std::vector<std::vector<double>>& vals) {
        rows = vals.size();
        cols = vals.empty() ? 0 : vals[0].size();
        data.reserve(rows * cols);
        for (const auto& row : vals) {
            if ((int)row.size() != cols)
                throw std::invalid_argument("All rows must have the same number of columns");
            data.insert(data.end(), row.begin(), row.end());
        }
    }

    void fill_uniform() {
      // Random engine
      static std::random_device rd;
      static std::mt19937 gen(rd());

      // Uniform distribution in [-1, 1]
      std::uniform_real_distribution<double> dist(-1.0, 1.0);

      for (auto& x : data) {
          x = dist(gen);
      }
    }

    void set_data(const std::vector<double>& values) {
        if (values.size() != rows * cols) {
            throw std::runtime_error("Wrong number of values");
        }
        data = values;
    }

    inline double& at(int r, int c) {
        return data[r * cols + c];
    }

    inline const double& at(int r, int c) const {
        return data[r * cols + c];
    }

    Matrix multiply(const Matrix& other) const {
        Matrix result(rows, other.cols);
        ::multiply_matrix(*this, other, &result);
        return result;
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


class Block {
protected:
  Matrix val;

public:
  Block(int r, int c) : val(r, c) {
  }

  const Matrix& GetVal() const {
    return val;
  }

  virtual void CalcVal() = 0;

  Matrix& GetVal() {
    return val;
  }
};

class DataBlock: public Block {
public: 
   DataBlock(int r, int c) : Block(r, c) {}
    
   void SetVal(const Matrix& m) {
     val.set_data(m.data);
   }
  
   void CalcVal() override {
      // nothing
   }
};


class MatMulBlock: public Block {
protected:
  std::vector<Block*> args;
public:
  MatMulBlock(Block* a1, Block* a2) : Block(a1->GetVal().rows, a2->GetVal().cols) {
    args.push_back(a1);
    args.push_back(a2);
  }

  void CalcVal() override {
    args[0]->CalcVal();
    args[1]->CalcVal();
    multiply_matrix(args[0]->GetVal(), args[1]->GetVal(), &val);
  }
};


class AddBlock: public Block {
protected:
  std::vector<Block*> args;
public:
  AddBlock(Block* a1, Block* a2) : Block(a1->GetVal().rows, a1->GetVal().cols) {
    // TODO: check dimensions
    args.push_back(a1);
    args.push_back(a2);
  }

  void CalcVal() override {
    args[0]->CalcVal();
    args[1]->CalcVal();
    sum_matrix(args[0]->GetVal(), args[1]->GetVal(), &val);
  }
};


using DifFu = std::function<double(double)>;

class Funcs {
public:
  static double square(double d) {
    return d * d;
  } 

  static DifFu get_mul_el(double n) {
    return [n](double d) {
      return n * d;
    };
  }
  

};

class ElFunBlock: public Block {
  Block *arg;

  DifFu fu;

public:
  ElFunBlock(Block *a, DifFu f) : Block(a->GetVal().rows, a->GetVal().cols) {
     arg = a;
     fu = f; 
  }

  void CalcVal() {
    arg->CalcVal();

    const auto& in = arg->GetVal();
    auto* out = &this->val;
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            out->at(i, j) = fu(in.at(i, j));
        }
    }
  }
};




class SqrtBlock: public ElFunBlock {
  Block *arg;

public:
  SqrtBlock(Block *a) : ElFunBlock(a, &Funcs::square) {
  }
};


class MulElBlock: public ElFunBlock {
  Block *arg;

public:
  MulElBlock(Block *a, double n) : ElFunBlock(a, Funcs::get_mul_el(n)) {
  }
};


class DifBlock: public AddBlock {
public:
  // TODO: fix the memory leak here:
  DifBlock(Block* a1, Block* a2) : AddBlock(a1, new MulElBlock(a2, -1)) {
  }
};


class SumBlock: public Block {
  Block *arg;

public:
  SumBlock(Block *a) : Block(1, 1) {
     arg = a;
  }

  void CalcVal() {
    arg->CalcVal();

    const auto& in = arg->GetVal();
    float s = 0.0;
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            s += in.at(i, j);
        }
    }
    val.at(0, 0) = s;
  }
};




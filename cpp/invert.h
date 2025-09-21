#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include <memory>

std::vector<int> invert(const std::vector<int> &v);

class Matrix;

void multiply_matrix(const Matrix& a, const Matrix& b, Matrix* c);
void sum_matrix(const Matrix& a, const Matrix& b, Matrix* c);

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

    void set_data(const std::vector<double>& values) {
        if (values.size() != rows * cols) {
            throw std::runtime_error("Wrong number of values");
        }
        *data = values;
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


class Block {
protected:

public:
  Matrix val;
  Matrix dval;


  Block(int r, int c) : val(r, c), dval(r, c) {
  }

  const Matrix& GetVal() const {
    return val;
  }

  Matrix& GetVal() {
    return val;
  }

  virtual void CalcVal() = 0;
  virtual void CalcDval() {}; // TODO = 0

  Matrix& GetDval() {
     return dval;
  }
  const Matrix& GetDval() const {
     return dval;
  }

  void ApplyGrad(float learning_rate) {
    for (int i = 0; i < val.rows; i++) {
        for (int j = 0; j < val.cols; j++) {
            val.at(i, j) += dval.at(i, j) * learning_rate;
        }
    }

  }


};

class DataBlock: public Block {
public: 
   DataBlock(int r, int c) : Block(r, c) {}
    
   void SetVal(const Matrix& m) {
     val.set_data(*m.data);
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

  static void for_each_el(const Matrix& in, Matrix* out, DifFu fu) {
    for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
            out->at(i, j) = fu(in.at(i, j));
        }
    }
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

    Funcs::for_each_el(arg->GetVal(), &this->val, fu);
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

// Sum Square Error
// TODO: this inheritance is no good. Use composition instead.
class SSEBlock: public SumBlock {
  Block *arg1; // model output
  Block *arg2; // labels 

public:

  SSEBlock(Block *a, Block *b) : SumBlock(new SqrtBlock(new DifBlock(a, b))) {
     arg1 = a;
     arg2 = b;
  }
  
  void CalcDval() override {
     dval = val;
     auto* dblock =  new MulElBlock(new DifBlock(arg2, arg1), 2);
     dblock->CalcVal();
     arg1->dval = dblock->GetVal();
  } 

};






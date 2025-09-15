#include <vector>
#include <iostream>
#include <random>

std::vector<int> invert(const std::vector<int> &v);

class Matrix {
public:
    int rows;
    int cols;

    std::vector<double> data;  // flat row-major storage

    Matrix() : rows(0), cols(0) {
    }

    // Constructor with rows, cols (zero initialized)
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}


    // Constructor with values (nested vector)
    Matrix(const std::vector<std::vector<double>>& vals) {
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

    Matrix(int rows, int cols, const std::vector<double>& values)
        : rows(rows), cols(cols), data(rows * cols) {
        if (!values.empty() && values.size() != rows * cols) {
            throw std::runtime_error("Wrong number of values");
        }
        if (!values.empty()) {
            data = values;
        }
    }


    inline double& at(int r, int c) {
        return data[r * cols + c];
    }

    inline const double& at(int r, int c) const {
        return data[r * cols + c];
    }

    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        Matrix result(rows, other.cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0.0;
                for (int k = 0; k < cols; k++) {
                    sum += at(i, k) * other.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
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
};

class DataBlock: public Block {
public: 
   DataBlock(int r, int c) : Block(r, c) {}
   //void SetVal(const std::vector<std::vector<double>>& vals) {
   void SetVal(const Matrix& m) {
     // TODO: make set value method in matrix
     val = m;
   }
};


class MulBlock: public Block {
protected:
  std::vector<Block> args;
public:
  MulBlock(const Block& a1, const Block& a2) : Block(a1.GetVal().rows, a2.GetVal().cols) {
    args.push_back(a1);
    args.push_back(a2);
  }

  void CalcVal() {
    val = args[0].GetVal().multiply(args[1].GetVal());
  }
};

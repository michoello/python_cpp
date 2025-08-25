// listinvert/invert.cpp
#include <vector>
#include <iostream>

#include "invert.h"

std::vector<int> invert(const std::vector<int> &v) {
    return std::vector<int>(v.rbegin(), v.rend());
}


/*
class Matrix {
public:
    int rows;
    int cols;
    std::vector<double> data;  // flat row-major storage

    Matrix() : rows(0), cols(0) {}

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



// IMPORTANT: the module name is "_listinvert" to match extension "listinvert._listinvert"
PYBIND11_MODULE(_listinvert, m) {
    m.doc() = "Matrix with flat vector storage";

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<>())  // empty
        .def(py::init<int,int>(), py::arg("rows"), py::arg("cols"))
        .def(py::init<const std::vector<std::vector<double>>&>(), py::arg("values"))
        .def(py::init([](py::kwargs kwargs) {
            if (kwargs.contains("values")) {
                return Matrix(kwargs["values"].cast<std::vector<std::vector<double>>>());
            }
            int r = kwargs.contains("rows") ? kwargs["rows"].cast<int>() : 0;
            int c = kwargs.contains("cols") ? kwargs["cols"].cast<int>() : 0;
            return Matrix(r, c);
        })
        )
        .def("multiply", &Matrix::multiply)
        .def("value", &Matrix::value)
        .def("at", (double& (Matrix::*)(int,int)) &Matrix::at,
             py::return_value_policy::reference_internal,
             py::arg("row"), py::arg("col"),
             "Get/set an element by (row, col)")
        ;
    
    // expose function
    m.def("invert", [](const std::vector<int>& input) {
        return std::vector<int>(input.rbegin(), input.rend());
    });
    //m.def("invert", &invert, "Invert a list of integers");
}
*/

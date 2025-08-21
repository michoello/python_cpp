// listinvert/invert.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::vector<int> invert(const std::vector<int> &v) {
    return std::vector<int>(v.rbegin(), v.rend());
}




class Matrix {
public:
    Matrix(const std::vector<std::vector<double>>& values)
        : data(values) {}

    Matrix multiply(const Matrix& other) const {
        size_t n = data.size();
        size_t m = other.data[0].size();
        size_t k = other.data[0].size();

        std::vector<std::vector<double>> result(n, std::vector<double>(m, 0.0));

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                for (size_t t = 0; t < other.data.size(); t++) {
                    result[i][j] += data[i][t] * other.data[t][j];
                }
            }
        }
        return Matrix(result);
    }

    std::vector<std::vector<double>> value() const {
        return data;
    }

private:
    std::vector<std::vector<double>> data;
};


// IMPORTANT: the module name is "_listinvert" to match extension "listinvert._listinvert"
PYBIND11_MODULE(_listinvert, m) {
    m.doc() = "Example module combining a class and a function";

    // expose function
    m.def("invert", &invert, "Invert a list of integers");

    // expose class
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<const std::vector<std::vector<double>>&>())
        .def("multiply", &Matrix::multiply)
        .def("value", &Matrix::value);
}



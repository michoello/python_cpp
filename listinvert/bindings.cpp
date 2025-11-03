// listinvert/invert.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

#include "invert.h"

namespace py = pybind11;


// IMPORTANT: the module name is "_listinvert" to match extension "listinvert._listinvert"
PYBIND11_MODULE(_listinvert, m) {
    m.doc() = "Matrix with flat vector storage";

    // expose class
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<int,int>(), py::arg("rows"), py::arg("cols"))
        .def(py::init<const Matrix&>(), py::arg("other"))
        .def("set_data", &Matrix::set_data)
        .def("fill_uniform", &Matrix::fill_uniform)
        .def("value", &Matrix::value)
        .def("at", (double& (Matrix::*)(int,int)) &Matrix::at,
             py::return_value_policy::reference_internal,
             py::arg("row"), py::arg("col"),
             "Get/set an element by (row, col)");
 

    m.def("multiply_matrix", &multiply_matrix<Matrix, Matrix>, "Multiplies two matrices and writes result into the third one");


    // expose function
    m.def("invert", [](const std::vector<int>& input) {
        return std::vector<int>(input.rbegin(), input.rend());
    });

    py::class_<Mod3l>(m, "Mod3l")
        .def(py::init<>())
        .def("set_data", &Mod3l::set_data);

    py::class_<Block>(m, "Block")
        //.def(py::init< const std::vector<Block*>, int, int>())
        .def("calc_fval", &Block::calc_fval)
        .def("calc_bval", &Block::calc_bval)
        .def("apply_bval", &Block::apply_bval)
        .def("fval", &Block::fval)
        .def("bval", &Block::bval);

    m.def("Data", &Data, py::return_value_policy::reference_internal, "Block with data (weights or inputs/outputs)");
    m.def("MatMul", &MatMul, py::return_value_policy::reference_internal, "Matrix multiplication");
    m.def("SSE", &SSE, py::return_value_policy::reference_internal, "SSE loss func");
}

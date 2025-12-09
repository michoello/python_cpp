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
        .def("at", (double& (Matrix::*)(int,int)) &Matrix::at,
             py::return_value_policy::reference_internal,
             py::arg("row"), py::arg("col"),
             "Get/set an element by (row, col)");
 

    m.def("value", &value<Matrix>, "Returns matrix value as vector of vectors");
    m.def("multiply_matrix", &multiply_matrix<Matrix, Matrix, Matrix>, "Multiplies two matrices and writes result into the third one");

    py::class_<Mod3l>(m, "Mod3l")
        .def(py::init<>())
        .def("set_data", &Mod3l::set_data);

    py::class_<Block>(m, "Block")
        .def("apply_bval", &Block::apply_bval)
        .def("fval", &Block::fval)
        .def("bval", &Block::bval);

    m.def("Data", &Data, py::return_value_policy::reference_internal, "Block with data (weights or inputs/outputs)");
    m.def("MatMul", &MatMul, py::return_value_policy::reference_internal, "Matrix multiplication");
    m.def("Add", &Add, py::return_value_policy::reference_internal, "Matrix sum");
    m.def("SSE", &SSE, py::return_value_policy::reference_internal, "SSE loss func");
    m.def("BCE", &BCE, py::return_value_policy::reference_internal, "BCE loss func");
    m.def("Sigmoid", &Sigmoid, py::return_value_policy::reference_internal, "Sigmoid applied to each element");
    m.def("Reshape", &Reshape, py::return_value_policy::reference_internal, "SSE loss func");
}

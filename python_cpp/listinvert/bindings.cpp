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
        .def("get", (double (Matrix::*)(int,int)) &Matrix::get,
             py::arg("row"), py::arg("col"),
             "Gets an element by (row, col)")
        .def("set", (void (Matrix::*)(int,int, double)) &Matrix::set,
             py::arg("row"), py::arg("col"), py::arg("value"),
             "Sets an element by (row, col)");
 

    m.def("value", &value<Matrix>, "Returns matrix value as vector of vectors");
    m.def("multiply_matrix", &multiply_matrix<Matrix, Matrix, Matrix>, "Multiplies two matrices and writes result into the third one");

    py::class_<Mod3l>(m, "Mod3l")
        .def(py::init<>())
        .def("set_data", &Mod3l::set_data)
        .def("global_grad_norm", &Mod3l::global_grad_norm);

    py::class_<Block>(m, "Block")
        .def("apply_bval", &Block::apply_bval)
        .def("rows", &Block::rows)
        .def("cols", &Block::cols)
        .def("fval", &Block::fval)
        .def("bval", &Block::bval);

    m.def("Data", &Data, py::return_value_policy::reference_internal, "Block with data (weights or inputs/outputs)");
    m.def("MatMul", &MatMul, py::return_value_policy::reference_internal, "Matrix multiplication");
    m.def("Add", &Add, py::return_value_policy::reference_internal, "Matrix sum");
    m.def("SSE", &SSE, py::return_value_policy::reference_internal, "SSE loss func");
    m.def("Abs", &Abs, py::return_value_policy::reference_internal, "Abs loss func (mostly for tests)");
    m.def("BCE", &BCE, py::return_value_policy::reference_internal, "BCE loss func");
    m.def("Sigmoid", &Sigmoid, py::return_value_policy::reference_internal, "Sigmoid applied to each element");
    m.def("Reshape", &Reshape, py::return_value_policy::reference_internal, "SSE loss func");
    m.def("Convo", &Convo, py::return_value_policy::reference_internal, "Convolution block");
    m.def("Convo2", &Convo2, py::return_value_policy::reference_internal, "Convolution block v2.0 - faster");
    m.def("GradClipper", &GradClipper, py::return_value_policy::reference_internal, "Gradient Clipper block");
    m.def("Explode", &Explode, py::return_value_policy::reference_internal, "Explode block for chaining convolutions");
    m.def("ReLU", &ReLU, py::return_value_policy::reference_internal, "ReLU (leaky) func");
    m.def("Tanh", &Tanh, py::return_value_policy::reference_internal, "Tanh func");
    m.def("SoftMax", &SoftMax, py::return_value_policy::reference_internal, "SoftMax func");
    m.def("SoftMaxCrossEntropy", &SoftMaxCrossEntropy, py::return_value_policy::reference_internal, "SoftMaxCrossEntropy loss");




}

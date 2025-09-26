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

    py::class_<Matrix>(m, "Matrix")
        //.def(py::init<>())  // empty
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
        .def("set_data", &Matrix::set_data)
        //.def("multiply", &Matrix::multiply)
        .def("fill_uniform", &Matrix::fill_uniform)
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

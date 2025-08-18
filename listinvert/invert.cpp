// listinvert/invert.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

std::vector<int> invert(const std::vector<int> &v) {
    return std::vector<int>(v.rbegin(), v.rend());
}

// IMPORTANT: the module name is "_listinvert" to match extension "listinvert._listinvert"
PYBIND11_MODULE(_listinvert, m) {
    m.doc() = "List inverter implemented in C++ with pybind11 (backend _listinvert)";
    m.def("invert", &invert, "Invert a list of integers");
}

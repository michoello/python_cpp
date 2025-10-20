// listinvert/invert.cpp
#include <vector>
#include <iostream>

#include "invert.h"

std::vector<int> invert(const std::vector<int> &v) {
    return std::vector<int>(v.rbegin(), v.rend());
}

void sum_matrix(const Matrix& a, const Matrix& b, Matrix* c) {
    if (!(a.rows == b.rows && a.rows == c->rows && b.cols == c->cols && a.cols == b.cols)) {
        throw std::invalid_argument("Matrix dimensions do not match for sum");
    }
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            c->at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }
}

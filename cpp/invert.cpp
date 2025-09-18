// listinvert/invert.cpp
#include <vector>
#include <iostream>

#include "invert.h"

std::vector<int> invert(const std::vector<int> &v) {
    return std::vector<int>(v.rbegin(), v.rend());
}

void multiply_matrix(const Matrix& a, const Matrix& b, Matrix* c) {
    if (a.cols != b.rows || a.rows != c->rows || b.cols != c->cols) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a.cols; k++) {
                sum += a.at(i, k) * b.at(k, j);
            }
            c->at(i, j) = sum;
        }
    }
}

void sum_matrix(const Matrix& a, const Matrix& b, Matrix* c) {
    // TODO: that's not enough, add conditions
    if (a.rows != b.rows || a.rows != c->rows || b.cols != c->cols) {
        throw std::invalid_argument("Matrix dimensions do not match for sum");
    }
    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < a.cols; j++) {
            c->at(i, j) = a.at(i, j) + b.at(i, j);
        }
    }
}


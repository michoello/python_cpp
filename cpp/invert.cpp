// listinvert/invert.cpp
#include <iostream>
#include <vector>

#include "invert.h"

std::vector<int> invert(const std::vector<int> &v) {
  return std::vector<int>(v.rbegin(), v.rend());
}

Block::Block(const std::vector<Block *> &argz, int r, int c): fowd_fun(r, c), bawd_fun(r, c)
{
  // TODO: This is very ugly, rewrite it
  for (Block *arg : argz) {
    if (model == nullptr && arg->model != nullptr) {
      arg->model->add(this);
      // TODO: check that all args belong to the same model
    }
  }
}


void Block::apply_bval(float learning_rate) {
    Matrix &val = fval();
    Matrix &grads = bval();
    for (int i = 0; i < val.rows; i++) {
      for (int j = 0; j < val.cols; j++) {
        val.at(i, j) -= grads.at(i, j) * learning_rate;
      }
    }
    // Now all funcs have to be recalculated. Or should reset_both_lazy_funcs() be called explicitly?
		model->reset_all_lazy_funcs();
}


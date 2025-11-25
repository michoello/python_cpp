// listinvert/invert.cpp
#include <iostream>
#include <vector>

#include "invert.h"

std::vector<int> invert(const std::vector<int> &v) {
  return std::vector<int>(v.rbegin(), v.rend());
}

Block::Block(const std::vector<Block *> &argz, int r, int c) {
  fowd_fun = new LazyFunc({}, r, c);
 
  // TODO: This is very ugly, rewrite it
  for (Block *arg : argz) {
    if (model == nullptr && arg->model != nullptr) {
      arg->model->add(this);
      // TODO: check that all args belong to the same model
    }
    fowd_fun->args.push_back(arg->fowd_fun);
    fowd_fun->arg_mats.push_back(arg->fowd_fun->val());
  }

  bawd_fun = new LazyFunc({}, r, c);
  for (Block *arg : argz) {
    arg->bawd_fun->args.push_back(this->bawd_fun);
    arg->bawd_fun->arg_mats.push_back(this->bawd_fun->val());
  }
}

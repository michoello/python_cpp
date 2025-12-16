#include <iostream>
#include <vector>

#include "invert.h"

Block::Block(const std::vector<Block *> &argz, int r, int c): fowd_fun(r, c), default_grads(r, c, 1.0)
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
    Matrix &val = fowd_fun.val();

    // Ugly. TODO: make it prettier
    size_t grads_count = bawd_funs.size();
    for(size_t g = 0; g < grads_count; ++g) {
      for_each_ella([learning_rate](double grads, double& val) {
          val -= grads * learning_rate; 
      }, bval(g), val);
    }

    // Now all funcs have to be recalculated. Or should reset_both_lazy_funcs() be called explicitly?
		model->reset_all_lazy_funcs();
}


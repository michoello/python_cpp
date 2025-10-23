// listinvert/invert.cpp
#include <vector>
#include <iostream>

#include "invert.h"

std::vector<int> invert(const std::vector<int> &v) {
    return std::vector<int>(v.rbegin(), v.rend());
}


Block::Block(const std::vector<Block *> &argz, int r, int c) { 
     fowd = new LazyFunc({}, r, c);
     for(Block* arg: argz) {
       if(model == nullptr && arg->model != nullptr) {
          arg->model->add(this);
          // TODO: check that they all belong to the same model
       }
       fowd->args.push_back(arg->fowd);
     }

     back = new LazyFunc({}, r, c);
     for(Block* arg: argz) {
         arg->back->args.push_back(this->back);
     }
  }


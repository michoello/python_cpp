#!/bin/bash

# Compile with optimizations
# g++ -std=c++17 -O2 cpp/invert.cpp cpp/test_matrix.cpp -o test_matrix
#
# Compile with debugging information
if g++ -std=c++17 -g -O0 cpp/invert.cpp cpp/test_matrix.cpp -o test_matrix; then
   time ./test_matrix $@
else
   echo "build failed"
fi


#!/bin/bash

if g++ -std=c++17 -O2 cpp/invert.cpp cpp/test_matrix.cpp -o test_matrix; then
   ./test_matrix
else
   echo "build failed"
fi


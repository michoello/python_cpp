#include "invert.h"
#include <chrono>
#include <iostream>

// Benchmark utility: takes any callable, returns elapsed seconds
template <typename Func> double benchmark(Func f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

// Your benchmark function
void benchmark_multiply() {
  Matrix A(100, 100);
  Matrix B(100, 100);
  A.fill_uniform();
  B.fill_uniform();

  Matrix C(100, 100);
  for (size_t i = 0; i < 5000; ++i) {
    multiply_matrix(A, B, &C);
    if (i % 1000 == 0) {
      std::cout << i << "\n";
    }
  }

  std::cout << "Benchmark completed ✅\n";
}

int main() {
  double seconds = benchmark(benchmark_multiply);
  std::cout << "Elapsed time: " << seconds << " s\n";
}

import random
import time

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        # initialize with zeros
        self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def fill_uniform(self):
        """Fill the matrix with random values uniformly distributed in [-1, 1]."""
        self.data = [[random.uniform(-1, 1) for _ in range(self.cols)]
                     for _ in range(self.rows)]

    def multiply(self, other):
        """Return the product of self * other as a new Matrix."""
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions do not align for multiplication.")

        result = Matrix(self.rows, other.cols)
        result.data = [
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ]
        return result

    def __str__(self):
        """Pretty-print matrix for debugging."""
        return "\n".join(" ".join(f"{val:7.3f}" for val in row) for row in self.data)


def benchmark(func):
    """Measure runtime of a function in seconds (float)."""
    start = time.perf_counter()
    func()
    end = time.perf_counter()
    return end - start


# Equivalent to your C++ benchmark_multiply
def benchmark_multiply():
    A = Matrix(100, 100)
    B = Matrix(100, 100)
    A.fill_uniform()
    B.fill_uniform()

    for i in range(50):
        A = A.multiply(B)
        if i % 10 == 0:
            print(i)

    print("Benchmark completed ✅")


if __name__ == "__main__":
    elapsed = benchmark(benchmark_multiply)
    print(f"Elapsed time: {elapsed:.3f} seconds")

import unittest
from listinvert import invert, Matrix

class TestInvert(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(invert([1, 2, 3]), [3, 2, 1])

    def test_empty(self):
        self.assertEqual(invert([]), [])

    def test_single_element(self):
        self.assertEqual(invert([42]), [42])

    def test_negative_numbers(self):
        self.assertEqual(invert([-1, -2, -3]), [-3, -2, -1])

    def test_mixed_numbers(self):
        self.assertEqual(invert([10, -20, 30]), [30, -20, 10])


# ----------------------------------------------------------------

def python_matmul(A, B):
    """Plain Python matrix multiplication for comparison"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, "Incompatible dimensions"

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

class TestMatrixMultiply(unittest.TestCase):
    def test_matrix_multiplication(self):
        # Define test matrices
        A_list = [
            [1, 2, 3],
            [4, 5, 6]
        ]
        B_list = [
            [7, 8],
            [9, 10],
            [11, 12]
        ]

        # Expected result (Python implementation)
        expected = python_matmul(A_list, B_list)

        # C++ Matrix version
        A_cpp = Matrix(A_list)
        B_cpp = Matrix(B_list)
        C_cpp = A_cpp.multiply(B_cpp)

        result = C_cpp.value()

        # Compare element-wise
        self.assertEqual(result, expected)



if __name__ == "__main__":
    unittest.main()

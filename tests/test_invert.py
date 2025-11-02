import unittest
from listinvert import invert, Matrix, multiply_matrix, Mod3l, Block, Data, MatMul

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
        A_cpp = Matrix(2, 3)
        A_cpp.set_data(A_list)
        self.assertEqual(A_cpp.value(), A_list)

        B_cpp = Matrix(3, 2)
        B_cpp.set_data(B_list)
        self.assertEqual(B_cpp.value(), B_list)

        C_cpp = Matrix(2, 2)
        multiply_matrix(A_cpp, B_cpp, C_cpp)

        result = C_cpp.value()

        # Compare element-wise
        self.assertEqual(result, expected)

        self.assertEqual(A_cpp.at(1, 1), 5)

        # This does not work, but ok for now
        #A_cpp.at(1, 1) = 3
        #self.assertEqual(A_cpp.at(1, 1), 3)


class TestMod3l(unittest.TestCase):
    # Simplest smoke test for model Data block
    def test_mod3l_data(self):
        m = Mod3l()
        da = Data(m, 2, 3)
        m.set_data(da, [[1, 2, 3], [4, 5, 6]])

        self.assertEqual(da.fval(), [[1, 2, 3], [4, 5, 6]])

        # TODO: error scenarios, like this:
        # m.set_data(da, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


    def test_mod3l_matmul(self):
      m = Mod3l()

      da = Data(m, 2, 3)
      m.set_data(da, [[1, 2, 3], [4, 5, 6]])

      db = Data(m, 3, 4)
      m.set_data(db, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

      dc = MatMul(da, db)

      self.assertEqual(da.fval(), [
                                               [1, 2, 3],
                                               [4, 5, 6],
                                           ])
      dc.calc_fval()

      self.assertEqual(dc.fval(), [
                                               [38, 44, 50, 56],
                                               [83, 98, 113, 128],
                                           ])


if __name__ == "__main__":
    unittest.main()

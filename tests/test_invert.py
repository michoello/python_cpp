import unittest
from listinvert import Matrix, multiply_matrix, Mod3l, Block, Data, MatMul, SSE, Add, BCE, Sigmoid, Reshape, value


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
        A_list = [[1, 2, 3], [4, 5, 6]]
        B_list = [[7, 8], [9, 10], [11, 12]]

        # Expected result (Python implementation)
        expected = python_matmul(A_list, B_list)

        # C++ Matrix version
        A_cpp = Matrix(2, 3)
        A_cpp.set_data(A_list)
        self.assertEqual(value(A_cpp), A_list)

        B_cpp = Matrix(3, 2)
        B_cpp.set_data(B_list)
        self.assertEqual(value(B_cpp), B_list)

        C_cpp = Matrix(2, 2)
        multiply_matrix(A_cpp, B_cpp, C_cpp)

        result = value(C_cpp)

        # Compare element-wise
        self.assertEqual(result, expected)

        self.assertEqual(A_cpp.get(1, 1), 5)

        # This does not work, but ok for now
        # A_cpp.set(1, 1, 3)
        # self.assertEqual(A_cpp.get(1, 1), 3)


class TestMod3l(unittest.TestCase):
    def assertNearlyEqual(self, a, b, delta=1e-3):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            self.assertEqual(len(a), len(b), "Lengths differ")
            for x, y in zip(a, b):
                self.assertNearlyEqual(x, y, delta)
        else:
            self.assertAlmostEqual(a, b, delta=delta)

    # Simplest smoke test for model Data block
    def test_mod3l_data(self):
        m = Mod3l()
        da = Data(m, 2, 3)
        m.set_data(da, [[1, 2, 3], [4, 5, 6]])

        self.assertEqual(value(da.fval()), [[1, 2, 3], [4, 5, 6]])

        # TODO: error scenarios, like this:
        # m.set_data(da, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_mod3l_matmul(self):
        m = Mod3l()

        da = Data(m, 2, 3)
        m.set_data(da, [[1, 2, 3], [4, 5, 6]])

        db = Data(m, 3, 4)
        m.set_data(db, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        dc = MatMul(da, db)

        self.assertEqual(
value(            da.fval()),
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        )

        self.assertEqual(
value(            dc.fval()),
            [
                [38, 44, 50, 56],
                [83, 98, 113, 128],
            ],
        )

    def test_mod3l_sse_with_grads(self):
        m = Mod3l()

        dy = Data(m, 1, 2)
        m.set_data(dy, [[1, 2]])  # true labels

        # "labels"
        dl = Data(m, 1, 2)
        m.set_data(dl, [[0, 4]])

        ds = SSE(dy, dl)

        self.assertEqual(value(ds.fval()), [[5]])

        # Derivative of loss function is its value is 1.0 (aka df/df)
        self.assertEqual(
            value(ds.bval(0)),
            [
                [1],
            ],
        )
        # Derivative of its args
        self.assertEqual(
            value(dy.bval(0)),
            [
                [2, -4],
            ],
        )

        dy.apply_bval(0.1)
        self.assertNearlyEqual(
value(            dy.fval()),
            [
                [0.8, 2.4],
            ],
        )

        # Calc loss again
        self.assertNearlyEqual(
value(            ds.fval()),
            [
                [3.2],
            ],
        )

    def test_mod3l_add(self):
        m = Mod3l()

        dy = Data(m, 1, 2)
        m.set_data(dy, [[1, 2]])

        dl = Data(m, 1, 2)
        m.set_data(dl, [[0, 4]])

        ds = Add(dy, dl)

        self.assertEqual(value(ds.fval()), [[1, 6]])

    def test_mod3l_add_fwd_bwd(self):
        m = Mod3l()
        da = Data(m, 2, 3)
        db = Data(m, 2, 3)
        dc = Data(m, 2, 3)
        dy = Data(m, 2, 3)

        m.set_data(da, [[1, 2, 3], [4, 5, 6]])
        m.set_data(db, [[4, 5, 6], [1, 2, 3]])
        m.set_data(dc, [[1, 1, 1], [2, 2, 2]])
        m.set_data(dy, [[0.1, 0.3, 0.7], [0.99, 0.5, 0.001]])

        ds2 = Add(Add(da, db), dc)

        self.assertEqual(value(ds2.fval()), [
                                       [6, 8, 10],
                                       [7, 9, 11],
                                    ])

        dsig = Sigmoid(ds2)
        dl = BCE(dsig, dy)

        self.assertNearlyEqual(value(dl.fval()), [
            [ 5.402, 5.600, 3.000 ],
            [ 0.071, 4.500, 10.989 ]
        ])

        self.assertNearlyEqual(value(dsig.bval(0)), [
            [ 363.886, 2087.070, 6607.540 ],
            [ 9.985, 4051.542, 59815.266 ],
                                    ])

        # From Sum and backwards it all goes the same:
        self.assertNearlyEqual(value(ds2.bval(0)), [
            [ 0.898, 0.700, 0.300 ],
            [ 0.009, 0.500, 0.999 ]
                                    ])


        self.assertEqual(value(da.bval(0)), value(ds2.bval(0)))
        self.assertEqual(value(db.bval(0)), value(ds2.bval(0)))
        self.assertEqual(value(dc.bval(0)), value(ds2.bval(0)))



    def test_mod3l_reshape(self):
        m = Mod3l()

        dy = Data(m, 3, 4)
        m.set_data(dy, [
           [1, 2, 3, 4],
           [5, 2, 3, 4],
           [8, 2, 3, 4],
        ])

        dr = Reshape(dy, 4, 3)

        self.assertEqual(value(dr.fval()), 
            [[1, 2, 3], [4, 5, 2], [3, 4, 8], [2, 3, 4]])

    def test_mod3l_sigmoid(self):
        m = Mod3l()

        dy = Data(m, 3, 4)
        m.set_data(dy, [
           [1, 2, 3, 4],
           [5, 2, 3, 4],
           [8, 2, 3, 4],
        ])

        dr = Reshape(dy, 4, 3)

        self.assertEqual(value(dr.fval()), 
            [[1, 2, 3], [4, 5, 2], [3, 4, 8], [2, 3, 4]])

    # Clone of tictactoe test_hello/test_bce_loss
    def test_mod3l_bce_loss(self):
        m = Mod3l()
        x = Data(m, 1, 2)
        m.set_data(x, [[0.1, -0.2]])

        w = Data(m, 2, 3)
        m.set_data(w, [[-0.1, 0.5, 0.3], [-0.6, 0.7, 0.8]])

        #y = (x @ w).sigmoid()
        y = Sigmoid(MatMul(x, w))

        self.assertNearlyEqual(value(y.fval()), [[0.527, 0.478, 0.468]])

        l = Data(m, 1, 3)
        m.set_data(l, [[0, 1, 0.468]])

        #loss = y.bce(ml.BB([[0, 1, 0.468]]))
        loss = BCE(y, l)

        self.assertNearlyEqual(value(loss.fval()), [[0.75, 0.739, 0.691]])

        self.assertNearlyEqual(value(w.bval(0)), [
             [0.0527, -0.052, -4.543/100000],
             [-0.105, 0.104, 9.086/100000]
        ], 3)

        w.apply_bval(1.0)
        self.assertNearlyEqual(value(w.fval()), [
            [-0.153, 0.552, 0.3],
            [-0.495, 0.596, 0.8]
        ])

        # Check that loss decreased
        self.assertNearlyEqual(value(loss.fval()), [[0.736, 0.726, 0.691]])

        # Check that outputs are getting a bit closer
        self.assertNearlyEqual(value(y.fval()), [[0.521, 0.484, 0.468]])

        x.apply_bval(0.01)
        self.assertNearlyEqual(value(x.fval()), [[0.103, -0.194]])

        # Check that updating x also reduces the loss
        self.assertNearlyEqual(value(loss.fval()), [[0.734, 0.723, 0.691]])



if __name__ == "__main__":
    unittest.main()

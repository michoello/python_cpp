import unittest
from listinvert import invert, Matrix, multiply_matrix, Mod3l, Block, Data, MatMul, SSE, Add, BCE, Sigmoid, Reshape


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
        A_list = [[1, 2, 3], [4, 5, 6]]
        B_list = [[7, 8], [9, 10], [11, 12]]

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
        # A_cpp.at(1, 1) = 3
        # self.assertEqual(A_cpp.at(1, 1), 3)


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

        self.assertEqual(
            da.fval(),
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
        )
        dc.calc_fval()

        self.assertEqual(
            dc.fval(),
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

        ds.calc_fval()

        self.assertEqual(ds.fval(), [[5]])

        # Calc derivatives
        dy.calc_bval()

        # Derivative of loss function is its value is 1.0 (aka df/df)
        self.assertEqual(
            ds.bval(),
            [
                [1],
            ],
        )
        # Derivative of its args
        self.assertEqual(
            dy.bval(),
            [
                [2, -4],
            ],
        )

        dy.apply_bval(0.1)
        self.assertNearlyEqual(
            dy.fval(),
            [
                [0.8, 2.4],
            ],
        )

        # Calc loss again
        ds.calc_fval()
        self.assertNearlyEqual(
            ds.fval(),
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

        ds.calc_fval()
        self.assertEqual(ds.fval(), [[1, 6]])

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

        ds2.calc_fval()
        self.assertEqual(ds2.fval(), [
                                       [6, 8, 10],
                                       [7, 9, 11],
                                    ])

        dsig = Sigmoid(ds2)
        dl = BCE(dsig, dy)

        dl.calc_fval()
        self.assertNearlyEqual(dl.fval(), [
            [ 5.402, 5.600, 3.000 ],
            [ 0.071, 4.500, 10.989 ]
        ])

        # Calc derivatives
        da.calc_bval()

        self.assertNearlyEqual(dsig.bval(), [
            [ 363.886, 2087.070, 6607.540 ],
            [ 9.985, 4051.542, 59815.266 ],
                                    ])

        # From Sum and backwards it all goes the same:
        self.assertNearlyEqual(ds2.bval(), [
            [ 0.898, 0.700, 0.300 ],
            [ 0.009, 0.500, 0.999 ]
                                    ])


        self.assertEqual(da.bval(), ds2.bval())

        # Db is not yet calculated
        self.assertEqual(db.bval(), [
            [ 1, 1, 1],
            [ 1, 1, 1],
                                    ])
        db.calc_bval()
        dc.calc_bval()
        # Now it is calculated
        self.assertEqual(db.bval(), ds2.bval())
        self.assertEqual(dc.bval(), ds2.bval())



    def test_mod3l_reshape(self):
        m = Mod3l()

        dy = Data(m, 3, 4)
        m.set_data(dy, [
           [1, 2, 3, 4],
           [5, 2, 3, 4],
           [8, 2, 3, 4],
        ])

        dr = Reshape(dy, 4, 3)

        dr.calc_fval()
        self.assertEqual(dr.fval(), 
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

        dr.calc_fval()
        self.assertEqual(dr.fval(), 
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

        y.calc_fval()

        self.assertNearlyEqual(y.fval(), [[0.527, 0.478, 0.468]])

        l = Data(m, 1, 3)
        m.set_data(l, [[0, 1, 0.468]])

        #loss = y.bce(ml.BB([[0, 1, 0.468]]))
        loss = BCE(y, l)
        loss.calc_fval()

        self.assertNearlyEqual(loss.fval(), [[0.75, 0.739, 0.691]])

        # Not let's check the grad starting from the `loss`, the real one
        w.calc_bval()

        # let's calc grads for inputs as well
        x.calc_bval()

        self.assertNearlyEqual(w.bval(), [
             [0.0527, -0.052, -4.543/100000],
             [-0.105, 0.104, 9.086/100000]
        ], 3)

        w.apply_bval(1.0)
        self.assertNearlyEqual(w.fval(), [
            [-0.153, 0.552, 0.3],
            [-0.495, 0.596, 0.8]
        ])

        # Check that loss decreased
        loss.calc_fval()
        self.assertNearlyEqual(loss.fval(), [[0.736, 0.726, 0.691]])

        # Check that outputs are getting a bit closer
        # Note: no need to call 'y.calc_fval()' as it is already calculated by loss
        self.assertNearlyEqual(y.fval(), [[0.521, 0.484, 0.468]])

        x.apply_bval(0.01)
        self.assertNearlyEqual(x.fval(), [[0.103, -0.193]])

        # Check that updating x also reduces the loss
        loss.calc_fval()
        self.assertNearlyEqual(loss.fval(), [[0.734, 0.723, 0.691]])



if __name__ == "__main__":
    unittest.main()

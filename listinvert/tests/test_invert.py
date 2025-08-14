import unittest
from listinvert import invert

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

if __name__ == "__main__":
    unittest.main()

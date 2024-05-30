import unittest
from context import utils


class test_is_triangle(unittest.TestCase):

    def test_valid_symmetric(self):
        a, c, e = 100, 80, 60
        b, d, f = 20, 30, 40

        trng = utils.is_triangle(a, b, c, d, e, f, 5)

        self.assertEqual(trng, "Symmetric")

    def test_perfectly_valid_ascending(self):
        a, c, e = 100, 100, 100  # Straight upper line
        b, d, f = 20, 30, 40  # Ascending lower line

        trng = utils.is_triangle(a, b, c, d, e, f, 10)

        self.assertEqual(trng, "Ascending")

    def test_valid_ascending_with_variance(self):
        # Slightly descending upper line but still within average candle range
        a, c, e = (100, 98, 95)
        b, d, f = 20, 30, 40  # Ascending lower line

        trng = utils.is_triangle(a, b, c, d, e, f, 10)

        self.assertEqual(trng, "Ascending")

    def test_perfectly_valid_descending(self):
        a, c, e = 100, 80, 70  # Descending upper line
        b, d, f = 50, 50, 50  # Straight lower line

        trng = utils.is_triangle(a, b, c, d, e, f, 10)

        self.assertEqual(trng, "Descending")

    def test_valid_descending_with_variance(self):
        # Slightly decreasing upper line but still within average candle range
        a, c, e = 100, 80, 70
        b, d, f = 50, 47, 48

        trng = utils.is_triangle(a, b, c, d, e, f, 10)

        self.assertEqual(trng, "Descending")

    def test_invalid_triangle(self):
        a, c, e = 100, 120, 130
        b, d, f = 20, 30, 40
        trng = utils.is_triangle(a, b, c, d, e, f, 10)

        self.assertEqual(trng, None)


if __name__ == "__main__":
    unittest.main()
